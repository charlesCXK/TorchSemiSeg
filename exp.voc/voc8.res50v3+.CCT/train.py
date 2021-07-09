from __future__ import division
import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from config import config
from dataloader import get_train_loader
from network import Network
from dataloader import VOC
from utils.init_func import init_weight, group_weight
from engine.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
# from seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d
from tensorboardX import SummaryWriter
from unsupervised_head import *

try:
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")

try:
    from azureml.core import Run
    azure = True
    run = Run.get_context()
except:
    azure = False

parser = argparse.ArgumentParser()

os.environ['MASTER_PORT'] = '169711'

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True

    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader + unsupervised data loader
    train_loader, train_sampler = get_train_loader(engine, VOC, train_source=config.train_source, \
                                                   unsupervised=False)
    unsupervised_train_loader, unsupervised_train_sampler = get_train_loader(engine, VOC, \
                train_source=config.unsup_source, unsupervised=True)

    if engine.distributed and (engine.local_rank == 0):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        logger = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
    criterion = abCE_loss(iters_per_epoch=config.niters_per_epoch, epochs=config.nepochs,
                                num_classes=config.num_classes)

    if engine.distributed:
        BatchNorm2d = SyncBatchNorm

    model = Network(config.num_classes, criterion=criterion,
                    pretrained_model=config.pretrained_model,
                    norm_layer=BatchNorm2d)
    init_weight(model.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr * engine.world_size

    params_list = []
    params_list = group_weight(params_list, model.backbone,
                               BatchNorm2d, base_lr)
    for module in model.business_layer:
        params_list = group_weight(params_list, module, BatchNorm2d,
                                   base_lr)

    optimizer = torch.optim.SGD(params_list,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        print('distributed !!')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DataParallelModel(model, device_ids=engine.devices)
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    model.train()
    print('begin train')

    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)

        dataloader = iter(train_loader)
        unsupervised_dataloader = iter(unsupervised_train_loader)

        sum_loss = 0
        sum_loss_unsup = 0

        for idx in pbar:
            optimizer.zero_grad()
            engine.update_iteration(epoch, idx)

            minibatch = dataloader.next()
            unsupervised_minibatch = unsupervised_dataloader.next()

            imgs = minibatch['data']
            gts = minibatch['label']
            unsupervised_imgs = unsupervised_minibatch['data']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            unsupervised_imgs = unsupervised_imgs.cuda(non_blocking=True)

            loss_sup, loss_unsup = model(imgs, unsupervised_imgs, gts, curr_iter=idx+1, epoch=epoch)

            # reduce the whole loss over multi-gpu
            dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
            loss_sup = loss_sup / engine.world_size
            dist.all_reduce(loss_unsup, dist.ReduceOp.SUM)
            loss_unsup = loss_unsup / engine.world_size


            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            unsup_weight = config.unsup_weight
            loss_unsup = loss_unsup * unsup_weight
            tot_loss = loss_sup + loss_unsup      # sup loss + unsup loss
            tot_loss.backward()
            optimizer.step()

            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.2f' % tot_loss.item() \
                        + ' loss_unsup=%.2f' % loss_unsup.item()

            sum_loss += tot_loss.item()
            sum_loss_unsup += loss_unsup.item()

            pbar.set_description(print_str, refresh=False)

        if engine.distributed and (engine.local_rank == 0):
            logger.add_scalar('train_loss', sum_loss / len(pbar), epoch)
            logger.add_scalar('train_loss_unsup', sum_loss_unsup / len(pbar), epoch)

            if azure:
                run.log(name='Training Loss', value=sum_loss / len(pbar))
                run.log(name='Unsupervised Training Loss', value=sum_loss_unsup / len(pbar))

        if (epoch > config.nepochs // 6) and (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
