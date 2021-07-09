from __future__ import division
import os.path as osp
import os
import sys
import time
import argparse
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
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

if os.getenv('debug') is not None:
    is_debug = os.environ['debug']
else:
    is_debug = False

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
    criterion_csst = nn.MSELoss(reduction='mean')

    if engine.distributed:
        BatchNorm2d = SyncBatchNorm

    model = Network(config.num_classes, criterion=criterion,
                    pretrained_model=config.pretrained_model,
                    norm_layer=BatchNorm2d)
    init_weight(model.branch1.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    init_weight(model.branch2.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr * engine.world_size

    params_list_l = []
    params_list_l = group_weight(params_list_l, model.branch1.backbone,
                               BatchNorm2d, base_lr)
    for module in model.branch1.business_layer:
        params_list_l = group_weight(params_list_l, module, BatchNorm2d,
                                   base_lr)        # head lr * 10

    optimizer_l = torch.optim.SGD(params_list_l,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    params_list_r = []
    params_list_r = group_weight(params_list_r, model.branch2.backbone,
                               BatchNorm2d, base_lr)
    for module in model.branch2.business_layer:
        params_list_r = group_weight(params_list_r, module, BatchNorm2d,
                                   base_lr)        # head lr * 10

    optimizer_r = torch.optim.SGD(params_list_r,
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
                          optimizer_l=optimizer_l, optimizer_r=optimizer_r)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    model.train()
    print('begin train')

    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'

        if is_debug:
            pbar = tqdm(range(10), file=sys.stdout, bar_format=bar_format)
        else:
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)


        dataloader = iter(train_loader)
        unsupervised_dataloader = iter(unsupervised_train_loader)

        sum_loss_sup = 0
        sum_loss_sup_t = 0
        sum_csst = 0

        ''' supervised part '''
        for idx in pbar:
            optimizer_l.zero_grad()
            optimizer_r.zero_grad()
            engine.update_iteration(epoch, idx)
            start_time = time.time()

            minibatch = dataloader.next()
            unsup_minibatch = unsupervised_dataloader.next()
            imgs = minibatch['data']
            gts = minibatch['label']
            unsup_imgs = unsup_minibatch['data']
            imgs = imgs.cuda(non_blocking=True)
            unsup_imgs = unsup_imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)

            cat_imgs = torch.cat([imgs, unsup_imgs], dim=0)

            current_idx = epoch * config.niters_per_epoch + idx
            s_sup_pred, t_sup_pred = model(imgs, step=1, cur_iter=current_idx)
            s_unsup_pred, t_unsup_pred = model(unsup_imgs, step=2, cur_iter=current_idx)
            s_pred = torch.cat([s_sup_pred, s_unsup_pred], dim=0)
            t_pred = torch.cat([t_sup_pred, t_unsup_pred], dim=0)

            softmax_pred_s = F.softmax(s_pred, 1)
            softmax_pred_t = F.softmax(t_pred, 1)

            ### Mean Teacher loss ###
            csst_loss = criterion_csst(softmax_pred_s, softmax_pred_t.detach())
            dist.all_reduce(csst_loss, dist.ReduceOp.SUM)
            csst_loss = csst_loss / engine.world_size
            csst_loss = csst_loss * config.unsup_weight

            ### Supervised loss For Student ###
            loss_sup = criterion(s_sup_pred, gts)
            dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
            loss_sup = loss_sup / engine.world_size
            
            ### Supervised loss For Teracher. No Backward ###
            loss_sup_t = criterion(t_sup_pred, gts)
            dist.all_reduce(loss_sup_t, dist.ReduceOp.SUM)
            loss_sup_t = loss_sup_t / engine.world_size

            unlabeled_loss = True

            lr = lr_policy.get_lr(current_idx)

            optimizer_l.param_groups[0]['lr'] = lr
            optimizer_l.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_l.param_groups)):
                optimizer_l.param_groups[i]['lr'] = lr

            loss = loss_sup + csst_loss
            loss.backward()
            optimizer_l.step()      # only the student model need to be updated by SGD.

            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss_sup=%.2f' % loss_sup.item() \
                        + ' loss_sup_t=%.2f' % loss_sup_t.item() \
                        + ' loss_csst=%.4f' % csst_loss.item()

            sum_loss_sup += loss_sup.item()
            sum_loss_sup_t += loss_sup_t.item()
            sum_csst += csst_loss.item()
            pbar.set_description(print_str, refresh=False)

            end_time = time.time()

        if engine.distributed and (engine.local_rank == 0):
            logger.add_scalar('train_loss_sup', sum_loss_sup / len(pbar), epoch)
            logger.add_scalar('train_loss_sup_t', sum_loss_sup_t / len(pbar), epoch)
            logger.add_scalar('train_loss_csst', sum_csst / len(pbar), epoch)

        if azure and engine.local_rank == 0:
            run.log(name='Supervised Training Loss', value=sum_loss_sup / len(pbar))
            run.log(name='Supervised Training Loss Teacher', value=sum_loss_sup_t / len(pbar))
            run.log(name='Supervised Training Loss CSST', value=sum_csst / len(pbar))

        if (epoch > config.nepochs // 2) and (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
