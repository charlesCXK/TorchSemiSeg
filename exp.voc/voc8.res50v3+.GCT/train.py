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
from gct_util import sigmoid_rampup, FlawDetectorCriterion

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

    # define and init the model
    model = Network(config.num_classes, criterion=criterion,
                    pretrained_model=config.pretrained_model,
                    norm_layer=BatchNorm2d)
    init_weight(model.l_model.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    init_weight(model.r_model.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    # define the learning rate
    base_lr = config.lr
    fd_lr = config.fd_lr
    if engine.distributed:
        base_lr = config.lr * engine.world_size
        fd_lr = config.fd_lr * engine.world_size

    # define the optimizers
    params_list_l = []
    params_list_l = group_weight(params_list_l, model.l_model.backbone,
                               BatchNorm2d, base_lr)
    for module in model.l_model.business_layer:
        params_list_l = group_weight(params_list_l, module, BatchNorm2d,
                                   base_lr)        # head lr * 10

    optimizer_l = torch.optim.SGD(params_list_l,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    params_list_r = []
    params_list_r = group_weight(params_list_r, model.r_model.backbone,
                               BatchNorm2d, base_lr)
    for module in model.r_model.business_layer:
        params_list_r = group_weight(params_list_r, module, BatchNorm2d,
                                   base_lr)        # head lr * 10

    optimizer_r = torch.optim.SGD(params_list_r,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    fd_optimizer = torch.optim.Adam(model.fd_model.parameters(),
                                    lr=fd_lr, betas=(0.9, 0.99))

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)
    lr_policy_fd = WarmUpPolyLR(fd_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

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
        model.train()

        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'

        if is_debug:
            pbar = tqdm(range(10), file=sys.stdout, bar_format=bar_format)
        else:
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)

        dataloader = iter(train_loader)
        unsupervised_dataloader = iter(unsupervised_train_loader)

        sum_loss_l = 0
        sum_loss_r = 0
        sum_loss_fd = 0

        ''' supervised part '''
        for idx in pbar:
            optimizer_l.zero_grad()
            optimizer_r.zero_grad()
            fd_optimizer.zero_grad()

            engine.update_iteration(epoch, idx)

            minibatch = dataloader.next()
            unsupervised_minibatch = unsupervised_dataloader.next()
            imgs = minibatch['data']
            gts = minibatch['label']
            imgs_unlabeled = unsupervised_minibatch['data']
            gts_unlabeled = (gts * 0 + 255).long()  # set the gt of unlabeled data to be 255

            imgs = imgs.cuda(non_blocking=True)
            imgs_unlabeled = imgs_unlabeled.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            gts_unlabeled = gts_unlabeled.cuda(non_blocking=True)

            l_inp = torch.cat([imgs, imgs_unlabeled], dim=0)
            l_gt = torch.cat([gts, gts_unlabeled], dim=0)

            r_inp = torch.cat([imgs, imgs_unlabeled], dim=0)
            r_gt = torch.cat([gts, gts_unlabeled], dim=0)

            current_idx = epoch * config.niters_per_epoch + idx
            total_steps = config.dc_rampup_epochs * config.niters_per_epoch
            dc_rampup_scale = sigmoid_rampup(current_idx, total_steps)

            # -----------------------------------------------------------------------------
            # step-1: train the task models
            # -----------------------------------------------------------------------------
            for param in model.module.fd_model.parameters():
                param.requires_grad = False

            pred_l, flawmap_l, l_fc_mask, l_dc_gt, pred_r, flawmap_r, r_fc_mask, r_dc_gt = model(l_inp, r_inp, l_gt,
                                                                                                 r_gt, current_idx,
                                                                                                 total_steps, step=1)

            # task loss
            fd_criterion = FlawDetectorCriterion()
            dc_criterion = torch.nn.MSELoss()

            b, c, h, w = pred_l.shape
            task_loss_l = criterion(pred_l[:b // 2], l_gt[:b // 2])
            dist.all_reduce(task_loss_l, dist.ReduceOp.SUM)
            task_loss_l = task_loss_l / engine.world_size

            fc_ssl_loss_l = fd_criterion(flawmap_l, torch.zeros(flawmap_l.shape).cuda(), is_ssl=True,
                                         reduction=False)
            fc_ssl_loss_l = l_fc_mask * fc_ssl_loss_l
            fc_ssl_loss_l = config.fc_ssl_scale * torch.mean(fc_ssl_loss_l)
            dist.all_reduce(fc_ssl_loss_l, dist.ReduceOp.SUM)
            fc_ssl_loss_l = fc_ssl_loss_l / engine.world_size

            dc_ssl_loss_l = dc_criterion(F.softmax(pred_l, dim=1), l_dc_gt)
            dc_ssl_loss_l = dc_rampup_scale * config.dc_ssl_scale * torch.mean(dc_ssl_loss_l)
            dist.all_reduce(dc_ssl_loss_l, dist.ReduceOp.SUM)
            dc_ssl_loss_l = dc_ssl_loss_l / engine.world_size

            loss_l = task_loss_l + fc_ssl_loss_l + dc_ssl_loss_l

            ''' train the 'r' task model '''
            b, c, h, w = pred_r.shape
            task_ross_r = criterion(pred_r[:b // 2], r_gt[:b // 2])
            dist.all_reduce(task_ross_r, dist.ReduceOp.SUM)
            task_ross_r = task_ross_r / engine.world_size

            fc_ssl_loss_r = fd_criterion(flawmap_r, torch.zeros(flawmap_r.shape).cuda(), is_ssl=True,
                                         reduction=False)
            fc_ssl_loss_r = r_fc_mask * fc_ssl_loss_r
            fc_ssl_loss_r = config.fc_ssl_scale * torch.mean(fc_ssl_loss_r)
            dist.all_reduce(fc_ssl_loss_r, dist.ReduceOp.SUM)
            fc_ssl_loss_r = fc_ssl_loss_r / engine.world_size

            dc_ssl_loss_r = dc_criterion(F.softmax(pred_r, dim=1), r_dc_gt)
            dc_ssl_loss_r = dc_rampup_scale * config.dc_ssl_scale * torch.mean(dc_ssl_loss_r)
            dist.all_reduce(dc_ssl_loss_r, dist.ReduceOp.SUM)
            dc_ssl_loss_r = dc_ssl_loss_r / engine.world_size

            loss_r = task_ross_r + fc_ssl_loss_r + dc_ssl_loss_r

            loss_task = loss_l + loss_r
            loss_task.backward()
            optimizer_l.step()
            optimizer_r.step()

            # -----------------------------------------------------------------------------
            # step-2: train the flaw detector
            # -----------------------------------------------------------------------------
            for param in model.module.fd_model.parameters():
                param.requires_grad = True

            l_flawmap, r_flawmap, l_flawmap_gt, r_flawmap_gt = model(l_inp, r_inp, l_gt, r_gt, current_idx, total_steps,
                                                                     step=2)

            # generate the ground truth for the flaw detector (on labeled data only)
            lbs = b // 2
            l_fd_loss = fd_criterion(l_flawmap[:lbs, ...], l_flawmap_gt)
            l_fd_loss = config.fd_scale * torch.mean(l_fd_loss)

            r_fd_loss = fd_criterion(r_flawmap[:lbs, ...], r_flawmap_gt)
            r_fd_loss = config.fd_scale * torch.mean(r_fd_loss)

            fd_loss = (l_fd_loss + r_fd_loss) / 2
            dist.all_reduce(fd_loss, dist.ReduceOp.SUM)
            fd_loss = fd_loss / engine.world_size

            fd_loss.backward()
            fd_optimizer.step()

            lr = lr_policy.get_lr(current_idx)
            fd_lr = lr_policy_fd.get_lr(current_idx)

            optimizer_l.param_groups[0]['lr'] = lr
            optimizer_l.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_l.param_groups)):
                optimizer_l.param_groups[i]['lr'] = lr

            optimizer_r.param_groups[0]['lr'] = lr
            optimizer_r.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_r.param_groups)):
                optimizer_r.param_groups[i]['lr'] = lr

            sum_loss_l += loss_l.item()
            sum_loss_r += loss_r.item()
            sum_loss_fd += fd_loss.item()

            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss_l=%.2f' % loss_l.item() \
                        + ' loss_r=%.2f' % loss_r.item() \
                        + ' loss_fd=%.2f' % fd_loss.item()

            pbar.set_description(print_str, refresh=False)

        if engine.distributed and (engine.local_rank == 0):
            logger.add_scalar('train_loss_l', sum_loss_l / len(pbar), epoch)
            logger.add_scalar('train_loss_r', sum_loss_r / len(pbar), epoch)
            logger.add_scalar('train_loss_fd', sum_loss_fd / len(pbar), epoch)

        if azure and engine.local_rank == 0:
            run.log(name='train_loss_l', value=sum_loss_l / len(pbar))
            run.log(name='train_loss_r', value=sum_loss_r / len(pbar))
            run.log(name='train_loss_fd', value=sum_loss_fd / len(pbar))

        if (epoch > config.nepochs // 2) and (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)