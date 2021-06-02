#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2018/8/2 下午3:23
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : engine.py
import os
import os.path as osp
import time
import argparse

import shutil
import torch
import torch.distributed as dist

from .logger import get_logger
from .version import __version__
from utils.pyt_utils import load_model, parse_devices, extant_file, link_file, \
    ensure_dir

logger = get_logger()


class State(object):
    def __init__(self):
        self.epoch = 0
        self.iteration = 0
        self.dataloader = None
        self.model = None
        self.optimizer = None
        self.optimizer_l = None
        self.optimizer_r = None

    def register(self, **kwargs):
        for k, v in kwargs.items():
            # assert k in ['epoch', 'iteration', 'dataloader', 'model',
            #              'optimizer']
            setattr(self, k, v)


class Engine(object):
    def __init__(self, custom_parser=None):
        self.version = __version__
        logger.info(
            "PyTorch Version {}, Furnace Version {}".format(torch.__version__,
                                                            self.version))
        self.state = State()
        self.devices = None
        self.distributed = False

        if custom_parser is None:
            self.parser = argparse.ArgumentParser()
        else:
            assert isinstance(custom_parser, argparse.ArgumentParser)
            self.parser = custom_parser

        self.inject_default_parser()
        self.args = self.parser.parse_args()

        if self.args.continue_fpath is not None and os.path.exists(self.args.continue_fpath):
            self.continue_state_object = self.args.continue_fpath
        else:
            self.continue_state_object = None
        print('continue_state_object: ', self.continue_state_object)

        if 'WORLD_SIZE' in os.environ:
            self.distributed = int(os.environ['WORLD_SIZE']) >= 1

        if self.distributed:
            self.local_rank = self.args.local_rank
            self.world_size = int(os.environ['WORLD_SIZE'])
            torch.cuda.set_device(self.local_rank)
            os.environ['MASTER_PORT'] = self.args.port
            dist.init_process_group(backend="nccl", init_method='env://')
            self.devices = [i for i in range(self.world_size)]
        else:
            self.devices = parse_devices(self.args.devices)

    def inject_default_parser(self):
        p = self.parser
        p.add_argument('-d', '--devices', default='',
                       help='set data parallel training')
        # p.add_argument('-c', '--continue', type=extant_file,
        #                metavar="FILE",
        #                dest="continue_fpath",
        #                help='continue from one certain checkpoint')
        p.add_argument('-c', '--continue', type=str,
                       dest="continue_fpath",
                       help='continue from one certain checkpoint')
        p.add_argument('--local_rank', default=0, type=int,
                       help='process rank on node')
        p.add_argument('-p', '--port', type=str,
                       default='16001',
                       dest="port",
                       help='port for init_process_group')
        p.add_argument('--debug', default=0, type=int,
                       help='whether to use the debug mode')

    def register_state(self, **kwargs):
        self.state.register(**kwargs)

    def update_iteration(self, epoch, iteration):
        self.state.epoch = epoch
        self.state.iteration = iteration

    def save_checkpoint(self, path):
        logger.info("Saving checkpoint to file {}".format(path))
        t_start = time.time()

        state_dict = {}

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in self.state.model.state_dict().items():
            key = k
            if k.split('.')[0] == 'module':
                key = k[7:]
            new_state_dict[key] = v
        state_dict['model'] = new_state_dict
        if self.state.optimizer is not None:
            state_dict['optimizer'] = self.state.optimizer.state_dict()
        if self.state.optimizer_l is not None:
            state_dict['optimizer_l'] = self.state.optimizer_l.state_dict()
        if self.state.optimizer_r is not None:
            state_dict['optimizer_r'] = self.state.optimizer_r.state_dict()
        state_dict['epoch'] = self.state.epoch
        state_dict['iteration'] = self.state.iteration

        t_iobegin = time.time()
        torch.save(state_dict, path)
        del state_dict
        del new_state_dict
        t_end = time.time()
        logger.info(
            "Save checkpoint to file {}, "
            "Time usage:\n\tprepare snapshot: {}, IO: {}".format(
                path, t_iobegin - t_start, t_end - t_iobegin))

    def link_tb(self, source, target):
        ensure_dir(source)
        ensure_dir(target)
        link_file(source, target)


    def save_and_link_checkpoint(self, snapshot_dir, log_dir, log_dir_link, name=None):
        ensure_dir(snapshot_dir)
        if not osp.exists(log_dir_link):
            link_file(log_dir, log_dir_link)
        if name is None:
            current_epoch_checkpoint = osp.join(snapshot_dir, 'epoch-{}.pth'.format(
                self.state.epoch))
        else:
            current_epoch_checkpoint = osp.join(snapshot_dir, '{}.pth'.format(
                name))

        ''' 如果旧文件存在，先删除 '''
        if os.path.exists(current_epoch_checkpoint):
            os.remove(current_epoch_checkpoint)

        self.save_checkpoint(current_epoch_checkpoint)
        last_epoch_checkpoint = osp.join(snapshot_dir,
                                         'epoch-last.pth')
        # link_file(current_epoch_checkpoint, last_epoch_checkpoint)
        try:
            shutil.copy(current_epoch_checkpoint, last_epoch_checkpoint)
        except:
            pass

    def restore_checkpoint(self):
        t_start = time.time()
        if self.distributed:
            tmp = torch.load(self.continue_state_object,
                             map_location=lambda storage, loc: storage.cuda(
                                 self.local_rank))
        else:
            tmp = torch.load(self.continue_state_object)
        t_ioend = time.time()

        self.state.model = load_model(self.state.model, tmp['model'],
                                      True)
        if 'optimizer_l' in tmp:
            self.state.optimizer_l.load_state_dict(tmp['optimizer_l'])
        if 'optimizer_r' in tmp:
            self.state.optimizer_r.load_state_dict(tmp['optimizer_r'])
        if 'optimizer' in tmp:
            self.state.optimizer.load_state_dict(tmp['optimizer'])
        self.state.epoch = tmp['epoch'] + 1
        self.state.iteration = tmp['iteration']
        del tmp
        t_end = time.time()
        logger.info(
            "Load checkpoint from file {}, "
            "Time usage:\n\tIO: {}, restore snapshot: {}".format(
                self.continue_state_object, t_ioend - t_start, t_end - t_ioend))

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            logger.warning(
                "A exception occurred during Engine initialization, "
                "give up running process")
            return False
