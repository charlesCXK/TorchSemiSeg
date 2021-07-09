# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from collections import OrderedDict
from config import config
from base_model import resnet50
from gct_util import *

class Network(nn.Module):
    def __init__(self, num_classes, criterion, norm_layer, pretrained_model=None):
        super(Network, self).__init__()
        self.l_model = SingleNetwork(num_classes, criterion, norm_layer, pretrained_model)
        self.r_model = SingleNetwork(num_classes, criterion, norm_layer, pretrained_model)
        self.fd_model = FlawDetector(config.num_classes+3, norm_layer)
        self.zero_df_gt = torch.zeros([config.batch_size, 1, config.image_height, config.image_width]).cuda()

        self.flawmap_handler = FlawmapHandler().cuda()
        self.dcgt_generator = DCGTGenerator().cuda()
        self.fdgt_generator = FDGTGenerator().cuda()

    def forward(self, l_inp, r_inp=None, l_gt=None, r_gt=None, current_idx=None, total_steps=None, step=1):
        if not self.training:
            l_pred = self.l_model(l_inp)
            return l_pred
            # -----------------------------------------------------------------------------
            # step-0: pre-forwarding to save GPU memory
            #   - forward the task models and the flaw detector
            #   - generate pseudo ground truth for the unlabeled data if the dynamic
            #     consistency constraint is enabled
            # -----------------------------------------------------------------------------

        with torch.no_grad():
            b, c, h, w = l_inp.shape
            l_pred_1 = self.l_model(l_inp[:b//2])
            l_pred_2 = self.l_model(l_inp[b // 2:])
            l_pred = torch.cat([l_pred_1, l_pred_2], dim=0)
            l_activated_pred = F.softmax(l_pred, 1)

            r_pred_1 = self.r_model(r_inp[:b//2])
            r_pred_2 = self.r_model(r_inp[b // 2:])
            r_pred = torch.cat([r_pred_1, r_pred_2], dim=0)
            r_activated_pred = F.softmax(r_pred, 1)

            # 'l_flawmap' and 'r_flawmap' will be used in step-2
        l_flawmap = self.fd_model(l_inp, l_activated_pred)
        r_flawmap = self.fd_model(r_inp, r_activated_pred)

        with torch.no_grad():
            l_handled_flawmap = self.flawmap_handler(l_flawmap)
            r_handled_flawmap = self.flawmap_handler(r_flawmap)
            l_dc_gt, r_dc_gt, l_fc_mask, r_fc_mask = self.dcgt_generator(
                l_activated_pred.detach(), r_activated_pred.detach(), l_handled_flawmap, r_handled_flawmap)

        # -----------------------------------------------------------------------------
        # step-1: train the task models
        # -----------------------------------------------------------------------------
        if step == 1:
            ''' train the 'l' task model '''
            b, c, h, w = l_inp.shape
            l_pred_1 = self.l_model(l_inp[:b//2])
            l_pred_2 = self.l_model(l_inp[b // 2:])
            pred_l = torch.cat([l_pred_1, l_pred_2], dim=0)
            activated_pred_l = F.softmax(pred_l, dim=1)
            flawmap_l = self.fd_model(l_inp, activated_pred_l)

            ''' train the 'r' task model '''
            r_pred_1 = self.r_model(r_inp[:b//2])
            r_pred_2 = self.r_model(r_inp[b // 2:])
            pred_r = torch.cat([r_pred_1, r_pred_2], dim=0)
            activated_pred_r = F.softmax(pred_r, dim=1)
            flawmap_r = self.fd_model(r_inp, activated_pred_r)

            return pred_l, flawmap_l, l_fc_mask, l_dc_gt, pred_r, flawmap_r, r_fc_mask, r_dc_gt
        elif step == 2:
            # -----------------------------------------------------------------------------
            # step-2: train the flaw detector
            # -----------------------------------------------------------------------------
            # for param in model.module.fd_model.parameters():
            #     param.requires_grad = True

            # generate the ground truth for the flaw detector (on labeled data only)
            b, h, w = l_gt.shape
            lbs = b // 2
            with torch.no_grad():
                l_flawmap_gt = self.fdgt_generator(
                    l_activated_pred[:lbs, ...].detach(), sslgct_prepare_task_gt_for_fdgt(l_gt[:lbs, ...]))
                r_flawmap_gt = self.fdgt_generator(
                    r_activated_pred[:lbs, ...].detach(), sslgct_prepare_task_gt_for_fdgt(r_gt[:lbs, ...]))

            return l_flawmap, r_flawmap, l_flawmap_gt, r_flawmap_gt

class SingleNetwork(nn.Module):
    def __init__(self, num_classes, criterion, norm_layer, pretrained_model=None):
        super(SingleNetwork, self).__init__()
        self.backbone = resnet50(pretrained_model, norm_layer=norm_layer,
                                  bn_eps=config.bn_eps,
                                  bn_momentum=config.bn_momentum,
                                  deep_stem=True, stem_width=64)
        self.dilate = 2
        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))
            self.dilate *= 2

        self.head = Head(num_classes, norm_layer, config.bn_momentum)
        self.business_layer = []
        self.business_layer.append(self.head)
        self.criterion = criterion

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)
        self.business_layer.append(self.classifier)


    def forward(self, data):
        b, c, h, w = data.shape
        blocks = self.backbone(data)

        v3plus_feature = self.head(blocks)
        pred = self.classifier(v3plus_feature)
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            return pred
        return pred

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_rates=(12, 24, 36),
                 hidden_channels=256,
                 norm_act=nn.BatchNorm2d,
                 pooling_size=None):
        super(ASPP, self).__init__()
        self.pooling_size = pooling_size

        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[0],
                      padding=dilation_rates[0]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[1],
                      padding=dilation_rates[1]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[2],
                      padding=dilation_rates[2])
        ])
        self.map_bn = norm_act(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)

        self.leak_relu = nn.LeakyReLU()

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.leak_relu(out)       # add activation layer
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)

        pool = self.leak_relu(pool)  # add activation layer

        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))

        out += pool
        out = self.red_bn(out)
        out = self.leak_relu(out)  # add activation layer
        return out

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            pooling_size = (min(try_index(self.pooling_size, 0), x.shape[2]),
                            min(try_index(self.pooling_size, 1), x.shape[3]))
            padding = (
                (pooling_size[1] - 1) // 2,
                (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1,
                (pooling_size[0] - 1) // 2,
                (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1
            )

            pool = nn.functional.avg_pool2d(x, pooling_size, stride=1)
            pool = nn.functional.pad(pool, pad=padding, mode="replicate")
        return pool

class Head(nn.Module):
    def __init__(self, classify_classes, norm_act=nn.BatchNorm2d, bn_momentum=0.0003):
        super(Head, self).__init__()

        self.classify_classes = classify_classes
        self.aspp = ASPP(2048, 256, [6, 12, 18], norm_act=norm_act)

        self.reduce = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            norm_act(48, momentum=bn_momentum),
            nn.ReLU(),
        )
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       )

    def forward(self, f_list):
        f = f_list[-1]
        f = self.aspp(f)

        low_level_features = f_list[0]
        low_h, low_w = low_level_features.size(2), low_level_features.size(3)
        low_level_features = self.reduce(low_level_features)

        f = F.interpolate(f, size=(low_h, low_w), mode='bilinear', align_corners=True)
        f = torch.cat((f, low_level_features), dim=1)
        f = self.last_conv(f)

        return f


if __name__ == '__main__':
    model = Network(40, criterion=nn.CrossEntropyLoss(),
                    pretrained_model=None,
                    norm_layer=nn.BatchNorm2d)
    left = torch.randn(2, 3, 128, 128)
    right = torch.randn(2, 3, 128, 128)

    print(model.backbone)

    out = model(left)
    print(out.shape)
