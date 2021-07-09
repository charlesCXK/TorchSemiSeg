# encoding: utf-8
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from collections import OrderedDict
from config import config
from base_model import resnet50
from unsupervised_head import *

class Network(nn.Module):
    def __init__(self, num_classes, criterion, norm_layer, pretrained_model=None):
        super(Network, self).__init__()
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

        # read the decoder config file
        unsup_config = json.load(open('unsup.json'))
        conf = unsup_config['model']

        if conf['un_loss'] == "KL":
            self.unsuper_loss = softmax_kl_loss
        elif conf['un_loss'] == "MSE":
            self.unsuper_loss = softmax_mse_loss
        elif conf['un_loss'] == "JS":
            self.unsuper_loss = softmax_js_loss

        self.confidence_th = conf['confidence_th']
        self.confidence_masking = conf['confidence_masking']

        upscale = 4     # v3+ only need 4
        num_out_ch = 256
        decoder_in_ch = num_out_ch

        self.main_decoder = MainDecoder(upscale, decoder_in_ch, num_classes=num_classes,
                                        norm_layer=norm_layer)  # pixel shuffle upsample
        self.business_layer.append(self.main_decoder)

        # The auxilary decoders
        drop_decoder = [DropOutDecoder(upscale, decoder_in_ch, num_classes,
                                       drop_rate=conf['drop_rate'], spatial_dropout=conf['spatial'], norm_layer=norm_layer)
                        for _ in range(conf['drop'])]
        cut_decoder = [CutOutDecoder(upscale, decoder_in_ch, num_classes, erase=conf['erase'], norm_layer=norm_layer)
                       for _ in range(conf['cutout'])]
        context_m_decoder = [ContextMaskingDecoder(upscale, decoder_in_ch, num_classes, norm_layer=norm_layer)
                             for _ in range(conf['context_masking'])]
        object_masking = [ObjectMaskingDecoder(upscale, decoder_in_ch, num_classes, norm_layer=norm_layer)
                          for _ in range(conf['object_masking'])]
        feature_drop = [FeatureDropDecoder(upscale, decoder_in_ch, num_classes, norm_layer=norm_layer)
                        for _ in range(conf['feature_drop'])]
        feature_noise = [FeatureNoiseDecoder(upscale, decoder_in_ch, num_classes,
                                             uniform_range=conf['uniform_range'], norm_layer=norm_layer)
                         for _ in range(conf['feature_noise'])]

        self.aux_decoders = nn.ModuleList([*drop_decoder, *cut_decoder,
                                *context_m_decoder, *object_masking, *feature_drop, *feature_noise])

        self.business_layer.append(self.aux_decoders)

    def forward(self, data, unsupervised_data=None, label=None, curr_iter=None, epoch=None):
        b, c, h, w = data.shape
        blocks = self.backbone(data)

        encoder_sup = self.head(blocks)

        if not self.training:
            return self.main_decoder(encoder_sup)

        # encoder output from labeled data
        output_sup = self.main_decoder(encoder_sup)
        loss_sup = self.criterion(output_sup, label, curr_iter=curr_iter, epoch=epoch,
                                 ignore_index=255)

        ''' Below is semi loss '''
        encoder_unsup = self.head(self.backbone(unsupervised_data))
        output_unsup_main = self.main_decoder(encoder_unsup)
        # output of aux decoders
        outputs_unsup = [aux_decoder(encoder_unsup, output_unsup_main.detach()) for aux_decoder in self.aux_decoders]
        targets = F.softmax(output_unsup_main.detach(), dim=1)

        # Compute unsupervised loss
        loss_unsup = sum([self.unsuper_loss(inputs=u, targets=targets, \
                                            conf_mask=self.confidence_masking, threshold=self.confidence_th,
                                            use_softmax=False) for u in outputs_unsup])
        loss_unsup = (loss_unsup / len(outputs_unsup))

        if label is not None:
            return loss_sup, loss_unsup

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