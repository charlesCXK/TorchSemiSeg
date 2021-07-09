import math, time
import torch
import torch.nn.functional as F
from torch import nn
from itertools import chain
import contextlib
import random
import numpy as np
import cv2
import ramps
from torch.distributions.uniform import Uniform

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    """
    Checkerboard artifact free sub-pixel convolution
    https://arxiv.org/abs/1707.02937
    """
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale ** 2))
    k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale ** 2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)


class PixelShuffle(nn.Module):
    """
    Real-Time Single Image and Video Super-Resolution
    https://arxiv.org/abs/1609.05158
    """

    def __init__(self, n_channels, scale):
        super(PixelShuffle, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels * (scale ** 2), kernel_size=1)
        icnr(self.conv.weight)
        self.shuf = nn.PixelShuffle(scale)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return x

class upsample(nn.Module):
    def __init__(self, in_channels, out_channels, upscale, norm_layer):
        super(upsample, self).__init__()
        self.classifier = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.upscale = upscale

    def forward(self, x):
        pred = self.classifier(x)
        return F.interpolate(pred, scale_factor=self.upscale, mode='bilinear', align_corners=True)


class MainDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, norm_layer):
        super(MainDecoder, self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, upscale, norm_layer=norm_layer)

    def forward(self, x):
        x = self.upsample(x)
        return x


class DropOutDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, drop_rate=0.3, spatial_dropout=True, norm_layer=None):
        super(DropOutDecoder, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)
        self.upsample = upsample(conv_in_ch, num_classes, upscale, norm_layer)

    def forward(self, x, _):
        x = self.upsample(self.dropout(x))
        return x


class FeatureDropDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, norm_layer):
        super(FeatureDropDecoder, self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, upscale, norm_layer)

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, x, _):
        x = self.feature_dropout(x)
        x = self.upsample(x)
        return x


class FeatureNoiseDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, uniform_range=0.3, norm_layer=None):
        super(FeatureNoiseDecoder, self).__init__()
        self.upsample = upsample(conv_in_ch, num_classes, upscale, norm_layer)
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x, _):
        x = self.feature_based_noise(x)
        x = self.upsample(x)
        return x


def _l2_normalize(d):
    # Normalizing per batch axis
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def get_r_adv(x, decoder, it=1, xi=1e-1, eps=10.0):
    """
    Virtual Adversarial Training
    https://arxiv.org/abs/1704.03976
    """
    x_detached = x.detach()
    with torch.no_grad():
        pred = F.softmax(decoder(x_detached), dim=1)

    d = torch.rand(x.shape).sub(0.5).to(x.device)
    d = _l2_normalize(d)

    for _ in range(it):
        d.requires_grad_()
        pred_hat = decoder(x_detached + xi * d)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
        adv_distance.backward()
        d = _l2_normalize(d.grad)
        decoder.zero_grad()

    r_adv = d * eps
    return r_adv


class VATDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, xi=1e-1, eps=10.0, iterations=1):
        super(VATDecoder, self).__init__()
        self.xi = xi
        self.eps = eps
        self.it = iterations
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x, _):
        r_adv = get_r_adv(x, self.upsample, self.it, self.xi, self.eps)
        x = self.upsample(x + r_adv)
        return x


def guided_cutout(output, upscale, resize, erase=0.4, use_dropout=False, norm_layer=None):
    if len(output.shape) == 3:
        masks = (output > 0).float()
    else:
        masks = (output.argmax(1) > 0).float()

    if use_dropout:
        p_drop = random.randint(3, 6) / 10
        maskdroped = (F.dropout(masks, p_drop) > 0).float()
        maskdroped = maskdroped + (1 - masks)
        maskdroped.unsqueeze_(0)
        maskdroped = F.interpolate(maskdroped, size=resize, mode='nearest')

    masks_np = []
    for mask in masks:
        mask_np = np.uint8(mask.cpu().numpy())
        mask_ones = np.ones_like(mask_np)
        try:  # Version 3.x
            _, contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except:  # Version 4.x
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polys = [c.reshape(c.shape[0], c.shape[-1]) for c in contours if c.shape[0] > 50]
        for poly in polys:
            min_w, max_w = poly[:, 0].min(), poly[:, 0].max()
            min_h, max_h = poly[:, 1].min(), poly[:, 1].max()
            bb_w, bb_h = max_w - min_w, max_h - min_h
            rnd_start_w = random.randint(0, int(bb_w * (1 - erase)))
            rnd_start_h = random.randint(0, int(bb_h * (1 - erase)))
            h_start, h_end = min_h + rnd_start_h, min_h + rnd_start_h + int(bb_h * erase)
            w_start, w_end = min_w + rnd_start_w, min_w + rnd_start_w + int(bb_w * erase)
            mask_ones[h_start:h_end, w_start:w_end] = 0
        masks_np.append(mask_ones)
    masks_np = np.stack(masks_np)

    maskcut = torch.from_numpy(masks_np).float().unsqueeze_(1)
    maskcut = F.interpolate(maskcut, size=resize, mode='nearest')

    if use_dropout:
        return maskcut.to(output.device), maskdroped.to(output.device)
    return maskcut.to(output.device)


class CutOutDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, drop_rate=0.3, spatial_dropout=True, erase=0.4, norm_layer=None):
        super(CutOutDecoder, self).__init__()
        self.erase = erase
        self.upscale = upscale
        self.upsample = upsample(conv_in_ch, num_classes, upscale, norm_layer)

    def forward(self, x, pred=None):
        maskcut = guided_cutout(pred, upscale=self.upscale, erase=self.erase, resize=(x.size(2), x.size(3)))
        x = x * maskcut
        x = self.upsample(x)
        return x


def guided_masking(x, output, upscale, resize, return_msk_context=True, norm_layer=None):
    if len(output.shape) == 3:
        masks_context = (output > 0).float().unsqueeze(1)
    else:
        masks_context = (output.argmax(1) > 0).float().unsqueeze(1)

    masks_context = F.interpolate(masks_context, size=resize, mode='nearest')

    x_masked_context = masks_context * x
    if return_msk_context:
        return x_masked_context

    masks_objects = (1 - masks_context)
    x_masked_objects = masks_objects * x
    return x_masked_objects


class ContextMaskingDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, norm_layer):
        super(ContextMaskingDecoder, self).__init__()
        self.upscale = upscale
        self.upsample = upsample(conv_in_ch, num_classes, upscale, norm_layer)

    def forward(self, x, pred=None):
        x_masked_context = guided_masking(x, pred, resize=(x.size(2), x.size(3)),
                                          upscale=self.upscale, return_msk_context=True)
        x_masked_context = self.upsample(x_masked_context)
        return x_masked_context


class ObjectMaskingDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, norm_layer):
        super(ObjectMaskingDecoder, self).__init__()
        self.upscale = upscale
        self.upsample = upsample(conv_in_ch, num_classes, upscale, norm_layer)

    def forward(self, x, pred=None):
        x_masked_obj = guided_masking(x, pred, resize=(x.size(2), x.size(3)),
                                      upscale=self.upscale, return_msk_context=False)
        x_masked_obj = self.upsample(x_masked_obj)

        return x_masked_obj


def softmax_mse_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    inputs = F.softmax(inputs, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)

    if conf_mask:
        loss_mat = F.mse_loss(inputs, targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.mean()
    else:
        return F.mse_loss(inputs, targets, reduction='mean')


def softmax_kl_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    input_log_softmax = F.log_softmax(inputs, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)

    if conf_mask:
        loss_mat = F.kl_div(input_log_softmax, targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.sum() / mask.shape.numel()
    else:
        return F.kl_div(input_log_softmax, targets, reduction='mean')


def softmax_js_loss(inputs, targets, **_):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    epsilon = 1e-5

    M = (F.softmax(inputs, dim=1) + targets) * 0.5
    kl1 = F.kl_div(F.log_softmax(inputs, dim=1), M, reduction='mean')
    kl2 = F.kl_div(torch.log(targets + epsilon), M, reduction='mean')
    return (kl1 + kl2) * 0.5


class abCE_loss(nn.Module):
    """
    Annealed-Bootstrapped cross-entropy loss
    """
    def __init__(self, iters_per_epoch, epochs, num_classes, weight=None,
                 reduction='mean', thresh=0.7, min_kept=1, ramp_type='log_rampup'):
        super(abCE_loss, self).__init__()
        self.weight = torch.FloatTensor(weight) if weight is not None else weight
        self.reduction = reduction
        self.thresh = thresh
        self.min_kept = min_kept
        self.ramp_type = ramp_type

        if ramp_type is not None:
            self.rampup_func = getattr(ramps, ramp_type)
            self.iters_per_epoch = iters_per_epoch
            self.num_classes = num_classes
            self.start = 1 / num_classes
            self.end = 0.99
            self.total_num_iters = (epochs - (0.6 * epochs)) * iters_per_epoch

    def threshold(self, curr_iter, epoch):
        cur_total_iter = self.iters_per_epoch * epoch + curr_iter
        current_rampup = self.rampup_func(cur_total_iter, self.total_num_iters)
        return current_rampup * (self.end - self.start) + self.start

    def forward(self, predict, target, ignore_index, curr_iter, epoch):
        batch_kept = self.min_kept * target.size(0)
        prob_out = F.softmax(predict, dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == ignore_index] = 0
        prob = prob_out.gather(1, tmp_target.unsqueeze(1))
        mask = target.contiguous().view(-1, ) != ignore_index
        sort_prob, sort_indices = prob.contiguous().view(-1, )[mask].contiguous().sort()

        if self.ramp_type is not None:
            thresh = self.threshold(curr_iter=curr_iter, epoch=epoch)
        else:
            thresh = self.thresh

        min_threshold = sort_prob[min(batch_kept, sort_prob.numel() - 1)] if sort_prob.numel() > 0 else 0.0
        threshold = max(min_threshold, thresh)
        loss_matrix = F.cross_entropy(predict, target,
                                      weight=self.weight.to(predict.device) if self.weight is not None else None,
                                      ignore_index=ignore_index, reduction='none')
        loss_matirx = loss_matrix.contiguous().view(-1, )
        sort_loss_matirx = loss_matirx[mask][sort_indices]
        select_loss_matrix = sort_loss_matirx[sort_prob < threshold]
        if self.reduction == 'sum' or select_loss_matrix.numel() == 0:
            return select_loss_matrix.sum()
        elif self.reduction == 'mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')

class consistency_weight(object):
    """
    ramp_types = ['sigmoid_rampup', 'linear_rampup', 'cosine_rampup', 'log_rampup', 'exp_rampup']
    """
    def __init__(self, final_w, iters_per_epoch, rampup_starts=0, rampup_ends=7, ramp_type='sigmoid_rampup'):
        self.final_w = final_w
        self.iters_per_epoch = iters_per_epoch
        self.rampup_starts = rampup_starts * iters_per_epoch
        self.rampup_ends = rampup_ends * iters_per_epoch
        self.rampup_length = (self.rampup_ends - self.rampup_starts)
        self.rampup_func = getattr(ramps, ramp_type)
        self.current_rampup = 0

    def __call__(self, epoch, curr_iter):
        cur_total_iter = self.iters_per_epoch * epoch + curr_iter
        if cur_total_iter < self.rampup_starts:
            return 0
        self.current_rampup = self.rampup_func(cur_total_iter - self.rampup_starts, self.rampup_length)
        return self.final_w * self.current_rampup