import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import scipy.ndimage
from config import config

class FlawDetector(nn.Module):
    """ The FC Discriminator proposed in paper:
        'Guided Collaborative Training for Pixel-wise Semi-Supervised Learning'
    """

    ndf = 64  # basic number of channels

    def __init__(self, in_channels, norm_layer=None):
        super(FlawDetector, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, self.ndf, kernel_size=4, stride=2, padding=1)
        self.ibn1 = IBNorm(self.ndf, norm_layer=norm_layer)
        self.conv2 = nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=4, stride=2, padding=1)
        self.ibn2 = IBNorm(self.ndf * 2, norm_layer=norm_layer)
        self.conv2_1 = nn.Conv2d(self.ndf * 2, self.ndf * 2, kernel_size=4, stride=1, padding=1)
        self.ibn2_1 = IBNorm(self.ndf * 2, norm_layer=norm_layer)
        self.conv3 = nn.Conv2d(self.ndf * 2, self.ndf * 4, kernel_size=4, stride=2, padding=1)
        self.ibn3 = IBNorm(self.ndf * 4, norm_layer=norm_layer)
        self.conv3_1 = nn.Conv2d(self.ndf * 4, self.ndf * 4, kernel_size=4, stride=1, padding=1)
        self.ibn3_1 = IBNorm(self.ndf * 4, norm_layer=norm_layer)
        self.conv4 = nn.Conv2d(self.ndf * 4, self.ndf * 8, kernel_size=4, stride=2, padding=1)
        self.ibn4 = IBNorm(self.ndf * 8, norm_layer=norm_layer)
        self.conv4_1 = nn.Conv2d(self.ndf * 8, self.ndf * 8, kernel_size=4, stride=1, padding=1)
        self.ibn4_1 = IBNorm(self.ndf * 8, norm_layer=norm_layer)
        self.classifier = nn.Conv2d(self.ndf * 8, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, task_inp, task_pred):
        resulter, debugger = {}, {}

        # task_inp = torch.cat(task_inp, dim=1)
        x = torch.cat((task_inp, task_pred), dim=1)
        x = self.leaky_relu(self.ibn1(self.conv1(x)))
        x = self.leaky_relu(self.ibn2(self.conv2(x)))
        x = self.leaky_relu(self.ibn2_1(self.conv2_1(x)))
        x = self.leaky_relu(self.ibn3(self.conv3(x)))
        x = self.leaky_relu(self.ibn3_1(self.conv3_1(x)))
        x = self.leaky_relu(self.ibn4(self.conv4(x)))
        x = self.leaky_relu(self.ibn4_1(self.conv4_1(x)))
        x = self.classifier(x)
        x = F.interpolate(x, size=(task_pred.shape[2], task_pred.shape[3]), mode='bilinear', align_corners=True)

        # x is not activated here since it will be activated by the criterion function
        assert x.shape[2:] == task_pred.shape[2:]
        resulter['flawmap'] = x
        return x


class IBNorm(nn.Module):
    """ This layer combines BatchNorm and InstanceNorm.
    """

    def __init__(self, num_features, split=0.5, norm_layer=None):
        super(IBNorm, self).__init__()

        self.num_features = num_features
        self.num_BN = int(num_features * split + 0.5)
        self.bnorm = norm_layer(num_features=self.num_BN, affine=True)
        self.inorm = nn.InstanceNorm2d(num_features=num_features - self.num_BN, affine=False)

    def forward(self, x):
        if self.num_BN == self.num_features:
            return self.bnorm(x.contiguous())
        else:
            xb = self.bnorm(x[:, 0:self.num_BN, :, :].contiguous())
            xi = self.inorm(x[:, self.num_BN:, :, :].contiguous())

            return torch.cat((xb, xi), 1)


class FlawDetectorCriterion(nn.Module):
    """ Criterion of the flaw detector.
    """

    def __init__(self):
        super(FlawDetectorCriterion, self).__init__()

    def forward(self, pred, gt, is_ssl=False, reduction=True):
        loss = F.mse_loss(pred, gt, reduction='none')
        if reduction:
            loss = torch.mean(loss, dim=(1, 2, 3))
        return loss


class FlawmapHandler(nn.Module):
    """ Post-processing of the predicted flawmap.

    This module processes the predicted flawmap to fix some special
    cases that may cause errors in the subsequent steps of generating
    pseudo ground truth.
    """

    def __init__(self):
        super(FlawmapHandler, self).__init__()
        self.clip_threshold = 0.1

        blur_ksize = config.image_height // 16
        blur_ksize = blur_ksize + 1 if blur_ksize % 2 == 0 else blur_ksize
        self.blur = GaussianBlurLayer(1, blur_ksize)

    def forward(self, flawmap):
        flawmap = flawmap.data

        # force all values to be larger than 0
        flawmap.mul_((flawmap >= 0).float())
        # smooth the flawmap
        flawmap = self.blur(flawmap)
        # if all values in the flawmap are less than 'clip_threshold'
        # set the entire flawmap to 0, i.e., no flaw pixel
        fmax = flawmap.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        fmin = flawmap.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        max_matrix = fmax.repeat(1, 1, flawmap.shape[2], flawmap.shape[3])
        flawmap.mul_((max_matrix > self.clip_threshold).float())        # maximum number is lower than threshold, set the error to 0.
        # normalize the flawmap
        flawmap = flawmap.sub_(fmin).div_(fmax - fmin + 1e-9)

        return flawmap


class DCGTGenerator(nn.Module):
    """ Generate the ground truth of the dynamic consistency constraint.
    """

    def __init__(self):
        super(DCGTGenerator, self).__init__()

    def forward(self, l_pred, r_pred, l_handled_flawmap, r_handled_flawmap):
        l_tmp = l_handled_flawmap.clone()
        r_tmp = r_handled_flawmap.clone()

        l_bad = l_tmp > config.dc_threshold
        r_bad = r_tmp > config.dc_threshold

        both_bad = (l_bad & r_bad).float()  # too high error rate

        l_handled_flawmap.mul_((l_tmp <= config.dc_threshold).float())
        r_handled_flawmap.mul_((r_tmp <= config.dc_threshold).float())

        l_handled_flawmap.add_((l_tmp > config.dc_threshold).float())
        r_handled_flawmap.add_((r_tmp > config.dc_threshold).float())

        l_mask = (r_handled_flawmap >= l_handled_flawmap).float()
        r_mask = (l_handled_flawmap >= r_handled_flawmap).float()

        l_dc_gt = l_mask * l_pred + (1 - l_mask) * r_pred
        r_dc_gt = r_mask * r_pred + (1 - r_mask) * l_pred

        return l_dc_gt, r_dc_gt, both_bad, both_bad


class FDGTGenerator(nn.Module):
    """ Generate the ground truth of the flaw detector,
        i.e., pipeline 'C' in the paper.
    """

    def __init__(self):
        super(FDGTGenerator, self).__init__()

        blur_ksize = int(config.image_height / 8)
        blur_ksize = blur_ksize + 1 if blur_ksize % 2 == 0 else blur_ksize
        self.blur = GaussianBlurLayer(1, blur_ksize)

        reblur_ksize = int(config.image_height / 4)
        reblur_ksize = reblur_ksize + 1 if reblur_ksize % 2 == 0 else reblur_ksize
        self.reblur = GaussianBlurLayer(1, reblur_ksize)

        self.dilate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        )

    def forward(self, pred, gt):
        diff = torch.abs_(gt - pred.detach())
        diff = torch.sum(diff, dim=1, keepdim=True).mul_(config.mu)

        diff = self.blur(diff)
        for _ in range(0, config.nu):
            diff = self.reblur(self.dilate(diff))

        # normlize each sample to [0, 1]
        dmax = diff.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        dmin = diff.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        diff.sub_(dmin).div_(dmax - dmin + 1e-9)

        flawmap_gt = diff
        return flawmap_gt


class GaussianBlurLayer(nn.Module):
    """ Add Gaussian Blur to a 4D tensor

    This layer takes a 4D tensor of {N, C, H, W} as input.
    The Gaussian blur will be performed in given channel number (C) splitly.
    """

    def __init__(self, channels, kernel_size):
        """
        Arguments:
            channels (int): Channel for input tensor
            kernel_size (int): Size of the kernel used in blurring
        """

        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0

        self.op = nn.Sequential(
            nn.ReflectionPad2d(math.floor(self.kernel_size / 2)),
            nn.Conv2d(channels, channels, self.kernel_size,
                      stride=1, padding=0, bias=None, groups=channels)
        )

        self._init_kernel()

    def forward(self, x):
        """
        Arguments:
            x (torch.Tensor): input 4D tensor

        Returns:
            torch.Tensor: Blurred version of the input
        """

        assert  len(list(x.shape)) == 4
        assert x.shape[1] == self.channels

        return self.op(x)

    def _init_kernel(self):
        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8

        n = np.zeros((self.kernel_size, self.kernel_size))
        i = math.floor(self.kernel_size / 2)
        n[i, i] = 1
        kernel = scipy.ndimage.gaussian_filter(n, sigma)

        for name, param in self.named_parameters():
            param.data.copy_(torch.from_numpy(kernel))

def sigmoid_rampup(current, rampup_length):
    """ Exponential rampup from https://arxiv.org/abs/1610.02242 .
    """
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def sslgct_prepare_task_gt_for_fdgt(task_gt):
    task_gt = task_gt.unsqueeze(1)
    gt_np = task_gt.data.cpu().numpy()
    shape = list(gt_np.shape)
    assert len(shape) == 4
    shape[1] = config.num_classes

    one_hot = torch.zeros(shape).cuda()
    for i in range(config.num_classes):
        one_hot[:, i:i+1, ...].add_((task_gt == i).float())
        # ignore segment boundary
        one_hot[:, i:i+1, ...].mul_((task_gt != 255).float())

    # return torch.FloatTensor(one_hot)
    return one_hot