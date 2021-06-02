#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-04-08 14:41
# @Author  : Jingbo Wang
# @E-mail    : wangjingbo1219@foxmail.com & wangjingbo@megvii.com
# @File    : geo_utils.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class Plane2Space(nn.Module):
    def __init__(self):
        super(Plane2Space, self).__init__()

    def forward(self, depth, coordinate, camera_params):
        valid_mask = 1-depth.eq(0.).to(torch.float32)

        depth = torch.clamp(depth, min=1e-5)
        N, H, W = depth.size(0), depth.size(2), depth.size(3)
        intrinsic = camera_params['intrinsic']

        K_inverse = depth.new_zeros(N, 3, 3)
        K_inverse[:,0,0] = 1./intrinsic['fx']
        K_inverse[:,1,1] = 1./intrinsic['fy']
        K_inverse[:,2,2] = 1.
        K_inverse[:,0,2] = -intrinsic['cx']/intrinsic['fx']
        K_inverse[:,1,2] = -intrinsic['cy']/intrinsic['fy']
        coord_3d = torch.matmul(K_inverse, (coordinate.float()*depth.float()).view(N,3,H*W)).view(N,3,H,W).contiguous()
        coord_3d *= valid_mask

        return coord_3d

class Space2Plane(nn.Module):
    def __init__(self):
        super(Space2Plane, self).__init__()

    def forward(self, depth, coord_3d, camera_params):
        N, C, H, W = torch.size()
        intrinsic = camera_params['intrinsic']
        # K_inverse = depth.new_zeros(N, 3, 3)
        # K_inverse[:,0,0] = 1./intrinsic['fx']
        # K_inverse[:,1,1] = 1./intrinsic['fy']
        # K_inverse[:,2,2] = 1.
        # K_inverse[:,0,2] = -intrinsic['cx']/intrinsic['fx']
        # K_inverse[:,1,2] = -intrinsic['cy']/intrinsic['fy']
        K = depth.new_zeros(N, 3, 3)
        K[:,0,0] = intrinsic['fx']
        K[:,1,1] = intrinsic['fy']


