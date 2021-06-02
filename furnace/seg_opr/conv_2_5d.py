#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-03-04 20:52
# @Author  : Jingbo Wang
# @E-mail    : wangjingbo1219@foxmail.com & wangjingbo@megvii.com
# @File    : conv_2.5d.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter



def _ntuple(n):
    def parse(x):
        if isinstance(x, list) or isinstance(x, tuple):
            return x
        return tuple([x]*n)
    return parse
_pair = _ntuple(2)


class Conv2_5D_disp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True,
                 pixel_size=16):
        super(Conv2_5D_disp, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_size_prod = self.kernel_size[0] * self.kernel_size[1]
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pixel_size = pixel_size
        assert self.kernel_size_prod % 2 == 1
        
        self.weight_0 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.weight_1 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.weight_2 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x, disp, camera_params):
        N, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
        out_H = (H + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        out_W = (W + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        intrinsic, extrinsic = camera_params['intrinsic'], camera_params['extrinsic']
        
        x_col = F.unfold(x, self.kernel_size, dilation=self.dilation, padding=self.padding,
                         stride=self.stride)  # (N, C*kh*kw, out_H*out_W)
        x_col = x_col.view(N, C, self.kernel_size_prod, out_H * out_W)
        
        disp_col = F.unfold(disp, self.kernel_size, dilation=self.dilation, padding=self.padding,
                            stride=self.stride)  # (N, kh*kw, out_H*out_W)
        valid_mask = 1 - disp_col.eq(0.).to(torch.float32)
        valid_mask *= valid_mask[:, self.kernel_size_prod // 2, :].view(N, 1, out_H * out_W)
        disp_col *= valid_mask
        depth_col = (extrinsic['baseline'] * intrinsic['fx']).view(N, 1, 1).cuda() / torch.clamp(disp_col, 0.01, 256)
        valid_mask = valid_mask.view(N, 1, self.kernel_size_prod, out_H * out_W)
        
        center_depth = depth_col[:, self.kernel_size_prod // 2, :].view(N, 1, out_H * out_W)
        grid_range = self.pixel_size * self.dilation[0] * center_depth / intrinsic['fx'].view(N, 1, 1).cuda()
        
        mask_0 = torch.abs(depth_col - (center_depth + grid_range)).le(grid_range / 2).view(N, 1, self.kernel_size_prod,
                                                                                            out_H * out_W).to(
            torch.float32)
        mask_1 = torch.abs(depth_col - (center_depth)).le(grid_range / 2).view(N, 1, self.kernel_size_prod,
                                                                               out_H * out_W).to(torch.float32)
        mask_1 = (mask_1 + 1 - valid_mask).clamp(min=0., max=1.)
        mask_2 = torch.abs(depth_col - (center_depth - grid_range)).le(grid_range / 2).view(N, 1, self.kernel_size_prod,
                                                                                            out_H * out_W).to(
            torch.float32)
        
        output = torch.matmul(self.weight_0.view(-1, C * self.kernel_size_prod),
                              (x_col * mask_0).view(N, C * self.kernel_size_prod, out_H * out_W))
        output += torch.matmul(self.weight_1.view(-1, C * self.kernel_size_prod),
                               (x_col * mask_1).view(N, C * self.kernel_size_prod, out_H * out_W))
        output += torch.matmul(self.weight_2.view(-1, C * self.kernel_size_prod),
                               (x_col * mask_2).view(N, C * self.kernel_size_prod, out_H * out_W))
        output = output.view(N, -1, out_H, out_W)
        if self.bias:
            output += self.bias.view(1, -1, 1, 1)
        return output
    
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Conv2_5D_depth(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False,
                 pixel_size=1, is_graph=False):
        super(Conv2_5D_depth, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_size_prod = self.kernel_size[0] * self.kernel_size[1]
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pixel_size = pixel_size
        assert self.kernel_size_prod % 2 == 1
        self.is_graph = is_graph
        if self.is_graph:
            self.weight_0 = Parameter(torch.Tensor(out_channels, 1, *kernel_size))
            self.weight_1 = Parameter(torch.Tensor(out_channels, 1, *kernel_size))
            self.weight_2 = Parameter(torch.Tensor(out_channels, 1, *kernel_size))
        else:
            self.weight_0 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            self.weight_1 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            self.weight_2 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, depth, camera_params):
        # if self.is_graph:
        #     weight_0 = self.weight_0.expand(self.out_channels, self.in_channels, *self.kernel_size).contiguous()
        #     weight_1 = self.weight_1.expand(self.out_channels, self.in_channels, *self.kernel_size).contiguous()
        #     weight_2 = self.weight_2.expand(self.out_channels, self.in_channels, *self.kernel_size).contiguous()
        # else:
        #     weight_0 = self.weight_0
        #     weight_1 = self.weight_1
        #     weight_2 = self.weight_2
        N, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
        out_H = (H + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        out_W = (W + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        intrinsic = camera_params['intrinsic']
        x_col = F.unfold(x, self.kernel_size, dilation=self.dilation, padding=self.padding,
                         stride=self.stride)  # N*(C*kh*kw)*(out_H*out_W)
        x_col = x_col.view(N, C, self.kernel_size_prod, out_H * out_W)
        depth_col = F.unfold(depth, self.kernel_size, dilation=self.dilation, padding=self.padding,
                             stride=self.stride)  # N*(kh*kw)*(out_H*out_W)
        center_depth = depth_col[:, self.kernel_size_prod // 2, :]
        #print(depth_col.size())
        center_depth  = center_depth.view(N, 1, out_H * out_W)
        # grid_range = self.pixel_size * center_depth / (intrinsic['fx'].view(N,1,1) * camera_params['scale'].view(N,1,1))
        grid_range = self.pixel_size * center_depth / intrinsic['fx'].cuda().view(N, 1, 1)

        mask_0 = torch.abs(depth_col - (center_depth + grid_range)).le(grid_range / 2).view(N, 1, self.kernel_size_prod,
                                                                                            out_H * out_W).to(
            torch.float32)
        mask_1 = torch.abs(depth_col - (center_depth)).le(grid_range / 2).view(N, 1, self.kernel_size_prod,
                                                                               out_H * out_W).to(torch.float32)
        mask_2 = torch.abs(depth_col - (center_depth - grid_range)).le(grid_range / 2).view(N, 1, self.kernel_size_prod,
                                                                                            out_H * out_W).to(
            torch.float32)
        output = torch.matmul(self.weight_0.view(-1, C * self.kernel_size_prod),
                              (x_col * mask_0).view(N, C * self.kernel_size_prod, out_H * out_W))
        output += torch.matmul(self.weight_1.view(-1, C * self.kernel_size_prod),
                               (x_col * mask_1).view(N, C * self.kernel_size_prod, out_H * out_W))
        output += torch.matmul(self.weight_2.view(-1, C * self.kernel_size_prod),
                               (x_col * mask_2).view(N, C * self.kernel_size_prod, out_H * out_W))
        output = output.view(N, -1, out_H, out_W)
        if self.bias:
            output += self.bias.view(1, -1, 1, 1)
        return output



    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)
