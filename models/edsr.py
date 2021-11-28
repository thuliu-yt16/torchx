# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register


def default_conv(in_channels, out_channels, kernel_size, bias=True, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

class EDSR(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(EDSR, self).__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        self.m_body = m_body
    
        if args.all_block:
            self.head = nn.ModuleList(m_head)
            self.body = nn.ModuleList(m_body)
        else:
            self.head = nn.Sequential(*m_head)
            self.body = nn.Sequential(*m_body)

        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            # define tail module
            m_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ]
            self.m_tail = m_tail
            self.tail = nn.Sequential(*m_tail)
        
        if args.all_block:
            self.out_dims = [args.n_colors]
            for _ in range(len(self.head)):
                self.out_dims.append(n_feats)
            for _ in range(len(self.body)):
                self.out_dims.append(n_feats)

    def forward(self, x):
        #x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        if self.args.no_upsampling:
            x = res
        else:
            x = self.tail(res)
        #x = self.add_mean(x)
        return x
    
    def forward_verbose(self, x):
        block_out = [x]
        for layer in self.head:
            x = layer(x)
            block_out.append(x)

        res = x
        for layer in self.body:
            res = layer(res) 
            block_out.append(res)

        # y = res + x
        # block_out.append(y)
        return block_out

    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


@register('edsr-baseline')
def make_edsr_baseline(n_resblocks=16, n_feats=64, res_scale=1,
                       scale=2, no_upsampling=False, rgb_range=1, all_block=False):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = 3
    args.all_block = all_block
    return EDSR(args)


@register('edsr')
def make_edsr(n_resblocks=32, n_feats=256, res_scale=0.1,
              scale=2, no_upsampling=False, rgb_range=1, all_block=False):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = 3
    args.all_block = all_block
    return EDSR(args)

class MyResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1, act_after_all=True):

        super(MyResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.act = act
        self.res_scale = res_scale
        self.act_after_all = act_after_all

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        if self.act_after_all:
            res = self.act(res)
        return res

class DSNet(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(DSNet, self).__init__()
        self.args = args

        n_high = args.n_high
        n_middle = args.n_middle
        n_low = args.n_low

        n_feats = args.n_feats
        if isinstance(n_feats, int):
            n_feats = [n_feats, n_feats, n_feats]

        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        resblock = args.resblock

        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats[0], kernel_size)]

        m_high_body = [
            resblock(
                conv, n_feats[0], kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_high)
        ]

        m_high2mid = [
            conv(n_feats[0], n_feats[1], kernel_size, stride=2),
        ]

        m_mid_body = [
            resblock(
                conv, n_feats[1], kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_middle)
        ]

        m_mid2low = [
            conv(n_feats[1], n_feats[2], kernel_size, stride=2),
        ]

        m_low_body = [
            resblock(
                conv, n_feats[2], kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_low)
        ]

        self.head = nn.Sequential(*m_head)
        self.high_body = nn.Sequential(*m_high_body)
        self.mid_body = nn.Sequential(*m_mid_body)
        self.low_body = nn.Sequential(*m_low_body)

        self.high2mid = nn.Sequential(*m_high2mid)
        self.mid2low = nn.Sequential(*m_mid2low)
        self.out_dims = n_feats
        if args.proj_conv:
            self.high_proj = conv(n_feats[0], n_feats[0], 1)
            self.mid_proj = conv(n_feats[1], n_feats[1], 1)
            self.low_proj = conv(n_feats[2], n_feats[2], 1)

    def forward(self, x):
        x = self.head(x)
        res = self.high_body(x)
        high = x + res

        mid = self.high2mid(high)
        mid_res = self.mid_body(mid)
        mid = mid_res + mid

        low = self.mid2low(mid)
        low_res = self.low_body(low)
        low = low_res + low

        if self.args.proj_conv:
            high = self.high_proj(high)
            mid = self.mid_proj(mid)
            low = self.low_proj(low)

        return [high, mid, low]

@register('dsnet')
def make_dsnet(n_high=12, n_middle=12, n_low=16, n_feats=64, res_scale=1, scale=2, rgb_range=1, resblock='MyResBlock', proj_conv=False):
    args = Namespace()
    args.n_high = n_high
    args.n_middle = n_middle
    args.n_low = n_low

    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.rgb_range = rgb_range
    args.n_colors = 3

    args.resblock = MyResBlock
    rb = {
        'MyResBlock': MyResBlock,
        'ResBlock': ResBlock,
    }
    args.resblock = rb[resblock]
    args.proj_conv = proj_conv
    return DSNet(args)

class EDSRForPooling(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(EDSRForPooling, self).__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))
    
        self.head = nn.Sequential(*m_head)
        self.body = nn.ModuleList(m_body)
        
        if args.proj_conv:
            self.proj = nn.ModuleList()
            for _ in args.select_layers:
                self.proj.append(conv(n_feats, n_feats, 1, bias=True, stride=1))

        assert(args.no_upsampling)
        self.out_dim = n_feats
        self.out_dims = [n_feats] * len(args.select_layers)

    def forward(self, x):
        x = self.head(x)
        cur_layer = 1
        out = []

        if cur_layer in self.args.select_layers:
            out.append(x)

        res = x
        for body_layer in self.body:
            res = body_layer(res)
            cur_layer += 1
            if cur_layer in self.args.select_layers:
                out.append(res)
            
        x = x + res
        # res = self.body(res)
        cur_layer += 1
        if cur_layer in self.args.select_layers:
            out.append(x)
        
        if self.args.proj_conv:
            assert(len(out) == len(self.proj))
            for i in range(len(out)):
                out[i] = self.proj[i](out[i])

        return out

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


@register('edsr-pooling')
def make_edsr_pooling(n_resblocks=16, n_feats=64, res_scale=1,
                       scale=2, no_upsampling=False, rgb_range=1, select_layers=[1, 10, 18], proj_conv=False):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.rgb_range = rgb_range
    args.n_colors = 3
    args.select_layers = select_layers
    args.proj_conv = proj_conv
    return EDSRForPooling(args)