'''
Author: supermantx
Date: 2024-10-08 15:48:48
LastEditTime: 2024-10-11 17:29:11
Description: 

An implementation of GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations. https://arxiv.org/abs/1911.11907
The train script of the model is similar to that of MobileNetV3
Original model: https://github.com/huawei-noah/CV-backbones/tree/master/ghostnet_pytorch

derived from noah-research/ghostnet change the model shape for face recognition
'''
import math
from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import  Linear, CondConv2d, HardSigmoid, make_divisible, DropPath
from timm.models.efficientnet import SqueezeExcite
from timm.models.helpers import build_model_with_cfg

__all__ = ['GhostNet']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'input_size': (3, 112, 112),
        'pool_size': (1, 1),
        'crop_pct': 0.875,
        'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem',
        'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    'ghostnet_100':
    _cfg(
        url=
        'https://github.com/huawei-noah/CV-backbones/releases/download/ghostnet_pth/ghostnet_1x.pth'
    ),
    'ghostnet':
    _cfg(url=''),
}

_SE_LAYER = partial(SqueezeExcite, gate_layer=HardSigmoid)


class DynamicConv(nn.Module):
    """ Dynamic Conv layer
    """

    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=1,
                 stride=1,
                 padding='',
                 dilation=1,
                 groups=1,
                 bias=False,
                 num_experts=4):
        super().__init__()
        self.routing = nn.Linear(in_features, num_experts)
        self.cond_conv = CondConv2d(in_features, out_features, kernel_size,
                                    stride, padding, dilation, groups, bias,
                                    num_experts)

    def forward(self, x):
        pooled_inputs = F.adaptive_avg_pool2d(x,
                                              1).flatten(1)  # CondConv routing
        routing_weights = torch.sigmoid(self.routing(pooled_inputs))
        x = self.cond_conv(x, routing_weights)
        return x


class ConvBnAct(nn.Module):
    """ Conv + Norm Layer + Activation w/ optional skip connection
    """

    def __init__(self,
                 in_chs,
                 out_chs,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 pad_type='',
                 skip=False,
                 act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2d,
                 drop_path_rate=0.,
                 num_experts=4):
        super(ConvBnAct, self).__init__()
        self.has_residual = skip and stride == 1 and in_chs == out_chs
        self.drop_path_rate = drop_path_rate
        # self.conv = create_conv2d(in_chs, out_chs, kernel_size, stride=stride, dilation=dilation, padding=pad_type)
        self.conv = DynamicConv(in_chs,
                                out_chs,
                                kernel_size,
                                stride,
                                dilation=dilation,
                                padding=pad_type,
                                num_experts=num_experts)
        self.bn1 = norm_layer(out_chs)
        self.act1 = act_layer()

    def feature_info(self, location):
        if location == 'expansion':  # output of conv after act, same as block coutput
            info = dict(module='act1',
                        hook_type='forward',
                        num_chs=self.conv.out_channels)
        else:  # location == 'bottleneck', block output
            info = dict(module='',
                        hook_type='',
                        num_chs=self.conv.out_channels)
        return info

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = DropPath(x, self.drop_path_rate, self.training)
            x += shortcut
        return x

class GDC(nn.Module):
    def __init__(self, embedding_size):
        super(GDC, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(512, 512, (7, 7), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(512),
            nn.Flatten(),
            Linear(512, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size))

    def forward(self, x):
        return self.layers(x)

class GhostModule(nn.Module):

    def __init__(self,
                 inp,
                 oup,
                 kernel_size=1,
                 ratio=2,
                 dw_size=3,
                 stride=1,
                 act_layer=nn.ReLU,
                 num_experts=4):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            DynamicConv(inp,
                        init_channels,
                        kernel_size,
                        stride,
                        kernel_size // 2,
                        bias=False,
                        num_experts=num_experts),
            nn.BatchNorm2d(init_channels),
            act_layer() if act_layer is not None else nn.Identity(),
        )

        self.cheap_operation = nn.Sequential(
            DynamicConv(init_channels,
                        new_channels,
                        dw_size,
                        1,
                        dw_size // 2,
                        groups=init_channels,
                        bias=False,
                        num_experts=num_experts),
            nn.BatchNorm2d(new_channels),
            act_layer() if act_layer is not None else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self,
                 in_chs,
                 mid_chs,
                 out_chs,
                 dw_kernel_size=3,
                 stride=1,
                 act_layer=nn.ReLU,
                 se_ratio=0.,
                 drop_path=0.,
                 num_experts=4):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs,
                                  mid_chs,
                                  act_layer=act_layer,
                                  num_experts=num_experts)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs,
                                     mid_chs,
                                     dw_kernel_size,
                                     stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs,
                                     bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)
        else:
            self.conv_dw = None
            self.bn_dw = None

        # Squeeze-and-excitation
        self.se = _SE_LAYER(mid_chs,
                            rd_ratio=se_ratio,
                            act_layer=act_layer if act_layer is not nn.GELU
                            else nn.ReLU) if has_se else None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs,
                                  out_chs,
                                  act_layer=None,
                                  num_experts=num_experts)

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                DynamicConv(in_chs,
                            in_chs,
                            dw_kernel_size,
                            stride=stride,
                            padding=(dw_kernel_size - 1) // 2,
                            groups=in_chs,
                            bias=False,
                            num_experts=num_experts),
                nn.BatchNorm2d(in_chs),
                DynamicConv(in_chs,
                            out_chs,
                            1,
                            stride=1,
                            padding=0,
                            bias=False,
                            num_experts=num_experts),
                nn.BatchNorm2d(out_chs),
            )

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.conv_dw is not None:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x = self.shortcut(shortcut) + self.drop_path(x)
        return x

def scale_ch(x, scale):
    return x * scale

class GhostNet(nn.Module):

    def __init__(self,
                 fp16,
                 cfgs,
                 width=1.0,
                 dropout=0.2,
                 in_chans=3,
                 output_stride=32,
                 drop_path=0.,
                 act_layer=nn.PReLU,
                 num_experts=4,
                 **kwargs):
        super(GhostNet, self).__init__()
        self.fp16 = fp16

        # setting of inverted residual blocks
        assert output_stride == 32, 'only output_stride==32 is valid, dilation not supported'
        self.cfgs = cfgs
        self.dropout = dropout
        self.feature_info = []
        scale = partial(scale_ch, scale=width)
        # building first layer
        stem_chs = scale(64)
        self.conv_stem = DynamicConv(in_chans,
                                     stem_chs,
                                     3,
                                     2,
                                     1,
                                     bias=False,
                                     num_experts=num_experts)
        self.feature_info.append(
            dict(num_chs=stem_chs, reduction=2, module=f'conv_stem'))
        self.bn1 = nn.BatchNorm2d(stem_chs)
        self.act1 = act_layer()
        prev_chs = stem_chs

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path, len(self.cfgs))
               ]  # stochastic depth decay rule

        # building inverted residual blocks
        stages = nn.ModuleList([])
        block = GhostBottleneck
        stage_idx = 0
        net_stride = 2
        for cfg in self.cfgs:
            layers = []
            s = 1
            for k, exp_size, c, se_ratio, s in cfg:
                out_chs = scale(c)
                mid_chs = scale(exp_size)
                layers.append(
                    block(prev_chs,
                          mid_chs,
                          out_chs,
                          k,
                          s,
                          se_ratio=se_ratio,
                          drop_path=dpr[stage_idx],
                          act_layer=act_layer,
                          num_experts=num_experts))
                prev_chs = out_chs
            stages.append(nn.Sequential(*layers))
            stage_idx += 1
        self.blocks = nn.Sequential(*stages)
        
        self.conv_sep = ConvBnAct(prev_chs, 512, kernel_size=1, act_layer=act_layer)
        self.features = GDC(512)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv_stem(x)
            x = self.bn1(x)
            x = self.act1(x)
            x = self.blocks(x)
        x = self.conv_sep(x.float() if self.fp16 else x)
        x = self.features(x)
        return x

def _create_parameternet_fr(variant, fp16, width=1.0, pretrained=False, **kwargs):
    """
    Construct a parameternet(base on GhostNet) for face recognition
    """
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3, 64, 64, 0.25, 1]],
        # stage2
        [[5, 64, 64, 0, 2]],
        [[3, 64, 64, 0.25, 1],
         [3, 64, 64, 0.25, 1]],
        # stage3
        [[3, 128, 128, 0, 2]],
        [[3, 128, 128, 0.25, 1], 
         [3, 128, 128, 0.25, 1], 
         [3, 128, 128, 0.25, 1]],
        # stage4
        [[5, 256, 128, 0, 2]],
        [[3, 128, 128, 0.25, 1]]
    ]
    model_kwargs = dict(
        cfgs=cfgs,
        width=width,
        fp16=fp16
    )
    return build_model_with_cfg(GhostNet,
                                variant,
                                pretrained,
                                pretrained_cfg=default_cfgs[variant],
                                feature_cfg=dict(flatten_sequential=True),
                                **model_kwargs)


def parameternet_fr(fp16, pretrained=False, **kwargs):
    """ parameternet_fr """
    model = _create_parameternet_fr('ghostnet',
                                    fp16,
                                    width=2,
                                    pretrained=pretrained,
                                    act_layer=nn.Hardswish,
                                    **kwargs)
    return model

if __name__ == "__main__":
    import timm
    from torchsummary import summary

    model = timm.create_model('parameternet_fr')
    
    summary(model, (3, 112, 112), device='cpu')    