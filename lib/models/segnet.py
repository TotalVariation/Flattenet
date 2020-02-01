import os
import logging
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import get_backbone
from core.psp import PyramidPooling
from core.self_attention import SelfAttentionBlock
from core.conv import conv1x1, conv3x3


BN_MOMENTUM = 0.01

logger = logging.getLogger(__name__)


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, planes):
        super(BatchNorm2d, self).__init__(planes, momentum=BN_MOMENTUM)


class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def extra_repr(self):
        return f'groups={self.groups}'

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels//self.groups

        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)

        return x


class FlattenModule(nn.Module):

    def __init__(self, inplanes, expansions, groups, upscale, pool_sizes=None,
                 norm_layer=BatchNorm2d):
        assert len(groups) == len(expansions)
        super(FlattenModule, self).__init__()
        self.groups = groups
        # context strengthening strategy
        if pool_sizes is None:
            self.ctx_module = SelfAttentionBlock(
                in_channels=inplanes,
                key_channels=inplanes//8,
                value_channels=inplanes//4,
                norm_layer=norm_layer
            )
        else:
            self.ctx_module = PyramidPooling(inplanes, pool_sizes, norm_layer)
        self.inp = inplanes * 2
        layers = []
        for i in range(len(groups)):
            """
            reference:
                complying with complementary condition proposed in IGCV2
            """
            conv1 = conv1x1(self.inp, self.inp, groups=groups[i])
            norm1 = norm_layer(self.inp)
            conv2 = conv1x1(self.inp, self.inp * expansions[i], groups=self.inp//groups[i])
            norm2 = norm_layer(self.inp * expansions[i])
            layers.append(
                nn.Sequential(OrderedDict([
                    ('conv1', conv1),
                    ('norm1', norm1),
                    ('cs', ChannelShuffle(groups[i])),
                    ('conv2', conv2),
                    ('norm2', norm2),
                    ('relu', nn.ReLU(inplace=True)),
                ]))
            )
            self.inp *= expansions[i]

        self.layers = nn.Sequential(*layers)
        """
        Rearrange pixel-wise visual descriptors
        from channel domain to spatial domain.
        """
        self.cs = ChannelShuffle(upscale**2)
        self.ps = nn.PixelShuffle(upscale)

    def forward(self, x):

        context = self.ctx_module(x)
        feat = torch.cat([x, context], dim=1)
        out = self.layers(feat)
        out = self.cs(out)
        out = self.ps(out)
        return out


class SegNet(nn.Module):

    def __init__(self, config):
        super(SegNet, self).__init__()
        extra = config.MODEL.EXTRA
        backbone = extra.BACKBONE
        self.nclass = config.DATASET.NUM_CLASSES

        _backbone_params = {
            'num_layers': extra.NUM_LAYERS,
            'pretrained': extra.BASE_PRETRAINED,
            'deep_stem': extra.DEEP_STEM,
            'stem_width': extra.STEM_WIDTH,
            'avg_down': extra.AVG_DOWN,
        }
        self.base = get_backbone(backbone, **_backbone_params)

        _flatten_layer = FlattenModule(
            inplanes=self.base.out_planes[1],
            expansions=extra.EXPS1,
            groups=extra.GROUPS1,
            upscale=extra.UPSCALE1,
            pool_sizes=extra.POOL_SIZES1,
        )
        self.head = nn.Sequential(OrderedDict([
            ('flatten', _flatten_layer),
            ('dropout', nn.Dropout2d(extra.DROP1)),
            ('pred', nn.Conv2d(_flatten_layer.inp // extra.UPSCALE1**2,
                               self.nclass, 1, bias=True)),
        ]))

        _aux_flatten_layer = FlattenModule(
            inplanes=self.base.out_planes[0],
            expansions=extra.EXPS2,
            groups=extra.GROUPS2,
            upscale=extra.UPSCALE2,
            pool_sizes=extra.POOL_SIZES2,
        )
        self.aux_head = nn.Sequential(OrderedDict([
            ('flatten', _aux_flatten_layer),
            ('dropout', nn.Dropout2d(extra.DROP2)),
            ('pred', nn.Conv2d(_aux_flatten_layer.inp // extra.UPSCALE2**2,
                               self.nclass, kernel_size=1, bias=True)),
        ]))

        self.head_alpha_nwd = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if name.find('base') == -1:
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=1e-3)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x3, x4 = self.base(x)
        aux = self.aux_head(x3)
        out = self.head(x4)
        return out + self.head_alpha_nwd * aux


def get_seg_model(config):
    return SegNet(config)
