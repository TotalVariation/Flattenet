from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import conv1x1


class PyramidPooling(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, pool_sizes, norm_layer=nn.BatchNorm2d):
        super(PyramidPooling, self).__init__()
        branches = []
        out_channels = int(in_channels/4)

        for s in pool_sizes:
            branches.append(
                nn.Sequential(OrderedDict([
                    ('pool', nn.AdaptiveAvgPool2d(s)),
                    ('conv', conv1x1(in_channels, out_channels)),
                    ('norm', norm_layer(out_channels)),
                    ('relu', nn.ReLU(True)),
                ]))
            )
        self.branches = nn.ModuleList(branches)
        # bilinear interpolation options
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': False}

    def forward(self, x):
        _, _, h, w = x.size()
        feat = []
        for b in self.branches:
            out = F.interpolate(b(x), size=(h, w), **self._up_kwargs)
            feat.append(out)
        return torch.cat(feat, 1)
