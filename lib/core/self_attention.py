from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import conv1x1


class SelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''
    def __init__(self, in_channels, key_channels, value_channels,
                 out_channels=None, scale=1, norm_layer=nn.BatchNorm2d):
        super(SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels is None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale)) if scale != 1 else None
        self.f_key = nn.Sequential(OrderedDict([
            ('conv', conv1x1(self.in_channels, self.key_channels)),
            ('norm', norm_layer(self.key_channels)),
            ('relu', nn.ReLU(inplace=True)),
        ]))
        self.f_query = nn.Sequential(OrderedDict([
            ('conv', conv1x1(self.in_channels, self.key_channels)),
            ('norm', norm_layer(self.key_channels)),
            ('relu', nn.ReLU(inplace=True)),
        ]))
        self.f_value = nn.Sequential(OrderedDict([
            ('conv', conv1x1(self.in_channels, self.value_channels)),
            ('norm', norm_layer(self.value_channels)),
            ('relu', nn.ReLU(inplace=True)),
        ]))
        self.W = nn.Sequential(OrderedDict([
            ('conv', conv1x1(self.value_channels, self.out_channels)),
            ('norm', norm_layer(self.out_channels)),
            ('relu', nn.ReLU(inplace=True)),
        ]))
        # bilinear interpolation options
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': False}

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.pool is not None:
            x = self.pool(x)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), **self._up_kwargs)

        return context
