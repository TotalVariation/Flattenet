from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import logging
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.conv import conv1x1, conv3x3

BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, planes):
        super(BatchNorm2d, self).__init__(planes, momentum=BN_MOMENTUM)


class BasicBlock(nn.Module):
    """
    reference: torchvision
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=BatchNorm2d):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    reference: torchvision
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=BatchNorm2d):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    refence: gluon-cv
    """
    def __init__(self, block, layers, groups=1, width_per_group=64,
                 deep_stem=False, stem_width=32, avg_down=False,
                 norm_layer=BatchNorm2d, pretrained=None, **kwargs):

        super(ResNet, self).__init__()
        self._norm_layer = norm_layer

        self.groups = groups
        self.base_width = width_per_group
        self.avg_down = avg_down
        self.out_planes = []

        if not deep_stem:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                                   stride=2, padding=3, bias=False)
            self.inplanes = 64
        else:
            self.conv1 = nn.Sequential(OrderedDict([
                ('conv1', conv3x3(3, stem_width, 2)),
                ('norm1', norm_layer(stem_width)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', conv3x3(stem_width, stem_width)),
                ('norm2', norm_layer(stem_width)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', conv3x3(stem_width, stem_width*2))
            ]))
            self.inplanes = stem_width*2
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.out_planes.append(self.inplanes)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.out_planes.append(self.inplanes)

        if pretrained:
            self._load_pretrained_model(pretrained)
        else:
            self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = [('conv', conv1x1(self.inplanes, planes * block.expansion)),
                              ('norm', norm_layer(planes * block.expansion)),]
                if stride != 1:
                    downsample.insert(0, ('pool', nn.AvgPool2d(stride, stride, ceil_mode=True)))
            else:
                downsample = [('conv', conv1x1(self.inplanes, planes * block.expansion, stride)),
                              ('norm', norm_layer(planes * block.expansion)),]
            downsample = nn.Sequential(OrderedDict(downsample))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        feats = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feats.append(x)
        x = self.layer4(x)
        feats.append(x)

        return tuple(feats)

    def _init_weights(self):
        logger.info('Training from Scratch!')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _load_pretrained_model(self, pretrained=''):
        """
        reference: xingyizhou github repo CenterNet
        """
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> Loading pretrained model {}.'.format(pretrained))
            model_dict = self.state_dict()
            # check loaded parameters size/shape compatibility
            for k in pretrained_dict:
                if k in model_dict:
                    if pretrained_dict[k].shape != model_dict[k].shape:
                        logger.info('=> Skip loading parameter {}, required shape {}, '\
                              'loaded shape {}.'.format(
                            k, model_dict[k].shape, pretrained_dict[k].shape))
                        pretrained_dict[k] = model_dict[k]
                else:
                    logger.info('=> Drop parameter {}.'.format(k))
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> Loading {} from pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            raise ValueError('Pretrained model {} does not exist!'.format(pretrained))


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_resnet(num_layers=50, **kwargs):
    block_class, layers = resnet_spec[num_layers]
    model = ResNet(block_class, layers, **kwargs)

    return model
