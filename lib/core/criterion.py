import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None, reduction='mean'):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction=reduction)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(score, size=(h, w), mode='bilinear',
                    align_corners=False)

        loss = self.criterion(score, target)

        return loss


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thresh=0.7, min_kept=100000,
                 weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thresh
        self.min_kept = int(min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(score, size=(h, w),
                                  mode='bilinear', align_corners=False)
        with torch.no_grad():
            pred = F.softmax(score.clone().detach(), dim=1)
            tmp_target = target.clone().detach()
            mask = tmp_target.contiguous().view(-1) != self.ignore_label
            tmp_target[tmp_target == self.ignore_label] = 0
            pred = pred.gather(1, tmp_target.unsqueeze(1))
            pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
            min_value = pred[min(self.min_kept, pred.numel()) - 1]
            threshold = max(min_value, self.thresh)

        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()
