# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cityscapes import Cityscapes as cityscapes
from .pascal_ctx import PASCALContext as pascal_ctx
from .pascal_aug import PASCALAug as pascal_aug
from .pascal_voc import PASCALVOC as pascal_voc
from .ade20k import ADE20KSegmentation as ade20k
from .lip import LIP as lip
from .mscoco import COCOSeg as coco_pretrain
