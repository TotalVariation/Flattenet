# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# Referring to the implementation in 
# https://github.com/zhanghang1989/PyTorch-Encoding
# ------------------------------------------------------------------------------

import os
import logging
import scipy.io

import cv2
import numpy as np
from PIL import Image

import torch

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class PASCALVOC(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=21,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=520,
                 crop_size=(480, 480),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],):

        self.Classes = [
            'background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
            'tv']
        super(PASCALVOC, self).__init__(ignore_label, base_size,
                                        crop_size, scale_factor, mean, std)

        self.root = os.path.abspath(os.path.join(root, 'pascal_voc/VOCdevkit/VOC2012'))
        self.split = list_path

        self.num_classes = num_classes
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        self.crop_size = crop_size

        # prepare data
        if 'val' in self.split:
            file_path = os.path.join(self.root, 'ImageSets/Segmentation', 'val.txt')
        elif 'train' in self.split:
            file_path = os.path.join(self.root, 'ImageSets/Segmentation', 'trainval.txt')
        elif 'test' in self.split:
            self.root = os.path.abspath(os.path.join(self.root, 'voc2012_test'))
            file_path = os.path.join(self.root, 'ImageSets/Segmentation', 'test.txt')
        else:
            raise ValueError('Unknown dataset split.')

        self.files = [line.rstrip('\n') for line in tuple(open(file_path, 'r'))]

        logger.info(f'The number of samples of {self.split} set: {len(self.files)}')

    def __getitem__(self, index):
        img_dir = os.path.join(self.root, 'JPEGImages')

        img_id = self.files[index]
        image = cv2.imread(os.path.join(img_dir, img_id + '.jpg'))
        size = image.shape

        if 'test' in self.split:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), img_id

        mask_dir = os.path.join(self.root, 'SegmentationClass')
        label = Image.open(os.path.join(mask_dir, img_id + '.png'))
        label = np.array(label)

        if 'train' in self.split:
            image, label = self.gen_sample(image, label, self.multi_scale, self.flip)
        elif 'val' in self.split:
            image = cv2.resize(image, self.crop_size[::-1],
                               interpolation=cv2.INTER_LINEAR)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            label = cv2.resize(label, self.crop_size[::-1],
                               interpolation=cv2.INTER_NEAREST)
            label = self.label_transform(label)
        else:
            raise NotImplementedError

        return image.copy(), label.copy(), np.array(size), img_id

    def label_transform(self, label):
        label = np.array(label).astype('int32')
        label[label == 255] = -1
        return label

    def save_pred(self, preds, sv_path, name):
        preds = preds.cpu().numpy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            save_img = Image.fromarray(preds[i])
            save_img.save(os.path.join(sv_path, name[i]+'.png'))
