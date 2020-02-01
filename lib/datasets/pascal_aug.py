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


class PASCALAug(BaseDataset):
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
        super(PASCALAug, self).__init__(ignore_label, base_size, crop_size,
                                        scale_factor, mean, std)

        self.root = os.path.abspath(os.path.join(root, 'VOCaug/dataset'))
        self.split = list_path

        self.num_classes = num_classes
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        self.crop_size = crop_size

        # prepare data
        if 'val' in self.split:
            file_path = os.path.join(self.root, 'val.txt')
        elif 'train' in self.split:
            file_path = os.path.join(self.root, 'trainval.txt')
        else:
            raise ValueError('Only supporting train and val set.')

        self.files = [line.rstrip('\n') for line in tuple(open(file_path, 'r'))]

        logger.info(f'The number of samples of {self.split} set: {len(self.files)}')

    def __getitem__(self, index):
        img_dir = os.path.join(self.root, 'img')
        mask_dir = os.path.join(self.root, 'cls')

        img_id = self.files[index]

        if 'train' in self.split:
            image = cv2.imread(os.path.join(img_dir, img_id + '.jpg'))
            mat = scipy.io.loadmat(os.path.join(mask_dir, img_id + '.mat'),
                                   mat_dtype=True, squeeze_me=True, struct_as_record=False)
            label = mat['GTcls'].Segmentation  # np.ndarray
            size = image.shape
            image, label = self.gen_sample(image, label, self.multi_scale, self.flip)
        elif 'val' in self.split:
            image = cv2.imread(os.path.join(img_dir, img_id + '.jpg'))
            size = image.shape
            image = cv2.resize(image, self.crop_size[::-1],
                               interpolation=cv2.INTER_LINEAR)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            mat = scipy.io.loadmat(os.path.join(mask_dir, img_id + '.mat'),
                                   mat_dtype=True, squeeze_me=True, struct_as_record=False)
            label = mat['GTcls'].Segmentation  # np.ndarray
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
