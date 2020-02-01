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


class ADE20KSegmentation(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=150,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=520,
                 crop_size=(480, 480),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],):

        super(ADE20KSegmentation, self).__init__(ignore_label, base_size,
                                                 crop_size, downsample_rate,
                                                 scale_factor, mean, std)

        self.root = os.path.abspath(os.path.join(root, 'ADEChallengeData2016'))
        self.split = list_path

        self.num_classes = num_classes
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        self.crop_size = crop_size

        # prepare data and labels
        self.images, self.masks = self._get_ade20k_pairs(self.root, self.split)

        logger.info(f'The number of samples of {self.split} set: {len(self.images)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_pth = self.images[index]
        mask_pth = self.masks[index]
        image = cv2.imread(img_pth)
        label = Image.open(mask_pth)
        label = np.asarray(label, dtype=np.int32)
        size = image.shape
        name = os.path.splitext(os.path.basename(img_pth))[0]

        if self.split == 'val':
            image = cv2.resize(image, self.crop_size[::-1],
                               interpolation=cv2.INTER_LINEAR)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            label = cv2.resize(label, self.crop_size[::-1],
                               interpolation=cv2.INTER_NEAREST)
            label = self.label_transform(label)
        elif self.split == 'testval':
            # evaluate model on val dataset
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            label = self.label_transform(label)
        else:
            image, label = self.gen_sample(image, label,
                                           self.multi_scale, self.flip)

        return image.copy(), label.copy(), np.array(size), name

    def label_transform(self, label):
        label = np.array(label).astype('int32') - 1
        label[label == -2] = -1
        return label

    def _get_ade20k_pairs(self, folder, split='train'):
        def get_path_pairs(img_folder, mask_folder):
            img_paths = []
            mask_paths = []
            for filename in os.listdir(img_folder):
                basename, _ = os.path.splitext(filename)
                if filename.endswith(".jpg"):
                    imgpath = os.path.join(img_folder, filename)
                    maskname = basename + '.png'
                    maskpath = os.path.join(mask_folder, maskname)
                    if os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        logger.info('cannot find the mask:', maskpath)
            return img_paths, mask_paths

        if split == 'train':
            img_folder = os.path.join(folder, 'images/training')
            mask_folder = os.path.join(folder, 'annotations/training')
            img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
            assert len(img_paths) == 20210
        elif split == 'val':
            img_folder = os.path.join(folder, 'images/validation')
            mask_folder = os.path.join(folder, 'annotations/validation')
            img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
            assert len(img_paths) == 2000
        else:
            assert split == 'trainval'
            train_img_folder = os.path.join(folder, 'images/training')
            train_mask_folder = os.path.join(folder, 'annotations/training')
            val_img_folder = os.path.join(folder, 'images/validation')
            val_mask_folder = os.path.join(folder, 'annotations/validation')
            train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
            val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
            img_paths = train_img_paths + val_img_paths
            mask_paths = train_mask_paths + val_mask_paths
            assert len(img_paths) == 22210
        return img_paths, mask_paths
