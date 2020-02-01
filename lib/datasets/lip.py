# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset


class LIP(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=20,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=473,
                 crop_size=(473, 473),
                 scale_factor=11,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(LIP, self).__init__(ignore_label, base_size, crop_size,
                                  scale_factor, mean, std)
        self.root = root
        self.img_dir = os.path.abspath(os.path.join(root, 'LIP/images'))
        self.mask_dir = os.path.abspath(os.path.join(root, 'LIP/TrainVal_parsing_annotations'))
        self.num_classes = num_classes
        self.list_path = list_path
        self.class_weights = None
        self.right_idx = [15, 17, 19]
        self.left_idx = [14, 16, 18]
        self.flip_idx = torch.tensor(list(range(14)) + [15, 14, 17, 16, 19, 18]).long()

        self.multi_scale = multi_scale
        self.flip = flip
        self.img_list = [line.strip().split() for line in
                         open(os.path.join(root, list_path), 'r')]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        files = []
        for item in self.img_list:
            image_path, label_path = item[:2]
            name = os.path.splitext(os.path.basename(label_path))[0]
            sample = {"img": image_path,
                      "label": label_path,
                      "name": name,
                      }
            files.append(sample)
        return files

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]

        image = cv2.imread(os.path.join(self.img_dir, item["img"]),
                           cv2.IMREAD_COLOR)
        label = cv2.imread(os.path.join(self.mask_dir, item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        label = np.asarray(label, dtype=np.int32)
        size = label.shape

        if 'val' in self.list_path:
            image = cv2.resize(image, self.crop_size[::-1],
                               interpolation=cv2.INTER_LINEAR)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            label = cv2.resize(label, self.crop_size[::-1],
                               interpolation=cv2.INTER_NEAREST)
            label = self.label_transform(label)
        elif 'train' in self.list_path:
            if self.flip:
                flip = np.random.choice(2) * 2 - 1
                image = image[:, ::flip, :]
                label = label[:, ::flip]

                if flip == -1:
                    tmp = label.copy()
                    for i in range(0, 3):
                        label[tmp == self.right_idx[i]] = self.left_idx[i]
                        label[tmp == self.left_idx[i]] = self.right_idx[i]

            image, label = self.resize_image(image, label, self.crop_size)
            image, label = self.gen_sample(image, label,
                                           self.multi_scale, False)

        return image.copy(), label.copy(), np.array(size), name

    def inference(self, model, image, flip=True):
        size = image.size()
        pred = model(image.cuda())
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        pred = F.interpolate(pred, size=(size[-2], size[-1]),
                             mode='bilinear', align_corners=False)
        if flip:
            flip_img = torch.flip(image, [3])
            flip_output = model(flip_img.cuda())
            if isinstance(flip_output, (list, tuple)):
                flip_output = flip_output[0]
            flip_output = F.interpolate(flip_output, size=(size[-2], size[-1]),
                                        mode='bilinear', align_corners=False)
            flip_pred = torch.index_select(flip_output, dim=1, index=self.flip_idx.cuda())
            flip_pred = torch.flip(flip_pred, [3])  # flip back
            pred += flip_pred
            pred = pred * 0.5
        pred = F.softmax(pred, dim=1)
        return pred
