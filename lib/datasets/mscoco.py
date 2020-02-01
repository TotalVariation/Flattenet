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
from tqdm import trange

import torch

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class COCOSeg(BaseDataset):
    NUM_CLASS = 21
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
                1, 64, 20, 63, 7, 72]
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

        super(COCOSeg, self).__init__(ignore_label, base_size, crop_size,
                                      scale_factor, mean, std)

        self.root = os.path.abspath(os.path.join(root, 'coco'))
        self.split = list_path

        self.num_classes = num_classes
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        self.crop_size = crop_size

        from pycocotools.coco import COCO
        from pycocotools import mask
        # prepare data
        if 'val' in self.split:
            ann_file = os.path.join(self.root, 'annotations/instances_val2017.json')
            ids_file = os.path.join(self.root, 'annotations/voc_coco_pretrain_val_ids.pth')
        elif 'train' in self.split:
            ann_file = os.path.join(self.root, 'annotations/instances_train2017.json')
            ids_file = os.path.join(self.root, 'annotations/voc_coco_pretrain_train_ids.pth')
        else:
            raise ValueError('Only supporting train and val set.')

        self.coco = COCO(ann_file)
        self.coco_mask = mask
        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)

        logger.info(f'The number of samples of {self.split} set: {len(self.ids)}')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        img_path = img_metadata['file_name']

        if 'train' in self.split:
            img_dir = os.path.join(self.root, 'images/train2017')
            image = cv2.imread(os.path.join(img_dir, img_path))
            size = image.shape
            cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
            label = Image.fromarray(self._gen_seg_mask(
                cocotarget, img_metadata['height'], img_metadata['width']))
            label = np.array(label)
            image, label = self.gen_sample(image, label, self.multi_scale, self.flip)
        elif 'val' in self.split:
            img_dir = os.path.join(self.root, 'images/val2017')
            image = cv2.imread(os.path.join(img_dir, img_path))
            size = image.shape
            image = cv2.resize(image, self.crop_size[::-1],
                               interpolation=cv2.INTER_LINEAR)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
            label = Image.fromarray(self._gen_seg_mask(
                cocotarget, img_metadata['height'], img_metadata['width']))
            label = np.array(label)
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

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while." +
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                      img_metadata['width'])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'.
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids

    @property
    def classes(self):
        """Category names."""
        return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
                'tv'
                )


"""
NUM_CHANNEL = 91
[] background
[5] airplane
[2] bicycle
[16] bird
[9] boat
[44] bottle
[6] bus
[3] car
[17] cat
[62] chair
[21] cow
[67] dining table
[18] dog
[19] horse
[4] motorcycle
[1] person
[64] potted plant
[20] sheep
[63] couch
[7] train
[72] tv
"""
