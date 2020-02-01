# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

import _init_paths
import datasets
import models
from config import config
from config import update_config
from core.criterion import CrossEntropy, OhemCrossEntropy
from core.function import train, validate
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel, get_optimizer, get_rank
from utils.lr_scheduler import get_lr_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description='Train Segmentation Network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    distributed = len(gpus) > 1
    device = torch.device('cuda:{}'.format(args.local_rank))

    # build model
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    if args.local_rank == 0:
        logger.info(model)
        tot_params = sum(p.numel() for p in model.parameters()) / 1000000.0
        logger.info(f">>> total params: {tot_params:.2f}M")

        # provide the summary of model
        dump_input = torch.rand(
            (1, 3, config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1])
        )
        logger.info(get_model_summary(model.to(device), dump_input.to(device)))

        # copy model file
        this_dir = os.path.dirname(__file__)
        models_dst_dir = os.path.join(final_output_dir, 'models')
        if os.path.exists(models_dst_dir):
            shutil.rmtree(models_dst_dir)
        shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )

    # prepare data
    train_dataset = eval('datasets.'+config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TRAIN_SET,
        num_samples=None,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=config.TRAIN.MULTI_SCALE,
        flip=config.TRAIN.FLIP,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TRAIN.BASE_SIZE,
        crop_size=tuple(config.TRAIN.IMAGE_SIZE),  # (height, width)
        scale_factor=config.TRAIN.SCALE_FACTOR
    )

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE and train_sampler is None,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler
    )

    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_samples=config.TEST.NUM_SAMPLES,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=tuple(config.TEST.IMAGE_SIZE),  # (height, width)
    )

    if distributed:
        test_sampler = DistributedSampler(test_dataset)
    else:
        test_sampler = None

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=test_sampler
    )

    # criterion
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     weight=train_dataset.class_weights,
                                     thresh=config.LOSS.OHEMTHRESH,
                                     min_kept=config.LOSS.OHEMKEEP
                                     )
    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                 weight=train_dataset.class_weights)

    model_state_file = config.MODEL.PRETRAINED
    logger.info('=> Loading model from {}'.format(model_state_file))
    pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info('=> Loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = FullModel(model, criterion)
    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank)

    # optimizer
    optimizer = get_optimizer(config, model)

    epoch_iters = np.int(train_dataset.__len__() /
                         config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    best_mIoU = 0
    last_epoch = 0
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file,
                                    map_location=lambda storage, loc: storage)
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters

    # learning rate scheduler
    lr_scheduler_dict = {
        'optimizer': optimizer,
        'milestones': [s * epoch_iters for s in config.TRAIN.LR_STEP],
        'gamma': config.TRAIN.LR_FACTOR,
        'max_iters': num_iters,
        'last_epoch': last_epoch,
        'epoch_iters': epoch_iters
    }
    lr_scheduler = get_lr_scheduler(config.TRAIN.LR_SCHEDULER, **lr_scheduler_dict)

    for epoch in range(last_epoch, end_epoch):
        if distributed:
            train_sampler.set_epoch(epoch)
        train(config, epoch, end_epoch, epoch_iters, trainloader, optimizer,
              lr_scheduler, model, writer_dict, device)

        valid_loss, mean_IoU = validate(config, testloader, model, writer_dict, device)

        if args.local_rank == 0:
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + '/checkpoint.pth.tar'))
            torch.save(
                {'epoch': epoch+1,
                 'best_mIoU': best_mIoU,
                 'state_dict': model.module.state_dict(),
                 'optimizer': optimizer.state_dict()},
                os.path.join(final_output_dir, 'checkpoint.pth.tar')
            )

            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                torch.save(model.module.state_dict(),
                           os.path.join(final_output_dir, 'best.pth'))
            msg = f'Loss: {valid_loss:.4f}, MeanIU: {mean_IoU: 4.4f}, \
                        Best_mIoU: {best_mIoU: 4.4f}'
            logger.info(msg)

            if epoch == end_epoch - 1:
                torch.save(model.module.state_dict(),
                           os.path.join(final_output_dir, 'final_state.pth'))

                writer_dict['writer'].close()
                end = timeit.default_timer()
                logger.info(f'Hours: {np.int((end-start)/3600)}')
                logger.info('Done!')


if __name__ == '__main__':
    main()
