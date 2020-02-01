# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from utils.utils import AverageMeter, get_world_size, get_rank
from utils.metric import batch_intersection_union, batch_pix_accuracy


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp


def train(config, epoch, num_epoch, epoch_iters, trainloader,
          optimizer, lr_scheduler, model, writer_dict, device):
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    rank = get_rank()
    world_size = get_world_size()

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, _, _ = batch
        labels = labels.long().to(device)
        images = images.to(device)

        loss, _ = model(images, labels)
        reduced_loss = reduce_tensor(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        lr = optimizer.param_groups[0]['lr']

        if i_iter % config.PRINT_FREQ == 0 and rank == 0:
            print_loss = ave_loss.average() / world_size
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), lr, print_loss)
            logging.info(msg)

    if rank == 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', ave_loss.average() / world_size, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1


def validate(config, testloader, model, writer_dict, device):
    rank = get_rank()
    world_size = get_world_size()
    model.eval()
    ave_loss = AverageMeter()
    tot_inter = np.zeros(config.DATASET.NUM_CLASSES)
    tot_union = np.zeros(config.DATASET.NUM_CLASSES)
    with torch.no_grad():
        for i_iter, batch in enumerate(testloader):
            image, label, _, _ = batch
            size = label.size()
            label = label.long().to(device)
            image = image.to(device)

            loss, pred = model(image, label)
            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(pred, size=(size[-2], size[-1]),
                                     mode='bilinear', align_corners=False)
            reduced_loss = reduce_tensor(loss)
            ave_loss.update(reduced_loss.item())

            batch_inter, batch_union = batch_intersection_union(pred, label,
                                                                config.DATASET.NUM_CLASSES)
            tot_inter += batch_inter
            tot_union += batch_union

            if i_iter % config.PRINT_FREQ == 0 and rank == 0:
                msg = f'Iter: {i_iter}, Loss: {ave_loss.average() / world_size:.6f}'
                logging.info(msg)

    tot_inter = torch.from_numpy(tot_inter).to(device)
    tot_union = torch.from_numpy(tot_union).to(device)
    tot_inter = reduce_tensor(tot_inter).cpu().numpy()
    tot_union = reduce_tensor(tot_union).cpu().numpy()
    IoU = np.float64(1.0) * tot_inter / (np.spacing(1, dtype=np.float64) + tot_union)
    mean_IoU = IoU.mean()
    print_loss = ave_loss.average() / world_size

    if rank == 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', print_loss, global_steps)
        writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return print_loss, mean_IoU


def testval(config, test_dataset, testloader, model, sv_dir='', sv_pred=False):
    model.eval()
    tot_inter, tot_union, tot_pix_correct, tot_pix_labeled = 0, 0, 0, 0
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name = batch
            size = label.size()
            pred = test_dataset.multi_scale_inference(
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(pred, (size[-2], size[-1]),
                                     mode='bilinear', align_corners=False)

            pix_correct, pix_labeled = batch_pix_accuracy(pred, label)
            tot_pix_correct += pix_correct
            tot_pix_labeled += pix_labeled
            inter, union = batch_intersection_union(pred, label,
                                                    config.DATASET.NUM_CLASSES)
            tot_inter += inter
            tot_union += union

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info(f'processing: {index} images')
                IoU = np.float64(1.0) * tot_inter / (np.spacing(1, dtype=np.float64) + tot_union)
                mean_IoU = IoU.mean()
                pixAcc = 1.0 * tot_pix_correct / (np.spacing(1, dtype=np.float64) + tot_pix_labeled)
                logging.info(f'mIoU: {mean_IoU:.4f}, pixAcc: {pixAcc:.4f}')

    pixel_acc = 1.0 * tot_pix_correct / (np.spacing(1, dtype=np.float64) + tot_pix_labeled)
    IoU = np.float64(1.0) * tot_inter / (np.spacing(1, dtype=np.float64) + tot_union)
    mean_IoU = IoU.mean()

    return mean_IoU, pixel_acc


def test(config, test_dataset, testloader, model, sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(pred, (size[-2], size[-1]),
                                     mode='bilinear', align_corners=False)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
