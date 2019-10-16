from __future__ import division
import os.path as osp
import sys
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from config import config
from dataloader import get_train_loader
from network import CPNet
from datasets import ADE
from utils.init_func import init_weight, group_weight
from engine.lr_policy import PolyLR
from engine.logger import get_logger
from engine.engine import Engine
from seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d
from seg_opr.seg_oprs import one_hot

try:
    from apex.parallel import SyncBatchNorm, DistributedDataParallel
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")

logger = get_logger()

torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)

parser = argparse.ArgumentParser()

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True
    if engine.distributed:
        torch.cuda.set_device(engine.local_rank)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, ADE)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean',
                                    ignore_index=-1)

    if engine.distributed:
        logger.info('Use the Multi-Process-SyncBatchNorm')
        BatchNorm2d = SyncBatchNorm
    else:
        BatchNorm2d = BatchNorm2d
    model = CPNet(config.num_classes, criterion=criterion,
                  pretrained_model=config.pretrained_model,
                  norm_layer=BatchNorm2d)
    init_weight(model.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    # group weight and config optimizer
    base_lr = config.lr
    # if engine.distributed:
    #     base_lr = config.lr * engine.world_size

    params_list = []
    params_list = group_weight(params_list, model.backbone,
                               BatchNorm2d, base_lr)
    for module in model.business_layer:
        params_list = group_weight(params_list, module, BatchNorm2d,
                                   base_lr)

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = PolyLR(base_lr, config.lr_power, total_iteration)
    optimizer = torch.optim.SGD(params_list,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    if engine.distributed:
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DataParallelModel(model, engine.devices)
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    model.train()

    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)
        for idx in pbar:
            engine.update_iteration(epoch, idx)

            minibatch = dataloader.next()
            imgs = minibatch['data']
            gts = minibatch['label']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)

            b, h, w = gts.size()
            scaled_gts = F.interpolate((gts.view(b, 1, h, w)).float(),
                                       scale_factor=0.125,
                                       mode="nearest")
            b, c, h, w = scaled_gts.size()
            scaled_gts = scaled_gts.squeeze_()
            C = config.num_classes + 1
            one_hot_gts = one_hot(scaled_gts, C).view(b, C, -1)
            similarity_gts = torch.bmm(one_hot_gts.permute(0, 2, 1),
                                       one_hot_gts)

            gts = gts - 1

            loss = model(imgs, gts, similarity_gts)

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                dist.all_reduce(loss, dist.ReduceOp.SUM)
                loss = loss / engine.world_size
            else:
                loss = Reduce.apply(*loss) / len(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            for i in range(0, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.2f' % loss.item()

            pbar.set_description(print_str, refresh=False)

        if (epoch >= config.nepochs - 20) or (
                epoch % config.snapshot_iter == 0):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
