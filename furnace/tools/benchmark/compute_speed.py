#!/usr/bin/env python

import time

import torch

from engine.logger import get_logger

logger = get_logger()


def compute_speed(model, input_size, device, iteration):
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True

    model.eval()
    model = model.cuda()

    input = torch.randn(*input_size, device=device)

    for _ in range(10):
        model(input)

    logger.info('=========Speed Testing=========')
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iteration):
        model(input)
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start
    logger.info(
        'Elapsed time: [%.2f s / %d iter]' % (elapsed_time, iteration))
    logger.info('Speed Time: %.2f ms / iter    FPS: %.2f' % (
        elapsed_time / iteration * 1000, iteration / elapsed_time))
