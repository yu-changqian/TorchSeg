#!/usr/bin/env python
# encoding=utf8

import time
import cv2
import numpy as np

import torch
import torch.nn.functional as F
import torchprof

from engine.logger import get_logger

logger = get_logger()


def compute_speed(model, input_size, device, iteration):
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True

    model.eval()
    model = model.cuda()

    input = torch.randn(*input_size, device=device)

    torch.cuda.synchronize()
    for _ in range(50):
        model(input)
        torch.cuda.synchronize()

    logger.info('=========Speed Testing=========')
    time_spent = []
    for _ in range(iteration):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.no_grad():
            model(input)
        torch.cuda.synchronize()
        time_spent.append(time.perf_counter() - t_start)
    torch.cuda.synchronize()
    elapsed_time = np.sum(time_spent)
    with torchprof.Profile(model, use_cuda=True) as prof:
        model(input)
    print(prof.display(show_events=False))
    logger.info(
        'Elapsed time: [%.2f s / %d iter]' % (elapsed_time, iteration))
    logger.info('Speed Time: %.2f ms / iter    FPS: %.2f' % (
        elapsed_time / iteration * 1000, iteration / elapsed_time))