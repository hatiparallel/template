import argparse
import os
import time
import sys
import csv
import numpy as np
from statistics import mean, median, variance, stdev

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import torch.autograd

import criteria
import misc
import logger

def validate(val_loader, model, epoch, datadir, write_to_file=True):
    result = logger.Result()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    targets, preds, uncertainties = [], [], []
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input)

        gpu_time = time.time() - end

        # measure accuracy and record loss
        target = target.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        result.update(target, pred)
        end = time.time()

    result.calculate()
    return result