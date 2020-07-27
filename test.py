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

def validate(val_loader : torch.utils.data.DataLoader, model : nn.Module, criterion : nn.Module, optimizer : torch.optim.Optimizer) -> logger.Result:
    """
    test for 1 epoch
    Args
        val_loader : a data loader for validation or test.
        model : a deep learning model
        criterion : a criterion for loss function
        optimizer : an optimizer
    Returns
        results of this epoch
    """

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
            loss = criterion(pred, target)

        gpu_time = time.time() - end

        # measure accuracy and record loss
        target = target.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        loss = loss.cpu().detach().item()
        result.update(target, pred, loss)
        end = time.time()

    result.calculate()
    return result