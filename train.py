import numpy as np
import torch
import torch.nn as nn
import torch.optim
immport torch.
import time

import logger

def train(train_loader : torch.utils.data.DataLoader, model : nn.Module, criterion : nn.Module, optimizer : torch.optim.Optimizer) -> logger.Result:
     """
    test for 1 epoch
    Args
        train_loader : a data loader for validation or test.
        model : a deep learning model
        criterion : a criterion for loss function
        optimizer : an optimizer
    Returns
        results of this epoch
    """
    
    # switch to train mode
    model.train()

    end = time.time()
    
    result = logger.Result()

    for i, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()
        data_time = time.time() - end

        # compute pred
        end = time.time()
        pred = model(input)
        loss = criterion(pred, target)

        optimizer.zero_grad()
        
        loss.backward()  # compute gradient and do SGD step

        optimizer.step()

        gpu_time = time.time() - end

        # measure accuracy and record loss
        target = target.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        loss = loss.cpu().detach().item()
        result.update(target, pred, loss)
        end = time.time()

    result.calculate()
    
    return result