import numpy as np
import torch
import time

import logger

def train(train_loader, model, criterion, optimizer, epoch, datadir):
    
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
        result.update(target, pred)
        end = time.time()

    result.calculate()
    return result