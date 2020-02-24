import numpy as np
import torch
import logger

def train(train_loader, model, criterion, optimizer, epoch, datadir):
    average_meter = AverageMeter()

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
        train_loss_f.write("{0}\n".format(loss.data))

        optimizer.zero_grad()
        
        loss.backward()  # compute gradient and do SGD step

        optimizer.step()

        gpu_time = time.time() - end

        # measure accuracy and record loss
        target = target.cpu().numpy()
        pred = pred.cpu().numpy()
        result.update(target, pred)
        end = time.time()

    result.calculate()
    return result