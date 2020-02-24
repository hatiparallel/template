import numpy as np
import torch
import csv

# base class for metrics
class Metric(object):
    def __init__(self, ini, updater, calculator):
        self.value = ini
        self.updater = updater
        self.calculator = calculator
    def update(self, target, pred):
        self.value += self.updater(target, pred)
    def calculate(self, count):
        return self.value = self.calculator(self.value, count)

# use once for every epoch
# have metrics
# write metrics for yourself
# 0. Do you need something except for the target and pred ?
#    Then rewrite Result and Metric.
# 1. define updater
# 2. define calculator
# 3. initialize the metric
class Result(object):
    def __init__(self):
        self.count = 0

        def mae_updater(target, pred):
            return np.abs(target - pred)
        def mae_calculator(value, count):
            return value / count
        self.mae = Metric(0, mae_updater, mae_calculator)

    def update(self, targets, preds):
        for target, pred in zip(targets, preds):
            for key, value in self.__dict__.items():
                if value.__class__.__name__ == "Metric":
                    getattr(self, key).update(target, pred)
            count += 1

    def calculate(self):
        for key, value in self.__dict__.items():
            if value.__class__.__name__ == "Metric":
                getattr(self, key).calculate(self, self.count)

# use once for one training
# save and visualize the results of each epoch
class Logger(object):
    def __init__(self, dir, start_epoch, result):
        self.dir = dir
        self.epochs = [start_epoch]
        for key, value in result.__dict__.items():
            if value.__class__.__name__ == "Metric":
                log = getattr(result, key).value
                setattr(self, key, [log])

    def append(result, file):
        log_dict = {}

        for key, value in result.__dict__.items():
            if value.__class__.__name__ == "Metric":
                log = getattr(result, key)
                logs = getattr(self, key).append(log)
                log_dict[key] = value
                setattr(self, key, logs)

        wwith open(file, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(log_dict)

    def write_into_csv(name):
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                                'data_time': avg.data_time, 'gpu_time': avg.gpu_time})