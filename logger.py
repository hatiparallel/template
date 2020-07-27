import numpy as np
import torch
import csv
import pandas as pd
import os
import matplotlib.pyplot as plt

# ALL YOU HAVE TO DO IS REWRITE Result.__init__

# base class for metrics
class Metric(object):
    def __init__(self, ini, updater, calculator):
        self.value = ini
        self.updater = updater
        self.calculator = calculator
    def update(self, target, pred):
        self.value += self.updater(target, pred)
    def calculate(self, count):
        self.value = self.calculator(self.value, count)
        return self.value


class LossMetric(object):
    def __init__(self, ini):
        self.value = ini
    def update(self, loss):
        self.value += loss
    def calculate(self, count):
        self.value = self.value / count
        return self.value    




class Result(object):
    """
    You define this object in train.py or test.py once for every epoch.
    This object has all the metrics to calculate.
    Write metrics for yourself.

    0. Do you need something except for the target and pred ?
       Then rewrite Result and Metric.
    1. define updater
    2. define calculator
    3. initialize the metric with the updater and the calculator

    update must be called after one batch calculation.
    calculate must be called after one epoch.
    """
    def __init__(self):
        self.count = 0

        def accuracy_updater(target, pred):
            if target == np.argmax(pred):
                return 1
            else:
                return 0

        def accuracy_calculator(value, count):
            return value / count

        self.accuracy = Metric(0, accuracy_updater, accuracy_calculator)

        self.loss = LossMetric(0)

    def update(self, targets : torch.Tensor, preds : torch.Tensor, loss : float) -> None:
        """
        update for one mini-batch
        Args
            targets : target tensor
            preds : model prediction tensor
            loss : loss value
        Returns 
            None
        """
        self.loss.update(loss)

        for target, pred in zip(targets, preds):
            for key, value in self.__dict__.items():
                if value.__class__.__name__ == "Metric":
                    getattr(self, key).update(target, pred)
            self.count += 1

    def calculate(self):
        """
        summarize one epoch
        Args
            None
        Returns 
            None
        """
        self.loss.calculate(self.count)

        for key, value in self.__dict__.items():
            if value.__class__.__name__ == "Metric" or value.__class__.__name__ == "LossMetric":
                getattr(self, key).calculate(self.count)

# use once for one training
# save and visualize the results of each epoch
class Logger(object):
    """
    You define this object in main.py once in all the training procedure.
    This object has all the results of the metrics.

    append must be called after one epoch.
    write_into_file must be called after all the training procedure.
    """
    def __init__(self, place : str, result : Result, file : str):
    """
    Args
        place : path to save the results
        result : result of the first epoch
        file : filename (without path)
    """
        self.file = file
        self.place = place
        self.fieldnames = []

        log_dict = {}

        for key, value in result.__dict__.items():
            if value.__class__.__name__ == "Metric" or value.__class__.__name__ == "LossMetric":
                log_dict[key] = value.value
                self.fieldnames.append(key)
                setattr(self, key, [value])

        with open(self.file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerow(log_dict)

    def append(self, result : Result) -> None:
        """
        Args
            result : the result of the latest epoch
        Returns
            None
        """
        log_dict = {}

        for key, value in result.__dict__.items():
            if value.__class__.__name__ == "Metric" or value.__class__.__name__ == "LossMetric":
                logs = getattr(self, key)
                logs.append(value)
                log_dict[key] = value.value
                setattr(self, key, logs)

        with open(self.file, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(log_dict)

    def write_into_file(self, name : str) -> None:
        """
        Args
            name : file name (without path) to save jpg
        Returns
            None
        """
        df = pd.read_csv(self.file)
        values = df.values.T
        columns = df.columns
        epochs = np.array([i for i in range(len(values[0]))])

        for i, c in enumerate(columns):
            filename = os.path.join(self.place, "{}_{}.jpg".format(name, c))
            plt.figure()
            plt.plot(epochs, values[i])
            plt.savefig(filename)