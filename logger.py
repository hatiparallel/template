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


class LossMetric(object):
    def __init__(self, ini):
        self.value = ini
    def update(self, loss):
        self.value += loss
    def calculate(self, count):
        self.value = self.value / count
        return self.value    



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

        self.loss = LossMetric(0)

    def update(self, targets, preds, loss):
        self.loss.update(loss)

        for target, pred in zip(targets, preds):
            for key, value in self.__dict__.items():
                if value.__class__.__name__ == "Metric":
                    getattr(self, key).update(target, pred)
            self.count += 1

    def calculate(self):
        self.loss.calculate(self.count)

        for key, value in self.__dict__.items():
            if value.__class__.__name__ == "Metric" or value.__class__.__name__ == "LossMetric":
                getattr(self, key).calculate(self, self.count)

# use once for one training
# save and visualize the results of each epoch
class Logger(object):
    def __init__(self, dir, start_epoch, result):
        self.dir = dir
        self.epochs = [start_epoch]
        for key, value in result.__dict__.items():
            if value.__class__.__name__ == "Metric" or value.__class__.__name__ == "LossMetric":
                log = getattr(result, key).value
                setattr(self, key, [log])

    def append(self, result, file):
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

    def write_into_file(self, name):
        df = pd.read_csv(self.file)
        values = df.values.T
        columns = df.columns
        epochs = np.array([i for i in range(len(values[0]))])

        for i, c in enumerate(columns):
            filename = os.path.join(self.place, "{}_{}.jpg".format(name, c))
            plt.figure()
            plt.plot(epochs, values[i])
            plt.savefig(filename)