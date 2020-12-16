from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import time

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count

class Profiler:
    def __init__(self):
        self.start_time = time.time()
        self.interval_times = {"total_time": AverageMeter()}
        self.last_time = self.start_time

    def elapsed_time(self):
        return time.time() - self.start_time

    def interval_trigger(self, interval_name):
        if interval_name not in self.interval_times:
            self.interval_times[interval_name] = AverageMeter()
        self.interval_times[interval_name].update(time.time()-self.last_time)

        self.last_time = time.time()

    def return_interval_times(self):
        sum = 0
        for name, time in interval_times.items():
            sum += time
        
        return self.interval_times

    def print_interval_times(self):
        sum = 0.
        for name, time in self.interval_times.items():
            if name is not "total_time":
                sum += time.avg

        string = "\nTotal time: {:.2f}s\n".format(self.interval_times["total_time"].avg)
        for name, time in self.interval_times.items():
            if name is "total_time":
                continue
            string += "{}: {:.2f}s ({:.2f}%)\n".format(name, time.avg, 100 * time.avg / sum)

        print(string)

    def start(self):
        self.start_time = time.time()
        self.last_time = time.time()

    def pause(self):
        self.interval_times["total_time"].update(time.time() - self.start_time)