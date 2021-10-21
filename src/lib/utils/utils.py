from __future__ import absolute_import, division, print_function

import time

import torch


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
        self.interval_times[interval_name].update(time.time() - self.last_time)

        self.last_time = time.time()

    def return_interval_times(self):
        sum = 0
        for name, time in interval_times.items():
            sum += time

        return self.interval_times

    def print_interval_times(self):
        sum = 0.0
        for name, time in self.interval_times.items():
            if name != "total_time":
                sum += time.avg

        string = "\nTotal time: {:.2f}s\n".format(self.interval_times["total_time"].avg)
        for name, time in self.interval_times.items():
            if name == "total_time":
                continue
            string += "{}: {:.6f}s ({:.2f}%)\n".format(
                name, time.avg, 100 * time.avg / sum
            )

        print(string)

    def start(self):
        self.start_time = time.time()
        self.last_time = time.time()

    def pause(self):
        self.interval_times["total_time"].update(time.time() - self.start_time)


def separate_batches(output, batch_size):
    ret = []
    for batch in range(batch_size):
        ret.append({})
    for key, val in output.items():
        for batch in range(batch_size):
            ret[batch][key] = val[batch].to("cpu")

    return ret


def attribute_lists_to_objects(detections, detection_thresh=0.4):
    objects = {}
    max_objects = detections["scores"].size
    for obj in range(max_objects):
        if detections["scores"][obj] < detection_thresh:
            continue
        attributes = {}
        attributes["location"] = detections["location_wcf"][obj]
        size = detections["size"][obj]
        size[0], size[2] = size[2], size[0]
        attributes["location"][2] = attributes["location"][2] - size[2] / 2
        attributes["size"] = size
        attributes["rot"] = detections["rot"][obj]

        objects[obj] = attributes

    return objects


def objects_to_attribute_list(objects):
    attributes = {}
    if isinstance(objects, list):
        objects = {i: objects[i] for i in range(len(objects))}
    for obj, obj_attributes in objects.items():
        for key, val in obj_attributes.items():
            if key in attributes:
                attributes[key].append(val)
            else:
                attributes[key] = [val]

    return attributes
