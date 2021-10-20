from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import torch
import json
import math
import os

from .datasets.coco import COCO
from .datasets.kitti import KITTI
from .datasets.wibam import WIBAM
from .datasets.coco_hp import COCOHP
from .datasets.mot import MOT
from .datasets.nuscenes import nuScenes
from .datasets.crowdhuman import CrowdHuman
from .datasets.kitti_tracking import KITTITracking
from .datasets.custom_dataset import CustomDataset
from .datasets.nuscenes_1cls import nuScenes_1cls


dataset_factory = {
  'custom': CustomDataset,
  'coco': COCO,
  'kitti': KITTI,
  'coco_hp': COCOHP,
  'mot': MOT,
  'nuscenes': nuScenes,
  'crowdhuman': CrowdHuman,
  'kitti_tracking': KITTITracking,
  'wibam': WIBAM,
  'nuscenes_1cls': nuScenes_1cls
}

class ConcatDatasets():
  def __init__(self, dataloaders):
    self.dataloaders = dataloaders
    self.batch_size = self.dataloaders[0].batch_size + self.dataloaders[1].batch_size

  def __iter__(self):
    self.loader_iter = []
    for data_loader in self.dataloaders:
      self.loader_iter.append(iter(data_loader))
    return self

  def __next__(self):
    out = []
    for data_iter in self.loader_iter:
      out.append(next(data_iter))
    return tuple(out)

  def __len__(self):
    length = min([len(self.dataloaders[0]),len(self.dataloaders[1])])
    return length

def get_dataset(dataset):
  return dataset_factory[dataset]

def mixed_dataset(dataset_1, dataset_2, batch_size1, batch_size2,
                  num_workers, task, drop_last=True, shuffle=True,
                  opt=None):
  Dataset1 = get_dataset(dataset_1)(opt, task)
  Dataset2 = get_dataset(dataset_2)(opt, task)

  if task == 'val':
    batch_size = math.ceil(len(Dataset1)/len(Dataset2))
    mult = 2 * math.floor(opt.batch_size/batch_size)
    batch_size1 = batch_size * mult
    batch_size2 = mult

  # opt = opts().update_dataset_info_and_set_heads(self.opt, Dataset1)

  DataLoader1 = torch.utils.data.DataLoader(
    Dataset1, batch_size=batch_size1, num_workers=num_workers,
    drop_last=drop_last, shuffle=shuffle
  )

  print("Main {} dataloader:\n {}\nIterations: {}\nSamples: {}\n".format(
      task, DataLoader1.dataset, len(DataLoader1), 
      len(DataLoader1.dataset))
  )

  DataLoader2 = torch.utils.data.DataLoader(
    Dataset2, batch_size=batch_size2, num_workers=num_workers,
    drop_last=drop_last, shuffle=shuffle
  )

  print("Mixed {} dataloader:\n {}\nIterations: {}\nSamples: {}\n".format(
      task, DataLoader2.dataset, len(DataLoader2), 
      len(DataLoader2.dataset))
  )

  MixedDataloader = ConcatDatasets([DataLoader1, DataLoader2])

  return MixedDataloader