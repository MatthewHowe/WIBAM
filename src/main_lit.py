from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

from torch.utils.tensorboard import SummaryWriter
from opts import opts
from model.model import create_model, load_model, save_model
from utils.collate import default_collate, instance_batching_collate
from dataset.dataset_factory import get_dataset
from utils.net import *
from utils.utils import Profiler
from trainer import MultiviewLoss

class LitWIBAM(pl.LightningModule):
	def __init__(self):
		super().__init__()
		opt = opts().parse()
		Dataset = get_dataset(opt.dataset)
		self.loss = MultiviewLoss(opt)
		opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
		self.model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)

	def forward(self, x):
		output = self.model(x)
		return output

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		x = train_batch['image']
		z = self.model(x)
		loss, loss_stats = self.loss(z, train_batch)
		for key, val in loss_stats.items():
			self.log("train_{}".format(key), val)
		return loss

	def validation_step(self, val_batch, batch_idx):
		x = val_batch['image']
		z = self.model(x)
		loss, loss_stats = self.loss(z, val_batch)
		for key, val in loss_stats.items():
			self.log("val_{}".format(key), val)

opt = opts().parse()

# data
Dataset = get_dataset(opt.dataset)
opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), batch_size=opt.batch_size,
	  num_workers=opt.num_workers)

val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), batch_size=opt.batch_size,
	  num_workers=opt.num_workers)

# model
model = LitWIBAM()
state_dict = torch.load(opt.load_model)
state_dict['state_dict'] = {'model.' + str(key) : val for key, val in state_dict['state_dict'].items()}
model.load_state_dict(state_dict['state_dict'])
# training
trainer = pl.Trainer(gpus=opt.gpus)
trainer.fit(model, train_loader, val_loader)