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
from trainer import MultiviewLoss, GenericLoss

class LitWIBAM(pl.LightningModule):
	def __init__(self):
		super().__init__()
		opt = opts().parse()
		Dataset = get_dataset(opt.dataset)
		opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
		self.opt = opt
		self.main_loss = GenericLoss(opt)
		self.mix_loss = MultiviewLoss(opt)
		self.model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)

	def forward(self, x):
		output = self.model(x)
		return output

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		main_out = self(train_batch[0]['image'])[0]
		mix_out = self(train_batch[1]['image'])[0]

		main_loss, main_loss_stats = self.main_loss(main_out, train_batch[0])
		mix_loss, mix_loss_stats = self.mix_loss(mix_out, train_batch[1])
		
		for key, val in main_loss_stats.items():
			self.log("train_main_{}".format(key), val)
		for key, val in mix_loss_stats.items():
			self.log("train_mix_{}".format(key), val)
		return main_loss + mix_loss

	def training_epoch_end(self, training_step_outputs):
		gsutil_sync(True, "aiml-reid-casr-data", Path("lightning_logs"),
					"", bucket_prefix_folder="lightning_experiments")

	def validation_epoch_end(self, validation_step_outputs):
		gsutil_sync(True, "aiml-reid-casr-data", Path("lightning_logs"),
					"", bucket_prefix_folder="lightning_experiments")

	def validation_step(self, val_batch, batch_idx):
		x = val_batch['image']
		z = self.model(x)
		loss, loss_stats = self.main_loss(z, val_batch)
		for key, val in loss_stats.items():
			self.log("val_{}".format(key), val)

class ConcatDatasets(torch.utils.data.Dataset):
	def __init__(self, dataloaders):
		self.dataloaders = dataloaders

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
		batch_size = self.dataloaders[0].batch_size + self.dataloaders[1].batch_size
		length = len(self.dataloaders[0].dataset) + len(self.dataloaders[1].dataset)
		return int(length/batch_size)

if __name__ == '__main__':
	opt = opts().parse()

	# data
	MainDataset = get_dataset(opt.dataset)
	MixedDataset = get_dataset(opt.mixed_dataset)
	opt = opts().update_dataset_info_and_set_heads(opt, MainDataset)

	training_loader = torch.utils.data.DataLoader(
		MainDataset(opt, 'train'), batch_size=opt.batch_size,
		num_workers=opt.num_workers
	)

	print(len(training_loader.dataset))

	mixed_loader = torch.utils.data.DataLoader(
		MixedDataset(opt, 'train'), batch_size=opt.mixed_batchsize,
		num_workers=opt.num_workers
	)

	val_loader = torch.utils.data.DataLoader(
		MainDataset(opt, 'val'), batch_size=opt.batch_size,
		num_workers=opt.num_workers
	)

	MixedDataloader = ConcatDatasets([training_loader, mixed_loader])

	# model
	model = LitWIBAM()
	state_dict = torch.load(opt.load_model)
	state_dict['state_dict'] = {'model.' + str(key) : val for key, val in state_dict['state_dict'].items()}
	model.load_state_dict(state_dict['state_dict'])
	# training
	trainer = pl.Trainer(gpus=opt.gpus, accelerator="ddp")
	trainer.fit(model, MixedDataloader, val_loader)