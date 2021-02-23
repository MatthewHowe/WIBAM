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
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin

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
		optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
						optimizer, mode='min', factor=0.1, patience=2,
						threshold=0.001, verbose=True)
		return {"optimizer": optimizer, "lr_scheduler": scheduler,
				"monitor": "val_tot"}

	def training_step(self, train_batch, batch_idx):
		main_out = self(train_batch[0]['image'])[0]
		mix_out = self(train_batch[1]['image'])[0]

		main_loss, main_loss_stats = self.main_loss(main_out, train_batch[0])
		mix_loss, mix_loss_stats = self.mix_loss(mix_out, train_batch[1])
		
		for key, val in main_loss_stats.items():
			self.log("train_main_{}".format(key), val, on_epoch=True)
		for key, val in mix_loss_stats.items():
			self.log("train_mix_{}".format(key), val, on_epoch=True)
		total_loss = main_loss + mix_loss
		self.log("train_tot", total_loss, on_epoch=True)
		return {"loss": total_loss}

	def training_epoch_end(self, training_step_outputs):
		format_dict = {}
		for key, val in [(key, dict[key]) for dict in training_step_outputs for key in dict]:
			if key not in format_dict:
				format_dict[key] = [val]
			else:
				format_dict[key].append(val)
		variance = torch.var(torch.stack(format_dict['loss']))
		self.log("train_variance", variance, on_epoch=True)

	def validation_step(self, val_batch, batch_idx):
		main_out = self(val_batch[0]['image'])[0]
		mix_out = self(val_batch[1]['image'])[0]

		main_loss, main_loss_stats = self.main_loss(main_out, val_batch[0])
		mix_loss, mix_loss_stats = self.mix_loss(mix_out, val_batch[1])

		for key, val in main_loss_stats.items():
			self.log("val_main_{}".format(key), val, on_epoch=True)
		for key, val in mix_loss_stats.items():
			self.log("val_mix_{}".format(key), val, on_epoch=True)
		total_val = main_loss + mix_loss
		self.log("val_tot", total_val, on_epoch=True)

		return main_loss

	def validation_epoch_end(self, validation_step_outputs):
		variance = torch.var(torch.stack(validation_step_outputs))
		mean = torch.mean(torch.stack(validation_step_outputs))
		self.log("val_variance", variance, on_epoch=True)

class MyDDP(DDPPlugin):

    def configure_ddp(self, model, device_ids=device_ids):
        model = LightningDistributedDataParallel(model, device_ids, find_unused_parameters=True)
        return model

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
		num_workers=opt.num_workers, drop_last=True
	)

	mixed_loader = torch.utils.data.DataLoader(
		MixedDataset(opt, 'train'), batch_size=opt.mixed_batchsize,
		num_workers=opt.num_workers, drop_last=True
	)

	MixedDataloader = ConcatDatasets([training_loader, mixed_loader])

	val_loader = torch.utils.data.DataLoader(
		MainDataset(opt, 'val'), batch_size=opt.batch_size,
		num_workers=opt.num_workers, drop_last=True
	)

	val_mixed_loader = torch.utils.data.DataLoader(
		MixedDataset(opt, 'val'), batch_size=opt.batch_size,
		num_workers=opt.num_workers, drop_last=True
	)

	MixedValLoader = ConcatDatasets([val_loader, val_mixed_loader])

	# model
	model = LitWIBAM()
	state_dict = torch.load(opt.load_model)
	state_dict['state_dict'] = {'model.' + str(key): val for key, val in state_dict['state_dict'].items()}
	model.load_state_dict(state_dict['state_dict'])

	checkpoint_callback = ModelCheckpoint(monitor="val_main_tot", save_last=True, 
										  save_top_k=2, mode='min', period=2
										  )

	my_ddp = MyDDP()

	# training
	trainer = pl.Trainer(checkpoint_callback=True,
						 callbacks=[checkpoint_callback],
						 default_root_dir=opt.output_path, 
						 gpus=opt.gpus, accelerator="ddp",
						 check_val_every_n_epoch=1,
						 plugins=[my_ddp]
						 )
	
	trainer.fit(model, MixedDataloader, MixedValLoader)