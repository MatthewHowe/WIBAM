from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import math
from pathlib import Path
import fsspec
import torch
from torch import nn
import cv2
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin

from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel

from torch.utils.tensorboard import SummaryWriter
from opts import opts
from model.model import create_model, load_model, save_model
from utils.mv_utils import test_post_process, compare_ground_truth
from model.decode import generic_decode
from utils.post_process import generic_post_process
from utils.collate import default_collate, instance_batching_collate
from dataset.dataset_factory import mixed_dataset, get_dataset
from dataset.datasets.wibam import WIBAM_test
from utils.net import *
from utils.utils import Profiler, separate_batches
from trainer import MultiviewLoss, GenericLoss

class LitWIBAM(pl.LightningModule):
	def __init__(self):
		super().__init__()
		opt = opts().parse()
		Dataset = get_dataset(opt.dataset)
		opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
		self.opt = opt
		if self.opt.dataset == "wibam":
			self.main_loss = MultiviewLoss(opt)
		else:
			self.main_loss = GenericLoss(opt)
		self.model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)

	def forward(self, x):
		output = self.model(x)
		return output

	def train_dataloader(self):
		if self.opt.mixed_dataset is not None:
			DataLoader = mixed_dataset(
				self.opt.dataset, self.opt.mixed_dataset,
				self.opt.batch_size, self.opt.mixed_batchsize,
				num_workers=self.opt.num_workers,
				task = "train", opt=self.opt,
				drop_last=True, shuffle=True
			)
		else:
			DataLoader = torch.utils.data.DataLoader(
				get_dataset(self.opt.dataset)(self.opt, 'train'), 
				batch_size=self.opt.batch_size,
				num_workers=self.opt.num_workers, 
				drop_last=True, shuffle=True
			)

		return DataLoader

	def val_dataloader(self):
		if self.opt.mixed_dataset is not None:
			DataLoader = mixed_dataset(
				self.opt.dataset, self.opt.mixed_dataset,
				self.opt.batch_size, self.opt.mixed_batchsize,
				num_workers=self.opt.num_workers,
				task = "val", opt=self.opt,
				drop_last=True, shuffle=False
			)
		else:
			DataLoader = torch.utils.data.DataLoader(
				get_dataset(self.opt.dataset)(self.opt, 'val'), 
				batch_size=self.opt.batch_size * 2,
				num_workers=self.opt.num_workers, 
				drop_last=True, shuffle=False
			)

		return DataLoader

	def test_dataloader(self):
		DataLoader = torch.utils.data.DataLoader(
			WIBAM_test(self.opt),
			batch_size=1,
			num_workers=0,
		   	drop_last=False, shuffle=False
		)
		return DataLoader

	def configure_optimizers(self):
		for name, param in self.model.named_parameters():
			# Setting rotation to not change weights
			if name.split(".")[0] == "rot":
				param.requires_grad=False
		optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
						optimizer, mode='min', factor=0.1, patience=2,
						threshold=0.001, verbose=True)
		return {"optimizer": optimizer, "lr_scheduler": scheduler,
				"monitor": "val_tot"}

	def training_step(self, train_batch, batch_idx):
		if self.opt.mixed_dataset is not None:
			main_out = self(train_batch[0]['image'])[0]
			mix_out = self(train_batch[1]['image'])[0]

			main_loss, main_loss_stats = self.main_loss(main_out, train_batch[0])
			mix_loss, mix_loss_stats = self.mix_loss(mix_out, train_batch[1])

			main_loss_stats = {'train_main_'+str(key): val for key, val in main_loss_stats.items()}
			mix_loss_stats = {'train_mix_'+str(key): val for key, val in mix_loss_stats.items()}
			self.log_dict(main_loss_stats, on_epoch=True)
			self.log_dict(mix_loss_stats, on_epoch=True)

			total_loss = main_loss + mix_loss
			self.log("train_tot", total_loss, on_epoch=True)

		else:
			main_out = self(train_batch['image'])[0]
			main_loss, main_loss_stats = self.main_loss(main_out, train_batch)

			main_loss_stats = {'train_main_'+str(key): val for key, val in main_loss_stats.items()}
			self.log_dict(main_loss_stats, on_epoch=True)
			total_loss = main_loss

		return total_loss

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
		if self.opt.mixed_dataset is not None:
			main_out = self(val_batch[0]['image'])[0]
			mix_out = self(val_batch[1]['image'])[0]

			main_loss, main_loss_stats = self.main_loss(main_out, val_batch[0])
			mix_loss, mix_loss_stats = self.mix_loss(mix_out, val_batch[1])
			
			main_loss_stats = {'val_main_'+str(key): val for key, val in main_loss_stats.items()}
			mix_loss_stats = {'val_mix_'+str(key): val for key, val in mix_loss_stats.items()}
			self.log_dict(main_loss_stats, on_epoch=True)
			self.log_dict(mix_loss_stats, on_epoch=True)

			total_val = main_loss + mix_loss
			self.log("val_tot", total_val, on_epoch=True)
		else:
			main_out = self(val_batch['image'])[0]
			main_loss, main_loss_stats = self.main_loss(main_out, val_batch)
			main_loss_stats = {'val_main_'+str(key): val for key, val in main_loss_stats.items()}
			self.log_dict(main_loss_stats, on_epoch=True)
			self.log("val_tot", main_loss, on_epoch=True)
		return main_loss

	def validation_epoch_end(self, validation_step_outputs):
		variance = torch.var(torch.stack(validation_step_outputs))
		mean = torch.mean(torch.stack(validation_step_outputs))
		self.log("val_variance", variance, on_epoch=True)

	def on_test_epoch_start(self):

		self.results = {}
		if opt.save_video:
			fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
			self.ImageWriter = cv2.VideoWriter(
				'image_out.avi', fourcc, 12, (1920,1080)
			)
			self.BevWriter = cv2.VideoWriter(
				'bev_out.avi', fourcc, 12, (800,800)
			)
			self.JointWriter = cv2.VideoWriter(
				'joint_out.avi', fourcc, 12, (3000, 1080)
			)

	def test_step(self, test_batch,  test_idx):
		if self.opt.test:
			out = self(test_batch['image'])[0]
			for key, val in out.items():
				out[key] = val.cpu()
			out = generic_decode(out, self.opt.K, self.opt)
			# test_batch['meta'] = separate_batches(test_batch['meta'], 
			# 									  opt.batch_size)
			out = generic_post_process(self.opt, out, 
				test_batch['meta']['c'].cpu(), test_batch['meta']['s'].cpu(),
				96,320,10, test_batch['meta']['calib'].cpu())
			for i in range(opt.batch_size):
				meta = test_batch['meta']
				out[i] = generic_post_process(self.opt, out, 
					test_batch['meta']['c'][i], test_batch['meta']['s'][i],
					96, 320, 10, test_batch['meta']['calib'][i])
			out = separate_batches(out, test_batch['image'].shape[0])
			# self.test_dataloader.dataloader.dataset.save_mini_result(
			# 	out, test_batch
			# )
			save_res = []
			for i in range(len(out)):
				self.results[test_batch['image_id'][i]] = out[i]
			self.test_dataloader.dataloader.dataset.save_mini_result(
				self.results
			)
		else:
			out = self(test_batch['image'])[0]
			labels, calibration = self.trainer.test_dataloaders[0].dataset.get_annotations(test_idx)
			detections = test_post_process(out, calibration)
			performance_stats, images, bev = compare_ground_truth(
				detections, labels, test_batch['drawing_image'], calibration, self.opt
			)
			for _, match in performance_stats.items():
				for stat, val in match.items():
					if stat in self.results:
						self.results[stat].append(val)
					else:
						self.results[stat] = [val]
			if images != None:
				self.ImageWriter.write(images[0])
				self.BevWriter.write(bev)
				bev = cv2.resize(bev, (1080,1080), fx=0, fy=0, interpolation= cv2.INTER_CUBIC)
				stack = np.hstack([images[0], bev])
				self.JointWriter.write(stack)
			return detections

	def test_epoch_end(self, test_step_outputs):
		np.printoptions(precision=2)
		for stat, val in self.results.items():
			if stat == 'size':
				print(f"l,w,h Average: {np.mean(val[0]):.2f}, {np.mean(val[1]):.2f}, {np.mean(val[2]):.2f}")
				continue
			print(f"{stat} Average: {np.mean(val):.2f}, Variance: {np.var(val):.2f}")
		self.ImageWriter.release()
		self.BevWriter.release()
		self.JointWriter.release()
		cv2.destroyAllWindows()
		print("FINISHED")

if __name__ == '__main__':

	opt = opts().parse()
	if opt.output_path is not None:
		gclout = fsspec.filesystem(opt.output_path.split(":", 1)[0])
		print(gclout.isdir(opt.output_path))
		print(gclout.isdir(opt.output_path))

	model = LitWIBAM()
	if opt.load_model != '':
		state_dict_ = torch.load(opt.load_model)['state_dict']
		state_dict = {}
		for k in state_dict_:
			if k.startswith('module') and not k.startswith('module_list'):
				state_dict[k[7:]] = state_dict_[k]
			else:
				state_dict[k] = state_dict_[k]
			if not k.startswith('model'):
				state_dict['model.' + k] = state_dict[k]
				del state_dict[k]
			if "base.fc" in k:
				del state_dict[k]

		model.load_state_dict(state_dict)

	checkpoint_callback = ModelCheckpoint(
		monitor="val_main_tot", save_last=True, 
		save_top_k=2, mode='min', period=2
	)
										  
	class MyDDP(DDPPlugin):
		def configure_ddp(self, model, device_ids=opt.gpus):
			model = LightningDistributedDataParallel(
				model, device_ids, find_unused_parameters=True
			)
			return model
	my_ddp = MyDDP()

	trainer = pl.Trainer(checkpoint_callback=True,
						 callbacks=[checkpoint_callback],
						 default_root_dir=opt.output_path, 
						 gpus=opt.gpus, accelerator="dp",
						 check_val_every_n_epoch=1,
						 plugins=[my_ddp],
						 max_epochs=50
						 )

	# trainer.test(model)
	trainer.fit(model)