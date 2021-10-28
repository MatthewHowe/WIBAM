from __future__ import absolute_import, division, print_function

import copy
import csv
import math
import os
from pathlib import Path

# import _init_paths
import fsspec
import torch

torch.manual_seed(10)
import cv2
from dataset.dataset_factory import get_dataset, mixed_dataset
from dataset.datasets.wibam import WIBAM_test
from model.decode import generic_decode
from model.model import create_model, load_model, save_model
import numpy as np
from opts import opts
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import MNIST
from trainer import GenericLoss, MultiviewLoss
from utils.collate import default_collate, instance_batching_collate
from utils.mv_utils import compare_ground_truth, test_post_process
from utils.net import *
from utils.post_process import generic_post_process
from utils.utils import Profiler, separate_batches


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
                self.opt.dataset,
                self.opt.mixed_dataset,
                self.opt.batch_size,
                self.opt.mixed_batchsize,
                num_workers=self.opt.num_workers,
                task="train",
                opt=self.opt,
                drop_last=True,
                shuffle=True,
            )
        else:
            DataLoader = torch.utils.data.DataLoader(
                get_dataset(self.opt.dataset)(self.opt, "train"),
                batch_size=self.opt.batch_size,
                num_workers=self.opt.num_workers,
                drop_last=True,
                shuffle=True,
            )

        return DataLoader

    def val_dataloader(self):
        if self.opt.mixed_dataset is not None:
            DataLoader = mixed_dataset(
                self.opt.dataset,
                self.opt.mixed_dataset,
                self.opt.batch_size,
                self.opt.mixed_batchsize,
                num_workers=self.opt.num_workers,
                task="val",
                opt=self.opt,
                drop_last=True,
                shuffle=False,
            )
        else:
            DataLoader = torch.utils.data.DataLoader(
                get_dataset(self.opt.dataset)(self.opt, "val"),
                batch_size=self.opt.batch_size,
                num_workers=self.opt.num_workers,
                drop_last=True,
                shuffle=True,
            )

        return DataLoader

    def test_dataloader(self):
        if self.opt.validate:
            return self.val_dataloader()
        DataLoader = torch.utils.data.DataLoader(
            WIBAM_test(self.opt),
            batch_size=1,
            num_workers=0,
            drop_last=False,
            shuffle=False,
        )
        return DataLoader

    def configure_optimizers(self):
        # for name, param in self.model.named_parameters():
        # 	# Setting rotation to not change weights
        # 	if name.split(".")[0] == "rot":
        # 		param.requires_grad=False
        optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        # 				optimizer, mode='min', factor=0.1, patience=4,
        # 				threshold=0.001, verbose=True)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler,
        # 		"monitor": "val_tot"}
        return optimizer

    def training_step(self, train_batch, batch_idx):
        if self.opt.mixed_dataset is not None:
            main_out = self(train_batch[0]["image"])[0]
            mix_out = self(train_batch[1]["image"])[0]

            main_loss, main_loss_stats = self.main_loss(main_out, train_batch[0])
            mix_loss, mix_loss_stats = self.mix_loss(mix_out, train_batch[1])

            main_loss_stats = {
                "train_main_" + str(key): val for key, val in main_loss_stats.items()
            }
            mix_loss_stats = {
                "train_mix_" + str(key): val for key, val in mix_loss_stats.items()
            }
            self.log_dict(main_loss_stats, on_epoch=True)
            self.log_dict(mix_loss_stats, on_epoch=True)

            total_loss = main_loss + mix_loss
            self.log("train_tot", total_loss, on_epoch=True)

        else:
            main_out = self(train_batch["image"])[0]
            main_loss, main_loss_stats = self.main_loss(main_out, train_batch)

            main_loss_stats = {
                "train_main_" + str(key): val for key, val in main_loss_stats.items()
            }
            self.log_dict(main_loss_stats, on_epoch=True)
            total_loss = main_loss

        return total_loss

    def training_epoch_end(self, training_step_outputs):
        format_dict = {}
        for key, val in [
            (key, dict[key]) for dict in training_step_outputs for key in dict
        ]:
            if key not in format_dict:
                format_dict[key] = [val]
            else:
                format_dict[key].append(val)
        variance = torch.var(torch.stack(format_dict["loss"]))
        self.log("train_variance", variance, on_epoch=True)

    def validation_step(self, val_batch, batch_idx):
        if self.opt.mixed_dataset is not None:
            main_out = self(val_batch[0]["image"])[0]
            mix_out = self(val_batch[1]["image"])[0]

            main_loss, main_loss_stats = self.main_loss(main_out, val_batch[0])
            mix_loss, mix_loss_stats = self.mix_loss(mix_out, val_batch[1])

            main_loss_stats = {
                "val_main_" + str(key): val for key, val in main_loss_stats.items()
            }
            mix_loss_stats = {
                "val_mix_" + str(key): val for key, val in mix_loss_stats.items()
            }
            self.log_dict(main_loss_stats, on_epoch=True)
            self.log_dict(mix_loss_stats, on_epoch=True)

            total_val = main_loss + mix_loss
            self.log("val_tot", total_val, on_epoch=True)
        else:
            main_out = self(val_batch["image"])[0]
            main_loss, main_loss_stats = self.main_loss(main_out, val_batch)
            main_loss_stats = {
                "val_main_" + str(key): val for key, val in main_loss_stats.items()
            }
            self.log_dict(main_loss_stats, on_epoch=True)
            self.log("val_tot", main_loss, on_epoch=True)
        return main_loss

    def validation_epoch_end(self, validation_step_outputs):
        variance = torch.var(torch.stack(validation_step_outputs))
        mean = torch.mean(torch.stack(validation_step_outputs))
        self.log("val_variance", variance, on_epoch=True)

    def on_test_epoch_start(self):
        self.count = 0
        self.results = {}
        model_name = self.opt.load_model.split("/")[-1].split(".")[0]
        field_names = [
            "visibility",
            "location",
            "l",
            "w",
            "h",
            "rot",
            "3D_iou",
            "2D_iou",
            "bev_iou",
            "scale",
            "volume",
            "area",
        ]
        self.CSVWriter = csv.DictWriter(
            open(f"csv_results/{model_name}.csv", "w"), fieldnames=field_names
        )
        self.CSVWriter.writeheader()
        if opt.save_video:
            fourcc = cv2.VideoWriter_fourcc("F", "M", "P", "4")
            self.ImageWriter = cv2.VideoWriter(
                "image_out.avi", fourcc, 12, (1920, 1080)
            )
            self.BevWriter = cv2.VideoWriter("bev_out.avi", fourcc, 12, (800, 800))

            self.JointWriter = cv2.VideoWriter("joint_out.avi", fourcc, 12, (1800, 648))

    def test_step(self, test_batch, test_idx):
        if self.opt.validate:
            return self.validation_step(test_batch, test_idx)
        if self.opt.test:
            out = self(test_batch["image"])[0]
            for key, val in out.items():
                out[key] = val.cpu()
            out = generic_decode(out, self.opt.K, self.opt)
            # test_batch['meta'] = separate_batches(test_batch['meta'],
            # 									  opt.batch_size)
            out = generic_post_process(
                self.opt,
                out,
                test_batch["meta"]["c"].cpu(),
                test_batch["meta"]["s"].cpu(),
                96,
                320,
                10,
                test_batch["meta"]["calib"].cpu(),
            )
            for i in range(opt.batch_size):
                meta = test_batch["meta"]
                out[i] = generic_post_process(
                    self.opt,
                    out,
                    test_batch["meta"]["c"][i],
                    test_batch["meta"]["s"][i],
                    96,
                    320,
                    10,
                    test_batch["meta"]["calib"][i],
                )
            out = separate_batches(out, test_batch["image"].shape[0])
            for i in range(len(out)):
                self.results[test_batch["image_id"][i]] = out[i]
            self.test_dataloader.dataloader.dataset.save_mini_result(self.results)
        else:
            out = self(test_batch["image"])[0]
            labels, calibration, cam = self.trainer.test_dataloaders[
                0
            ].dataset.get_annotations(test_idx)
            detections = test_post_process(out, calibration)
            # if self.count % 7 == 0 and self.count >=(1*49) + 1 and self.count< 2*49:

            performance_stats, images, bev = compare_ground_truth(
                detections,
                labels,
                test_batch["drawing_image"],
                calibration,
                cam,
                self.opt,
            )
            for _, match in performance_stats.items():
                self.CSVWriter.writerows([match])
                for stat, val in match.items():
                    if stat in self.results:
                        self.results[stat].append(val)
                    else:
                        self.results[stat] = [val]
            if images != None:
                self.ImageWriter.write(images[0])
                self.BevWriter.write(bev)
                bev = cv2.resize(
                    bev, (1080, 1080), fx=0, fy=0, interpolation=cv2.INTER_CUBIC
                )
                stack = np.hstack([images[0], bev])
                stack = cv2.resize(
                    stack, (1800, 648), fx=0, fy=0, interpolation=cv2.INTER_CUBIC
                )
                self.JointWriter.write(stack)
            self.count += 1
            return detections

    def test_epoch_end(self, test_step_outputs):
        if self.opt.validate:
            return self.validation_epoch_end()
        np.printoptions(precision=2)
        # model_name = self.opt.load_model.split("/")[-1].split(".")[:1]
        # with open(f"{model_name}.csv", "w") as file:
        # 	writer = csv.writer(file)
        # 	for key, val in self.results.items():

        for stat, val in self.results.items():
            if stat == "size":
                print(
                    f"l,w,h Average: {np.mean(val[0]):.2f}, {np.mean(val[1]):.2f}, {np.mean(val[2]):.2f}"
                )
                continue
            if stat == "rot":
                val = np.array(val)
                aliased = copy.deepcopy(val[(val >= 150) | (val <= -150)])
                val = val[(val <= 150) & (val >= -150)]
                print(
                    f"{stat} Average: {np.mean(val):.2f}, Variance: {np.var(val):.2f}"
                )
                print(f"{stat}_alias %: {aliased.size/(val.size + aliased.size)}")
                continue
            print(f"{stat} Average: {np.mean(val):.2f}, Variance: {np.var(val):.2f}")
        self.ImageWriter.release()
        self.BevWriter.release()
        self.JointWriter.release()
        # self.CSVWriter.close()
        cv2.destroyAllWindows()
        print("FINISHED")


if __name__ == "__main__":

    opt = opts().parse()
    if opt.output_path is not None:
        gclout = fsspec.filesystem(opt.output_path.split(":", 1)[0])
        print(gclout.isdir(opt.output_path))
        print(gclout.isdir(opt.output_path))

    model = LitWIBAM()
    if opt.load_model != "":
        state_dict_ = torch.load(opt.load_model)["state_dict"]
        state_dict = {}
        for k in state_dict_:
            if k.startswith("module") and not k.startswith("module_list"):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
            if not k.startswith("model"):
                state_dict["model." + k] = state_dict[k]
                del state_dict[k]
            if "base.fc" in k:
                del state_dict[k]

        model.load_state_dict(state_dict)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_main_tot", save_last=True, save_top_k=3, mode="min", period=5
    )

    earlystop_callback = EarlyStopping(monitor="val_tot", min_delta=0.001, patience=10)

    class MyDDP(DDPPlugin):
        def configure_ddp(self, model, device_ids=opt.gpus):
            model = LightningDistributedDataParallel(
                model, device_ids, find_unused_parameters=True
            )
            return model

    my_ddp = MyDDP()

    trainer = pl.Trainer(
        checkpoint_callback=True,
        callbacks=[checkpoint_callback, earlystop_callback],
        default_root_dir=opt.output_path,
        gpus=opt.gpus,
        accelerator="dp",
        check_val_every_n_epoch=2,
        plugins=[my_ddp],
    )

    # trainer.test(model)
    trainer.fit(model)
