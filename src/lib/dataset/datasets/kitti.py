from __future__ import absolute_import, division, print_function

import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pycocotools.coco as coco
import torch
from utils.ddd_utils import compute_box_3d, project_to_image

from ..generic_dataset import GenericDataset


class KITTI(GenericDataset):

    default_resolution = [384, 1280]

    # ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck',  'Person_sitting',
    #       'Tram', 'Misc', 'DontCare']

    # num_categories = 3
    # class_name = ['Pedestrian', 'Car', 'Cyclist']
    # # negative id is for "not as negative sample for abs(id)".
    # # 0 for ignore losses for all categories in the bounding box region
    # cat_ids = {1:1, 2:2, 3:3, 4:-2, 5:-2, 6:-1, 7:-9999, 8:-9999, 9:0}
    # max_objs = 50

    num_categories = 10
    class_name = [
        "Car",
        "truck",
        "bus",
        "trailer",
        "construction_vehicle",
        "Pedestrian",
        "motorcycle",
        "Cyclist",
        "traffic_cone",
        "barrier",
    ]
    cat_ids = {i + 1: i + 1 for i in range(num_categories)}

    def __init__(self, opt, split):
        data_dir = os.path.join(opt.data_dir, "kitti")
        img_dir = os.path.join(data_dir, "training", "image_2")
        if opt.trainval:
            split = "trainval" if split == "train" else "test"
            img_dir = os.path.join(data_dir, "images", split)
            ann_path = os.path.join(data_dir, "annotations", "kitti_{}.json").format(
                split
            )
        else:
            ann_path = os.path.join(data_dir, "annotations", "kitti_{}_{}.json").format(
                opt.kitti_split, split
            )

        self.images = None
        # load image list and coco
        super(KITTI, self).__init__(opt, split, ann_path, img_dir)
        self.alpha_in_degree = False
        self.num_samples = len(self.images)
        self.meta = {}

        print("Loaded {} {} samples".format(split, self.num_samples))

    def __len__(self):
        return self.num_samples

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        pass

    def save_results(self, results, save_dir):
        results_dir = os.path.join(save_dir, "results_kitti")
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        for img_id in results.keys():
            out_path = os.path.join(results_dir, "{:06d}.txt".format(img_id))
            f = open(out_path, "w")
            for i in range(len(results[img_id])):
                item = results[img_id][i]
                category_id = item["class"]
                cls_name_ind = category_id
                class_name = self.class_name[cls_name_ind - 1]
                if not ("alpha" in item):
                    item["alpha"] = -1
                if not ("rot_y" in item):
                    item["rot_y"] = -1
                if "dim" in item:
                    item["dim"] = [
                        max(item["dim"][0], 0.01),
                        max(item["dim"][1], 0.01),
                        max(item["dim"][2], 0.01),
                    ]
                if not ("dim" in item):
                    item["dim"] = [-1000, -1000, -1000]
                if not ("loc" in item):
                    item["loc"] = [-1000, -1000, -1000]
                f.write("{} 0.0 0".format(class_name))
                f.write(" {:.2f}".format(item["alpha"]))
                f.write(
                    " {:.2f} {:.2f} {:.2f} {:.2f}".format(
                        item["bbox"][0],
                        item["bbox"][1],
                        item["bbox"][2],
                        item["bbox"][3],
                    )
                )
                f.write(
                    " {:.2f} {:.2f} {:.2f}".format(
                        item["dim"][0], item["dim"][1], item["dim"][2]
                    )
                )
                f.write(
                    " {:.2f} {:.2f} {:.2f}".format(
                        item["loc"][0], item["loc"][1], item["loc"][2]
                    )
                )
                f.write(" {:.2f} {:.2f}\n".format(item["rot_y"], item["score"]))
            f.close()

    def save_mini_result(self, results):
        root_dir = Path("/home/matthew/Documents/phd_projects/WIBAM/")
        save_dir = os.path.join(root_dir, "exp/test")
        results_dir = os.path.join(save_dir, "results_kitti")
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        for img_id in results.keys():
            out_path = os.path.join(results_dir, "{}.txt".format(img_id))
            f = open(out_path, "w")
            for key, item in results[img_id].items():
                item = results[img_id][i]
                category_id = item["class"]
                cls_name_ind = category_id
                class_name = self.class_name[cls_name_ind - 1]
                if not ("alpha" in item):
                    item["alpha"] = -1
                if not ("rot_y" in item):
                    item["rot_y"] = -1
                if "dim" in item:
                    item["dim"] = [
                        max(item["dim"][0], 0.01),
                        max(item["dim"][1], 0.01),
                        max(item["dim"][2], 0.01),
                    ]
                if not ("dim" in item):
                    item["dim"] = [-1000, -1000, -1000]
                if not ("loc" in item):
                    item["loc"] = [-1000, -1000, -1000]
                f.write("{} 0.0 0".format(class_name))
                f.write(" {:.2f}".format(item["alpha"]))
                f.write(
                    " {:.2f} {:.2f} {:.2f} {:.2f}".format(
                        item["bbox"][0],
                        item["bbox"][1],
                        item["bbox"][2],
                        item["bbox"][3],
                    )
                )
                f.write(
                    " {:.2f} {:.2f} {:.2f}".format(
                        item["dim"][0], item["dim"][1], item["dim"][2]
                    )
                )
                f.write(
                    " {:.2f} {:.2f} {:.2f}".format(
                        item["loc"][0], item["loc"][1], item["loc"][2]
                    )
                )
                f.write(" {:.2f} {:.2f}\n".format(item["rot_y"], item["score"]))
            f.close()

    def run_eval(self, results, save_dir):
        # import pdb; pdb.set_trace()
        self.save_results(results, save_dir)
        print("\n\n")
        print(os.system("ls"))
        print("Results of IoU threshold 0.7")
        os.system(
            "src/tools/kitti_eval/evaluate_object_3d_offline_07 "
            + "data/kitti/training/label_2/  "
            + "{}/results_kitti".format(save_dir)
        )
        print("Results of IoU threshold 0.5")
        os.system(
            "src/tools/kitti_eval/evaluate_object_3d_offline "
            + "data/kitti/training/label_2/ "
            + "{}/results_kitti".format(save_dir)
        )
