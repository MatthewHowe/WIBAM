from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
import numpy as np
import torch
import json
import cv2
import os
import math
import copy

from utils.image import get_affine_transform, affine_transform
from ..generic_dataset import GenericDataset
from utils.ddd_utils import compute_box_3d, project_to_image
from utils.image import gaussian_radius, draw_umich_gaussian

class WIBAM(GenericDataset):
  num_categories = 1
  default_resolution = [1088, 1920]
  # ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck',  'Person_sitting',
  #       'Tram', 'Misc', 'DontCare']
  class_name = ['Car']
  # negative id is for "not as negative sample for abs(id)".
  # 0 for ignore losses for all categories in the bounding box region
  cat_ids = {1:1, 2:2, 3:3, 4:-2, 5:-2, 6:-1, 7:-9999, 8:-9999, 9:0}
  max_objs = 50

  # Initialisation function
  def __init__(self, opt, split):
    # Define specific locations of dataset files
    data_dir = os.path.join(opt.data_dir, 'wibam')
    img_dir = os.path.join(data_dir, 'frames')
    # Where instance sets are kept
    split_dir = os.path.join(data_dir, 'image_sets')
    self.instances = []
    # Count the number of lines in the image set, which indicates time instances
    with open(os.path.join(split_dir, '{}.txt'.format(split))) as image_set:
      for line in image_set:
        self.instances.append(int(line))

    # Get annotation path
    ann_path = os.path.join(data_dir,
      'annotations', 'wibam_{}.json').format(split)

    self.images = None
    # load image list and coco
    super(WIBAM, self).__init__(opt, split, ann_path, img_dir)

    # Dataset can be loaded as either single images or all camera
    # images at an instance
    if self.opt.instance_batching: 
      self.instance_batching = True
      self.num_samples = len(self.instances)
    else:
      self.instance_batching = False
      self.num_samples = len(self.images)

    print('[INFO] Loaded {} split with {} images and {} instances'.format( \
          split, self.images, self.instances))
    print("[INFO] Loading with instance_batching: {} in {} samples".format( \
          self.opt.instance_batching, self.num_samples))

  # Override functions
  # Called to load image and annotations, option to do all instances at time
  def __getitem__(self, index):
    # Get option arguments
    opt = self.opt

    # Load information from annotations file including image
    # output is a list of outputs for each image at instances
    # in the case of instance_batching, else just a list of one
    # image and annotaitons
    imgs, imgs_anns, imgs_info, imgs_path = self._load_data(index)

    ###############
    # All data augmentation code was here - removed
    ###############

    rets = None

    for i in range(len(imgs)):
      img = imgs[i]
      height, width = img.shape[0], img.shape[1]
      center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
      scale = max(img.shape[0], img.shape[1]) * 1.0 if not self.opt.not_max_crop \
        else np.array([img.shape[1], img.shape[0]], np.float32)
      aug_s, rot, flipped = 1, 0, 0

      # Calculate transformation on image
      trans_input = get_affine_transform(
        center, scale, rot, [opt.input_w, opt.input_h])
      trans_output = get_affine_transform(
        center, scale, rot, [opt.output_w, opt.output_h])

      inp = self._get_input(img, trans_input)
      # initialise return variable
      ret = {'image': inp}
      # initialise gt dictionary
      gt_det = {'bboxes': [], 'scores': [], 'clses': [], 'cts': []}

      ### init samples
      self._init_ret(ret, gt_det)

      # Get calibration information from image info
      # Dictionary of all calibraiton information
      calib = self._get_calib(imgs_info[i])

      # Number of objects minimum of detections or max objects
      num_objs = min(len(imgs_anns[i]), self.max_objs)

      # For all objects
      for k in range(num_objs):
        # Get indexed annotation
        ann = imgs_anns[i][k]

        # Get annotation class ID
        cls_id = int(self.cat_ids[ann['category_id']])
        
        # Skip if class outsize of number of clasaug_sses or false
        if cls_id > self.opt.num_classes or cls_id <= -999:
          continue

        # Convert bounding box from coco
        bbox, bbox_amodal = self._get_bbox_output(ann['bbox'], trans_output)
        
        # Create mask for objects to ignore
        if cls_id <= 0 or ('iscrowd' in ann and ann['iscrowd'] > 0):
          self._mask_ignore_or_crowd(ret, cls_id, bbox)
          continue

        # Add information to ret
        # GT_det dict is used for debugging
        self._add_instance(
          ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output, calib)

      if self.opt.debug > 0:
        gt_det = self._format_gt_det(gt_det)
        meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_info['id'],
                'img_path': img_path, 'calib': calib,
                'flipped': flipped}
        ret['meta'] = meta
      
      # If instance batching, need to gather all rets
      if not self.instance_batching:
        return ret
      else:
        # If rets not initialised, initialise it by creating dict with
        # lists for each
        if rets is None:
          rets = {}
          for key, val in ret.items():
            rets["{}".format(key)] = []
        for key, val in ret.items():
          rets["{}".format(key)].append(val)

    for key, val in rets.items():
      rets["{}".format(key)] = np.array(rets["{}".format(key)])

    return rets

  # Transforms input for data augmentation
  def _get_input(self, img, trans_input):
    inp = cv2.warpAffine(img, trans_input,
                        (self.opt.input_w, self.opt.input_h),
                        flags=cv2.INTER_LINEAR)

    inp = (inp.astype(np.float32) / 255.)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)
    return inp

  # Used to compile the ret variable with information
  def _add_instance(
    self, ret, gt_det, k, cls_id, bbox_hm, bbox_input, ann,
    trans_output, calib):

    # Get size of bounding box
    h, w = bbox_hm[3] - bbox_hm[1], bbox_hm[2] - bbox_hm[0]
    # Ensure bounding box is positive
    if h <= 0 or w <= 0:
      return

    # Create a gaussian for the heat map for objects
    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
    radius = max(0, int(radius))

    # Get the centre of the object
    ct = np.array(
      [(bbox_hm[0] + bbox_hm[2]) / 2, (bbox_hm[1] + bbox_hm[3]) / 2], dtype=np.float32)
    ct_int = ct.astype(np.int32)
    
    # Input the category
    ret['cat'][k] = cls_id - 1
    # Mask indicated that a valid object is at instance k in ret
    ret['mask'][k] = 1

    # wh refers to width and height of bounding box
    if 'wh' in ret:
      ret['wh'][k] = 1. * w, 1. * h
      ret['wh_mask'][k] = 1
    
    # Add original bbox for loss
    ret['bboxes'][k] = bbox_input

    # Index of centre location
    ret['ind'][k] = ct_int[1] * self.opt.output_w + ct_int[0]
    # Offset of int and float (decimal part of centre)
    ret['reg'][k] = ct - ct_int
    # Mask indicating that a reg is present?
    ret['reg_mask'][k] = 1
    # Create heatmap
    draw_umich_gaussian(ret['hm'][cls_id - 1], ct_int, radius)

    # Put bbox into gt_boxes
    gt_det['bboxes'].append(
      np.array([ct[0] - w / 2, ct[1] - h / 2,
                ct[0] + w / 2, ct[1] + h / 2], dtype=np.float32))
    # Set detection score
    gt_det['scores'].append(ann['score'])
    # Set classes list to gt
    gt_det['clses'].append(cls_id - 1)
    # Add centers to gt
    gt_det['cts'].append(ct)

    # Removed depth, rot, etc since they dont exist for this dataset

  # Initialise return before filling in
  def _init_ret(self, ret, gt_det):
    max_objs = self.max_objs * self.opt.dense_reg
    ret['hm'] = np.zeros(
      (self.opt.num_classes, self.opt.output_h, self.opt.output_w), 
      np.float32)
    ret['ind'] = np.zeros((max_objs), dtype=np.int64)
    ret['cat'] = np.zeros((max_objs), dtype=np.int64)
    ret['mask'] = np.zeros((max_objs), dtype=np.float32)

    ret['bboxes'] = np.zeros((max_objs, 4), dtype=np.float32)

    regression_head_dims = {
      'reg': 2, 'wh': 2}

    for head in regression_head_dims:
      if head in self.opt.heads:
        ret[head] = np.zeros(
          (max_objs, regression_head_dims[head]), dtype=np.float32)
        ret[head + '_mask'] = np.zeros(
          (max_objs, regression_head_dims[head]), dtype=np.float32)
        gt_det[head] = []

    if 'hm_hp' in self.opt.heads:
      num_joints = self.num_joints
      ret['hm_hp'] = np.zeros(
        (num_joints, self.opt.output_h, self.opt.output_w), dtype=np.float32)
      ret['hm_hp_mask'] = np.zeros(
        (max_objs * num_joints), dtype=np.float32)
      ret['hp_offset'] = np.zeros(
        (max_objs * num_joints, 2), dtype=np.float32)
      ret['hp_ind'] = np.zeros((max_objs * num_joints), dtype=np.int64)
      ret['hp_offset_mask'] = np.zeros(
        (max_objs * num_joints, 2), dtype=np.float32)
      ret['joint'] = np.zeros((max_objs * num_joints), dtype=np.int64)
    
    if 'rot' in self.opt.heads:
      ret['rotbin'] = np.zeros((max_objs, 2), dtype=np.int64)
      ret['rotres'] = np.zeros((max_objs, 2), dtype=np.float32)
      ret['rot_mask'] = np.zeros((max_objs), dtype=np.float32)
      gt_det.update({'rot': []})

  # Get calibration information
  def _get_calib(self, img_info):
    calibration_info = {}
    calibration_info["P"] = img_info['P']
    calibration_info["dist_coefs"] = img_info['dist_coefs']
    calibration_info["rvec"] = img_info['rvec']
    calibration_info["tvec"] = img_info['tvec']
    calibration_info["theta_X_d"] = img_info['theta_X_d']

  # Loading data, use dataloader index to get image id
  # Function for loading single image with all annotations
  def _load_data(self, index):
    if self.instance_batching:
      id = self.instances[index]
    else:
      id = self.images[index]
    img, anns, img_info, img_path = self._load_image_anns(id, self.coco, self.img_dir)

    return img, anns, img_info, img_path

  # Loads image annotations from file
  def _load_image_anns(self, id, coco, img_dir):
    # Get number of cameras to load data from irregardless of instance batching
    # Single camera still needs other views to formulate loss
    # Will be image info for cam0 if instance
    img_info = coco.loadImgs(ids=[id])[0]
    num_cams = img_info['num_cams']
    # Get empty lists to return
    # In case of instance batching, will contain all images and anns
    # For single image will contain just its own batch
    imgs_info = []
    imgs_path = []
    imgs_anns = []
    imgs = []

    # Need to decipher between loading all instances at a time and singular image
    if self.instance_batching:
      num_loads = num_cams
    else:
      num_loads = 1

    # For each instance being loaded retrieve information from annotations
    for i in range(num_loads):
      # Special case when instance batching that time ID corresponds to cam 0
      # of that instance, and each cam after is id + 1,2,3
      img_id = id + i

      img_info = coco.loadImgs(ids=[img_id])[0]
      imgs_info.append(img_info)

      # Cam num is also = i if instance batching
      cam_num = img_info['cam']

      # Get image path
      file_name = img_info['file_name']
      img_path = os.path.join(img_dir, file_name)
      imgs_path.append(img_path)

      # Load all annotations for that instance
      for j in range(num_cams):
        # Calculate the id number of the other camera images
        # ie if cam_num is 2 and ID = 10, loading cam 0 annotations 
        # needs ID = 8 (ID=9 is cam 1). 
        # Therefore cam_ID = current_ID + index - cam
        ann_img_id = img_id + j - cam_num

        # Use coco utils to get annotation IDs for image instance
        ann_ids = coco.getAnnIds(imgIds=[ann_img_id])
        # Copy recursively all the annotations for that image
        anns = copy.deepcopy(coco.loadAnns(ids=ann_ids))
        # Add this to the annotations for this camera
        imgs_anns.append(anns)

        # TODO: Add grabbing of calibration info for each camera

      # Load and append image
      imgs.append(cv2.imread(img_path))

    # Return results
    return imgs, imgs_anns, imgs_info, imgs_path

  # # May need to rewrite flip annotations due to other camera annotations being wrong
  # # TODO: Workout how it affects the system and calibrations, probably set all augmentation off initially
  # def flip_anns():

  # Dataloader uses this function to create the iterable dataset
  def __len__(self):
    return self.num_samples

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    pass

  def save_results(self, results, save_dir):
    results_dir = os.path.join(save_dir, 'results_kitti')
    if not os.path.exists(results_dir):
      os.mkdir(results_dir)
    for img_id in results.keys():
      out_path = os.path.join(results_dir, '{:06d}.txt'.format(img_id))
      f = open(out_path, 'w')
      for i in range(len(results[img_id])):
        item = results[img_id][i]
        category_id = item['class']
        cls_name_ind = category_id
        class_name = self.class_name[cls_name_ind - 1]
        if not ('alpha' in item):
          item['alpha'] = -1
        if not ('rot_y' in item):
          item['rot_y'] = -1
        if 'dim' in item:
          item['dim'] = [max(item['dim'][0], 0.01), 
            max(item['dim'][1], 0.01), max(item['dim'][2], 0.01)]
        if not ('dim' in item):
          item['dim'] = [-1000, -1000, -1000]
        if not ('loc' in item):
          item['loc'] = [-1000, -1000, -1000]
        f.write('{} 0.0 0'.format(class_name))
        f.write(' {:.2f}'.format(item['alpha']))
        f.write(' {:.2f} {:.2f} {:.2f} {:.2f}'.format(
          item['bbox'][0], item['bbox'][1], item['bbox'][2], item['bbox'][3]))
        f.write(' {:.2f} {:.2f} {:.2f}'.format(
          item['dim'][0], item['dim'][1], item['dim'][2]))
        f.write(' {:.2f} {:.2f} {:.2f}'.format(
          item['loc'][0], item['loc'][1], item['loc'][2]))
        f.write(' {:.2f} {:.2f}\n'.format(item['rot_y'], item['score']))
      f.close()

  def run_eval(self, results, save_dir):
    print("[ERROR] Not implemented")