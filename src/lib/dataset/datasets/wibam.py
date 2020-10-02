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

from ..generic_dataset import GenericDataset
from utils.ddd_utils import compute_box_3d, project_to_image
from utils.image import gaussian_radius, draw_umich_gaussian

class WIBAM(GenericDataset):
  num_categories = 1
  default_resolution = [1080, 1920]
  # ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck',  'Person_sitting',
  #       'Tram', 'Misc', 'DontCare']
  class_name = ['Car']
  # negative id is for "not as negative sample for abs(id)".
  # 0 for ignore losses for all categories in the bounding box region
  cat_ids = {1:1, 2:2, 3:3, 4:-2, 5:-2, 6:-1, 7:-9999, 8:-9999, 9:0}
  max_objs = 50
  def __init__(self, opt, split):
    data_dir = os.path.join(opt.data_dir, 'wibam')
    img_dir = os.path.join(data_dir, 'frames')

    ann_path = os.path.join(data_dir,
    'annotations', 'wibam_{}.json').format(split)

    self.images = None
    # load image list and coco
    super(WIBAM, self).__init__(opt, split, ann_path, img_dir)

    # Dataset can be loaded as either single images or all camera
    # images at an instance
    if self.opt.instance_batching:
      self.num_samples = len(self.images)
    else:
      self.num_samples = len(self.instances)

    print('Loaded {} split with {} samples'.format(split, self.num_samples))

  # Override functions
  # Called to load image and annotations, option to do all instances at time
  def __getitem__(self, index):
    # Get option arguments
    opt = self.opt

    # Load information from annotations file including image
    if self.opt.instance_batching:
      img, anns, img_info, img_path = self._load_data_instance_batch(index)
    else:
      img, anns, img_info, img_path = self._load_data(index)

    # Get image size
    height, width = img.shape[0], img.shape[1]

    # Get the centre location of the image
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)

    # Get the max size fo the image and convert to float unless dataset has imbalanced aspect ratio
    # then use
    s = max(img.shape[0], img.shape[1]) * 1.0 if not self.opt.not_max_crop \
      else np.array([img.shape[1], img.shape[0]], np.float32)

    #Initialise parameter for when validation is being performed (no augmentation)
    aug_s, rot, flipped = 1, 0, 0

    # If training split perform flip augmentation
    if self.split == 'train':
      # Override parameters if autmnetation being done
      c, aug_s, rot = self._get_aug_param(c, s, width, height)

      # New max size of image
      s = s * aug_s

      # Determin whether to flip or not
      if np.random.random() < opt.flip:
        # If random variable greater than flip we flip the image and set indicator
        flipped = 1
        img = img[:, ::-1, :]
        # Need to flip the annotations also
        anns = self._flip_anns(anns, width)

    # Calculate transformation on image
    trans_input = get_affine_transform(
      c, s, rot, [opt.input_w, opt.input_h])
    trans_output = get_affine_transform(
      c, s, rot, [opt.output_w, opt.output_h])

    # Resize and re colour image for data augmentation
    inp = self._get_input(img, trans_input)
    ret = {'image': inp}
    gt_det = {'bboxes': [], 'scores': [], 'clses': [], 'cts': []}

    # Tracking parameters
    pre_cts, track_ids = None, None
    if opt.tracking:
      pre_image, pre_anns, frame_dist = self._load_pre_data(
        img_info['video_id'], img_info['frame_id'], 
        img_info['sensor_id'] if 'sensor_id' in img_info else 1)
      if flipped:
        pre_image = pre_image[:, ::-1, :].copy()
        pre_anns = self._flip_anns(pre_anns, width)
      if opt.same_aug_pre and frame_dist != 0:
        trans_input_pre = trans_input
        trans_output_pre = trans_output
      else:
        c_pre, aug_s_pre, _ = self._get_aug_param(
          c, s, width, height, disturb=True)
        s_pre = s * aug_s_pre
        trans_input_pre = get_affine_transform(
          c_pre, s_pre, rot, [opt.input_w, opt.input_h])
        trans_output_pre = get_affine_transform(
          c_pre, s_pre, rot, [opt.output_w, opt.output_h])
      pre_img = self._get_input(pre_image, trans_input_pre)
      pre_hm, pre_cts, track_ids = self._get_pre_dets(
        pre_anns, trans_input_pre, trans_output_pre)
      ret['pre_img'] = pre_img
      if opt.pre_hm:
        ret['pre_hm'] = pre_hm

    ### init samples
    self._init_ret(ret, gt_det)

    # Get calibration information from image info
    calib = self._get_calib(img_info, width, height)

    num_objs = min(len(anns), self.max_objs)
    for k in range(num_objs):
      ann = anns[k]
      cls_id = int(self.cat_ids[ann['category_id']])
      if cls_id > self.opt.num_classes or cls_id <= -999:
        continue
      bbox, bbox_amodal = self._get_bbox_output(
        ann['bbox'], trans_output, height, width)
      if cls_id <= 0 or ('iscrowd' in ann and ann['iscrowd'] > 0):
        self._mask_ignore_or_crowd(ret, cls_id, bbox)
        continue
      self._add_instance(
        ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output, aug_s, 
        calib, pre_cts, track_ids)

    if self.opt.debug > 0:
      gt_det = self._format_gt_det(gt_det)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_info['id'],
              'img_path': img_path, 'calib': calib,
              'flipped': flipped}
      ret['meta'] = meta
    return ret

  # Called by get_item to add data to the return variable
  def _add_instance(
    self, ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output,
    aug_s, calib, pre_cts=None, track_ids=None):

    # Get size of bounding box
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    # Return if the bounding box size is negative or zero
    if h <= 0 or w <= 0:
      return

    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
    radius = max(0, int(radius)) 

    ct = np.array(
      [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)

    ct_int = ct.astype(np.int32)

    # Add category to return dict
    ret['cat'][k] = cls_id - 1

    # Indicate there is an annotation at this position
    ret['mask'][k] = 1
    if 'wh' in ret:
      # Add width and height to return dict
      ret['wh'][k] = 1. * w, 1. * h
      # Indicate object has width and height data
      ret['wh_mask'][k] = 1

    # 
    ret['ind'][k] = ct_int[1] * self.opt.output_w + ct_int[0]
    ret['reg'][k] = ct - ct_int
    ret['reg_mask'][k] = 1
    draw_umich_gaussian(ret['hm'][cls_id - 1], ct_int, radius)

    gt_det['bboxes'].append(
      np.array([ct[0] - w / 2, ct[1] - h / 2,
                ct[0] + w / 2, ct[1] + h / 2], dtype=np.float32))
    gt_det['scores'].append(1)
    gt_det['clses'].append(cls_id - 1)
    gt_det['cts'].append(ct)

    if 'tracking' in self.opt.heads:
      if ann['track_id'] in track_ids:
        pre_ct = pre_cts[track_ids.index(ann['track_id'])]
        ret['tracking_mask'][k] = 1
        ret['tracking'][k] = pre_ct - ct_int
        gt_det['tracking'].append(ret['tracking'][k])
      else:
        gt_det['tracking'].append(np.zeros(2, np.float32))

    if 'ltrb' in self.opt.heads:
      ret['ltrb'][k] = bbox[0] - ct_int[0], bbox[1] - ct_int[1], \
        bbox[2] - ct_int[0], bbox[3] - ct_int[1]
      ret['ltrb_mask'][k] = 1

    if 'ltrb_amodal' in self.opt.heads:
      ret['ltrb_amodal'][k] = \
        bbox_amodal[0] - ct_int[0], bbox_amodal[1] - ct_int[1], \
        bbox_amodal[2] - ct_int[0], bbox_amodal[3] - ct_int[1]
      ret['ltrb_amodal_mask'][k] = 1
      gt_det['ltrb_amodal'].append(bbox_amodal)

    if 'nuscenes_att' in self.opt.heads:
      if ('attributes' in ann) and ann['attributes'] > 0:
        att = int(ann['attributes'] - 1)
        ret['nuscenes_att'][k][att] = 1
        ret['nuscenes_att_mask'][k][self.nuscenes_att_range[att]] = 1
      gt_det['nuscenes_att'].append(ret['nuscenes_att'][k])

    if 'velocity' in self.opt.heads:
      if ('velocity' in ann) and min(ann['velocity']) > -1000:
        ret['velocity'][k] = np.array(ann['velocity'], np.float32)[:3]
        ret['velocity_mask'][k] = 1
      gt_det['velocity'].append(ret['velocity'][k])

    if 'hps' in self.opt.heads:
      self._add_hps(ret, k, ann, gt_det, trans_output, ct_int, bbox, h, w)

    if 'rot' in self.opt.heads:
      self._add_rot(ret, ann, k, gt_det)

    if 'dep' in self.opt.heads:
      if 'depth' in ann:
        ret['dep_mask'][k] = 1
        ret['dep'][k] = ann['depth'] * aug_s
        gt_det['dep'].append(ret['dep'][k])
      else:
        gt_det['dep'].append(2)

    if 'dim' in self.opt.heads:
      if 'dim' in ann:
        ret['dim_mask'][k] = 1
        ret['dim'][k] = ann['dim']
        gt_det['dim'].append(ret['dim'][k])
      else:
        gt_det['dim'].append([1,1,1])
    
    if 'amodel_offset' in self.opt.heads:
      if 'amodel_center' in ann:
        amodel_center = affine_transform(ann['amodel_center'], trans_output)
        ret['amodel_offset_mask'][k] = 1
        ret['amodel_offset'][k] = amodel_center - ct_int
        gt_det['amodel_offset'].append(ret['amodel_offset'][k])
      else:
        gt_det['amodel_offset'].append([0, 0])

  # Loads image annotations from file
  def _load_image_anns():
    img_info = coco.loadImgs(ids=[img_id])[0]
    file_name = img_info['file_name']
    img_path = os.path.join(img_dir, file_name)
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = copy.deepcopy(coco.loadAnns(ids=ann_ids))
    img = cv2.imread(img_path)
    return img, anns, img_info, img_path

  # Loads calibration information
  def _get_calib():
    if 'calib' in img_info:
      calib = np.array(img_info['calib'], dtype=np.float32)
    else:
      calib = np.array([[self.rest_focal_length, 0, width / 2, 0], 
                        [0, self.rest_focal_length, height / 2, 0], 
                        [0, 0, 1, 0]])
    return calib
  
  # May need to rewrite flip annotations due to other camera annotations being wrong
  # TODO: Workout how it affects the system and calibrations, probably set all augmentation off initially
  def flip_anns():

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