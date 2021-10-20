from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import pycocotools.coco as coco
from .coco_tools.coco import COCO as coco
import numpy as np
import torch
import json
import cv2
import os
import math
import copy
from torch.utils.data import Dataset, DataLoader

from utils.image import get_affine_transform, affine_transform
from ..generic_dataset import GenericDataset
from utils.ddd_utils import compute_box_3d, project_to_image
from utils.image import gaussian_radius, draw_umich_gaussian

class WIBAM(GenericDataset):
  # ## ORIGINAL
  default_resolution = [448, 800]
  num_categories = 10
  class_name = [
    'car', 'truck', 'bus', 'trailer', 
    'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
    'traffic_cone', 'barrier']
  cat_ids = {i + 1: i + 1 for i in range(num_categories)}
  focal_length = 1024
  original_resolution = [1080,1920]
  max_objs = 128
  _tracking_ignored_class = ['construction_vehicle', 'traffic_cone', 'barrier']
  _vehicles = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle']
  _cycles = ['motorcycle', 'bicycle']
  _pedestrians = ['pedestrian']
  attribute_to_id = {
    '': 0, 'cycle.with_rider' : 1, 'cycle.without_rider' : 2,
    'pedestrian.moving': 3, 'pedestrian.standing': 4, 
    'pedestrian.sitting_lying_down': 5,
    'vehicle.moving': 6, 'vehicle.parked': 7, 
    'vehicle.stopped': 8}
  id_to_attribute = {v: k for k, v in attribute_to_id.items()}

  # # WIBAM
  # num_categories = 1
  # focal_length = 1046
  # default_resolution = [448 , 800]
  # # ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck',  'Person_sitting',
  # #       'Tram', 'Misc', 'DontCare']
  # class_name = ['car']
  # # negative id is for "not as negative sample for abs(id)".
  # # 0 for ignore losses for all categories in the bounding box region
  # cat_ids = {1:1, 2:2, 3:3, 4:-2, 5:-2, 6:-1, 7:-9999, 8:-9999, 9:0}
  # max_objs = 50

  # Initialisation function
  def __init__(self, opt, split):
    # Define specific locations of dataset files
    data_dir = os.path.join(opt.data_dir, 'wibam_published')
    img_dir = os.path.join(data_dir, 'frames')
    # Where instance sets are kept
    split_dir = os.path.join(data_dir, 'image_sets')
    
    with open(os.path.join(split_dir, '{}.txt'.format(split))) as image_set:
      self.instances = [int(line) for line in image_set]

    if split == 'train':
      limit = int(opt.trainset_percentage * len(self.instances))
      self.instances = self.instances[:limit]

    # Get annotation path
    if opt.dataset_version == '':
      ann_path = os.path.join(data_dir,
        'annotations', 'wibam_all.json').format(split, opt.dataset_version)
    else:
      ann_path = os.path.join(data_dir,
        'annotations', 'wibam_{}_{}.json').format(split, opt.dataset_version) 

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

  # Override functions
  # Called to load image and annotations, option to do all instances at time
  def __getitem__(self, index):
    # Get option arguments
    opt = self.opt

    # Load information from annotations file including image
    # output is a list of outputs for each image at instances
    # in the case of instance_batching, else just a list of one
    # image and annotaitons
    imgs, cams, imgs_anns, imgs_info, imgs_path, drawing_images = self._load_data(index)

    ###############
    # All data augmentation code was here - removed
    ###############

    num_cams = len(imgs_anns)

    rets = None

    for i in range(len(imgs)):
      img = imgs[i]
      cam_num = cams[i]
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
      
      # inp = ((inp / 255. - np.array([0.40789655, 0.44719303, 0.47026116]) / np.array([0.2886383, 0.27408165, 0.27809834]) ))

      # initialise return variable
      ret = {'image': inp}
      ret['image_info'] = imgs_info
      # Cam number for this image
      ret['cam_num'] = cam_num
      ret['drawing_images'] = np.array(drawing_images)

      # Get calibration information and put into np.arrays for collation
      P, dist_coefs, rvec, tvec, theta_X_d = self._get_calib(imgs_info)
      ret['P'] = np.array(P).astype(np.float64)
      ret['dist_coefs'] = np.array(dist_coefs)
      ret['rvec'] = np.array(rvec)
      ret['tvec'] = np.array(tvec)
      ret['theta_X_d'] = np.array(theta_X_d)

      ret['P_det'] = np.array(P[cam_num]).astype(np.float32)
      ret['dist_coefs_det'] = np.array(dist_coefs[cam_num])
      ret['rvec_det'] = np.array(rvec[cam_num])
      ret['tvec_det'] = np.array(tvec[cam_num])
      ret['theta_X_d_det'] = np.array(theta_X_d[cam_num])
      

      # initialise gt dictionary
      gt_det = {'bboxes': [], 'scores': [], 'clses': [], 'cts': []}

      ### init samples
      self._init_ret(ret, gt_det, num_cams)

      

      # For each camera add the annotation to ret
      for cam in range(num_cams):
        # Get calibration information from image info
        # Dictionary of all calibraiton information
        

        # Number of objects minimum of detections or max objects
        num_objs = min(len(imgs_anns[cam]), self.max_objs)
          
        # For all objects detected at each camera
        for obj in range(num_objs):
          # Get indexed annotation
          ann = imgs_anns[cam][obj]

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

          pred_cam = (cam == cam_num)

          # Add information to ret
          # GT_det dict is used for debugging
          self._add_instance(
            ret, gt_det, cam, obj, cls_id, bbox, ann['bbox'], ann, pred_cam)

      ret['mask_det'] = ret['mask'][cam_num]
      ret['cat_det'] = ret['cat'][cam_num]

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
    self, ret, gt_det, cam, obj, cls_id, bbox_hm, bbox_input, ann, pred_cam):

    # Get size of bounding box
    h, w = bbox_hm[3] - bbox_hm[1], bbox_hm[2] - bbox_hm[0]
    # Ensure bounding box is positive
    if h <= 0 or w <= 0:
      return

    # Create a gcamsaussian for the heat map for objects
    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
    radius = max(0, int(radius))

    # Get the centre of the object
    ct = np.array(
      [(bbox_hm[0] + bbox_hm[2]) / 2, (bbox_hm[1] + bbox_hm[3]) / 2], dtype=np.float32)
    ct_int = ct.astype(np.int32)
    
    # Input the category
    ret['cat'][cam][obj] = int(cls_id - 1)

    # Mask indicated that a valid object is at instance k in ret
    ret['mask'][cam][obj] = 1

    # Add original bbox for loss
    ret['bboxes'][cam][obj] = bbox_input

    if ann['obj_id'] is None:
      ret['obj_id'][cam][obj] = -1
    else:
      ret['obj_id'][cam][obj] = ann['obj_id']

    ret['score'][cam][obj] = ann['score']

    # If this camera number is the prediction camera generate heatmap etc
    # Heatmap, index, etc from other cams not used in losses
    if pred_cam:
      # wh refers to width and height of bounding box
      if 'wh' in ret:
        ret['wh'][obj] = 1. * w, 1. * h
        ret['wh_mask'][obj] = 1

      # Index of centre location
      ret['ind'][obj] = ct_int[1] * self.opt.output_w + ct_int[0]

      # ret['ctr'][obj] = np.array(
      #     [(bbox_input[0] + bbox_input[2] / 2), 
      #     (bbox_input[1] + bbox_input[3] / 2)], 
      #     dtype=np.float32)
      ret['ctr'][obj] = ct

      # Offset of int and float (decimal part of centre)
      ret['reg'][obj] = ct - ct_int
    
      # Mask indicating that a reg is present?
      ret['reg_mask'][obj] = 1
    
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
  def _init_ret(self, ret, gt_det, num_cams):
    max_objs = self.max_objs * self.opt.dense_reg
    # Heat maps
    ret['hm'] = np.zeros(
      (self.opt.num_classes, self.opt.output_h, self.opt.output_w), 
      np.float32)
    
    # Centre location index
    ret['ind'] = np.zeros((max_objs), dtype=np.int64)

    # Center location
    ret['ctr'] = np.zeros((max_objs, 2), dtype=np.int64)
    
    # Categories
    ret['cat'] = np.zeros((num_cams, max_objs), dtype=np.int64)
    
    # Scores
    ret['score'] = np.zeros((num_cams, max_objs), dtype=np.float32)

    # Masks
    ret['mask'] = np.zeros((num_cams, max_objs), dtype=np.bool)  

    # Bounding boxes
    ret['bboxes'] = np.zeros((num_cams, max_objs, 4), dtype=np.float32)

    # Object instance id
    ret['obj_id'] = np.full((num_cams, max_objs), -1, dtype=np.int64)

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
  def _get_calib(self, imgs_info):
    P = []
    dist_coefs = []
    rvec = []
    tvec = []
    theta_X_d = []
    for img_info in imgs_info:
      calibration_info = {}
      P.append(img_info['P'])
      dist_coefs.append(img_info['dist_coefs'])
      rvec.append(img_info['rvec'])
      tvec.append(img_info['tvec'])
      theta_X_d.append(img_info['theta_X_d'])
    return P, dist_coefs, rvec, tvec, theta_X_d

  # Loading data, use dataloader index to get image id
  # Function for loading single image with all annotations
  def _load_data(self, index):
    if self.instance_batching:
      id = self.instances[index]
    else:
      # id = self.images[index]
      id = self.instances[index]
    img, cams, anns, img_info, img_path, drawing_images = self._load_image_anns(id, self.coco, self.img_dir)

    return img, cams, anns, img_info, img_path, drawing_images

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
    cam_nums = []
    imgs = []
    drawing_images = []

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

      # Cam num is also = i if instance batching
      cam_num = img_info['cam']

      # Get image path
      file_name = img_info['file_name']
      img_path = os.path.join(img_dir, file_name)
      imgs_path.append(img_path)

      # Load and append image
      imgs.append(cv2.imread(img_path))
      # Append corredsponding cam number
      cam_nums.append(cam_num)

      # Load all annotations for that instance
      for j in range(num_cams):
        # Calculate the id number of the other camera images
        # ie if cam_num is 2 and ID = 10, loading cam 0 annotations 
        # needs ID = 8 (ID=9 is cam 1). 
        # Therefore cam_ID = current_ID + index - cam
        ann_img_id = img_id + j - cam_num
        img_info = coco.loadImgs(ids=[ann_img_id])[0]

        file_name = img_info['file_name']
        img_path = os.path.join(img_dir, file_name)
        if self.opt.show_repro is True:
          drawing_images.append(cv2.imread(img_path))

        # Use coco utils to get annotation IDs for image instance
        ann_ids = coco.getAnnIds(imgIds=[ann_img_id])
        # Copy recursively all the annotations for that image
        anns = copy.deepcopy(coco.loadAnns(ids=ann_ids))
        # Add this to the annotations and calibration for this camera
        imgs_anns.append(anns)
        imgs_info.append(img_info)

    return imgs, cam_nums, imgs_anns, imgs_info, imgs_path, drawing_images

  def __len__(self):
    return len(self.instances)

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    pass

  # def save_results(self, results, save_dir):
  #   results_dir = os.path.join(save_dir, 'results_kitti')
  #   if not os.path.exists(results_dir):
  #     os.mkdir(results_dir)
  #   for img_id in results.keys():
  #     out_path = os.path.join(results_dir, '{:06d}.txt'.format(img_id))
  #     f = open(out_path, 'w')
  #     for i in range(len(results[img_id])):
  #       item = results[img_id][i]
  #       category_id = item['class']
  #       cls_name_ind = category_id
  #       class_name = self.class_name[cls_name_ind - 1]
  #       if not ('alpha' in item):
  #         item['alpha'] = -1
  #       if not ('rot_y' in item):
  #         item['rot_y'] = -1
  #       if 'dim' in item:
  #         item['dim'] = [max(item['dim'][0], 0.01), 
  #           max(item['dim'][1], 0.01), max(item['dim'][2], 0.01)]
  #       if not ('dim' in item):
  #         item['dim'] = [-1000, -1000, -1000]
  #       if not ('loc' in item):
  #         item['loc'] = [-1000, -1000, -1000]
  #       f.write('{} 0.0 0'.format(class_name))
  #       f.write(' {:.2f}'.format(item['alpha']))
  #       f.write(' {:.2f} {:.2f} {:.2f} {:.2f}'.format(
  #         item['bbox'][0], item['bbox'][1], item['bbox'][2], item['bbox'][3]))
  #       f.write(' {:.2f} {:.2f} {:.2f}'.format(
  #         item['dim'][0], item['dim'][1], item['dim'][2]))
  #       f.write(' {:.2f} {:.2f} {:.2f}'.format(
  #         item['loc'][0], item['loc'][1], item['loc'][2]))
  #       f.write(' {:.2f} {:.2f}\n'.format(item['rot_y'], item['score']))
  #     f.close()

  def save_results(self, results, save_dir, task):
    json.dump(self.convert_eval_format(results), 
                open('{}/results_nuscenes_{}.json'.format(save_dir, task), 'w'))


  def run_eval(self, results, save_dir):
    task = 'tracking' if self.opt.tracking else 'det'
    self.save_results(results, save_dir, task)
    if task == 'det':
      os.system('python ' + \
        'src/tools/nuscenes-devkit/python-sdk/nuscenes/eval/detection/evaluate.py ' +\
        '{}/results_nuscenes_{}.json '.format(save_dir, task) + \
        '--output_dir {}/nuscenes_eval_det_output/ '.format(save_dir) + \
        '--dataroot data/nuscenes/v1.0-trainval/ ' + \
        '--plot_examples=0')
    else:
      os.system('python ' + \
        'src/tools/nuscenes-devkit/python-sdk/nuscenes/eval/tracking/evaluate.py ' +\
        '{}/results_nuscenes_{}.json '.format(save_dir, task) + \
        '--output_dir {}/nuscenes_evaltracl__output/ '.format(save_dir) + \
        '--dataroot ../data/nuscenes/v1.0-trainval/')
      os.system('python ' + \
        'src/tools/nuscenes-devkit/python-sdk-alpha02/nuscenes/eval/tracking/evaluate.py ' +\
        '{}/results_nuscenes_{}.json '.format(save_dir, task) + \
        '--output_dir {}/nuscenes_evaltracl__output/ '.format(save_dir) + \
        '--dataroot ../data/nuscenes/v1.0-trainval/')


class WIBAM_test(Dataset):
  mean = np.array([0.40789655, 0.44719303, 0.47026116],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.2886383, 0.27408165, 0.27809834],
                   dtype=np.float32).reshape(1, 1, 3)
  def __init__(self, opt=None):
    self.data_dir = os.path.join(opt.data_dir, 'wibam')
    self.images_dir = os.path.join(self.data_dir, 'frames')
    self.labels_dir = os.path.join(self.data_dir, 'annotations/hand_labels')
    self.paths = {}
    self.image_paths = []
    self.opt = opt
    idx = 0
    for dirpath, dirnames, filenames in os.walk(self.labels_dir):
      if len(dirnames) == 0:
        cam = dirpath.split("/")[-1]
        filenames = [int(x.split('.')[0]) for x in filenames]
        filenames.sort()
        for filename in filenames:
          self.paths[idx] = {"image_path": os.path.join(self.images_dir, cam, str(filename) + ".jpg"),
                             "label_path": os.path.join(dirpath, str(filename) + ".json")}
          idx += 1

  def __getitem__(self, index):
    image_path = self.paths[index]["image_path"]
    image_num = image_path.split("/")[-1].split(".jpg")[0]
    cam_num = image_path.split("/")[-2]
    image = cv2.imread(image_path)
    drawing_image = np.copy(image)
    center = np.array(
      [image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32
    )
    trans_input = get_affine_transform(
      center , 1920, 0, [self.opt.input_w, self.opt.input_h]
    )

    image = self._get_input(image, trans_input)
    
    return {"image": image, "cam_num": cam_num, "image_num": image_num,
            "index": index, "drawing_image": drawing_image}

  def _get_input(self, img, trans_input):
    inp = cv2.warpAffine(img, trans_input,
                        (self.opt.input_w, self.opt.input_h),
                        flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    inp = (inp - self.mean) / self.std
    
    inp = inp.transpose(2, 0, 1)
    return inp
  
  def get_annotations(self, index):
    label_path = self.paths[index]['label_path']
    cam = label_path.split('/')[-2]
    with open(label_path, "r") as file:
      label = json.load(file)
    calib_dir = os.path.join(
      self.data_dir, "calib/calibration_{}.npz".format(cam)
    )
    calib = np.load(calib_dir)
    return label, calib, int(cam)

  def __len__(self):
    return len(self.paths)