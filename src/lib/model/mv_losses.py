# File to contain functions for calculating multi view losses
import cv2
import copy
import random
import torch
from utils.ddd_utils import draw_box_3d, ddd2locrot, project_3d_bbox, ddd2locrot
from utils.post_process import generic_post_process
import torch.nn as nn
from utils.mv_utils import *
from .decode import generic_decode
def generate_colors(n): 
  rgb_values = [] 
  rgb_01 = []
  hex_values = [] 
  r = int(random.random() * 256) 
  g = int(random.random() * 256) 
  b = int(random.random() * 256) 
  step = 256 / n
  for _ in range(n): 
    r += step * 5
    g -= step * 3
    b += step * 1
    r = int(r) % 256 
    g = int(g) % 256 
    b = int(b) % 256 
    r_hex = hex(r)[2:] 
    g_hex = hex(g)[2:] 
    b_hex = hex(b)[2:] 
    hex_values.append('#' + r_hex + g_hex + b_hex) 
    rgb_values.append((r,g,b))
    rgb_01.append((r/256,g/256,b/256))
  return rgb_values, rgb_01, hex_values 

def generalized_iou_loss(gt_bboxes, pr_bboxes, reduction='mean'):
  r"""
  GIoU function, takes in ground truth bounding boxes and predicted
  bounding boxes. Uses BB[min_x,min_y,max_x,max_y]
  Arguments:
    gt_bboxes (np.array, (#_detections,4)): Bounding boxes for ground truth data
    pr_bboxes (np.array, (#_detections,4)): Bounding boxes for prediction
  Returns:
    loss (float): Return loss
  """
  # TODO: Better variable names
  # TODO: Improve comments
  size = [1920,1080]

  # Convert bbox format
  gt_bboxes[:,2] = gt_bboxes[:,0] + gt_bboxes[:,2]
  gt_bboxes[:,3] = gt_bboxes[:,1] + gt_bboxes[:,3]

  pr_bboxes[:,2] = pr_bboxes[:,0] + pr_bboxes[:,2]
  pr_bboxes[:,3] = pr_bboxes[:,1] + pr_bboxes[:,3]
  # C
  x1 = torch.clamp(pr_bboxes[:,0], 0, size[0])
  x2 = torch.clamp(pr_bboxes[:,2], 0, size[0])

  y1 = torch.clamp(pr_bboxes[:,1], 0, size[1])
  y2 = torch.clamp(pr_bboxes[:,3], 0, size[1])

  pr_bboxes = torch.stack((x1,y1,x2,y2),1)

  gt_area = (gt_bboxes[:, 2]-gt_bboxes[:, 0])*(gt_bboxes[:, 3]-gt_bboxes[:, 1])
  pr_area = (pr_bboxes[:, 2]-pr_bboxes[:, 0])*(pr_bboxes[:, 3]-pr_bboxes[:, 1])

  # iou
  top_left = torch.max(gt_bboxes[:, :2], pr_bboxes[:, :2])
  bottom_right = torch.min(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
  TO_REMOVE = 1
  wh = (bottom_right - top_left + TO_REMOVE).clamp(min=0)
  inter = wh[:, 0] * wh[:, 1]
  union = gt_area + pr_area - inter
  iou = inter / union
  # enclosure
  top_left = torch.min(gt_bboxes[:, :2], pr_bboxes[:, :2])
  bottom_right = torch.max(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
  wh = (bottom_right - top_left + TO_REMOVE).clamp(min=0)
  enclosure = wh[:, 0] * wh[:, 1]

  giou = iou - (enclosure-union)/enclosure
  loss = 1. - giou
  if reduction == 'mean':
    loss = loss.mean()
  elif reduction == 'sum':
    loss = loss.sum()
  elif reduction == 'none':
    pass
  return loss

class ReprojectionLoss(nn.Module):
  r"""
  Class should house a forward function which when given the output from ddd
  detection model and the batch calculate the reprojection loss using GIoU
  on each other camera in the batch.
  Arguments:
    output (dict): Should contain all information raw from the model
    batch (dict): Contain all raw information from the dataloader including
      calibration information, annotations for every other multi-view camera,
      and information of which camera was used for detection.
  Returns:
    loss_tot (flaot): The total reprojection loss over all cameras
    losses (list): The component losses for each reprojected view
  """  
  def __init__(self, opt=None):
    super(ReprojectionLoss, self).__init__()
    self.opt = opt

  def forward(self, output, batch):
    detections = {}
    calibrations = {}
    BN = len(batch['cam_num'])
    max_objects = self.opt.K
    num_cams = batch['P'].shape[1]

    decoded_output = decode_output(output, self.opt.K)
    # Post processing code

    centers = decoded_output['bboxes'].reshape(BN,max_objects,2, 2).mean(axis=2)
    centers_offset = centers + decoded_output['amodel_offset']

    centers = translate_centre_points(centers, np.array([960,540]), 1920, 
                                      (200,112), BN, max_objects)

    centers_offset = translate_centre_points(centers_offset, np.array([960,540]), 
                                             1920, (200,112), BN, max_objects)
    
    # Put detections into their own dictionary with required information
    # Scale depth with focal length
    detections['depth'] = decoded_output['dep'] * 1046/1266
    detections['size'] = decoded_output['dim'] 
    detections['rot'] = decoded_output['rot']
    detections['center'] = centers_offset

    det_cam_to_det_3D_ccf(detections,batch)

    dets_3D_ccf_to_dets_3D_wcf(detections, batch)

    dets_3D_wcf_to_dets_2D(detections, batch)

    # gt_centers = batch['ctr'].type(torch.float)
    # temp_gt = torch.cat((gt_centers,torch.ones(BN, max_objects,1).to('cuda')), 2)
    # gt_centers = torch.matmul(temp_gt, trans.T)

    gt_centers = translate_centre_points(batch['ctr'].type(torch.float), np.array([960,540]),
                                         1920, (200,112), BN, max_objects)

    cost_matrix, gt_indexes = match_predictions_ground_truth(centers, 
                          gt_centers, batch['mask'], batch['cam_num'])
    
    gt_dict = {}
    pr_dict = {}

    if self.opt.show_repro:
      drawing_images = batch['drawing_images'].detach().cpu().numpy()

    for B in range(BN):
      det_cam = int(batch['cam_num'][B].item())
      colours, _, _ = generate_colors(max_objects)
      for pr_index in range(max_objects):
        gt_index = gt_indexes[B, pr_index]

        if cost_matrix[B, pr_index, gt_index] < 50 and batch['mask'][B, det_cam, gt_index].item() is True:
          
          obj_id = batch['obj_id'][B, det_cam, gt_index]

          gt_box_T = batch['bboxes'][B, det_cam, gt_index]
          pr_box_T = detections['2D_bounding_boxes'][B,det_cam,pr_index]
          if 'det' not in gt_dict:
            gt_dict['det'] = [gt_box_T]
            pr_dict['det'] = [pr_box_T]
          else:
            gt_dict['det'].append(gt_box_T)
            pr_dict['det'].append(pr_box_T)

          # Drawing functions
          if self.opt.show_repro:
            img = drawing_images[B,det_cam]
            cv2.putText(img, "detection_cam", (0,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
            ddd_box = detections['proj_3D_boxes'][B, det_cam, pr_index].detach().cpu().numpy().astype(int)
            gt_box = torch.clone(gt_box_T).detach().cpu().numpy().astype(int)
            pr_box = torch.clone(pr_box_T).detach().cpu().numpy().astype(int)
            draw_box_3d(img, ddd_box, colours[pr_index])
            cv2.rectangle(img, (gt_box[0],gt_box[1]), (gt_box[0]+gt_box[2],gt_box[1]+gt_box[3]), colours[pr_index], 2)
            cv2.rectangle(img, (pr_box[0],pr_box[1]), (pr_box[0]+pr_box[2],pr_box[1]+pr_box[3]), colours[pr_index], 2)

          if obj_id != -1:
            
            for cam in range(num_cams):
              if cam == det_cam:
                continue
              
              try:
                obj_id_list = batch['obj_id'][B][cam].tolist()
                gt_ind = obj_id_list.index(obj_id)
              except:
                continue

              gt_box_T = batch['bboxes'][B,cam,gt_ind]
              pr_box_T = detections['2D_bounding_boxes'][B,cam,pr_index]
              if cam not in gt_dict:
                gt_dict[cam] = [gt_box_T]
                pr_dict[cam] = [pr_box_T]
              else:
                gt_dict[cam].append(gt_box_T)
                pr_dict[cam].append(pr_box_T)

              # Drawing functions
              if self.opt.show_repro:
                img = drawing_images[B,cam]
                ddd_box = detections['proj_3D_boxes'][B, cam, pr_index].detach().cpu().numpy().astype(int)
                gt_box = torch.clone(gt_box_T).detach().cpu().numpy().astype(int)
                pr_box = torch.clone(pr_box_T).detach().cpu().numpy().astype(int)
                draw_box_3d(img, ddd_box, colours[pr_index])
                cv2.rectangle(img, (gt_box[0],gt_box[1]), (gt_box[0]+gt_box[2],gt_box[1]+gt_box[3]), colours[pr_index], 2)
                cv2.rectangle(img, (pr_box[0],pr_box[1]), (pr_box[0]+pr_box[2],pr_box[1]+pr_box[3]), colours[pr_index], 2)

    mv_loss = {'tot': 0}
    for key, val in gt_dict.items():
      gt_boxes = torch.stack(val, 0)
      pr_boxes = torch.stack(pr_dict[key], 0)
      loss = generalized_iou_loss(gt_boxes, pr_boxes, 'mean')
      mv_loss[key] = loss
      if key == 'det' and self.opt.no_det:
        continue
      elif key == 'det' and self.opt.det_only:
        mv_loss['tot'] += loss
        break
      elif not self.opt.det_only:
        mv_loss['tot'] += loss

    # # Make sure that number of detections is equal to number of gt detections
    # mv_loss['mult'] = multiplier = pow((torch.sum(batch['mask_det']) - len(pr_dict['det'])),2) + 1
    # mv_loss['tot_GIoU'] = mv_loss['tot']
    # mv_loss['tot'] = mv_loss['tot'] * mv_loss['mult']
    

    if self.opt.show_repro:
      for B in range(BN):
        composite = return_four_frames(drawing_images[B])
        cv2.namedWindow("Batch {}".format(B), cv2.WINDOW_NORMAL)
        cv2.imshow("Batch {}".format(B), composite)
        cv2.waitKey(0)

    return mv_loss
