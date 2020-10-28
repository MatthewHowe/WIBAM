# File to contain functions for calculating multi view losses
import torch
import torch.nn as nn
from utils.mv_utils import *



def generalized_iou_loss(gt_bboxes, pr_bboxes, reduction='mean'):
  r"""
  GIoU function, takes in ground truth bounding boxes and predicted
  bounding boxes.
  Arguments:
    gt_bboxes (np.array, (#_detections,4)): Bounding boxes for ground truth data
    pr_bboxes (np.array, (#_detections,4)): Bounding boxes for prediction
  Returns:
    loss (float): Return loss
  """
  # TODO: Better variable names
  # TODO: Improve comments

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

  def forward(self, output, batch):
    detections = {}
    calibrations = {}

    # Get predictions in format (BN, objects, dim) for each of dep,rot,dim
    pred_dep = tranpose_and_gather_feat(output['dep'], batch['ind'])
    pred_size = tranpose_and_gather_feat(output['dim'], batch['ind'])
    pred_rot = tranpose_and_gather_feat(output['rot'], batch['ind'])

    decoded_output = decode_output(output, pred_dep.shape[1])

    # Put detections into their own dictionary with required information
    # detections['centers'] = 
    detections['depth'] = pred_dep
    detections['size'] = pred_size
    detections['rot'] = pred_rot
    detections['center'] = decoded_output['centers'] * 4
    calibrations['cam_num'] = batch['cam_num']
    calibrations['P'] = batch['P']
    calibrations['dist_coefs'] = batch['dist_coefs']
    calibrations['tvec'] = batch['tvec']
    calibrations['rvec'] = batch['rvec']
    calibrations['theta_X_d'] = batch['theta_X_d']

    test_calculations(batch, calibrations)

    # Get predictions in camera.c.f loc(x,y,z), rot(alpha),size(l,w,h)
    # Adds them to detections dict
    det_cam_to_det_3D_ccf(detections,calibrations)

    # Put the predictions into world coordinate frame
    # Adds them to detections dict
    dets_3D_ccf_to_dets_3D_wcf(detections, calibrations)

    # Produce all the reprojections for every other camera
    dets_3D_wcf_to_dets_2D(detections, calibrations)

    cost_matrix, gt_indexes = match_predictions_ground_truth(detections['center'], 
                          batch['ctr'], batch['mask'], batch['cam_num'])
    matched_obj_ids = []
    matched_det_loc = []

    BN, max_objs = gt_indexes.shape
    num_cams = calibrations['P'].shape[1]
    
    for B in range(BN):
      cam = batch['cam_num'][B]
      for obj in range(max_objs):
        if cost_matrix[B,obj,gt_indexes[B,obj]] < 100:
          obj_id = batch['obj_id'][B,cam,gt_indexes[B,obj]]
          if obj_id != -1:
            matched_obj_ids.append(obj_id)
            matched_det_loc.append(obj)

    if len(matched_det_loc) != 0:
      gt_bboxes = []
      pr_bboxes = []
      # Put together the ground truth boxes and predicted boxes
      # For each batch
      for B in range(BN):
        for match in range(len(matched_det_loc)):
          pr_ind = matched_det_loc[B, match]
          obj_id = matched_obj_ids[B, match]
          for cam in range(num_cams):
            obj_id_list = batch['obj_ids'][cam].tolist()
            gt_ind = obj_id_list.index(obj_id)
            gt_bboxes.append(batch['bboxes'][B,cam,gt_ind])
            pr_bboxes.append(detections['2D_bounding_boxes'][B,cam,pr_ind])
      gt_bboxes = torch.cat(gt_bboxes)
      pr_bboxes = torch.cat(pr_bboxes)
      mv_loss = generalized_iou_loss(gt_bboxes,pr_bboxes)

      return mv_loss
    else:
      return 0