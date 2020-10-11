# File to contain functions for calculating multi view losses
import torch
import torch.nn as nn
from utils.mv_utils import det_cam_to_det_3D_ccf, dets_3D_ccf_to_dets_3D_wcf
from utils.mv_utils import dets_3D_wcf_to_dets_2D, decode_output
from utils.mv_utils import _gather_feat, _tranpose_and_gather_feat



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
  lt = torch.max(gt_bboxes[:, :2], pr_bboxes[:, :2])
  rb = torch.min(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
  TO_REMOVE = 1
  wh = (rb - lt + TO_REMOVE).clamp(min=0)
  inter = wh[:, 0] * wh[:, 1]
  union = gt_area + pr_area - inter
  iou = inter / union
  # enclosure
  lt = torch.min(gt_bboxes[:, :2], pr_bboxes[:, :2])
  rb = torch.max(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
  wh = (rb - lt + TO_REMOVE).clamp(min=0)
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
    pred_dep = _tranpose_and_gather_feat(output['dep'], batch['ind'])
    pred_size = _tranpose_and_gather_feat(output['dim'], batch['ind'])
    pred_rot = _tranpose_and_gather_feat(output['rot'], batch['ind'])

    decoded_output = decode_output(output, pred_dep.shape[1])

    # Put detections into their own dictionary with required information
    # detections['centers'] = 
    detections['depth'] = pred_dep
    detections['size'] = pred_size
    detections['rot'] = pred_rot
    detections['center'] = decoded_output['centers']
    calibrations['cam_num'] = batch['cam_num']
    calibrations['P'] = batch['P']
    calibrations['dist_coefs'] = batch['dist_coefs']
    calibrations['tvec'] = batch['tvec']
    calibrations['rvec'] = batch['rvec']
    calibrations['theta_X_d'] = batch['theta_X_d']

    # Get predictions in camera.c.f loc(x,y,z), rot(alpha),dim(l,w,h)
    # Adds them to detections dict
    det_cam_to_det_3D_ccf(detections,calibrations)

    # Put the predictions into world coordinate frame
    # Adds them to detections dict
    dets_3D_ccf_to_dets_3D_wcf(detections, calibrations)

    # Produce all the reprojections for every other camera
    reprojections = dets_3D_wcf_to_dets_2D(detections, calibrations)

    # # For each sample in the batch
    # for sample in len(range(batch['images'])):
    #   # For each camera in the batch
    #   for cam in len(range(batch['cams'])
    #     # Reproject all objects onto camera view and get boxes, clipped


        # Calculate GIoU from 2D bounding boxes and reprojections

        # Add loss to list of losses for this camera

    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')

    # Sum all the losses from each camera
    loss = -1
    # loss = loss / (mask.sum() + 1e-4)
    return loss