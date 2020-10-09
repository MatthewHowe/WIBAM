# File to contain functions for calculating multi view losses
import torch
import torch.nn as nn
from utils.mv_utils import det_cam_to_det_3D_ccf, dets_3D_ccf_to_dets_3D_wcf
from utils.mv_utils import dets_3D_wcf_to_dets_2D

def _gather_feat(feat, ind):
  r"""
  Composes a tensor from the variable prediction heatmap and the indexes from the 
  ground truth data.
  Arguments:
    feat (tensor, (BN,hm_x*hm_y,dim)): predictions for variable for each point 
      on the output heatmap that has been permuted
    ind (tensor, (BN, max_objects, dim)): indexes for the location for each 
      object
  Returns:
    feat (tensor, (BN,max_objects,dim)): the depth prediction at each gt heat map
      index
  """
  dim = feat.size(2)
  ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
  feat = feat.gather(1, ind)
  return feat

def _tranpose_and_gather_feat(feat, ind):
  r"""
  Composes the prediction data from a heat map size tensor that contains predictions
  to an output which is peridctions of max_object size for each dimension
  Arguments:
    feat (tensor, (BN,dim,hm_x,hm_y)): predictions for variable for each point 
      on the output heatmap
    ind (tensor, (BN, max_objects, dim)): indexes for the location for each 
      object
  Returns:
    feat (tensor, (BN,max_objects,dim)): the depth prediction at each gt heat map
      index
  """
  feat = feat.permute(0, 2, 3, 1).contiguous()
  feat = feat.view(feat.size(0), -1, feat.size(3))
  feat = _gather_feat(feat, ind)
  return feat

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

    # Put detections into their own dictionary with required information
    detections['depth'] = pred_dep
    detections['size'] = pred_size
    detections['rot'] = pred_rot
    calibrations['P'] = batch['P']
    calibrations['dist_coefs'] = batch['dist_coefs']
    calibrations['tvec'] = batch['tvex']
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