from __future__ import division
import scipy.optimize
import numpy as np
import torch
from shapely.geometry import Polygon

def bbox_iou(boxA, boxB):
  # Determine the (x, y)-coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  interW = xB - xA + 1
  interH = yB - yA + 1

  # Correction: reject non-overlapping boxes
  if interW <=0 or interH <=0 :
    return -1.0

  interArea = interW * interH
  boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
  iou = interArea / float(boxAArea + boxBArea - interArea)
  return iou

def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.5):
    '''
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.
    Parameters
    ----------
    bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2]. 
      The number of bboxes, N1 and N2, need not be the same.
    
    Returns
    -------
    (idxs_true, idxs_pred, ious, labels)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
        labels: vector of 0/1 values for the list of detections
    '''
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]
    MAX_DIST = 1.0
    MIN_IOU = 0.0

    bbox_gt[:,2] = [bbox_gt[i,2]+bbox_gt[i,0] for i in range(n_true)]
    bbox_gt[:,3] = [bbox_gt[i,3]+bbox_gt[i,1] for i in range(n_true)]
    bbox_pred[:,2] = [bbox_pred[i,2]+bbox_pred[i,0] for i in range(n_pred)]
    bbox_pred[:,3] = [bbox_pred[i,3]+bbox_pred[i,1] for i in range(n_pred)]

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = bbox_iou(bbox_gt[i,:], bbox_pred[j,:])

    if n_pred > n_true:
      # there are more predictions than ground-truth - add dummy rows
      diff = n_pred - n_true
      iou_matrix = np.concatenate( (iou_matrix, 
                                    np.full((diff, n_pred), MIN_IOU)), 
                                  axis=0)

    if n_true > n_pred:
      # more ground-truth than predictions - add dummy columns
      diff = n_true - n_pred
      iou_matrix = np.concatenate( (iou_matrix, 
                                    np.full((n_true, diff), MIN_IOU)), 
                                  axis=1)

    # call the Hungarian matching
    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]

    # remove dummy assignments
    sel_pred = idxs_pred<n_pred
    idx_pred_actual = idxs_pred[sel_pred] 
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label

def cal_batch_iou_3d(box3d1:torch.Tensor, box3d2:torch.Tensor, verbose=False):
    """calculated 3d iou. assume the 3d bounding boxes are only rotated around z axis
    Args:
        box3d1 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
        box3d2 (torch.Tensor): (B, N, 3+3+1),  (x,y,z,w,h,l,alpha)
    """
    box1 = box3d1[..., [0,1,3,4,6]]     # 2d box
    box2 = box3d2[..., [0,1,3,4,6]]
    zmax1 = box3d1[..., 2] + box3d1[..., 5] * 0.5
    zmin1 = box3d1[..., 2] - box3d1[..., 5] * 0.5
    zmax2 = box3d2[..., 2] + box3d2[..., 5] * 0.5
    zmin2 = box3d2[..., 2] - box3d2[..., 5] * 0.5
    z_overlap = (torch.min(zmax1, zmax2) - torch.max(zmin1, zmin2)).clamp_min(0.)
    iou_2d, corners1, corners2, u = cal_iou(box1, box2)        # (B, N)
    intersection_3d = iou_2d * u * z_overlap
    v1 = box3d1[..., 3] * box3d1[..., 4] * box3d1[..., 5]
    v2 = box3d2[..., 3] * box3d2[..., 4] * box3d2[..., 5]
    u3d = v1 + v2 - intersection_3d
    if verbose:
        z_range = (torch.max(zmax1, zmax2) - torch.min(zmin1, zmin2)).clamp_min(0.)
        return intersection_3d / u3d, corners1, corners2, z_range, u3d
    else:
        return intersection_3d / u3d

def calculate_3D_iou(box3d1, box3d2):
  box2d1 = Polygon(box3d1[:4, :2].tolist())
  box2d2 = Polygon(box3d2[:4, :2].tolist())

  min_z = np.min((np.max(box3d1[:, 2]), np.max(box3d2[:, 2])))
  max_z = np.max((np.min(box3d1[:, 2]), np.min(box3d2[:, 2])))
  z_overlap = np.clip((min_z - max_z), 0., 1e10)

  intersection_2d = box2d1.intersection(box2d2).area
  iou_2d = intersection_2d / box2d1.union(box2d2).area
  intersection_3d = intersection_2d * z_overlap

  volume1 = box2d1.area * (box3d1[-1, 2] - box3d1[0, 2])
  volume2 = box2d2.area * (box3d2[-1, 2] - box3d2[0, 2])
  
  union_3d = volume1 + volume2 - intersection_2d
  iou_3d = intersection_3d / union_3d

  return iou_3d, iou_2d

def cal_iou(box1:torch.Tensor, box2:torch.Tensor):
    """calculate iou
    Args:
        box1 (torch.Tensor): (B, N, 5)
        box2 (torch.Tensor): (B, N, 5)
    
    Returns:
        iou (torch.Tensor): (B, N)
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners1 (torch.Tensor): (B, N, 4, 2)
        U (torch.Tensor): (B, N) area1 + area2 - inter_area
    """
    corners1 = box2corners_th(box1)
    corners2 = box2corners_th(box2)
    inter_area, _ = oriented_box_intersection_2d(corners1, corners2)        #(B, N)
    area1 = box1[:, :, 2] * box1[:, :, 3]
    area2 = box2[:, :, 2] * box2[:, :, 3]
    u = area1 + area2 - inter_area
    iou = inter_area / u
    return iou, corners1, corners2, u

