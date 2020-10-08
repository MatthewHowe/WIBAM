def _gather_feat(feat, ind):
  dim = feat.size(2)
  ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
  feat = feat.gather(1, ind)
  return feat

def _tranpose_and_gather_feat(feat, ind):
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