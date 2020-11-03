# File to contain functions for calculating multi view losses
import cv2
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

  # Convert bbox format
  gt_bboxes[:,2] = gt_bboxes[:,0] + gt_bboxes[:,2]
  gt_bboxes[:,3] = gt_bboxes[:,1] + gt_bboxes[:,3]

  pr_bboxes[:,2] = pr_bboxes[:,0] + pr_bboxes[:,2]
  pr_bboxes[:,3] = pr_bboxes[:,1] + pr_bboxes[:,3]


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
    BN = self.opt.batch_size
    max_objects = self.opt.K
    num_cams = batch['P'][0].shape[1]

    calibs_ten = []
    calibs_arr = []
    for B in range(BN):
      cam = batch['cam_num'][B]
      calib = batch['P'][B][cam].to('cpu').numpy()
      trans = np.array([[0,0,0]]).T
      calib = np.concatenate((calib,trans), axis=1)
      calibs_ten.append(torch.Tensor(calib))
      calibs_arr.append(calib)

    decoded_output = decode_output(output, self.opt.K)

    ctrs = np.zeros((BN,max_objects,2))
    trans = get_affine_transform(
      np.array([960,540]), 1920, 0, (200, 112), inv=1).astype(np.float32)
    trans = torch.Tensor(trans)

    homog = torch.ones((len(decoded_output['cts'][0]),1)).reshape(50,1)
    for B in range(BN):
      cts = torch.cat((decoded_output['cts'][B].cpu(),homog),1)
      ctrs[B] = torch.mm(cts, trans.T)
    # Put detections into their own dictionary with required information
    # Scale depth with focal length
    detections['depth'] = decoded_output['dep'] * 1046/1266
    detections['size'] = decoded_output['dim'] 
    detections['rot'] = decoded_output['rot']
    detections['center'] = torch.Tensor(ctrs).to('cuda')
    calibrations['cam_num'] = batch['cam_num']
    calibrations['P'] = batch['P']
    calibrations['dist_coefs'] = batch['dist_coefs']
    calibrations['tvec'] = batch['tvec']
    calibrations['rvec'] = batch['rvec']
    calibrations['theta_X_d'] = batch['theta_X_d']

    drawing_images = batch['drawing_images'].to('cpu').numpy()
  
    # Get predictions in camera.c.f loc(x,y,z), rot(alpha),size(l,w,h)
    # Adds them to detections dict
    det_cam_to_det_3D_ccf(detections,calibrations)

    # Put the predictions into world coordinate frame
    # Adds them to detections dict
    dets_3D_ccf_to_dets_3D_wcf(detections, calibrations)

    # Produce all the reprojections for every other camera
    dets_3D_wcf_to_dets_2D(detections, calibrations)


    gt_centers = batch['ctr'].type(torch.float)
    for B in range(BN):
      temp_gt = torch.cat((gt_centers[B], homog.to('cuda')), 1)
      gt_centers[B] = torch.mm(temp_gt, trans.T.to('cuda'))

    cost_matrix, gt_indexes = match_predictions_ground_truth(detections['center'], 
                          gt_centers, batch['mask'], batch['cam_num'])
    matched_obj_ids = []
    matched_det_loc = []

    BN, max_objs = gt_indexes.shape
    num_cams = calibrations['P'].shape[1]
    
    for B in range(BN):
      cam = calibrations['cam_num'][B]
      P = calibrations['P'][B][cam].to('cpu').numpy()
      trans = np.array([[0,0,0]]).T
      P = np.concatenate((P,trans), axis=1)
      for det in range(len(detections['depth'][B])):
        loc, rot = ddd2locrot(detections['center'][B,det].to('cpu').detach().numpy(), detections['alpha'][B,det].to('cpu').detach().numpy(),
                              detections['size'][B,det].to('cpu').detach().numpy(), detections['depth'][B,det].to('cpu').detach().numpy(), P)
        box_2d = project_3d_bbox(loc, detections['size'][B,det].to('cpu').detach().numpy(), rot, P)

    # Draw centers
    centers = detections['center'].cpu().numpy().astype(int)
    for B in range(BN):
      for center in centers[B]:
        cv2.circle(drawing_images[B,batch['cam_num'][B]], tuple(center), 3, (0,0,255), -1)
      cv2.namedWindow('Detection centers', cv2.WINDOW_NORMAL)
      cv2.imshow('Detection centers', drawing_images[B,batch['cam_num'][B]])
      cv2.waitKey(0)

    for B in range(BN):
      matches_obj = []
      matches_det = []
      cam = batch['cam_num'][B]
      for obj in range(max_objs):
        if cost_matrix[B,obj,gt_indexes[B,obj]] < 100:
          obj_id = batch['obj_id'][B,cam,gt_indexes[B,obj]]
          if obj_id != -1:
            matches_obj.append(obj_id)
            matches_det.append(obj)
      matched_obj_ids.append(matches_obj)
      matched_det_loc.append(matches_det)

    mv_loss = 0
    
    # Put together the ground truth boxes and predicted boxes
    # For each batch
    for B in range(BN):
      gt_bboxes = []
      pr_bboxes = []
      num_matches = len(matched_det_loc[B])
      if num_matches != 0:
        colours, _, _ = generate_colors(num_matches)
        for match in range(num_matches):
          pr_ind = matched_det_loc[B][match]
          obj_id = matched_obj_ids[B][match]
          for cam in range(num_cams):
            obj_id_list = batch['obj_id'][B][cam].tolist()
            try:
              gt_ind = obj_id_list.index(obj_id)
            except:
              continue
            gt_box = batch['bboxes'][B,cam,gt_ind].to('cpu').numpy().astype(int)
            pr_box = detections['2D_bounding_boxes'][B,cam,pr_ind].to('cpu').numpy().astype(int)
            pr_3D_box = detections['proj_3D_boxes'][B,cam,pr_ind].to('cpu').detach().numpy()
            gt_bboxes.append(batch['bboxes'][B,cam,gt_ind])
            pr_bboxes.append(detections['2D_bounding_boxes'][B,cam,pr_ind])
            img = drawing_images[B,cam]
            draw_box_3d(img, pr_3D_box, colours[match])
            cv2.rectangle(img, (gt_box[0],gt_box[1]), (gt_box[0]+gt_box[2],gt_box[1]+gt_box[3]), colours[match], 2)
            cv2.rectangle(img, (pr_box[0],pr_box[1]), (pr_box[0]+pr_box[2],pr_box[1]+pr_box[3]), colours[match], 2)

        gt_bboxes = torch.stack(gt_bboxes)
        pr_bboxes = torch.stack(pr_bboxes)
        mv_loss += generalized_iou_loss(gt_bboxes,pr_bboxes)
        cv2.namedWindow("GT and Predicted bounding boxes", cv2.WINDOW_NORMAL)
        for image in drawing_images[B]:
          
          cv2.imshow("GT and Predicted bounding boxes", image)
          cv2.waitKey(0)

    return mv_loss
