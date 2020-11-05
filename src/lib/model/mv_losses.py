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
    BN = len(batch['cam_num'])
    max_objects = self.opt.K
    num_cams = batch['P'].shape[1]

    decoded_output = decode_output(output, self.opt.K)

    # Post processing code
    ctrs = torch.zeros((BN,max_objects,2)).to('cuda')
    trans = get_affine_transform(
      np.array([960,540]), 1920, 0, (200, 112), inv=1).astype(np.float32)
    trans = torch.Tensor(trans).to('cuda')

    homog = torch.ones(max_objects,1).to('cuda')
    
    for B in range(BN):
      ct_output = decoded_output['bboxes'][B].reshape(max_objects,2, 2).mean(axis=1)
      amodel_ct_output = ct_output + decoded_output['amodel_offset'][B]
      cts = amodel_ct_output.reshape(max_objects, 2)
      cts = torch.cat((cts,homog),1)
      ctrs[B] = torch.mm(cts, trans.T)

    # Put detections into their own dictionary with required information
    # Scale depth with focal length
    detections['depth'] = decoded_output['dep'] * 1046/1266
    detections['size'] = decoded_output['dim'] 
    detections['rot'] = decoded_output['rot']
    detections['center'] = ctrs
    calibrations['cam_num'] = batch['cam_num']
    calibrations['P'] = batch['P']
    calibrations['dist_coefs'] = batch['dist_coefs']
    calibrations['tvec'] = batch['tvec']
    calibrations['rvec'] = batch['rvec']
    calibrations['theta_X_d'] = batch['theta_X_d']
  
    # Get predictions in camera.c.f loc(x,y,z), rot(alpha),size(l,w,h)
    # Adds them to detections dict
    det_cam_to_det_3D_ccf(detections,calibrations)

    # Put the predictions into world coordinate frame
    # Adds them to detections dict
    dets_3D_ccf_to_dets_3D_wcf(detections, calibrations)

    # Produce all the reprojections for every other camera
    dets_3D_wcf_to_dets_2D(detections, calibrations)

    # Get the ground truth centers
    gt_centers = batch['ctr'].type(torch.float)
    for B in range(BN):
      temp_gt = torch.cat((gt_centers[B], homog), 1)
      gt_centers[B] = torch.mm(temp_gt, trans.T)

    cost_matrix, gt_indexes = match_predictions_ground_truth(detections['center'], 
                          gt_centers, batch['mask'], batch['cam_num'])
    empty = []
    gt_matched_boxes = [[empty.copy()]*num_cams]*BN 
    pr_matched_boxes = [[empty.copy()]*num_cams]*BN 
    ddd_matched_boxes = [[empty.copy()]*num_cams]*BN
    matches = [0]*BN
    mv_loss = {}
    for cam in range(num_cams):
      mv_loss[cam] = 0
    mv_loss['det'] = 0
    mv_loss['tot'] = 0

    if self.opt.show_repro:
      drawing_images = batch['drawing_images'].detach().cpu().numpy()

    for B in range(BN):
      det_cam = int(batch['cam_num'][B].item())
      colours, _, _ = generate_colors(max_objects)
      for pr_index in range(max_objects):
        gt_index = gt_indexes[B, pr_index]

        if cost_matrix[B, pr_index, gt_index] < 50:
          gt_box_T = batch['bboxes'][B, det_cam, gt_index]
          gt_matched_boxes[B][det_cam].append(gt_box_T)
          pr_box_T = detections['2D_bounding_boxes'][B,det_cam,pr_index]
          pr_matched_boxes[B][det_cam].append(pr_box_T)
          obj_id = batch['obj_id'][B, det_cam, gt_index]

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
            matches[B] += 1
            
            for cam in range(num_cams):
              if cam == det_cam:
                continue
              
              try:
                obj_id_list = batch['obj_id'][B][cam].tolist()
                gt_ind = obj_id_list.index(obj_id)
              except:
                continue
              
              gt_box_T = batch['bboxes'][B,cam,gt_ind]
              gt_matched_boxes[B][cam].append(gt_box_T)
              pr_box_T = detections['2D_bounding_boxes'][B,cam,pr_index]
              pr_matched_boxes[B][cam].append(pr_box_T)

              # Drawing functions
              if self.opt.show_repro:
                img = drawing_images[B,cam]
                ddd_box = detections['proj_3D_boxes'][B, cam, pr_index].detach().cpu().numpy().astype(int)
                gt_box = torch.clone(gt_box_T).detach().cpu().numpy().astype(int)
                pr_box = torch.clone(pr_box_T).detach().cpu().numpy().astype(int)
                draw_box_3d(img, ddd_box, colours[pr_index])
                cv2.rectangle(img, (gt_box[0],gt_box[1]), (gt_box[0]+gt_box[2],gt_box[1]+gt_box[3]), colours[pr_index], 2)
                cv2.rectangle(img, (pr_box[0],pr_box[1]), (pr_box[0]+pr_box[2],pr_box[1]+pr_box[3]), colours[pr_index], 2)

    # Calculate GIoU for all matches
    for B in range(BN):
      for cam in range(num_cams):
        if len(gt_matched_boxes[B][cam]) > 0:
          gt_bboxes = torch.stack(gt_matched_boxes[B][cam])
          pr_bboxes = torch.stack(pr_matched_boxes[B][cam])
          loss = generalized_iou_loss(gt_bboxes,pr_bboxes, 'mean')
          if cam == batch['cam_num'][B]:
            mv_loss['det'] += loss
          else:
            mv_loss[cam] += loss  
          mv_loss['tot'] += loss

    if self.opt.show_repro:
      for B in range(BN):
        composite = return_four_frames(drawing_images[B])
        cv2.namedWindow("Batch {}".format(B), cv2.WINDOW_NORMAL)
        cv2.imshow("Batch {}".format(B), composite)
        cv2.waitKey(0)

    #   # Drawing function
    #   drawing_images = batch['drawing_images'].numpy()
    #   # Draw centers
    #   centers = detections['center'].cpu().detach().numpy().astype(int)
    #   for B in range(BN):
    #     for center in centers[B]:
    #       cv2.circle(drawing_images[B,batch['cam_num'][B]], tuple(center), 3, (0,0,255), -1)

    #   for B in range(BN):
    #     num_matches = len(matched_det_loc[B])
    #     if num_matches != 0:
    #       colours, _, _ = generate_colors(num_matches)
    #       for match in range(num_matches):
    #         pr_ind = matched_det_loc[B][match]
    #         obj_id = matched_obj_ids[B][match]
    #         for cam in range(num_cams):
    #           obj_id_list = batch['obj_id'][B][cam].tolist()
    #           try:
    #             gt_ind = obj_id_list.index(obj_id)
    #           except:
    #             continue
    #           gt_box = batch['bboxes'][B,cam,gt_ind].cpu().detach().numpy().astype(int)
    #           pr_box = detections['2D_bounding_boxes'][B,cam,pr_ind].cpu().detach().numpy().astype(int)
    #           pr_3D_box = detections['proj_3D_boxes'][B,cam,pr_ind].cpu().detach().numpy()
    #           img = drawing_images[B,cam]
    #           draw_box_3d(img, pr_3D_box, colours[match])
    #           cv2.rectangle(img, (gt_box[0],gt_box[1]), (gt_box[0]+gt_box[2],gt_box[1]+gt_box[3]), colours[match], 2)
    #           cv2.rectangle(img, (pr_box[0],pr_box[1]), (pr_box[0]+pr_box[2],pr_box[1]+pr_box[3]), colours[match], 2)

    #       cv2.namedWindow("GT and Predicted bounding boxes", cv2.WINDOW_NORMAL)
    #       for image in drawing_images[B]:
    #         cv2.imshow("GT and Predicted bounding boxes", image)
    #         cv2.waitKey(0)

    return mv_loss
