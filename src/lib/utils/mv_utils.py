# Multi-view utilities file
from os import stat
from numpy.lib.type_check import imag
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch.nn as nn
from model.utils import _sigmoid
from utils.ddd_utils import draw_box_3d, ddd2locrot, project_3d_bbox, ddd2locrot
from utils.drawing_utils import draw_results, draw_birds_eye, initialise_birds_eye_image
from utils.utils import attribute_lists_to_objects, objects_to_attribute_list
from GUI.utils import draw_3D_labels
from utils.bbox_iou_evaluation import match_bboxes, bbox_iou, calculate_3D_iou
import torch
import math
import copy
import time
import cv2

def test_calculations(batch, calib):
  location_wcf = np.array([10,10,1.25]).reshape((3,1))
  cam_num = calib['cam_num'][0]
  # Get calibration information
  rvec = calib['rvec'][0][cam_num].cpu().numpy()
  R = cv2.Rodrigues(rvec)[0]
  tvec = calib['tvec'][0][cam_num].cpu().numpy()
  P = calib['P'][0][cam_num].cpu().numpy()
  dist_coefs = calib['dist_coefs'][0][cam_num].cpu().numpy()

  center = cv2.projectPoints(location_wcf.reshape((1,1,3)), rvec, tvec,
                             P, dist_coefs)[0]

  z = (np.dot(R, location_wcf) + tvec)[2]

  z = torch.Tensor([[z]])
  center = torch.Tensor(center)

  locations = unproject_2d_to_3d(center, z, calib)

  print(locations)

  detections = {"location": locations, "alpha": torch.Tensor([[0]])}

  detections = dets_3D_ccf_to_dets_3D_wcf(detections, calib)
  reprojected = detections['location_wcf'][0][0].cpu().numpy()
  difference = reprojected - location_wcf.reshape((3))

  if np.max(difference) > 1:
    print("[WARNING] Large error in reprojections, check calculations")

def draw_detections(batch, model_detections, calib):
  
  print("[ERROR] Not implemented")

def det_cam_to_det_3D_ccf(model_detections, calib):
  r"""
  Detections from model are formatted with depth, size, rotation(8),
  center location on the frame and must be converted to a location 
  in camera coordinate frame, and rotation(1) in camera coordinate
  frame. Function takes dictionary of detections from camera to 3D detections
  in camera coordinate frame, adding them to the dict
  Arguments:
    model_detections (dict): list of dictionaries for detections from the 
      model
      format: [{center(2),depth(1),size(3),rot(8)}, ...]
    calib (dict): dictionary of calibration information for camera
      format: {P, dist_coefs, rvec, tvec, theta_X_d}
  Returns:
    camera_d etections (dict): list of dictionaries for detections in camera
        coordinate frame.
        format: [{loc(3), size(3), rot(1)}]
  """
  # Get locations of objects
  locations = unproject_2d_to_3d(
    model_detections['center'], 
    model_detections['depth'], 
    calib["P_det"]
  )

  # Get rotation of objects
  alphas = get_alpha(model_detections['rot'])

  model_detections['alpha'] = alphas
  model_detections['location'] = locations

  return model_detections

def unproject_2d_to_3d(center, depth, calib):
  r"""
  Function takes argyment of the center location of object on frame
  and the predicted depth of the object, then using the camera calibration
  will get the x,y,z location of the objects.
  Arguments:
    center (tuple): Center location of object on camera frame
      format: (u,v)
    depth (float): Predicted depth of the object
    calib (dict): dictionary of calibration information for camera
      format: {P, dist_coefs, rvec, tvec, theta_X_d}
  Returns:
    location_3D (np.array, float32): location of the object in camera
      coordinate frame
      format: (x,y,z)
  """

  u = center[:,:,0]
  v = center[:,:,1]

  z = torch.squeeze(depth,-1)
  x = ((u - calib[:, 0, 2, None]) * z) / calib[:, 0, 0, None]
  y = ((v - calib[:, 1, 2, None]) * z) / calib[:, 1, 1, None]

  return torch.stack((x,y,z)).permute(1,2,0).to(device='cuda')

def get_alpha(bin_rotation):
  r"""
  Gets the multi-bin output and converts to a single angle alpha. CenterNet
  implementation doesn't use bin#_cls[0].
  Arguments:
    bin_rotation (np.array, (batch_size,8)): multi-bin rotation prediction
      format: [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
               bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  Returns:
    alpha (float): angle in radians of vehicle relative to camera
  """
  # True if bin1, else bin2
  bin_index = bin_rotation[:, :, 1] > bin_rotation[:, :, 5]

  pi =  torch.Tensor([math.pi]).to(device="cuda")

  alpha_bin1 = torch.atan2(bin_rotation[:, :, 2], bin_rotation[:, :, 3]) + (-0.5 * pi)
  alpha_bin2 = torch.atan2(bin_rotation[:, :, 6], bin_rotation[:, :, 7]) + ( 0.5 * pi)

  # Return only the predicted bin ~ inverts mask
  return alpha_bin1 * bin_index + alpha_bin2 * (~bin_index)

def dets_3D_ccf_to_dets_3D_wcf(detections, calib):
  r"""
  Convert detections from camera coordinate frame to world coordinate
  frame.
  W.C.F: World coordinate frame
  C.C.F: Camera coordinate frame
  Arguments:
      dets3Dc: 3D detections in the camera coordinate frame
          format: [[loc,size,rot], ...]
      calib (dict): dictionary of calibration information for camera detections
          were performed on.
          format: {P, dist_coefs, rvec, tvec, theta_X_d}
  Returns:
      dets3Dw: 3D detections 
  """
  # Initialise list to keep output
  BN, objs, dims = detections['location'].shape
  locations_wcf = torch.zeros((BN, objs, 3)).to(device="cuda")

  # Repeat process for each detection
  for batch in range(BN):
    cam_num = calib['cam_num'][batch]

    # Translation from camera to world origin .in c.c.f.
    tvec = calib['tvec'][batch,cam_num]
    rvec = calib['rvec'][batch,cam_num]
    # Calibration rotation is from world to camera, inv needed
    R_cw = np.linalg.inv(cv2.Rodrigues(rvec.cpu().numpy())[0])
    R_cw = torch.Tensor(R_cw).to(device="cuda")

    for obj in range(objs):
      # Convert detected location from ccf to wcf
      loc_ccf = detections['location'][batch][obj].reshape((3,1))
      loc_wcf = torch.sub(loc_ccf, tvec).float()
      locations_wcf[batch,obj] = torch.mm(R_cw,loc_wcf).reshape((3))

  detections['location_wcf'] = locations_wcf
  return detections

def dets_3D_wcf_to_dets_2D(detections, calib):
  r"""
  Reproject 3D detections in world coordinate frame to 2D bounding boxes
  on all other cameras, trimmed to fit onto frame.
  W.C.F: World coordinate frame
  C.C.F: Camera coordinate frame
  Arguments:
      dets_3D_wcf (np.array, float): list of 3D detections in world.c.f. [[loc,size,rot],...]
      calib (dict): dictionary of calibration information for all cameras
          {P, dist_coefs, rvec, tvec, theta_X_d}
  Returns:
      dets_2D (np.array, float): Detections in 3D on the given camera calibration
          format: 
  """
  BN, objs, dims = detections['location_wcf'].shape
  num_cams = calib['P'].shape[1]
  detections_2D = torch.zeros((BN, num_cams, objs, 4))
  detections_projected3Dbb = torch.zeros((BN, num_cams, objs, 8, 2))
  
  # Convert 3D detections to 3D bounding boxes
  det_3D_to_BBox_3D(detections, calib)

  for batch in range(BN):
    for cam in range(num_cams):
      P = calib['P'][batch][cam]
      rvec = calib['rvec'][batch][cam]
      R_wc = cv2.Rodrigues(rvec.cpu().numpy())[0]
      R_wc = torch.Tensor(R_wc).to(device="cuda")
      tvec = calib['tvec'][batch][cam]

      bounding_box_wcf = detections['3D_bounding_boxes'][batch]
      bounding_box_wcf = bounding_box_wcf.transpose(-2,-1)
      bounding_box_ccf = torch.matmul(R_wc, bounding_box_wcf)
      bounding_box_ccf = torch.add(bounding_box_ccf, tvec)

      bounding_box_cam = torch.matmul(P, bounding_box_ccf)

      bounding_box_cam = torch.div(bounding_box_cam[:],
                                   bounding_box_cam[:,2][:,None,:])

      bounding_box_cam = bounding_box_cam[:,:2].transpose(-2,-1)

      detections_projected3Dbb[batch][cam] = bounding_box_cam

      # Find the minimum rectangle fit around the 3D bounding box
      min_bb = torch.min(bounding_box_cam, axis=1)[0]
      max_bb = torch.max(bounding_box_cam, axis=1)[0]

      detections_2D[batch,cam,:,0] = min_bb[:,0]
      detections_2D[batch,cam,:,1] = min_bb[:,1]
      detections_2D[batch,cam,:,2] = max_bb[:,0]-min_bb[:,0]
      detections_2D[batch,cam,:,3] = max_bb[:,1]-min_bb[:,1]

  detections['proj_3D_boxes'] = detections_projected3Dbb
  detections['2D_bounding_boxes'] = detections_2D.to(device='cuda')

  return detections

def det_3D_to_BBox_3D(detections, calib):
  r"""
  Convert 3D detection to list of 8 x 3D points for bounding boxes
  Arguments:
    dets (np.array, float32): list of detections with format location on ground plane,
      size of object, and rotation +ve is anti-clockwise (right 
      hand rule around z-axis positive up)
      format: [[loc[x,y,0], size[l,w,h], rot[deg]], ...]
  Returns:
    BB3D (np.array, float32): list of 3D bounding boxes in order bottom to top, front to back, 
      right to left
      format: [[btm_fr,btm_fl,btm_rr,btm_rl,top_fr, ...], ...]
  """
  BN, objs, dim = detections['location_wcf'].shape
  bounding_box_3D = torch.zeros((BN, objs, 8, 3))

  for batch in range(BN):
    
    P = calib['P_det'][batch]
    center_x = P[0,2]
    f_x = P[0,0]


    h, w, l = detections['size'][batch].transpose(-1,-2)
    h, w, l = h[:, None], w[:, None], l[:, None]
    x_corners = torch.cat(( l/2,  l/2, -l/2, -l/2,  l/2, l/2, -l/2, -l/2), 1)
    y_corners = torch.cat((-w/2,  w/2,  w/2, -w/2, -w/2, w/2,  w/2, -w/2), 1)
    z_corners = torch.cat((-h/2, -h/2, -h/2, -h/2,  h/2, h/2,  h/2,  h/2), 1)
    points = torch.stack((x_corners, y_corners, z_corners), 1)

    alpha = detections['alpha'][batch]
    ct = detections['center'][batch]
    rotation_object = alpha + torch.atan((ct[:,0]-center_x)/f_x)
    rotation_object = torch.where(rotation_object > np.pi, 
                                  rotation_object - 2 * np.pi, 
                                  rotation_object)
    rotation_object = torch.where(rotation_object < -np.pi, 
                                  rotation_object + 2 * np.pi, 
                                  rotation_object)
    rotation_camera = calib['theta_X_d_det'][batch]*math.pi/180 - math.pi/2
    rotation_object = rotation_camera - rotation_object
    c, s = torch.cos(rotation_object), torch.sin(rotation_object)
    rotation_matrix = torch.zeros((objs, 3, 3))
    rotation_matrix[:, 0, 0] = c
    rotation_matrix[:, 0, 1] = -s
    rotation_matrix[:, 1, 0] = s
    rotation_matrix[:, 1, 1] = c
    rotation_matrix[:, 2, 2] = 1
    rotation_matrix = rotation_matrix.to(device='cuda')

    points = torch.bmm(rotation_matrix, points)
    location = detections['location_wcf'][batch][:,:,None]
    points_a = torch.add(points, location)

    points_b = torch.transpose(points_a, -2, -1)
  
    bounding_box_3D[batch] = points_b

  detections['3D_bounding_boxes'] = bounding_box_3D.to(device='cuda')

  return detections

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

def tranpose_and_gather_feat(feat, ind):
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

def _nms(heat, kernel=3):
  pad = (kernel - 1) // 2

  hmax = nn.functional.max_pool2d(
      heat, (kernel, kernel), stride=1, padding=pad)
  keep = (hmax == heat).float()
  return heat * keep

def _topk_channel(scores, K=100):
  batch, cat, height, width = scores.size()
  
  topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

  topk_inds = topk_inds % (height * width)
  topk_ys   = (topk_inds / width).int().float()
  topk_xs   = (topk_inds % width).int().float()

  return topk_scores, topk_inds, topk_ys, topk_xs

def _topk(scores, K=100):
  batch, cat, height, width = scores.size()
    
  topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

  topk_inds = topk_inds % (height * width)
  topk_ys   = (topk_inds / width).int().float()
  topk_xs   = (topk_inds % width).int().float()
    
  topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
  topk_clses = (topk_ind / K).int()
  topk_inds = _gather_feat(
      topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

  return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def decode_output(output, K=100, opt=None):
  heat = output['hm']
  batch, cat, height, width = heat.size()
  heat = _nms(heat)
  scores, inds, clses, ys0, xs0 = _topk(heat, K=K)

  clses  = clses.view(batch, K)
  scores = scores.view(batch, K)
  bboxes = None
  cts = torch.cat([xs0.unsqueeze(2), ys0.unsqueeze(2)], dim=2)
  ret = {'scores': scores, 'clses': clses.float(), 
         'xs': xs0, 'ys': ys0, 'cts': cts}

  if 'reg' in output:
    reg = output['reg']
    reg = tranpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs0.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys0.view(batch, K, 1) + reg[:, :, 1:2]
  else:
    xs = xs0.view(batch, K, 1) + 0.5
    ys = ys0.view(batch, K, 1) + 0.5

  if 'wh' in output:
    wh = output['wh']
    wh = tranpose_and_gather_feat(wh, inds) # B x K x (F)
    # wh = wh.view(batch, K, -1)
    wh = wh.view(batch, K, 2)
    wh[wh < 0] = 0
    if wh.size(2) == 2 * cat: # cat spec
      wh = wh.view(batch, K, -1, 2)
      cats = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2)
      wh = wh.gather(2, cats.long()).squeeze(2) # B x K x 2
    else:
      pass
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)
    ret['bboxes'] = bboxes
    # print('ret bbox', ret['bboxes'])
 
  if 'ltrb' in output:
    ltrb = output['ltrb']
    ltrb = tranpose_and_gather_feat(ltrb, inds) # B x K x 4
    ltrb = ltrb.view(batch, K, 4)
    bboxes = torch.cat([xs0.view(batch, K, 1) + ltrb[..., 0:1], 
                        ys0.view(batch, K, 1) + ltrb[..., 1:2],
                        xs0.view(batch, K, 1) + ltrb[..., 2:3], 
                        ys0.view(batch, K, 1) + ltrb[..., 3:4]], dim=2)
    ret['bboxes'] = bboxes

 
  regression_heads = ['tracking', 'dep', 'rot', 'dim', 'amodel_offset',
    'nuscenes_att', 'velocity']

  for head in regression_heads:
    if head in output:
      ret[head] = tranpose_and_gather_feat(
        output[head], inds).view(batch, K, -1)

  return ret

def translate_centre_points(points, 
                            img_ctr, 
                            scale, 
                            out_size, 
                            BN, 
                            max_objects, 
                            rot=0):

  trans = torch.Tensor(get_affine_transform(
                img_ctr, scale, rot, out_size,
                inv=1).astype(np.float32)).to('cuda')

  points = torch.cat((points.reshape(BN, max_objects, 2),
                         torch.ones(BN, max_objects,1).to('cuda')),
                         2)

  return torch.matmul(points, trans.T)

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):

  if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
    scale = np.array([scale, scale], dtype=np.float32)

  scale_tmp = scale
  src_w = scale_tmp[0]
  dst_w = output_size[0]
  dst_h = output_size[1]

  rot_rad = np.pi * rot / 180
  src_dir = get_dir([0, src_w * -0.5], rot_rad)
  dst_dir = np.array([0, dst_w * -0.5], np.float32)

  src = np.zeros((3, 2), dtype=np.float32)
  dst = np.zeros((3, 2), dtype=np.float32)
  src[0, :] = center + scale_tmp * shift
  src[1, :] = center + src_dir + scale_tmp * shift
  dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
  dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

  src[2:, :] = get_3rd_point(src[0, :], src[1, :])
  dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

  if inv:
    trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
  else:
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

  return trans

def transform_preds_with_trans(coords, trans):
    # target_coords = np.concatenate(
    #   [coords, np.ones((coords.shape[0], 1), np.float32)], axis=1)
    target_coords = np.ones((coords.shape[0], 3), np.float32)
    target_coords = torch.Tensor(target_coords)
    target_coords[:, :2] = coords.cpu()
    target_coords = torch.mm(trans, target_coords.transpose(0,1)).transpose(0,1)
    return target_coords[:, :2]

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def match_predictions_ground_truth(predicted_centers, gt_centers, gt_mask, cams):
  num_preds = predicted_centers.shape
  init_value = 1e6
  match_indexes = np.zeros((num_preds[0],num_preds[1]), dtype=int)
  match_indexes = np.zeros((num_preds[0],num_preds[1]), dtype=int)
  
  cost_matrix = torch.cdist(predicted_centers, gt_centers, p=2.0).cpu().detach().numpy()

  for batch in range(num_preds[0]):
    mask = gt_mask[batch,cams[batch]].cpu().detach().numpy()
    cost_matrix[batch, :] = cost_matrix[batch, :] * mask + np.bitwise_not(mask) * init_value

    row_indxs, col_indxs = linear_sum_assignment(cost_matrix[batch])
    match_indexes[batch] = col_indxs

  return cost_matrix, match_indexes

def return_four_frames(images, resize=False):
  # Arrange frames
  img_top = np.hstack((images[0],images[1]))
  img_bot = np.hstack((images[2],images[3]))
  all_imgs = np.vstack((img_top,img_bot))

  # Resize and show images
  h, w, d = all_imgs.shape
  ratio = w / h

  if resize:
    all_imgs = cv2.resize(all_imgs,(resize,int(resize/ratio)))

  return all_imgs

def test_post_process(model_output, calib):
  detections = {}
  max_objects = 50
  BN = 1
  model_output = _sigmoid_output(model_output)
  decoded_output = decode_output(model_output, max_objects)
  centers = decoded_output['bboxes'].reshape(BN,max_objects,2, 2).mean(axis=2)
  centers_offset = centers + decoded_output['amodel_offset']

  centers = translate_centre_points(
    centers, np.array([960,540]), 1920, 
    (200,112), BN, max_objects
  )

  centers_offset = translate_centre_points(
    centers_offset, np.array([960,540]), 
    1920, (200,112), BN, max_objects
  )
  detections['scores'] = decoded_output['scores']
  detections['depth'] = decoded_output['dep'] * 0.8 # * (1266 * 64.57)/(1024 * 86.30)
  # detections['depth'] = torch.sigmoid((decoded_output['dep'] * 0.80 - 30) / 10) * 60
  detections['size'] = decoded_output['dim']
  # detections['size'][:, :, 0] = torch.sigmoid(detections['size'][:, :, 0] - 1.5) * 3 # h 0-3m
  # detections['size'][:, :, 1] = torch.sigmoid(detections['size'][:, :, 1] - 2) * 4 # w 0-4m
  # detections['size'][:, :, 2] = torch.sigmoid(detections['size'][:, :, 2] - 4.5) * 9 # l 0-9m
  detections['rot'] = decoded_output['rot']
  detections['center'] = centers_offset

  P = calib["camera_matrix"]

  u = detections['center'][:,:,0]
  v = detections['center'][:,:,1]

  z = torch.squeeze(detections['depth'],-1)
  x = ((u - P[0, 2]) * z) / P[0, 0]
  y = ((v - P[1, 2]) * z) / P[1, 1]

  detections['locations'] = torch.stack((x,y,z)).permute(1,2,0).to(device='cuda')

  detections['alpha'] = get_alpha(detections['rot'])
  locations_wcf = torch.zeros((max_objects, 3)).to(device="cuda")

  ct = detections['center'][0]
  cam_center_x = P[0,2]
  focal_length_x = P[0,0]
  rotation_offset = torch.atan(
    (ct[:,0]-cam_center_x)/focal_length_x
  )
  rotation_object = detections['alpha'] + rotation_offset
  rotation_object = torch.where(rotation_object > np.pi, 
                                rotation_object - 2 * np.pi, 
                                rotation_object)
  rotation_object = torch.where(rotation_object < -np.pi, 
                                rotation_object + 2 * np.pi, 
                                rotation_object)
  rotation_camera = calib['theta_X_d']*math.pi/180 - math.pi/2
  detections['rot'] = rotation_camera - rotation_object

  tvec = torch.Tensor(calib['tvec']).to(device="cuda")
  rvec = calib['rvec']
  R_cw = np.linalg.inv(cv2.Rodrigues(rvec)[0])
  R_cw = torch.Tensor(R_cw).to(device="cuda")

  for obj in range(max_objects):
    loc_ccf = detections['locations'][0][obj].reshape((3,1))
    loc_wcf = torch.sub(loc_ccf, tvec).float()
    locations_wcf[obj] = torch.mm(R_cw,loc_wcf).reshape((3))

  detections['location_wcf'] = locations_wcf.reshape(BN, max_objects, 3)
  return detections

def _sigmoid_output(output):
    if 'hm' in output:
      output['hm'] = _sigmoid(output['hm'])
    if 'hm_hp' in output:
      output['hm_hp'] = _sigmoid(output['hm_hp'])
    if 'dep' in output:
      output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
    return output


over = True
BEV_image = copy.deepcopy(initialise_birds_eye_image())
if over:
  BEV_image['image'] = cv2.imread("output/restrictedBEV_1_123_prewib_ALT.png")

def compare_ground_truth(detection_attributes, annotated_objects, image, calib, cam, opt):
  cv2.imshow("plain", np.array(image.cpu())[0])
  cv2.waitKey(0)
  max_objects = 50
  stats = {}
  for key, val in detection_attributes.items():
    detection_attributes[key] = np.array(val.cpu())[0]
  predicted_objects = attribute_lists_to_objects(detection_attributes)
  predicted_objects_n = {}

  
  for key, val in predicted_objects.items():
    if not np.max(val['size']) > 6:
      predicted_objects_n[key] = val
  predicted_objects = predicted_objects_n
  if len(predicted_objects) < 1:
    return stats, None, None
  images, bev, predictions, ground_truths = draw_results(
    [np.array(image.cpu())[0]], [calib], 
    predicted_objects, annotated_objects
  )

  annotation_attributes = objects_to_attribute_list(ground_truths)
  prediction_attributes = objects_to_attribute_list(predictions)
  
  prediction_attributes['dd_bb_image'] = np.squeeze(np.array(prediction_attributes['dd_bb_image']), -2)
  annotation_attributes['dd_bb_image'] = np.squeeze(np.array(annotation_attributes['dd_bb_image']), -2)
  gt_idx, pred_idx, iou, label = match_bboxes(
    np.array(annotation_attributes['dd_bb_image']), 
    np.array(prediction_attributes['dd_bb_image'])
  )

  predicted_objects = [predicted_objects[i] for i, _ in predicted_objects.items()]
  annotated_objects = [annotated_objects[i] for i, _ in annotated_objects.items()]

  colour = [73,136,255] # ORANGE
  if not over:
    colour = [228,183,61] # BLUE

  for i in range(len(gt_idx)):
    draw_idx = [0,1,2]
    if gt_idx[i] not in draw_idx:
      continue
    if not over:
      draw_birds_eye(ground_truths[gt_idx[i]], BEV_image, tuple([40,190,105]))
    draw_birds_eye(predictions[pred_idx[i]], BEV_image, tuple(colour))
    matching_stats = compare_pred_gt(
      predicted_objects[pred_idx[i]],
      annotated_objects[gt_idx[i]],
      cam
    )
    matching_stats['2D_iou'] = iou[i]
    stats[i] = matching_stats

  if opt.show_repro:
    for image in images:
      cv2.namedWindow("result", cv2.WINDOW_NORMAL)
      cv2.imshow("result", image)
      cv2.imshow("bev", bev)
      cv2.imshow("restrictedBEV", BEV_image['image'])
      cv2.waitKey(0)

  return stats, images, bev

def compare_pred_gt(prediction, ground_truth, cam):
  stats = {}
  stats['visibility'] = ground_truth['visibility'][cam]
  for key, val in prediction.items():
    if key in ground_truth:
      if key == 'location':
        stats[key] = np.linalg.norm(val - ground_truth[key])
      elif key == 'size':
        error = val - ground_truth[key]
        # stats[key] = val - ground_truth[key]
        stats['l'], stats['w'], stats['h'] = error[0], error[1], error[2]
      elif key == 'rot':
        stats[key] = abs(((((val - ground_truth[key])*180/math.pi)+180) % 360) - 180)
      elif key == 'ddd_bb_world':
        stats['3D_iou'], stats['bev_iou'] = calculate_3D_iou(val, ground_truth[key])

  return stats
