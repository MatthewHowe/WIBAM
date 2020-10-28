# Multi-view utilities file
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch.nn as nn
import torch
import math
import shapely
import cv2

def test_calculations(batch, calib):
  location_wcf = np.array([10,10,1.25]).reshape((3,1))

  # Get calibration information
  rvec = calib['rvec'][0][0].cpu().numpy()
  R = cv2.Rodrigues(rvec)[0]
  tvec = calib['tvec'][0][0].cpu().numpy()
  P = calib['P'][0][0].cpu().numpy()
  dist_coefs = calib['dist_coefs'][0][0].cpu().numpy()

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
  locations = unproject_2d_to_3d(model_detections['center'], 
                                 model_detections['depth'], 
                                 calib)

  # Get rotation of objects
  alphas = get_alpha(model_detections['rot'])

  model_detections['alpha'] = alphas
  model_detections['location'] = locations

  return model_detections

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
  alphas_wcf = torch.zeros((BN, objs)).to(device="cuda")

  # Repeat process for each detection
  for batch in range(BN):
    cam_num = calib['cam_num'][batch]

    # Translation from camera to world origin .in c.c.f.
    tvec = calib['tvec'][batch,cam_num]
    rvec = calib['rvec'][batch,cam_num]
    theta = calib['theta_X_d'][batch,cam_num]
    # Calibration rotation is from world to camera, inv needed
    R_cw = np.linalg.inv(cv2.Rodrigues(rvec.cpu().numpy())[0])
    R_cw = torch.Tensor(R_cw).to(device="cuda")

    for obj in range(objs):
      # Convert detected location from ccf to wcf
      loc_ccf = detections['location'][batch][obj].reshape((3,1))
      loc_wcf = torch.sub(loc_ccf, tvec).float()
      locations_wcf[batch,obj] = torch.mm(R_cw,loc_wcf).reshape((3))

      # Convert detected angle from rad to degrees
      alpha_ccf = detections['alpha'][batch][obj] * 180/math.pi
      alphas_wcf[batch, obj] = alpha_ccf + theta

  detections['location_wcf'] = locations_wcf
  detections['alpha_wcf'] = alphas_wcf
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

  # Convert 3D detections to 3D bounding boxes
  detections = det_3D_to_BBox_3D(detections)

  # Repeat process for each detection
  for batch in range(BN):
    for cam in range(num_cams):
      for obj in range(objs):
        # Project 3D bounding box points to camera frame points
        bounding_box_wcf = detections['3D_bounding_boxes'][batch][obj]
        P = calib['P'][batch][cam]
        rvec = calib['rvec'][batch][cam]
        R_wc = cv2.Rodrigues(rvec.cpu().numpy())[0]
        R_wc = torch.Tensor(R_wc).to(device="cuda")
        tvec = calib['tvec'][batch][cam]

        bounding_box_wcf = bounding_box_wcf.transpose(0,1)
        bounding_box_ccf = torch.add(torch.mm(R_wc, bounding_box_wcf), tvec)

        bounding_box_cam = torch.mm(P, bounding_box_ccf)

        bounding_box_cam = torch.div(bounding_box_cam,bounding_box_cam[2])

        bounding_box_cam = bounding_box_cam[:2].transpose(0,1)

        # Find the minimum rectangle fit around the 3D bounding box
        min_x, min_y = torch.min(bounding_box_cam, axis=0)[0]
        max_x, max_y = torch.max(bounding_box_cam, axis=0)[0]

        # Append result to list
        detections_2D[batch][cam][obj] = torch.Tensor([min_x,min_y,max_x,max_y])

  detections['2D_bounding_boxes'] = detections_2D.to(device='cuda')

  return detections

# Function to get rotation matrix fiven three rotations
def get_rotation_matrix(rotations, degrees=True):
  r"""
  Creates a 3x3 rotation matrix from three angles given in
  tuple/array format
  Arguments:
      rot: three angles given in tuple or array format (float)
          format: (rot_x,rot_y,rot_z)
      degrees: Whether or not angles are given in degrees (bool)
          format: bool
  Returns:
      R: 3x3 rotation matrix (numpy 2D array)
  """

  if degrees:
      rotations *= math.pi/180

  X_rotation = np.array([
    [1, 0, 0],
    [0, math.cos(rotations[0]), math.sin(rotations[0])],
    [0,-math.sin(rotations[0]), math.cos(rotations[0])]
  ])

  Y_rotation = np.array([
    [math.cos(rotations[1]), 0,-math.sin(rotations[1])],
    [0, 1, 0],
    [math.sin(rotations[1]), 0, math.cos(rotations[1])]
  ])

  Z_rotation = np.array([
    [math.cos(rotations[2]),-math.sin(rotations[2]), 0],
    [math.sin(rotations[2]), math.cos(rotations[2]), 0],
    [0,0,1]
  ])

  R = np.dot(X_rotation, Y_rotation)
  R = np.dot(R, Z_rotation)
  return R

def det_3D_to_BBox_3D(detections):
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
    for obj in range(objs):
      l, w, h = detections['size'][batch,obj]
      alpha = detections['alpha'][batch,obj]
      loc = detections['location_wcf'][batch,obj].reshape((3,1)).cpu()

      # Get rotation around Z axis
      rotation_matrix = get_rotation_matrix(np.array([0,0,alpha]))
      rotation_matrix = torch.Tensor(rotation_matrix)

      # Get bounding box points
      bounding_box_3D_points = torch.Tensor([
        [l/2,-w/2,0],
        [l/2,w/2,0],
        [-l/2,-w/2,0],
        [-l/2,w/2,0],
        [l/2,-w/2,h],
        [l/2,w/2,h],
        [-l/2,-w/2,h],
        [-l/2,w/2,h],
      ]).transpose(0,1)

      # Rotate bounding box
      bounding_box_3D_points = torch.mm(rotation_matrix, bounding_box_3D_points)

      # Translate bounding box
      bounding_box_3D_points = torch.add(bounding_box_3D_points, loc)

      # Append to return variable
      bounding_box_3D[batch,obj] = torch.transpose(bounding_box_3D_points, 0, 1)

    detections['3D_bounding_boxes'] = bounding_box_3D.to(device='cuda')

    return detections

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
  locations = torch.zeros((center.shape[0], 3, center.shape[1]))

  for batch in range(center.shape[0]):
    camera_number = int(calib['cam_num'][batch])
    rvec = calib['rvec'][batch][camera_number].cpu().numpy()

    P = calib['P'][batch][camera_number]

    u = center[batch,:,0]
    v = center[batch,:,1]

    z = (depth[batch].reshape((center.shape[1])))
    x = (((u - P[0, 2]) * z) / P[0, 0])
    y = (((v - P[1, 2]) * z) / P[1, 1])

    locations[batch][0] = x
    locations[batch][1] = y 
    locations[batch][2] = z

  return locations.permute(0, 2, 1).contiguous().to(device='cuda')
  
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

# Finds top 100 objects on heatmap
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
         'xs': xs0, 'ys': ys0, 'centers': cts}

  return ret

def match_predictions_ground_truth(predicted_centers, gt_centers, gt_mask, cams):
  num_preds = predicted_centers.shape
  num_gt = gt_centers.shape[1] 
  init_value = 1000000
  match_indexes = np.zeros((num_preds[0],num_preds[1]), dtype=int)
  cost_matrix = np.full((num_preds[0], num_preds[1], num_gt), init_value, dtype=float)
  for batch in range(num_preds[0]):
    # Produce cost matrix
    for i in range(num_preds[1]):
      for j in range(num_gt):
        if gt_mask[batch,cams[batch],j] != 0:
          cost_matrix[batch,i,j] = torch.norm(predicted_centers[batch, i]-gt_centers[batch, j])

    row_indxs, col_indxs = linear_sum_assignment(cost_matrix[batch])
    print(cost_matrix[batch,row_indxs,col_indxs].sum())
    print(cost_matrix[batch,row_indxs,col_indxs])
    match_indexes[batch] = col_indxs
    

  return cost_matrix, match_indexes