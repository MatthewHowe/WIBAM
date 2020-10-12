# Multi-view utilities file

import numpy as np
import torch.nn as nn
import torch
import math
import shapely
import cv2

def draw_detections(batch, model_detections, calib):


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
      loc_ccf = detections['location'][batch][obj].reshape((3,1))
      loc_wcf = torch.sub(loc_ccf, tvec).float()
      locations_wcf[batch,obj] = torch.mm(R_cw,loc_wcf).reshape((3))

      # Convert detected angle from rad to degrees
      alpha_ccf = detections['alpha'][batch][obj] * 180/math.pi
      alpha_wcf = alpha_ccf + theta

      # Apply rotation to the given angle (theta_x_d + rot_d)

      # Append result to list
      
  detections['location_wcf'] = locations_wcf
  return detections

def dets_3D_wcf_to_dets_2D(dets_3D_wcf, calib):
  r"""
  Reproject 3D detections in world coordinate frame to 2D bounding boxes
  on the given calibrations, trimmed to fit onto frame.
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
  detections_2D = []

  # Convert 3D detections to 3D bounding boxes
  bounding_boxes_3D = det2bb3D(detections_3D_wcf)

  # Repeat process for each detection
  for bounding_box in bounding_boxes_3D:
    # Project 3D bounding box points to camera frame points
    img_pts = cv2.projectPoints(bounding_box, calib['rvec'], calib['tvec'], 
                                calib['P'], calib['dist_coefs'])

    # Find the minimum rectangle fit around the 3D bounding box
    min_x = np.min(img_pts, axis=0)
    min_y = np.min(img_pts, axis=1)
    max_x = np.max(img_pts, axis=0)
    max_y = np.max(img_pts, axis=1)

    # Append result to list
    detections_2D.append(np.array([min_x,min_y,max_x,max_y]))

  return np.array(detections_2D)

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

  R = np.dot(X, Y)
  R = np.dot(R, Z)
  return R

def det_3D_to_BBox_3D(detections_3D):
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
  bounding_box_3D_list = []

  for detection in detections_3D:
    l, w, h = detection[1]

    # Get rotation around Z axis
    rotation_matrix = get_rotation_matrix(np.array([0,0,rot]), detection[2])

    # Get bounding box points
    bounding_box_3D_points = np.array([
      [l/2,-w/2,0],
      [l/2,w/2,0],
      [-l/2,-w/2,0],
      [-l/2,w/2,0],
      [l/2,-w/2,h],
      [l/2,w/2,h],
      [-l/2,-w/2,h],
      [-l/2,w/2,h],
    ])

    # Rotate bounding box
    for point_idx in range(len(bounding_box_3D_points)):
      bounding_box_3D_points[point_idx] = np.dot(rotation_matrix, bounding_box_3D_points[point_idx])

    # Translate bounding box
    bounding_box_3D_points = np.add(bounding_box_3D_points, detection[0])

    # Append to return variable
    bounding_box_3D_list.append(bounding_box_3D_points)

  return np.array(bounding_box_3D_list)

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

    z = (depth[batch].reshape((50)))
    x = ((u * z) / P[0, 0])
    y = ((v * z) / P[1, 1])

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