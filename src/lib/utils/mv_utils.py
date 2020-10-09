# Multi-view utilities file

import numpy as np
import shapely
import cv2

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

def reproject_3D_detection(detections_3D_wcf, calib):
  r"""
  Reproject 3D detections in world coordinate frame to 2D bounding boxes
  on the given image, trimmed to fit onto frame.
  W.C.F: World coordinate frame
  C.C.F: Camera coordinate frame
  Arguments:
      dets3Dw (np.array, float): list of 3D detections in world.c.f. [[loc,size,rot],...]
      calib (dict): dictionary of calibration information for camera
          {P, dist_coefs, rvec, tvec, theta_X_d}
  Returns:
      dets2d (np.array, float): Detections in 3D on the given camera calibration
          format: [[x_min,y_min,x_max,y_max], ...]
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

def cam_to_world(detections_3D_ccf, calib):
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
    detections_3D_wcf = []
    # Repeat process for each detection
    for detection_3D_wcf in detections_3D_ccf:
        # Apply translation for location
        
        # Convert detected angle from rad to degrees

        # Apply rotation to the given angle (theta_x_d + rot_d)

        # Append result to list
        detections_3D_wcf.append()

    return detections_3D_wcf

def detection_to_BBox_3D(detections_3D):
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

def detection_to_3D(model_detections, calib):
  r"""
  Detections from model are formatted with depth, size, rotation(8),
  center location on the frame and must be converted to a location 
  in camera coordinate frame, and rotation(1) in camera coordinate
  frame.
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
  unproject_2d_to_3d()

  # Get rotation of objects
  get_aplha()

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

  # Get Rotation|translation matrix (3x4)
  R_t = cv2.composeRT(calib['rvec'], calib['tvec'])

  z = depth - R_t[2, 3]
  x = (center[0] * depth - R_t[0, 3] - R_t[0, 2] * z) / R_t[0, 0]
  y = (center[1] * depth - R_t[1, 3] - R_t[1, 2] * z) / R_t[1, 1]
  location_3D = np.array([x, y, z], dtype=np.float32).reshape(3)
  return location_3D

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
  bin_index = bin_rotation[:, 1] > bin_rotation[:, 5]
  alpha_bin1 = np.arctan2(bin_rotation[:, 2], bin_rotation[:, 3]) + (-0.5 * np.pi)
  alpha_bin2 = np.arctan2(bin_rotation[:, 6], bin_rotation[:, 7]) + ( 0.5 * np.pi)

  # Return only the predicted bin
  return alpha_bin1 * bin_index + alpha_bin2 * (1 - bin_index)
