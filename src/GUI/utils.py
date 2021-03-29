# from .lib.dataset.datasets.coco_tools import coco
# from ..lib.dataset.datasets.coco_tools import coco
from lib.dataset.datasets.coco_tools import coco
import numpy as np
import cv2
import math
import copy
import random

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
  objs = len(detections)
  bounding_box_3D = np.zeros((objs, 8, 3))

  for i in range(len(detections)):
    obj = detections[i]
    l = obj['l']
    w = obj['w']
    h = obj['h']
    x = obj['x']
    y = obj['y']
    z = obj['z']
    rot = obj['rot']
    
    x_corners = np.array([ l/2,  l/2, -l/2, -l/2,  l/2, l/2, -l/2, -l/2])
    y_corners = np.array([-w/2,  w/2,  w/2, -w/2, -w/2, w/2,  w/2, -w/2])
    z_corners = np.array([0, 0, 0, 0,  h, h,  h,  h])
    points = np.stack((x_corners, y_corners, z_corners), 1)

    rotation_object = rot * math.pi/180
    c, s = math.cos(rotation_object), math.sin(rotation_object)
    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[0, 0] = c
    rotation_matrix[0, 1] = -s
    rotation_matrix[1, 0] = s
    rotation_matrix[1, 1] = c
    rotation_matrix[2, 2] = 1

    points = np.transpose(points)
    points = np.matmul(rotation_matrix, points)
    location = np.array([[x],[y],[z]])
    points = np.add(location, points)
    points = np.transpose(points)

    bounding_box_3D[i] = points

  return bounding_box_3D

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
  objs = len(detections)
  num_cams = len(calib)
  detections_2D = np.zeros((num_cams, objs, 4))
  detections_projected3Dbb = np.zeros((num_cams, objs, 8, 2))
  
  # Convert 3D detections to 3D bounding boxes
  detections_3Dbb = det_3D_to_BBox_3D(detections, calib)

  for cam in range(num_cams):
    cam_calib = calib[cam]
    P = cam_calib['camera_matrix']
    rvec = cam_calib['rvec']
    R_wc = cv2.Rodrigues(rvec)[0]
    tvec = cam_calib['tvec']

    for obj in range(len(detections_3Dbb)):
      box = detections_3Dbb[obj]
      bounding_box_ccf = box.transpose()
      bounding_box_ccf = np.matmul(R_wc, bounding_box_ccf)
      bounding_box_ccf = np.add(bounding_box_ccf, tvec)
      if True in (bounding_box_ccf[2] < 0):
        # print("Behind camera {} {}".format(cam, obj))
        continue
      bounding_box_cam = np.matmul(P, bounding_box_ccf)
      divide = bounding_box_cam[2,:]
      bounding_box_cam = np.divide(
        bounding_box_cam[:],
        bounding_box_cam[2,:]
      )

      bounding_box_cam = bounding_box_cam[:2,:].transpose()

      detections_projected3Dbb[cam,obj] = bounding_box_cam

    # detections_2D[batch,cam,:,0] = min_bb[:,0]
    # detections_2D[batch,cam,:,1] = min_bb[:,1]
    # detections_2D[batch,cam,:,2] = max_bb[:,0]-min_bb[:,0]
    # detections_2D[batch,cam,:,3] = max_bb[:,1]-min_bb[:,1]

  # detections['proj_3D_boxes'] = detections_projected3Dbb
  # detections['2D_bounding_boxes'] = detections_2D.to(device='cuda')

  return detections_projected3Dbb, detections_3Dbb

def comput_corners_3d(dim, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  l, w, h = dim[2], dim[1], dim[0]
  x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
  y_corners = [0,0,0,0,-h,-h,-h,-h]
  z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

  corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
  corners_3d = np.dot(R, corners).transpose(1, 0)
  return corners_3d

def compute_box_3d(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  corners_3d = comput_corners_3d(dim, rotation_y)
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(1, 3)
  return corners_3d

def draw_box_3d(image, corners, c=(255, 0, 255), same_color=False):
  face_idx = [[0,1,5,4], #FLB, FRB, FRT, FLT
              [1,2,6, 5],
              [3,0,4,7],
              [2,3,7,6]] #RRB, RLB, RLT, RRT
  right_corners = [1, 2, 6, 5] if not same_color else []
  left_corners = [0, 3, 7, 4] if not same_color else []
  thickness = 4 if same_color else 2
  corners = corners.astype(np.int32)
  for ind_f in range(3, -1, -1):
    f = face_idx[ind_f]
    for j in range(4):
      # print('corners', corners)
      cc = c
      # if (f[j] in left_corners) and (f[(j+1)%4] in left_corners):
      #   cc = (255, 0, 0)
      # if (f[j] in right_corners) and (f[(j+1)%4] in right_corners):
      #   cc = (0, 0, 255)
      # try:
      cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
          (corners[f[(j+1)%4], 0], corners[f[(j+1)%4], 1]), cc, thickness, lineType=cv2.LINE_AA)

    if ind_f == 0:
      try:
        cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
                 (corners[f[2], 0], corners[f[2], 1]), c, 1, lineType=cv2.LINE_AA)
        cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
                 (corners[f[3], 0], corners[f[3], 1]), c, 1, lineType=cv2.LINE_AA)
      except:
        pass
    # top_idx = [0, 1, 2, 3]
  return image

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

def draw_3D_labels(images, labels, calib):
    projected_boxes, boxes = dets_3D_wcf_to_dets_2D(labels, calib)
    
    for i in range(len(images[:4])):
      for j in range(len(labels)):
        if labels[j]['current']:
          colour = (255,0,0)
        else:
          colour = (0,0,255)
        draw_box_3d(images[i], projected_boxes[i,j], c=colour)

    if len(images) > 4:
        images[-1] = return_four_frames(images)
    else:
        images.append(return_four_frames(images) )
    return images, boxes, projected_boxes.astype(int)

def get_annotations(idx, ann_path):
  CoCo = coco.COCO(ann_path)
  annotation_id = CoCo.getAnnIds(imgIds=[idx])
  annotation = copy.deepcopy(
    CoCo.loadAnns(ids=annotation_id)
  )

  return annotation

