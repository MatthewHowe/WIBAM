# Multi-view utilities file

import numpy as np
import cv2

# Function to get rotation matrix fiven three rotations
def R_matrix(rot, degrees=True):
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

    # If angle in degrees convert to radians
    if degrees:
        rot *= math.pi/180

    # X rotation
    X = np.array([
        [1, 0, 0],
        [0, math.cos(rot[0]), math.sin(rot[0])],
        [0,-math.sin(rot[0]), math.cos(rot[0])]
    ])

    # Y rotation matrix
    Y = np.array([
        [math.cos(rot[1]), 0,-math.sin(rot[1])],
        [0, 1, 0],
        [math.sin(rot[1]), 0, math.cos(rot[1])]
    ])

    # Z rotation matrix
    Z = np.array([
        [math.cos(rot[2]),-math.sin(rot[2]), 0],
        [math.sin(rot[2]), math.cos(rot[2]), 0],
        [0,0,1]
        ])

    # Combine all rotations
    R = np.dot(X, Y)
    R = np.dot(R, Z)
    return R

def reproject3Ddet(dets3Dw, calib):
    r"""
    Reproject 3D detections in world coordinate frame to 2D bounding boxes
    on the given image, trimmed to fit onto frame
    Arguments:
        dets3Dw: list of 3D detections in world.c.f. [[loc,size,rot],...]
        calib: dictionary of calibration information for camera
            {P, dist_coefs, rvec, tvec, theta_X_d}
    Returns:
        dets2d: Detections in 3D on the given camera calibration
            format: [[x_min,y_min,x_max,y_max], ...]
    """
    # Initialise list to keep 2D detections
    dets2D = []

    # Repeat process for each detection
    for det3D in dets3Dw:
        # Convert 3D detections to 3D bounding boxes


        # Project 3D bounding box points to camera frame points

        # Find the minimum rectangle fit around the 3D bounding box

        # Append result to list
        dets2D.append()

    return dets2D

def cam2world(dets3Dc, calib):
    r"""
    Convert detections from camera coordinate frame to world coordinate
    frame.
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
    dets3Dw = []
    # Repeat process for each detection
    for det3D in dets3Dc:
        # Apply translation for location
        
        # Convert detected angle from rad to degrees

        # Apply rotation to the given angle (theta_x_d + rot_d)

        # Append result to list
        dets3Dw.append()

    return dets3Dw

# Convert 3D detection to list of 8 x 3D points for bounding boxes
def det2bb3D(dets):
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
    # Initialise return variable
    BB3D = []

    # For each detection given
    for det in dets:
        # Decompose size
        l, w, h = det[1]

        # Get rotation around Z axis
        R = R_matrix(np.array([0,0,rot]), det[2])

        # Get bounding box points
        BB = np.array([
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
        for i in range(len(BB)):
            BB[i] = np.dot(R, BB[i])

        # Translate bounding box
        BB = np.add(BB, det[0])

        # Append to return variable
        BB3D.append(BB)

    return np.array(BB3D)

# Function summary template
r"""
Description
Arguments:
    Variable name: description of variable
        format of variable
"""