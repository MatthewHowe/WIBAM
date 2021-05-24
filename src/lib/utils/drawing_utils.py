import numpy as np
import random
import cv2

def draw_results(images, calibrations, predictions=[], ground_truths=[],
                 matches = [], colour_format='red-green', birds_eye=None, 
                 draw=['ddd_boxes', 'birds_eye', 'dd_boxes', 'gt']):

    if isinstance(predictions, dict):
        predictions = dict_to_list(predictions)
    if isinstance(ground_truths, dict):
        ground_truths = dict_to_list(ground_truths)

    if 'birds_eye' in draw and birds_eye is None:
        birds_eye = initialise_birds_eye_image()

    pr_colours, gt_colours = get_colours(
        len(predictions), len(ground_truths), colour_format, matches)
    
    for pr in predictions:
        draw_object(pr, images, calibrations, draw, pr_colours[predictions.index(pr)], birds_eye)
    
    if 'gt' in draw:
        for gt in ground_truths:
            draw_object(gt, images, calibrations, draw, gt_colours[ground_truths.index(gt)], birds_eye)

    return images, birds_eye['image'], predictions, ground_truths

BEV_image = cv2.imread("data/wibam/cleanBEV.png")

def dict_to_list(dict):
    temp = []
    for key, val in dict.items():
        temp.append(val)
    return temp

def initialise_birds_eye_image(path=None):
    meter_bounds = np.array([[20,-5], [40,40]])
    scale = 17.45 # pixes/meter
    offset = np.array([3,-2]) # ()
    pixel_bounds = (meter_bounds + offset) * scale
    # image = np.zeros((np.max(pixel_bounds[:,0]), np.max(pixel_bounds[:,1]), 3), dtype=np.uint8)
    # image = cv2.imread("data/wibam/cleanBEV.png")
    return {'image': BEV_image, 'scale': scale, 'offset': offset, 'bounds': meter_bounds}

def get_colours(number_pr, number_gt, colour_format='red-green', matches=[]):
    if colour_format == 'unique':
        pr_colours, _, _ = generate_colors(number_pr)
        gt_colours = [(0,0,0)]*number_pr
        for match in matches:
            gt_colours[match[1]] = pr_colours[match[0]]
    else:
        pr_colours = [(0,0,255)]*number_pr
        gt_colours = [(0,255,0)]*number_gt

    return pr_colours, gt_colours

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

def draw_object(obj, images, calibrations, draw, colour, birds_eye):
    if 'ddd_boxes' in draw and 'location' in obj:
        obj['ddd_bb_world'], obj['ddd_bb_image'] = object_to_ddd_bb(obj, calibrations)
        draw_ddd_box(obj['ddd_bb_image'], images, colour)

    if 'dd_boxes' in draw and ('dd_bb_image' in obj or 'ddd_bb_world' in obj):
        if 'dd_bb_image' not in obj:
            obj['dd_bb_image'] = get_dd_bb_image(obj['ddd_bb_image'])
        # draw_dd_box(obj['dd_bb_image'], images, colour)

    if 'birds_eye' in draw and 'location' in obj:
        if 'ddd_bb_world' not in obj:
            obj['ddd_bb_world'], obj['ddd_bb_image'] = object_to_ddd_bb(obj, calibrations)
        draw_birds_eye(obj, birds_eye, colour)

def object_to_ddd_bb(ddd_object, calibrations):
    ddd_bb_world = get_ddd_bb_world(ddd_object)
    ddd_bb_image = get_ddd_bb_image(ddd_bb_world, calibrations)

    return ddd_bb_world, ddd_bb_image   

def get_ddd_bb_world(obj):
    c, s = np.cos(obj['rot']), np.sin(obj['rot'])
    R = np.array([[ c,-s, 0],
                  [ s, c, 0],
                  [ 0, 0, 1]], dtype=np.float32)
    l, w, h = obj['size']
    x_corners = np.array([ l/2,  l/2, -l/2, -l/2,  l/2, l/2, -l/2, -l/2])
    y_corners = np.array([-w/2,  w/2,  w/2, -w/2, -w/2, w/2,  w/2, -w/2])
    z_corners = np.array([0, 0, 0, 0,  h, h,  h,  h])
    corners = np.stack((x_corners, y_corners, z_corners), 0).astype(np.float32)

    ddd_box_world = np.dot(R, corners).transpose(1, 0) 
    ddd_box_world = ddd_box_world + np.array(obj['location'])[None,:]

    return ddd_box_world

def get_ddd_bb_image(ddd_bb_world, calibrations):
    image_points = []
    for calib in calibrations:
        R_wc = cv2.Rodrigues(calib['rvec'])[0]
        ddd_bb_world = ddd_bb_world.transpose(1,0)
        ddd_bb_camera = np.matmul(R_wc, ddd_bb_world)
        ddd_bb_camera = np.add(ddd_bb_camera, calib['tvec'])

        dd_bb = np.matmul(calib['camera_matrix'], ddd_bb_camera)
        dd_bb = np.divide(dd_bb, dd_bb[2])
        dd_bb = dd_bb[:2].transpose(1, 0).astype(int)
        image_points.append(dd_bb)

    return image_points

def get_dd_bb_image(ddd_bb_image):
    dd_bbs = []
    for i in range(len(ddd_bb_image)):
        min_bb = np.min(ddd_bb_image[i], axis=0)
        max_bb = np.max(ddd_bb_image[i], axis=0)

        dd_bb = np.zeros((4))
        dd_bb[0] = min_bb[0]
        dd_bb[1] = min_bb[1]
        dd_bb[2] = max_bb[0] - min_bb[0]
        dd_bb[3] = max_bb[1] - min_bb[1]
        dd_bbs.append(dd_bb.astype(int))

    return dd_bbs


def draw_ddd_box(image_points, images, colour=(255,0,0), same_color=True):
    for img in range(len(images)):
        image = images[img]
        corners = image_points[img]
        face_indices = [[0, 1, 5, 4], #FLB, FRB, FRT, FLT
                        [1, 2, 6, 5],
                        [3, 0, 4, 7], 
                        [2, 3, 7, 6]] #RRB, RLB, RLT, RRT
        right_corners = face_indices[1] if not same_color else []
        left_corners = face_indices[2] if not same_color else []
        thickness = 4
        corners = corners.astype(np.int32)
        for face_index in range(3, -1, -1):
            face = face_indices[face_index]
            for i in range(4):
                face_colour = colour
                if (face[i] in left_corners) and (face[(i+1)%4] in left_corners):
                    face_colour = (255, 0, 0)
                if (face[i] in right_corners) and (face[(i+1)%4] in right_corners):
                    face_colour = (0, 0, 255)
                cv2.line(image, (corners[face[i], 0], corners[face[i], 1]),
                        (corners[face[(i+1)%4], 0], corners[face[(i+1)%4], 1]),
                        face_colour, thickness, lineType=cv2.LINE_AA)
            if face_index == 0:
                cv2.line(image, (corners[face[0], 0], corners[face[0], 1]),
                        (corners[face[2], 0], corners[face[2], 1]),
                        colour, 1, lineType=cv2.LINE_AA)
                cv2.line(image, (corners[face[1], 0], corners[face[1], 1]),
                        (corners[face[3], 0], corners[face[3], 1]),
                        colour, 1, lineType=cv2.LINE_AA)

def draw_dd_box(dd_bb, images, colour=(255,0,0)):
    for img in range(len(images)):
        image = images[img]
        box = dd_bb[img]
        cv2.rectangle(
            image, (box[0], box[1]), 
            (box[0] + box[2], box[1] + box[3]),
            colour, 2
        )

def draw_birds_eye(obj, birds_eye, colour=(0, 255, 0)):
    overlay = birds_eye['image'].copy()
    ground_point = obj['location'][:2]
    ddd_bb_bev = (obj['ddd_bb_world'][:4,:2] + birds_eye['offset']) * birds_eye['scale']
    arrow = np.array([
        ddd_bb_bev[0] + (ddd_bb_bev[3] - ddd_bb_bev[0]) * 0.33, 
        ddd_bb_bev[0] + (ddd_bb_bev[1] - ddd_bb_bev[0]) * 0.5, 
        ddd_bb_bev[1] + (ddd_bb_bev[2] - ddd_bb_bev[1]) * 0.33, 
    ])
    ddd_bb_bev = ddd_bb_bev.reshape((-1, 1, 2)).astype(int)
    arrow = arrow.reshape((-1, 1, 2)).astype(int)
    bev_point = tuple(((ground_point + birds_eye['offset']) * birds_eye['scale']).astype(int))
    cv2.circle(overlay, bev_point, 2, colour, -1)
    cv2.fillPoly(
        overlay, [ddd_bb_bev], colour, 1
    )

    alpha = 0.6
    cv2.addWeighted(overlay, alpha, birds_eye['image'], 1-alpha, 0, birds_eye['image'])
    cv2.polylines(
        birds_eye['image'], [ddd_bb_bev,arrow], True, colour, 1
    )

def attribute_lists_to_objects(detections, detection_thresh=0.4):
    objects = {}
    max_objects = detections['scores'].size
    for obj in range(max_objects):
        if detections['scores'][obj] < detection_thresh:
            continue
        attributes = {}
        attributes['location'] = detections['location_wcf'][obj]
        size = detections['size'][obj]
        size[0], size[2] = size[2], size[0]
        attributes['location'][2] = attributes['location'][2] - size[2]/2
        attributes['size'] = size
        attributes['rot'] = detections['rot'][obj]

        objects[obj] = attributes

    return objects