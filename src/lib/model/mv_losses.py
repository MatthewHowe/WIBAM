# File to contain functions for calculating multi view losses
import copy
import random

import cv2
from lib.utils.mv_utils import draw_detections
import torch
import torch.nn as nn
from utils.ddd_utils import ddd2locrot, draw_box_3d, project_3d_bbox
from utils.mv_utils import *
from utils.post_process import generic_post_process
from utils.utils import Profiler

from .decode import generic_decode

colours = [
    "#7ED379",
    "#FFFF00",
    "#1CE6FF",
    "#FF34FF",
    "#FF4A46",
    "#008941",
    "#006FA6",
    "#A30059",
    "#FFDBE5",
    "#7A4900",
    "#0000A6",
    "#63FFAC",
    "#B79762",
    "#004D43",
    "#8FB0FF",
    "#997D87",
    "#5A0007",
    "#809693",
    "#FEFFE6",
    "#1B4400",
    "#4FC601",
    "#3B5DFF",
    "#4A3B53",
    "#FF2F80",
    "#61615A",
    "#BA0900",
    "#6B7900",
    "#00C2A0",
    "#FFAA92",
    "#FF90C9",
    "#B903AA",
    "#D16100",
    "#DDEFFF",
    "#000035",
    "#7B4F4B",
    "#A1C299",
    "#300018",
    "#0AA6D8",
    "#013349",
    "#00846F",
    "#372101",
    "#FFB500",
    "#C2FFED",
    "#A079BF",
    "#CC0744",
    "#C0B9B2",
    "#C2FF99",
    "#001E09",
    "#00489C",
    "#6F0062",
    "#0CBD66",
    "#EEC3FF",
    "#456D75",
    "#B77B68",
    "#7A87A1",
    "#788D66",
    "#885578",
    "#FAD09F",
    "#FF8A9A",
    "#D157A0",
    "#BEC459",
    "#456648",
    "#0086ED",
    "#886F4C",
    "#34362D",
    "#B4A8BD",
    "#00A6AA",
    "#452C2C",
    "#636375",
    "#A3C8C9",
    "#FF913F",
    "#938A81",
    "#575329",
    "#00FECF",
    "#B05B6F",
    "#8CD0FF",
    "#3B9700",
    "#04F757",
    "#C8A1A1",
    "#1E6E00",
    "#7900D7",
    "#A77500",
    "#6367A9",
    "#A05837",
    "#6B002C",
    "#772600",
    "#D790FF",
    "#9B9700",
    "#549E79",
    "#FFF69F",
    "#201625",
    "#72418F",
    "#BC23FF",
    "#99ADC0",
    "#3A2465",
    "#922329",
    "#5B4534",
    "#FDE8DC",
    "#404E55",
    "#0089A3",
    "#CB7E98",
    "#A4E804",
    "#324E72",
    "#6A3A4C",
    "#83AB58",
    "#001C1E",
    "#D1F7CE",
    "#004B28",
    "#C8D0F6",
    "#A3A489",
    "#806C66",
    "#222800",
    "#BF5650",
    "#E83000",
    "#66796D",
    "#DA007C",
    "#FF1A59",
    "#8ADBB4",
    "#1E0200",
    "#5B4E51",
    "#C895C5",
    "#320033",
    "#FF6832",
    "#66E1D3",
    "#CFCDAC",
    "#D0AC94",
    "#7ED379",
    "#012C58",
]


def get_colours():
    rgb_values = []
    rgb_01 = []
    hex_values = []
    for colour in colours:
        hex_values.append(colour.lstrip("#"))
        rgb_values.append(
            tuple(int(colour.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
        )
        # rgb_01.append(RGB % 256)

    return rgb_values, rgb_01, hex_values


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
        hex_values.append("#" + r_hex + g_hex + b_hex)
        rgb_values.append((r, g, b))
        rgb_01.append((r / 256, g / 256, b / 256))

    return rgb_values, rgb_01, hex_values


def generalized_iou_loss(gt_bboxes, pr_bboxes, reduction="mean", loss_function="giou"):
    r"""
    GIoU function, takes in ground truth bounding boxes and predicted
    bounding boxes. Uses BB[min_x,min_y,max_x,max_y]
    Arguments:
      gt_bboxes (np.array, (#_detections,4)): Bounding boxes for ground truth data
      pr_bboxes (np.array, (#_detections,4)): Bounding boxes for prediction
    Returns:
      loss (float): Return loss
    """
    size = [1920, 1080]

    # Convert bbox format
    gt_bboxes[:, 2] = gt_bboxes[:, 0] + gt_bboxes[:, 2]
    gt_bboxes[:, 3] = gt_bboxes[:, 1] + gt_bboxes[:, 3]

    pr_bboxes[:, 2] = pr_bboxes[:, 0] + pr_bboxes[:, 2]
    pr_bboxes[:, 3] = pr_bboxes[:, 1] + pr_bboxes[:, 3]
    # C
    x1 = torch.clamp(pr_bboxes[:, 0], 0, size[0])
    x2 = torch.clamp(pr_bboxes[:, 2], 0, size[0])

    y1 = torch.clamp(pr_bboxes[:, 1], 0, size[1])
    y2 = torch.clamp(pr_bboxes[:, 3], 0, size[1])

    pr_bboxes = torch.stack((x1, y1, x2, y2), 1)

    gt_area = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
    pr_area = (pr_bboxes[:, 2] - pr_bboxes[:, 0]) * (pr_bboxes[:, 3] - pr_bboxes[:, 1])

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

    # distance
    gt_centers = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
    pr_centers = (pr_bboxes[:, :2] + pr_bboxes[:, 2:]) / 2
    center_dists = torch.diagonal(torch.cdist(gt_centers, pr_centers), 0)
    enclosure_diag_lengths = torch.diagonal(torch.cdist(top_left, bottom_right), 0)

    diou = iou + torch.pow(center_dists, 2) / torch.pow(enclosure_diag_lengths, 2)
    giou = iou - (enclosure - union) / enclosure

    if loss_function == "giou":
        loss = 1.0 - giou
    elif loss_function == "diou":
        loss = 1.0 - diou

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    elif reduction == "none":
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
        self.profiler = Profiler()
        super(ReprojectionLoss, self).__init__()
        self.opt = opt

    def forward(self, output, batch):
        self.profiler.start()
        detections = {}
        calibrations = {}
        BN = len(batch["cam_num"])
        max_objects = self.opt.K
        num_cams = batch["P"].shape[1]

        decoded_output = decode_output(output, self.opt.K)
        centers = decoded_output["bboxes"].reshape(BN, max_objects, 2, 2).mean(axis=2)
        centers_offset = centers + decoded_output["amodel_offset"]

        centers = translate_centre_points(
            centers, np.array([960, 540]), 1920, (200, 112), BN, max_objects
        )

        centers_offset = translate_centre_points(
            centers_offset, np.array([960, 540]), 1920, (200, 112), BN, max_objects
        )

        # detections['depth'] = decoded_output['dep'] * (1266 * 64.57)/(1024 * 86.30);
        # detections['depth'] = torch.sigmoid((decoded_output['dep'] * 0.80 - 30) / 10) * 60
        detections["depth"] = decoded_output["dep"] * 0.8
        detections["size"] = decoded_output["dim"]
        # Restrict size
        # detections['size'][:, :, 0] = torch.sigmoid(detections['size'][:, :, 0] - 1.5) * 3 # h 0-3m
        # detections['size'][:, :, 1] = torch.sigmoid(detections['size'][:, :, 1] - 2) * 4 # w 0-4m
        # detections['size'][:, :, 2] = torch.sigmoid(detections['size'][:, :, 2] - 4.5) * 9 # l 0-9m

        detections["rot"] = decoded_output["rot"]
        detections["center"] = centers_offset

        self.profiler.interval_trigger("Pre-processing")

        det_cam_to_det_3D_ccf(detections, batch)
        self.profiler.interval_trigger("det_cam_to_det_3D_ccf")

        dets_3D_ccf_to_dets_3D_wcf(detections, batch)
        self.profiler.interval_trigger("dets_3D_ccf_to_dets_3D_wcf")

        dets_3D_wcf_to_dets_2D(detections, batch)
        self.profiler.interval_trigger("dets_3D_wcf_to_dets_2D")

        gt_centers = translate_centre_points(
            batch["ctr"].type(torch.float),
            np.array([960, 540]),
            1920,
            (200, 112),
            BN,
            max_objects,
        )

        self.profiler.interval_trigger("translate_3")

        cost_matrix, gt_indexes = match_predictions_ground_truth(
            centers, gt_centers, batch["mask"], batch["cam_num"]
        )
        self.profiler.interval_trigger("matching")

        gt_dict = {}
        pr_dict = {}

        # if self.opt.show_repro:
        #   counts = [0,0,0,0]
        #   drawing_images = batch['drawing_images'].detach().cpu().numpy()
        #   det_cam = int(batch['cam_num'][0].item())
        #   id = str(batch['image_info'][0]['im_num'].detach().cpu().numpy()[0])
        #   print(id)
        #   while True:
        #     for i in range(len(drawing_images[0])):
        #       cv2.namedWindow(f"{i}", cv2.WINDOW_NORMAL)
        #       cv2.imshow(f"{i}", drawing_images[0][i])
        #       print(counts)

        #     if cv2.waitKey(0) & 0xFF == ord('q'):
        #       cam = 0
        #       if cam == det_cam:
        #         continue
        #       counts[cam] += 1
        #       im_num = str(batch['image_info'][cam]['im_num'].detach().cpu().numpy()[0] + counts[cam])
        #       drawing_images[0][cam] = cv2.imread(f"data/wibam/frames/{cam}/{im_num}.jpg")
        #     if cv2.waitKey(0) & 0xFF == ord('a'):
        #       cam = 0
        #       if cam == det_cam:
        #         continue
        #       counts[cam] -= 1
        #       im_num = str(batch['image_info'][cam]['im_num'].detach().cpu().numpy()[0] + counts[cam])
        #       drawing_images[0][cam] = cv2.imread(f"data/wibam/frames/{cam}/{im_num}.jpg")

        #     if cv2.waitKey(0) & 0xFF == ord('w'):
        #       cam = 1
        #       if cam == det_cam:
        #         continue
        #       counts[cam] += 1
        #       im_num = str(batch['image_info'][cam]['im_num'].detach().cpu().numpy()[0] + counts[cam])
        #       drawing_images[0][cam] = cv2.imread(f"data/wibam/frames/{cam}/{im_num}.jpg")
        #     if cv2.waitKey(0) & 0xFF == ord('s'):
        #       cam = 1
        #       if cam == det_cam:
        #         continue
        #       counts[cam] -= 1
        #       im_num = str(batch['image_info'][cam]['im_num'].detach().cpu().numpy()[0] + counts[cam])
        #       drawing_images[0][cam] = cv2.imread(f"data/wibam/frames/{cam}/{im_num}.jpg")

        #     if cv2.waitKey(0) & 0xFF == ord('e'):
        #       cam = 2
        #       if cam == det_cam:
        #         continue
        #       counts[cam] += 1
        #       im_num = str(batch['image_info'][cam]['im_num'].detach().cpu().numpy()[0] + counts[cam])
        #       drawing_images[0][cam] = cv2.imread(f"data/wibam/frames/{cam}/{im_num}.jpg")
        #     if cv2.waitKey(0) & 0xFF == ord('d'):
        #       cam = 2
        #       if cam == det_cam:
        #         continue
        #       counts[cam] -= 1
        #       im_num = str(batch['image_info'][cam]['im_num'].detach().cpu().numpy()[0] + counts[cam])
        #       drawing_images[0][cam] = cv2.imread(f"data/wibam/frames/{cam}/{im_num}.jpg")

        #     if cv2.waitKey(0) & 0xFF == ord('r'):
        #       cam = 3
        #       if cam == det_cam:
        #         continue
        #       counts[cam] += 1
        #       im_num = str(batch['image_info'][cam]['im_num'].detach().cpu().numpy()[0] + counts[cam])
        #       drawing_images[0][cam] = cv2.imread(f"data/wibam/frames/{cam}/{im_num}.jpg")
        #     if cv2.waitKey(0) & 0xFF == ord('f'):
        #       cam = 3
        #       if cam == det_cam:
        #         continue
        #       counts[cam] -= 1
        #       im_num = str(batch['image_info'][cam]['im_num'].detach().cpu().numpy()[0] + counts[cam])
        #       drawing_images[0][cam] = cv2.imread(f"data/wibam/frames/{cam}/{im_num}.jpg")

        #     if cv2.waitKey(0) & 0xFF == ord('p'):
        #       print(counts)
        #       print(id)
        #       break

        for B in range(BN):
            det_cam = int(batch["cam_num"][B].item())
            colours, _, _ = get_colours()
            for pr_index in range(max_objects):
                gt_index = gt_indexes[B, pr_index]

                if (
                    cost_matrix[B, pr_index, gt_index] < 50
                    and batch["mask"][B, det_cam, gt_index].item() is True
                ):

                    obj_id = batch["obj_id"][B, det_cam, gt_index]

                    gt_box_T = batch["bboxes"][B, det_cam, gt_index]
                    pr_box_T = detections["2D_bounding_boxes"][B, det_cam, pr_index]
                    if "det" not in gt_dict:
                        gt_dict["det"] = [gt_box_T]
                        pr_dict["det"] = [pr_box_T]
                    else:
                        gt_dict["det"].append(gt_box_T)
                        pr_dict["det"].append(pr_box_T)

                    # Drawing functions
                    if self.opt.show_repro:
                        img = drawing_images[B, det_cam]
                        cv2.putText(
                            img,
                            "detection_cam",
                            (0, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 0, 255),
                            2,
                        )
                        ddd_box = (
                            detections["proj_3D_boxes"][B, det_cam, pr_index]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(int)
                        )
                        gt_box = (
                            torch.clone(gt_box_T).detach().cpu().numpy().astype(int)
                        )
                        pr_box = (
                            torch.clone(pr_box_T).detach().cpu().numpy().astype(int)
                        )
                        draw_box_3d(img, ddd_box, colours[pr_index], same_color=True)
                        # cv2.rectangle(img, (gt_box[0],gt_box[1]), (gt_box[0]+gt_box[2],gt_box[1]+gt_box[3]), colours[pr_index], 2)
                        # cv2.rectangle(img, (pr_box[0],pr_box[1]), (pr_box[0]+pr_box[2],pr_box[1]+pr_box[3]), colours[pr_index], 2)

                    if obj_id != -1:

                        for cam in range(num_cams):
                            if cam == det_cam:
                                continue

                            try:
                                obj_id_list = batch["obj_id"][B][cam].tolist()
                                gt_ind = obj_id_list.index(obj_id)
                            except:
                                continue

                            gt_box_T = batch["bboxes"][B, cam, gt_ind]
                            pr_box_T = detections["2D_bounding_boxes"][B, cam, pr_index]
                            if cam not in gt_dict:
                                gt_dict[cam] = [gt_box_T]
                                pr_dict[cam] = [pr_box_T]
                            else:
                                gt_dict[cam].append(gt_box_T)
                                pr_dict[cam].append(pr_box_T)

                            # Drawing functions
                            if self.opt.show_repro:
                                img = drawing_images[B, cam]
                                ddd_box = (
                                    detections["proj_3D_boxes"][B, cam, pr_index]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .astype(int)
                                )
                                gt_box = (
                                    torch.clone(gt_box_T)
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .astype(int)
                                )
                                pr_box = (
                                    torch.clone(pr_box_T)
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .astype(int)
                                )
                                draw_box_3d(
                                    img, ddd_box, colours[pr_index], same_color=True
                                )
                                # cv2.rectangle(img, (gt_box[0],gt_box[1]), (gt_box[0]+gt_box[2],gt_box[1]+gt_box[3]), colours[pr_index], 2)
                                # cv2.rectangle(img, (pr_box[0],pr_box[1]), (pr_box[0]+pr_box[2],pr_box[1]+pr_box[3]), colours[pr_index], 2)

        self.profiler.interval_trigger("Constructing boxes")

        mv_loss = {}
        for key, val in gt_dict.items():
            gt_boxes = torch.stack(val, 0)
            pr_boxes = torch.stack(pr_dict[key], 0)
            loss = generalized_iou_loss(
                gt_boxes, pr_boxes, "mean", self.opt.reprojection_loss_function
            )
            mv_loss[key] = loss
            if key == "det" and self.opt.no_det:
                continue
            elif key == "det" and self.opt.det_only:
                self.add_to_total_loss(mv_loss, loss)
                break
            elif not self.opt.det_only:
                self.add_to_total_loss(mv_loss, loss)

        self.profiler.interval_trigger("Calculating loss")

        # Make sure that number of detections is equal to number of gt detections
        if "det" in pr_dict:
            mv_loss["mult"] = (
                pow((torch.sum(batch["mask_det"]) - len(pr_dict["det"])), 2) + 1.0
            )
        else:
            mv_loss["mult"] = pow((torch.sum(batch["mask_det"]) - 0), 2) + 1.0
        # mv_loss['tot_GIoU'] = mv_loss['tot']
        # self.add_to_total_loss(mv_loss, mv_loss['mult'])

        self.profiler.interval_trigger("Multipling loss")

        if self.opt.show_repro:
            for B in range(BN):
                composite = return_four_frames(drawing_images[B])
                cv2.namedWindow("Batch {}".format(B), cv2.WINDOW_NORMAL)
                cv2.imshow("Batch {}".format(B), composite)
                cv2.waitKey(0)

        self.profiler.pause()
        # self.profiler.print_interval_times()
        return mv_loss

    def add_to_total_loss(self, losses, loss):
        if "tot" in losses:
            losses["tot"] += loss
        else:
            losses["tot"] = loss
