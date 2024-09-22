import torch
import numpy as np


def truncated_mean_decorator(func):
    def wrapper(*args, **kwargs):
        depth = args[1]
        bboxes = args[2]
        depth = depth.cpu().numpy().squeeze()

        values = []
        scales = []

        for box in bboxes:
            box = box.cpu().numpy().astype(np.int)
            depth_box = depth[box[1]: box[3], box[0]: box[2]]
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            w = box[2] - box[0]
            h = box[3] - box[1]

            d_v = depth_box[(depth_box < 150) & (depth_box>0)]  # `150` denotes the maximum depth value
            len_d = len(d_v)
            if len_d < 1 or w > 800:
                values.append(-1)
                scales.append(1.)
                continue
            d_sorted = np.sort(d_v, axis=None)

            # truncated mean
            truncate_factor = 0.1
            truncate_start = int(truncate_factor * len_d)
            truncate_end = int((1 - truncate_factor) * len_d)
            d_seg = d_sorted[truncate_start: truncate_end]
            if len(d_seg) == 0:
                d_seg = d_sorted[:-1]
            d = np.mean(d_seg)

            values.append(d)

            scale = min(d * d / 400, 3.)  # scale mustn't larger than 3.
            scale = max(scale, 1.)  # scale must larger than 1.
            scales.append(scale)
        
        return values, scales
    return wrapper


def mean_decorator(func):
    def wrapper(*args, **kwargs):
        depth = args[1]
        bboxes = args[2]
        depth = depth.cpu().numpy().squeeze()

        values = []
        scales = []

        for box in bboxes:
            box = box.cpu().numpy().astype(np.int)
            depth_box = depth[box[1]: box[3], box[0]: box[2]]
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            w = box[2] - box[0]
            h = box[3] - box[1]

            d_v = depth_box[(depth_box < 150) & (depth_box>0)]  # `150` denotes the maximum depth value
            len_d = len(d_v)
            if len_d < 1 or w > 800:
                values.append(-1)
                scales.append(1.)
                continue
            d = np.mean(d_v)

            values.append(d)

            scale = min(d * d / 400, 3.)  # scale mustn't larger than 3.
            scale = max(scale, 1.)  # scale must larger than 1.
            scales.append(scale)
        
        return values, scales
    return wrapper


def median_decorator(func):
    def wrapper(*args, **kwargs):
        depth = args[1]
        bboxes = args[2]
        depth = depth.cpu().numpy().squeeze()

        values = []
        scales = []

        for box in bboxes:
            box = box.cpu().numpy().astype(np.int)
            depth_box = depth[box[1]: box[3], box[0]: box[2]]
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            w = box[2] - box[0]
            h = box[3] - box[1]

            d_v = depth_box[(depth_box < 150) & (depth_box>0)]  # `150` denotes the maximum depth value
            len_d = len(d_v)
            if len_d < 1 or w > 800:
                values.append(-1)
                scales.append(1.)
                continue
            
            d = np.median(d_v)

            values.append(d)

            scale = min(d * d / 400, 3.)  # scale mustn't larger than 3.
            scale = max(scale, 1.)  # scale must larger than 1.
            scales.append(scale)
        
        return values, scales
    return wrapper

def center_decorator(func):
    def wrapper(*args, **kwargs):
        depth = args[1]
        bboxes = args[2]
        depth = depth.cpu().numpy().squeeze()

        values = []
        scales = []

        for box in bboxes:
            box = box.cpu().numpy().astype(np.int)
            depth_box = depth[box[1]: box[3], box[0]: box[2]]
            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2
            w = box[2] - box[0]
            h = box[3] - box[1]

            d_v = depth_box[(depth_box < 150) & (depth_box>0)]  # `150` denotes the maximum depth value
            len_d = len(d_v)
            if len_d < 1 or w > 800:
                values.append(-1)
                scales.append(1.)
                continue
            # get the center value
            d = depth[cy, cx]

            values.append(d)

            scale = min(d * d / 400, 3.)  # scale mustn't larger than 3.
            scale = max(scale, 1.)  # scale must larger than 1.
            scales.append(scale)
        
        return values, scales
    return wrapper