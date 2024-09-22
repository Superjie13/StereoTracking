import cv2
import numpy as np
import torch


def GLME(curr_img, prev_img, metainfo, step=16):
	"""Calculate global-to-local motion estimation (GLME) between two images."""
	curr_img = np.transpose(curr_img, (2, 3, 1, 0)).squeeze(-1).astype(np.uint8)
	prev_img = np.transpose(prev_img, (2, 3, 1, 0)).squeeze(-1).astype(np.uint8)

	ori_h, ori_w = metainfo['img_shape']
	curr_img = curr_img[: ori_h, : ori_w, :3][:, :, ::-1]
	prev_img = prev_img[: ori_h, : ori_w, :3][:, :, ::-1]

	scale = (255, 255)  # w, h
	scale_fy = ori_h / scale[1]
	scale_fx = ori_w / scale[0]

	curr_img = cv2.resize(curr_img, scale)
	prev_img = cv2.resize(prev_img, scale)

	curr_img_gray = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)
	prev_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY)

	curr_img_gray = cv2.equalizeHist(curr_img_gray)
	prev_img_gray = cv2.equalizeHist(prev_img_gray)

	flow = cv2.calcOpticalFlowFarneback(prev_img_gray, curr_img_gray, None,
										pyr_scale=0.5,
										levels=5,
										winsize=128,
										iterations=3,
										poly_n=5,
										poly_sigma=1.2,
										flags=0)
	flow = flow * np.array([scale_fx, scale_fy])

	h, w = curr_img_gray.shape[: 2]
	y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
	fx, fy = flow[y, x].T

	speed = np.vstack((fx, fy)).T
	norm = np.linalg.norm(speed, axis=-1)[:, None]
	angle_cos = speed @ speed.T / (norm @ norm.T)
	angle_thr = np.cos(25 / 180 * np.pi)
	mask = angle_cos > angle_thr
	iou = np.sum(mask) / (mask.shape[0] * mask.shape[1])

	mask = np.sum(mask, axis=-1) > len(y) * 0.7

	if sum(mask) > 0:
		avg_speed = np.sum((speed * mask[:, None]), axis=0) / sum(mask)
	else:
		avg_speed = np.array([0.0, 0.0])

	return iou, avg_speed



def scale_bbox(bboxes, scales):
	cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
	cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
	w = bboxes[:, 2] - bboxes[:, 0]
	h = bboxes[:, 3] - bboxes[:, 1]

	w = w * scales
	h = h * scales

	x1 = cx - w/2
	x2 = cx + w/2
	y1 = cy - h/2
	y2 = cy + h/2

	new_bboxes = torch.cat((x1[:, None], y1[:, None], x2[:, None], y2[:, None]), dim=-1).reshape(-1, 4)
	return new_bboxes