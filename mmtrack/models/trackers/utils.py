import cv2
import numpy as np
import torch


def GLME_affine(curr_img, prev_img, metainfo, step=16, winsize=31,
				ransac_thr=5.0, min_inlier_ratio=0.3):
	"""Mesh-Affine CMAC: dense flow -> per-cell median mesh flows -> robust
	4-DoF affine fit.

	This keeps CMAC's identity (dense Farneback flow partitioned into an
	``m x n`` grid with a per-cell **median** mesh flow rejecting foreground)
	but replaces the translation-only averaging back-end with a RANSAC
	``estimateAffinePartial2D`` fit over the mesh-flow correspondences, so
	camera **rotation and scale** -- which the direction-consistency filter of
	plain GLME is structurally blind to -- are recovered as well. The returned
	affine is meant to be applied to the Kalman states exactly like the sparse
	CMC warp.

	Args:
		step (int): Mesh cell size in the resized (255x255) frame. Defaults
			to 16 (a ~15x15 mesh, as in plain GLME).
		winsize (int): Farneback window. Smaller than plain GLME's 128 so the
			flow field keeps the *spatial variation between cells* that
			encodes rotation/scale; ~2x the cell size a priori. Defaults 31.
		ransac_thr (float): RANSAC reprojection threshold in original-image
			pixels; 5.0 corresponds to ~1px flow noise under the ~5x upscale
			from the 255px estimation frame. Not tuned.
		min_inlier_ratio (float): Reliability gate (the paper's tau2 concept):
			the fit is trusted only if this fraction of mesh cells are RANSAC
			inliers. Defaults to 0.3.

	Returns:
		tuple: ``(H, inlier_ratio)`` where ``H`` is a 2x3 affine mapping
		previous-frame to current-frame coordinates in the original image
		scale, or ``None`` when the estimate is unreliable.
	"""
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

	curr_img_gray = cv2.equalizeHist(cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY))
	prev_img_gray = cv2.equalizeHist(cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY))

	flow = cv2.calcOpticalFlowFarneback(prev_img_gray, curr_img_gray, None,
										pyr_scale=0.5,
										levels=5,
										winsize=winsize,
										iterations=3,
										poly_n=5,
										poly_sigma=1.2,
										flags=0)
	flow = flow * np.array([scale_fx, scale_fy])

	h, w = curr_img_gray.shape[: 2]
	gh, gw = h // step, w // step
	cells = flow[:gh * step, :gw * step].reshape(gh, step, gw, step, 2)
	mesh = np.median(cells, axis=(1, 3)).reshape(-1, 2)

	# mesh cell centers in original-image coordinates
	ys, xs = np.mgrid[0:gh, 0:gw]
	cx = (xs + 0.5) * step * scale_fx
	cy = (ys + 0.5) * step * scale_fy
	src = np.stack([cx, cy], axis=-1).reshape(-1, 2).astype(np.float32)
	dst = (src + mesh).astype(np.float32)

	H, inliers = cv2.estimateAffinePartial2D(
		src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thr)
	if H is None or inliers is None:
		return None, 0.0
	ratio = float(inliers.sum()) / len(src)
	if ratio < min_inlier_ratio:
		return None, ratio
	return H.astype(np.float32), ratio


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