# Copyright (c) OpenMMLab. All rights reserved.
"""Mesh-Affine CMAC camera-motion compensation.

Estimates the frame-to-frame background motion as a 4-DoF similarity transform
(translation + rotation + uniform scale) from the dense-flow mesh described in
:func:`.utils.GLME_affine`, and applies it to the persistent Kalman state of
every live track right after the prediction step.
"""
import numpy as np

from .utils import GLME_affine


def glme_affine_warp(curr_img, prev_img, metainfo: dict,
                     glme_kwargs: dict = None):
    """Mesh-Affine CMAC warp (see :func:`.utils.GLME_affine`).

    Returns:
        ndarray | None: A 2x3 affine (prev->curr, original image scale) to be
        applied to the Kalman states, or ``None`` on the first frame / when
        the mesh-flow fit is unreliable.
    """
    if prev_img is None:
        return None
    warp, _ = GLME_affine(
        curr_img=curr_img.detach().cpu().numpy(),
        prev_img=prev_img.detach().cpu().numpy(),
        metainfo=metainfo,
        **(glme_kwargs or {}))
    return warp


def apply_gmc_to_tracks_cxcyah(tracks: dict, ids: list,
                               warp_matrix: np.ndarray) -> None:
    """Apply a GMC affine transform to cxcyah-state Kalman tracks (OC-SORT).

    OC-SORT's :class:`.motion.kalman_filter.KalmanFilter` uses the vanilla
    DeepSORT/SORT state ``(x, y, a, h, vx, vy, va, vh)`` -- center position,
    aspect ratio, height, and their velocities -- unlike BoT-SORT's
    ``(x, y, w, h, ...)`` state handled by :func:`apply_gmc_to_tracks`. Both
    :class:`GMC` and :func:`.utils.GLME_affine` are restricted to a 4-DoF
    similarity transform (uniform scale ``s`` + rotation + translation, via
    ``cv2.estimateAffinePartial2D``), so the aspect ratio ``a`` is invariant
    under the warp and only the height ``h`` (and ``vh``) need to be scaled by
    ``s``; the center and its velocity transform as ordinary 2-D vectors (the
    velocity only through the linear part, no translation).

    Args:
        tracks (dict): The tracker's track buffer (``id -> track``).
        ids (list[int]): Track ids whose Kalman states should be compensated.
        warp_matrix (ndarray): The ``2x3`` affine matrix from :meth:`GMC.apply`
            or :func:`.gmc.glme_affine_warp`.
    """
    R = warp_matrix[:2, :2].astype(np.float64)
    t = warp_matrix[:2, 2].astype(np.float64)
    s = float(np.sqrt(max(np.linalg.det(R), 1e-12)))
    for track_id in ids:
        track = tracks[track_id]
        if not hasattr(track, 'mean') or track.mean is None:
            continue
        mean = track.mean.copy()
        mean[0:2] = R.dot(mean[0:2]) + t
        mean[3] *= s
        mean[4:6] = R.dot(mean[4:6])
        mean[7] *= s
        track.mean = mean

        cov = track.covariance
        R8x8 = np.eye(8, dtype=float)
        R8x8[0:2, 0:2] = R
        R8x8[4:6, 4:6] = R
        # scalar rows/cols (a, va untouched; h, vh scaled by s) transform via
        # a diagonal Jacobian entry -- fold that into the same linear map.
        R8x8[3, 3] = s
        R8x8[7, 7] = s
        track.covariance = R8x8.dot(cov).dot(R8x8.T)
