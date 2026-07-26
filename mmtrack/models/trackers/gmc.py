# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from .utils import GLME_affine


def glme_affine_warp(curr_img, prev_img, metainfo: dict,
                     glme_kwargs: dict = None):
    """Mesh-Affine CMAC warp."""
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
    """Apply a GMC affine transform to cxcyah-state Kalman tracks (OC-SORT)."""
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
