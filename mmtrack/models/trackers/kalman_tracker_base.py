# Copyright (c) OpenMMLab. All rights reserved.
"""Shared Kalman-filter track-management base for the trackers in this repo.

Provides the bookkeeping that every Kalman-based tracker here needs -- track
initialisation/update of the ``(x, y, a, h)`` Kalman state, tentative-track
confirmation and stale-track eviction -- while leaving the association logic
(the ``track()`` method) entirely to the subclass.
"""
from typing import List, Tuple

from torch import Tensor

from mmtrack.registry import MODELS
from mmtrack.structures.bbox import bbox_xyxy_to_cxcyah
from .base_tracker import BaseTracker


@MODELS.register_module()
class KalmanTrackerBase(BaseTracker):
    """Kalman-state track management shared by the trackers in this repo.

    Args:
        obj_score_thr (float, optional): Threshold to filter the objects.
            Defaults to 0.3.
        reid (dict, optional): Kept for config compatibility; unused by the
            trackers in this repo, which are appearance-free.
        match_iou_thr (float, optional): Threshold of the IoU matching process.
            Defaults to 0.7.
        num_tentatives (int, optional): Number of continuous frames to confirm
            a track. Defaults to 3.
    """

    def __init__(self,
                 obj_score_thr: float = 0.3,
                 reid: dict = dict(
                     num_samples=10,
                     img_scale=(256, 128),
                     img_norm_cfg=None,
                     match_score_thr=2.0),
                 match_iou_thr: float = 0.7,
                 num_tentatives: int = 3,
                 **kwargs):
        super().__init__(**kwargs)
        self.obj_score_thr = obj_score_thr
        self.reid = reid
        self.match_iou_thr = match_iou_thr
        self.num_tentatives = num_tentatives

    @property
    def confirmed_ids(self) -> List:
        """Confirmed ids in the tracker."""
        ids = [id for id, track in self.tracks.items() if not track.tentative]
        return ids

    def init_track(self, id: int, obj: Tuple[Tensor]) -> None:
        """Initialize a track."""
        super().init_track(id, obj)
        self.tracks[id].tentative = True
        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.initiate(
            bbox)

    def update_track(self, id: int, obj: Tuple[Tensor]) -> None:
        """Update a track."""
        super().update_track(id, obj)
        if self.tracks[id].tentative:
            if len(self.tracks[id]['bboxes']) >= self.num_tentatives:
                self.tracks[id].tentative = False
        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.update(
            self.tracks[id].mean, self.tracks[id].covariance, bbox)

    def pop_invalid_tracks(self, frame_id: int) -> None:
        """Pop out invalid tracks."""
        invalid_ids = []
        for k, v in self.tracks.items():
            # case1: disappeared frames >= self.num_frames_retrain
            case1 = frame_id - v['frame_ids'][-1] >= self.num_frames_retain
            # case2: tentative tracks but not matched in this frame
            case2 = v.tentative and v['frame_ids'][-1] != frame_id
            if case1 or case2:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

