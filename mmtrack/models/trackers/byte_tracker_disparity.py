# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import lap
import numpy as np
import torch
from mmdet.structures.bbox import bbox_overlaps
from mmengine.structures import InstanceData

from mmtrack.registry import MODELS
from mmtrack.structures import TrackDataSample
from mmtrack.structures.bbox import bbox_cxcyah_to_xyxy, bbox_xyxy_to_cxcyah
from .byte_tracker import ByteTracker
from .utils import GLME, scale_bbox


@MODELS.register_module()
class ByteTracker_Disparity(ByteTracker):
    """Multi-modal tracker for ByteTrack.
    """

    def __init__(self,
                 obj_score_thrs: dict = dict(high=0.6, low=0.1),
                 init_track_thr: float = 0.7,
                 weight_iou_with_det_scores: bool = True,
                 match_iou_thrs: dict = dict(high=0.1, low=0.5, tentative=0.3),
                 num_tentatives: int = 3,
                 **kwargs):
        super().__init__(**kwargs)

    def last_obs(self, track):
        """extract the last associated observation."""
        for bbox in track.obs[::-1]:
            if bbox is not None:
                return bbox
            
    def track(self, model: torch.nn.Module, img: torch.Tensor,
              feats: List[torch.Tensor], data_sample: TrackDataSample,
              **kwargs) -> InstanceData:
        """Tracking forward function.

        Args:
            model (nn.Module): MOT model.
            img (Tensor): of shape (T, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.
                The T denotes the number of key images and usually is 1 in
                ByteTrack method.
            feats (list[Tensor]): Multi level feature maps of `img`.
            data_sample (:obj:`TrackDataSample`): The data sample.
                It includes information such as `pred_det_instances`.

        Returns:
            :obj:`InstanceData`: Tracking results of the input images.
            Each InstanceData usually contains ``bboxes``, ``labels``,
            ``scores`` and ``instances_id``.
        """
        metainfo = data_sample.metainfo
        bboxes = data_sample.pred_det_instances.bboxes
        labels = data_sample.pred_det_instances.labels
        scores = data_sample.pred_det_instances.scores
        scales = data_sample.pred_det_instances.scales
        depth = data_sample.pred_det_instances.depth

        frame_id = metainfo.get('frame_id', -1)
        self.img = img
        
        if frame_id == 0:
            self.reset()
        if not hasattr(self, 'kf'):
            self.kf = model.motion

        if self.empty or bboxes.size(0) == 0:
            valid_inds = scores > self.init_track_thr
            scores = scores[valid_inds]
            bboxes = bboxes[valid_inds]
            labels = labels[valid_inds]
            scales = scales[valid_inds]
            depth = depth[valid_inds]
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(self.num_tracks,
                               self.num_tracks + num_new_tracks).to(labels)
            self.num_tracks += num_new_tracks
            self.last_img = img

        else:
            # 0. init
            ids = torch.full((bboxes.size(0), ),
                             -1,
                             dtype=labels.dtype,
                             device=labels.device)

            # get the detection bboxes for the first association
            first_det_inds = scores > self.obj_score_thrs['high']
            first_det_bboxes = bboxes[first_det_inds]
            first_det_labels = labels[first_det_inds]
            first_det_scores = scores[first_det_inds]
            first_det_scales = scales[first_det_inds]
            first_det_depth = depth[first_det_inds]
            first_det_ids = ids[first_det_inds]

            # get the detection bboxes for the second association
            second_det_inds = (~first_det_inds) & (
                scores > self.obj_score_thrs['low'])
            second_det_bboxes = bboxes[second_det_inds]
            second_det_labels = labels[second_det_inds]
            second_det_scores = scores[second_det_inds]
            second_det_scales = scales[second_det_inds]
            second_det_depth = depth[second_det_inds]
            second_det_ids = ids[second_det_inds]

            # 1. use Kalman Filter to predict current location
            for id in self.confirmed_ids:
                # track is lost in previous frame
                if self.tracks[id].frame_ids[-1] != frame_id - 1:
                    self.tracks[id].mean[7] = 0
                (self.tracks[id].mean,
                 self.tracks[id].covariance) = self.kf.predict(
                     self.tracks[id].mean, self.tracks[id].covariance)

            # 2. first match
            first_match_track_inds, first_match_det_inds = self.assign_ids_mc(
                self.confirmed_ids, first_det_bboxes, first_det_labels,
                first_det_scores, first_det_scales, self.weight_iou_with_det_scores,
                self.match_iou_thrs['high'])
            # '-1' mean a detection box is not matched with tracklets in
            # previous frame
            valid = first_match_det_inds > -1
            first_det_ids[valid] = torch.tensor(
                self.confirmed_ids)[first_match_det_inds[valid]].to(labels)

            first_match_det_bboxes = first_det_bboxes[valid]
            first_match_det_labels = first_det_labels[valid]
            first_match_det_scores = first_det_scores[valid]
            first_match_det_scales = first_det_scales[valid]
            first_match_det_depth = first_det_depth[valid]
            first_match_det_ids = first_det_ids[valid]
            assert (first_match_det_ids > -1).all()

            first_unmatch_det_bboxes = first_det_bboxes[~valid]
            first_unmatch_det_labels = first_det_labels[~valid]
            first_unmatch_det_scores = first_det_scores[~valid]
            first_unmatch_det_scales = first_det_scales[~valid]
            first_unmatch_det_depth = first_det_depth[~valid]
            first_unmatch_det_ids = first_det_ids[~valid]
            assert (first_unmatch_det_ids == -1).all()

            # 3. use unmatched detection bboxes from the first match to match
            # the unconfirmed tracks
            (tentative_match_track_inds,
             tentative_match_det_inds) = self.assign_ids_mc(
                 self.unconfirmed_ids, first_unmatch_det_bboxes,
                 first_unmatch_det_labels, first_unmatch_det_scores, first_unmatch_det_scales,
                 self.weight_iou_with_det_scores,
                 self.match_iou_thrs['tentative'])
            valid = tentative_match_det_inds > -1
            first_unmatch_det_ids[valid] = torch.tensor(self.unconfirmed_ids)[
                tentative_match_det_inds[valid]].to(labels)

            cam_motion_valuable = False
            cam_iou, cam_speed = GLME(curr_img=self.img.detach().cpu().numpy(),
                                      prev_img=self.last_img.detach().cpu().numpy(),
                                      metainfo=metainfo)
            # print('Iou: ', cam_iou, cam_speed)
            cam_speed *= 1.5
            cam_v = np.linalg.norm(cam_speed)
            if cam_iou > 0.6 and cam_v > 2.0:
                cam_motion_valuable = True
            
            if cam_motion_valuable:
                offset = torch.tensor([cam_speed[0], cam_speed[1], cam_speed[0], cam_speed[1]])
            else:
                offset = torch.tensor([0,0,0,0])

            # 4. second match for unmatched tracks from the first match
            first_unmatch_track_ids = []
            for i, id in enumerate(self.confirmed_ids):
                # tracklet is not matched in the first match
                case_1 = first_match_track_inds[i] == -1
                # tracklet is not lost in the previous frame
                case_2 = self.tracks[id].frame_ids[-1] == frame_id - 1
                if case_1 and case_2:
                    first_unmatch_track_ids.append(id)

            second_match_track_inds, second_match_det_inds = self.assign_ids_mc(
                first_unmatch_track_ids, second_det_bboxes, second_det_labels,
                second_det_scores, second_det_scales, 
                self.weight_iou_with_det_scores, self.match_iou_thrs['low'], offset)
            valid = second_match_det_inds > -1
            second_det_ids[valid] = torch.tensor(first_unmatch_track_ids)[
                second_match_det_inds[valid]].to(ids)

            # 5. gather all matched detection bboxes from step 2-4
            # we only keep matched detection bboxes in second match, which
            # means the id != -1
            valid = second_det_ids > -1
            bboxes = torch.cat(
                (first_match_det_bboxes, first_unmatch_det_bboxes), dim=0)
            bboxes = torch.cat((bboxes, second_det_bboxes[valid]), dim=0)

            labels = torch.cat(
                (first_match_det_labels, first_unmatch_det_labels), dim=0)
            labels = torch.cat((labels, second_det_labels[valid]), dim=0)

            scores = torch.cat(
                (first_match_det_scores, first_unmatch_det_scores), dim=0)
            scores = torch.cat((scores, second_det_scores[valid]), dim=0)

            scales = torch.cat(
                (first_match_det_scales, first_unmatch_det_scales), dim=0)
            scales = torch.cat((scales, second_det_scales[valid]), dim=0)

            depth = torch.cat(
                (first_match_det_depth, first_unmatch_det_depth), dim=0)
            depth = torch.cat((depth, second_det_depth[valid]), dim=0)

            ids = torch.cat((first_match_det_ids, first_unmatch_det_ids),
                            dim=0)
            ids = torch.cat((ids, second_det_ids[valid]), dim=0)

            # 6. assign new ids
            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum()).to(labels)
            self.num_tracks += new_track_inds.sum()

        self.update(
            ids=ids,
            bboxes=bboxes,
            scores=scores,
            labels=labels,
            scales=scales,
            depth=depth,
            frame_ids=frame_id)
        
        self.last_img = img

        # update pred_track_instances
        pred_track_instances = InstanceData()
        pred_track_instances.bboxes = bboxes
        pred_track_instances.labels = labels
        pred_track_instances.scores = scores
        pred_track_instances.scales = scales
        pred_track_instances.depth = depth
        pred_track_instances.instances_id = ids

        return pred_track_instances


    def assign_ids_mc(
            self,
            ids: List[int],
            det_bboxes: torch.Tensor,
            det_labels: torch.Tensor,
            det_scores: torch.Tensor,
            det_scales: torch.Tensor,
            weight_iou_with_det_scores: Optional[bool] = False,
            match_iou_thr: Optional[float] = 0.5,
            offset: torch.Tensor = torch.tensor([0,0,0,0]),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assign ids.

        Args:
            ids (list[int]): Tracking ids.
            det_bboxes (Tensor): of shape (N, 4)
            det_labels (Tensor): of shape (N,)
            det_scores (Tensor): of shape (N,)
            det_scales (Tensor): of shape (N,)
            weight_iou_with_det_scores (bool, optional): Whether using
                detection scores to weight IOU which is used for matching.
                Defaults to False.
            match_iou_thr (float, optional): Matching threshold.
                Defaults to 0.5.

        Returns:
            tuple(np.ndarray, np.ndarray): The assigning ids.
        """
        # get track_bboxes
        track_bboxes = np.zeros((0, 4))
        track_scales = torch.zeros((0)).to(det_bboxes)
        for id in ids:
            track_bboxes = np.concatenate(
                (track_bboxes, self.tracks[id].mean[:4][None]), axis=0)
            track_scales = torch.concat(
                (track_scales, self.tracks[id].scales[-1]), dim=0)
        track_bboxes = torch.from_numpy(track_bboxes).to(det_bboxes)
        track_bboxes = bbox_cxcyah_to_xyxy(track_bboxes)

        scaled_det_bboxes = scale_bbox(det_bboxes, det_scales)
        scaled_track_bboxes = scale_bbox(track_bboxes, track_scales)

        # compute distance
        # ious2 = bbox_overlaps(track_bboxes, det_bboxes)
        ious = bbox_overlaps(scaled_track_bboxes + offset.to(scaled_det_bboxes), scaled_det_bboxes)
        # if offset.sum() > 0:
        #     print(f'\noffset: {offset}')
        #     print('new_iou: ', ious, '\nold_iou: ', ious2)

        if weight_iou_with_det_scores:
            ious *= det_scores
        # support multi-class association
        track_labels = torch.tensor([
            self.tracks[id]['labels'][-1] for id in ids
        ]).to(det_bboxes.device)

        cate_match = det_labels[None, :] == track_labels[:, None]
        # to avoid det and track of different categories are matched
        cate_cost = (1 - cate_match.int()) * 1e6

        dists = (1 - ious + cate_cost).cpu().numpy()

        # bipartite match
        if dists.size > 0:
            cost, row, col = lap.lapjv(
                dists, extend_cost=True, cost_limit=1 - match_iou_thr)
        else:
            row = np.zeros(len(ids)).astype(np.int32) - 1
            col = np.zeros(len(det_bboxes)).astype(np.int32) - 1
        return row, col
