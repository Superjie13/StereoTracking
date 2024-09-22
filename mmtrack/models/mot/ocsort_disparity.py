# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional
import numpy as np
import torch
from torch import Tensor

from mmtrack.registry import MODELS, TASK_UTILS
from mmtrack.utils import OptConfigType, OptMultiConfig, SampleList
from .ocsort import OCSORT
from mmtrack.utils.collect_results import save_prediction_results
from .depth_extraction_comparison import truncated_mean_decorator, mean_decorator, median_decorator, center_decorator
from ..trackers.utils import scale_bbox


@MODELS.register_module()
class OCSORT_Disparity(OCSORT):
    """OCOSRT: Observation-Centric SORT: Rethinking SORT for Robust
    Multi-Object Tracking

    This multi object tracker is the implementation of `OC-SORT
    <https://arxiv.org/abs/2203.14360>`_.

    Args:
        detector (dict): Configuration of detector. Defaults to None.
        tracker (dict): Configuration of tracker. Defaults to None.
        motion (dict): Configuration of motion. Defaults to None.
        init_cfg (dict): Configuration of initialization. Defaults to None.
    """

    def __init__(self,
                 detector: Optional[dict] = None,
                 tracker: Optional[dict] = None,
                 motion: Optional[dict] = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 baseline: float = 0.25,
                 focal_length: int = 640):
        super().__init__(detector, tracker, motion, data_preprocessor, init_cfg)
        self.baseline = baseline
        self.focal_length = focal_length

    def loss(self, inputs: Dict[str, Tensor], data_samples: SampleList,
             **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        # modify the inputs shape to fit mmdet
        inputs, data_samples = self.parse_train_input(inputs, data_samples)

        return self.detector.loss(inputs, data_samples, **kwargs)

    @save_prediction_results(file_path='results.csv')
    def predict(self, inputs: Dict[str, Tensor], data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        img = inputs['img']
        disp_postp = inputs['disp_postp']
        disp_mask = inputs['disp_mask']
        depth_postp = inputs.get('depth_postp', None)

        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(0) == 1, \
            'OCSort inference only support 1 batch size per gpu for now.'
        img = img[0]
        disp_postp = disp_postp[0]
        disp_mask = disp_mask[0]
        if depth_postp is not None:
            depth_postp = depth_postp[0]

        assert len(data_samples) == 1, \
            'OCSort inference only support 1 batch size per gpu for now.'

        track_data_sample = data_samples[0]
        data = dict(
            img=img,
            disp_postp=disp_postp,
            disp_mask=disp_mask,
        )

        det_results = self.detector.predict(data, data_samples)
        assert len(det_results) == 1, 'Batch inference is not supported.'

        pred_instances_scaled, _ = self.bbox_postp_depth(
            det_results[0].pred_instances.clone(), disp_postp, depth_postp)


        track_data_sample.pred_det_instances = pred_instances_scaled

        pred_track_instances = self.tracker.track(
            model=self,
            img=img,
            feats=None,
            data_sample=track_data_sample,
            **kwargs)

        # unscale the bboxes
        scales = pred_track_instances.scales
        pred_track_instances.bboxes = scale_bbox(pred_track_instances.bboxes, 1/scales)

        # Only for comparing depth differences
        _, depth = self.bbox_postp_depth(
            pred_track_instances.clone(), disp_postp, depth_postp)

        pred_track_instances['depth'] = depth['d_values']
        pred_track_instances['gt_depth'] = depth.get('gt_d_values', depth['d_values'])

        # track_data_sample
        track_data_sample.pred_det_instances = \
            det_results[0].pred_instances.clone()
        track_data_sample.pred_track_instances = pred_track_instances

        return [track_data_sample]

    def bbox_postp_depth(self, pred_instances, disp, gt_depth=None):
        depth_values = {}
        depth = self.disp2depth(disp[:,0:1,:,:])
        bboxes = pred_instances['bboxes']  # xyxy
        d_value, scales = self.extract_depth(depth, bboxes)
        depth_values['d_values'] = d_value
        # print(f'\ndepth: {d_value}')
        if gt_depth is not None:
            gt_d_value, _ = self.extract_depth(gt_depth, bboxes)
            depth_values['gt_d_values'] = gt_d_value
            # print(f'gt_depth: {gt_d_value}')
        scales = torch.Tensor(scales).to(bboxes)
        new_bboxes = scale_bbox(bboxes, scales)

        pred_instances['bboxes'] = new_bboxes
        pred_instances['scales'] = scales
        pred_instances['depth'] = torch.Tensor(d_value).to(bboxes)
        return pred_instances, depth_values

    def disp2depth(self, disp) -> Tensor:
        # depth = bl * f / disparity
        return self.baseline * self.focal_length / (disp + 1e-6)

    # @truncated_mean_decorator
    # @mean_decorator
    # @median_decorator
    # @center_decorator
    def extract_depth(self, depth, bboxes):
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
            d_mid = d_sorted[len_d//2]

            v_tl = np.mean(depth[box[1]: box[1] + 2, box[0]: box[0] + 2])
            v_tr = np.mean(depth[box[1]: box[1] + 2, box[2] - 2: box[2]])
            v_bl = np.mean(depth[box[3] - 2: box[3], box[0]: box[0] + 2])
            v_br = np.mean(depth[box[3] - 2: box[3], box[2] - 2: box[2]])

            w_start = min(1-sum([v_tl, v_tr, v_bl, v_br] > d_mid)/4, 0.4) * len_d
            w_end = w_start + 0.6 * len_d
            d_seg = d_sorted[int(w_start): int(w_end)]
            if len(d_seg) == 0:
                d_seg = d_sorted[:-1]
            d = np.mean(d_seg)
            # if sum([v_tl, v_tr, v_bl, v_br] > d_mid) >= 2:
            #     d_seg = d_sorted[int(0.3 * len_d): int(0.4 * len_d)]
            #     if len(d_seg) == 0:
            #         d_seg = d_sorted[:-1]
            #     d = np.mean(d_seg)
            # else:
            #     d_seg = d_sorted[int(0.6 * len_d): int(0.7 * len_d)]
            #     if len(d_seg) == 0:
            #         d_seg = d_sorted[:-1]
            #     d = np.mean(d_seg)

            values.append(d)

            scale = min(d * d / 400, 3.)  # scale mustn't larger than 3.
            scale = max(scale, 1.)  # scale must larger than 1.
            scales.append(scale)

        return values, scales

    def get_tracked_depth(self, track_instances, det_instances):
        """This func is temporarily find the depth value from
         det_instances, as some objects may not be tracked.
         The depth should be associated in tracker instead of here.
         TODO: remove this func
         """
        pass

    def parse_train_input(self, inputs: Dict[str, Tensor],
             data_samples: SampleList,):

        img = inputs['img']
        disp_postp = inputs['disp_postp']
        disp_mask = inputs['disp_mask']
        depth_postp = inputs.get('depth_postp', None)
        disp_cut_mask = inputs.get('disp_cut_mask', None)

        assert img.size(1) == 1
        # convert 'inputs' shape to (N, C, H, W)
        img = torch.squeeze(img, dim=1)
        disp_postp = torch.squeeze(disp_postp, dim=1)
        disp_mask = torch.squeeze(disp_mask, dim=1)
        if depth_postp is not None:
            depth_postp = torch.squeeze(depth_postp, dim=1)
        if disp_cut_mask is not None:
            disp_cut_mask = torch.squeeze(disp_cut_mask, dim=1)

        disp_cutout = disp_postp.clone()
        if disp_cut_mask is not None:
            _mask = disp_mask + disp_cut_mask  # equivalent to 'or' operation
            _mask = _mask >= 2
            _mask = torch.tile(_mask, [1, 3, 1, 1])
            disp_cutout[_mask] = 0.

        data = dict(
            img=img,
            disp_postp=disp_postp,
            disp_mask=disp_mask,
            disp_gt=depth_postp,
            disp_cut_mask=disp_cut_mask,
            disp_cutout=disp_cutout
        )

        return data, data_samples