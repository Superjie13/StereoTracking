# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import numpy as np

import torch
from torch import Tensor

from mmtrack.registry import MODELS, TASK_UTILS
from mmtrack.utils import OptConfigType, SampleList
from .deep_sort import DeepSORT


@MODELS.register_module()
class DeepSORT_Disparity(DeepSORT):
    """Simple online and realtime tracking with a deep association metric.

    Details can be found at `DeepSORT<https://arxiv.org/abs/1703.07402>`_.
    """

    def __init__(self,
                 detector: Optional[dict] = None,
                 reid: Optional[dict] = None,
                 tracker: Optional[dict] = None,
                 motion: Optional[dict] = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptConfigType = None,
                 baseline: float = 0.25,
                 focal_length: int = 640):
        super().__init__(detector, reid, tracker, motion, data_preprocessor, init_cfg)
        self.baseline = baseline
        self.focal_length = focal_length

    def loss(self, inputs: Dict[str, Tensor], data_samples: SampleList,
             **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        # modify the inputs shape to fit mmdet
        inputs, data_samples = self.parse_train_input(inputs, data_samples)

        return self.detector.loss(inputs, data_samples, **kwargs)

    def predict(self,
                inputs: Dict[str, Tensor],
                data_samples: SampleList,
                rescale: bool = True,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Dict[str, Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size.The T denotes the number of
                key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.
            rescale (bool, Optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to True.

        Returns:
            SampleList: Tracking results of the
                input images. Each TrackDataSample usually contains
                ``pred_det_instances`` or ``pred_track_instances``.
        """
        img = inputs['img']
        disp_postp = inputs['disp_postp']
        disp_mask = inputs['disp_mask']
        depth_postp = inputs.get('depth_postp', None)

        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(0) == 1, \
            'ByteTrack inference only support 1 batch size per gpu for now.'
        img = img[0]
        disp_postp = disp_postp[0]
        disp_mask = disp_mask[0]
        if depth_postp is not None:
            depth_postp = depth_postp[0]

        assert len(data_samples) == 1, \
            'SORT/DeepSORT inference only support ' \
            '1 batch size per gpu for now.'

        track_data_sample = data_samples[0]
        data = dict(
            img=img,
            disp_postp=disp_postp,
            disp_mask=disp_mask,
        )

        det_results = self.detector.predict(data, data_samples)
        assert len(det_results) == 1, 'Batch inference is not supported.'
        pred_instances, _ = self.bbox_postp_depth(
            det_results[0].pred_instances.clone(), disp_postp, depth_postp)


        track_data_sample.pred_det_instances = pred_instances

        pred_track_instances = self.tracker.track(
            model=self,
            img=img,
            feats=None,
            data_sample=track_data_sample,
            data_preprocessor=self.preprocess_cfg,
            rescale=rescale,
            **kwargs)

        # # unscale the bboxes
        # scales = pred_track_instances.scales
        # pred_track_instances.bboxes = self.scale_bbox(pred_track_instances.bboxes, 1/scales)

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

    def scale_bbox(self, bboxes, scales):
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

        pred_instances['scales'] = scales
        pred_instances['depth'] = torch.Tensor(d_value).to(bboxes)
        return pred_instances, depth_values

    def disp2depth(self, disp) -> Tensor:
        # depth = bl * f / disparity
        return self.baseline * self.focal_length / (disp + 1e-6)

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

            scale = min(d * d / 400, 3.)  # scale mustn't larger than 2.
            scale = max(scale, 1.)  # scale must larger than 1.
            scales.append(scale)
            # print(w)
            # print(scale)
        return values, scales

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
