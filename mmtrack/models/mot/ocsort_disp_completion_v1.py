# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List, Optional, Tuple, Union
from mmengine.optim import OptimWrapper
import matplotlib.pyplot as plt

import torch
from torch import Tensor

from mmtrack.visualization.visualization import subplotimg
from mmtrack.registry import MODELS, TASK_UTILS
from mmtrack.utils import OptConfigType, OptMultiConfig, SampleList
from .base import BaseMultiObjectTracker


@MODELS.register_module()
class OCSORT_Disp_Completion_V1(BaseMultiObjectTracker):
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
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor, init_cfg)

        if detector is not None:
            self.detector = MODELS.build(detector)

        if motion is not None:
            self.motion = TASK_UTILS.build(motion)

        if tracker is not None:
            self.tracker = MODELS.build(tracker)

        self.local_iter = 0


    def loss(self, inputs: Dict[str, Tensor],
             data_samples: SampleList,
             **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples.

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

        Returns:
            dict: A dictionary of loss components.
        """
        inputs, data_samples = self.parse_train_input(inputs, data_samples)
        # disp_gt = inputs['disp_gt']
        # depth_mask = torch.ones_like(disp_gt)
        # depth_mask[disp_gt < 10] = 0.1
        # inputs['pixels_weight'] = depth_mask
        losses = self.detector.loss(inputs, data_samples)
        pred_disp = losses.pop('pred_disp')
        if (self.local_iter+1) % 1000 == 0 or self.local_iter==0:
            self.vis_disp(inputs,
                          pred_disp,
                          outdir='~/Documents/debug_disp_completion')
        self.local_iter += 1
        return losses


    def predict(self, inputs: Dict[str, Tensor], data_samples: SampleList,
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

        Returns:
            SampleList: Tracking results of the input images.
            Each TrackDataSample usually contains ``pred_det_instances``
            or ``pred_track_instances``.
        """
        img = inputs['img']
        disp_postp = inputs['disp_postp']
        disp_mask = inputs['disp_mask']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        assert img.size(0) == 1, \
            'OCSort inference only support 1 batch size per gpu for now.'
        img = img[0]
        disp_postp = disp_postp[0]
        disp_mask = disp_mask[0]

        assert len(data_samples) == 1, \
            'OCSort inference only support 1 batch size per gpu for now.'

        track_data_sample = data_samples[0]
        data = dict(
            img=img,
            disp_postp=disp_postp,
            disp_mask=disp_mask
        )

        det_results, results_list_disp = self.detector.predict(data, data_samples)
        assert len(det_results) == 1, 'Batch inference is not supported.'
        track_data_sample.pred_det_instances = \
            det_results[0].pred_instances.clone()

        pred_track_instances = self.tracker.track(
            model=self,
            img=img,
            feats=None,
            data_sample=track_data_sample,
            **kwargs)
        track_data_sample.pred_track_instances = pred_track_instances
        track_data_sample.dense_depth_map = results_list_disp[0]

        return [track_data_sample]

    def parse_train_input(self, inputs: Dict[str, Tensor],
             data_samples: SampleList,):

        img = inputs['img']
        disp_postp = inputs['disp_postp']
        disp_mask = inputs['disp_mask']
        depth_postp = inputs['depth_postp']

        assert img.size(1) == 1
        # convert 'inputs' shape to (N, C, H, W)
        img = torch.squeeze(img, dim=1)
        disp_postp = torch.squeeze(disp_postp, dim=1)
        disp_mask = torch.squeeze(disp_mask, dim=1)
        depth_postp = torch.squeeze(depth_postp, dim=1)
        data = dict(
            img=img,
            disp_postp=disp_postp,
            disp_mask=disp_mask,
            disp_gt=depth_postp,
        )

        return data, data_samples

    def vis_disp(self, inputs: Dict, pred_disp: Tensor, outdir: str):
        out_dir = os.path.join(os.path.expanduser(outdir), )
        os.makedirs(out_dir, exist_ok=True)
        src_img = inputs['img']
        src_disp_postp = inputs['disp_postp']
        src_disp_gt = inputs.get('disp_gt', None)

        src_vis_img = src_img/255.
        src_vis_d = src_disp_postp[:, 0, ...]
        if src_disp_gt is not None:
            src_vis_d_gt = src_disp_gt
        src_vis_d_pred = pred_disp

        b, c, h, w = src_img.shape

        for j in range(b):
            # if j >= 2:
            #     break
            rows, cols = 2, 2
            fig, axs = plt.subplots(
                rows,
                cols,
                figsize=(3 * cols, 3 * rows),
                gridspec_kw={
                    'hspace': 0.05,
                    'wspace': 0,
                    'top': 0.95,
                    'bottom': 0,
                    'right': 1,
                    'left': 0
                },
            )
            subplotimg(axs[0][0], src_vis_img[j], 'Source Image', bgr2rgb=True)
            subplotimg(axs[1][0], src_vis_d[j], 'Source Disp')
            if src_disp_gt is not None:
                subplotimg(axs[0][1], src_vis_d_gt[j], 'Source GT Depth')
            subplotimg(axs[1][1], src_vis_d_pred[j], 'Source Pred Depth')

            for ax in axs.flat:
                ax.axis('off')
            plt.savefig(
                os.path.join(out_dir,
                             f'{(self.local_iter + 1):06d}_{j}.png'))
            plt.close()
