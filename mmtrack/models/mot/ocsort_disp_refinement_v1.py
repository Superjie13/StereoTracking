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
class OCSORT_Disp_Refinement_V1(BaseMultiObjectTracker):
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

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self.loss(**data)  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        # optim_wrapper.update_params(parsed_losses)
        optim_wrapper.backward(parsed_losses)
        optim_wrapper.step()
        optim_wrapper.zero_grad(set_to_none=True)
        return log_vars
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
        # if inputs.get('disp_gt', None) is not None:
        #     disp_gt = inputs['disp_gt']
        #     depth_mask = torch.ones_like(disp_gt)
        #     depth_mask[disp_gt < 10] = 0.01
        #     inputs['pixels_weight'] = depth_mask

        losses = self.detector.loss(inputs, data_samples)
        if losses.get('pred_disp', None) is not None:
            pred_disp = losses.pop('pred_disp')
            if (self.local_iter+1) % 1000 == 0:
                self.vis_disp(inputs,
                              pred_disp,
                              outdir='~/Documents/debug_disp_refinement_v1')
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
        # track_data_sample.pred_det_instances = \
        #     det_results[0].pred_instances.clone()

        track_data_sample.pred_det_instances = self.bbox_postp(
            det_results[0].pred_instances.clone())

        pred_track_instances = self.tracker.track(
            model=self,
            img=img,
            feats=None,
            data_sample=track_data_sample,
            **kwargs)
        track_data_sample.pred_track_instances = pred_track_instances
        track_data_sample.dense_depth_map = results_list_disp[0]

        return [track_data_sample]

    def bbox_postp(self, pred_instances):
        bboxes = pred_instances['bboxes']
        cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
        cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]

        w = w * 3
        h = h * 3

        x1 = cx - w/2
        x2 = cx + w/2
        y1 = cy - h/2
        y2 = cy + h/2

        new_bboxes = torch.cat((x1[:, None], y1[:, None], x2[:, None], y2[:, None]), dim=-1).reshape(-1, 4)
        pred_instances['bboxes'] = new_bboxes
        return pred_instances

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

    def vis_disp(self, inputs: Dict, pred_disp: Tensor, outdir: str):
        out_dir = os.path.join(os.path.expanduser(outdir), )
        os.makedirs(out_dir, exist_ok=True)
        src_img = inputs['img']
        src_disp_postp = inputs['disp_postp']
        src_disp_gt = inputs.get('disp_gt', None)
        src_disp_cutout = inputs.get('disp_cutout', None)

        src_vis_img = src_img/255.
        src_vis_d = src_disp_postp[:, 0, ...]
        if src_disp_gt is not None:
            src_vis_d_gt = src_disp_gt
        if src_disp_cutout is not None:
            src_vis_d_cutout = src_disp_cutout[:, 0, ...]
            src_vis_d_cutout = torch.clip(src_vis_d_cutout, 0, 255)
        src_vis_d_pred = pred_disp

        b, c, h, w = src_img.shape

        for j in range(b):
            # if j >= 2:
            #     break
            rows, cols = 2, 3
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
            if src_disp_cutout is not None:
                subplotimg(axs[0][1], src_vis_d_cutout[j], 'Source Disp Cutout')
            if src_disp_gt is not None:
                subplotimg(axs[0][2], src_vis_d_gt[j], 'Source GT Depth')
            subplotimg(axs[1][2], src_vis_d_pred[j], 'Source Pred Depth')

            for ax in axs.flat:
                ax.axis('off')
            plt.savefig(
                os.path.join(out_dir,
                             f'{(self.local_iter + 1):06d}_{j}.png'))
            plt.close()
