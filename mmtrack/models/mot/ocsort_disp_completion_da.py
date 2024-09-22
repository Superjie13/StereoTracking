# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union
from mmengine.optim import OptimWrapper

import torch
from torch import Tensor

from mmtrack.registry import MODELS, TASK_UTILS
from mmtrack.utils import OptConfigType, OptMultiConfig, SampleList
from .base import BaseMultiObjectTracker


@MODELS.register_module()
class OCSORT_Disp_Completion_DA(BaseMultiObjectTracker):
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
                 adaptor: Optional[dict] = None,
                 tracker: Optional[dict] = None,
                 motion: Optional[dict] = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor, init_cfg)

        if adaptor is not None:
            self.adaptor = MODELS.build(adaptor)

        if motion is not None:
            self.motion = TASK_UTILS.build(motion)

        if tracker is not None:
            self.tracker = MODELS.build(tracker)

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
        # optim_wrapper.backward(parsed_losses)
        optim_wrapper.step()
        optim_wrapper.zero_grad()
        return log_vars

    def loss(self, inputs: Dict[str, Dict[str, Tensor]],
             data_samples: Dict[str, Dict[str, SampleList]],
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
        # modify the inputs shape to fit mmdet
        multi_data, multi_data_samples = self.parse_train_input(inputs, data_samples)
        return self.adaptor.loss(multi_data, multi_data_samples)

    def parse_train_input(self,
                          inputs: Dict[str, Dict[str, Tensor]],
                          data_samples: Dict[str, Dict[str, SampleList]]):

        source_disp = inputs['src']['source_disp']
        target_disp = inputs['tar']['target_disp']
        target_sup_det = inputs['tar']['target_sup_det']

        src_disp_img = source_disp['img']
        src_disp_disp = source_disp['disp_postp']
        src_disp_disp_mask = source_disp['disp_mask']
        src_disp_disp_gt = source_disp['depth_postp']

        tar_disp_img = target_disp['img']
        tar_disp_disp = target_disp['disp_postp']
        tar_disp_disp_mask = target_disp['disp_mask']

        tar_det_img = target_sup_det['img']
        tar_det_disp = target_sup_det['disp_postp']
        tar_det_disp_mask = target_sup_det['disp_mask']

        source_disp_data_samples = data_samples['src']['source_disp']
        target_disp_data_samples = data_samples['tar']['target_disp']
        tar_det_data_samples = data_samples['tar']['target_sup_det']

        assert src_disp_img.size(1) == 1
        # convert 'inputs' shape to (N, C, H, W)
        src_disp_img = torch.squeeze(src_disp_img, dim=1)
        src_disp_disp = torch.squeeze(src_disp_disp, dim=1)
        src_disp_disp_mask = torch.squeeze(src_disp_disp_mask, dim=1)
        src_disp_disp_gt = torch.squeeze(src_disp_disp_gt, dim=1)

        assert tar_det_img.size(1) == 1
        # convert 'inputs' shape to (N, C, H, W)
        tar_det_img = torch.squeeze(tar_det_img, dim=1)
        tar_det_disp = torch.squeeze(tar_det_disp, dim=1)
        tar_det_disp_mask = torch.squeeze(tar_det_disp_mask, dim=1)

        assert tar_disp_img.size(1) == 1
        tar_disp_img = torch.squeeze(tar_disp_img, dim=1)
        tar_disp_disp = torch.squeeze(tar_disp_disp, dim=1)
        tar_disp_disp_mask = torch.squeeze(tar_disp_disp_mask, dim=1)

        b, _, h, w = tar_disp_img.shape
        _, _, src_h, src_w = src_disp_img.shape
        src_disp_data = dict(
            img=src_disp_img[:, :,
                src_h // 2 - h // 2: src_h // 2 + h // 2,
                src_w // 2 - w // 2: src_w // 2 + w // 2],
            disp_postp=src_disp_disp[:, :,
                       src_h // 2 - h // 2: src_h // 2 + h // 2,
                       src_w // 2 - w // 2: src_w // 2 + w // 2],
            disp_mask=src_disp_disp_mask[:, :,
                      src_h // 2 - h // 2: src_h // 2 + h // 2,
                      src_w // 2 - w // 2: src_w // 2 + w // 2],
            disp_gt=src_disp_disp_gt[:, :,
                    src_h // 2 - h // 2: src_h // 2 + h // 2,
                    src_w // 2 - w // 2: src_w // 2 + w // 2]
        )

        tar_disp_data = dict(
            img=tar_disp_img,
            disp_postp=tar_disp_disp,
            disp_mask=tar_disp_disp_mask
        )

        tar_det_data = dict(
            img=tar_det_img,
            disp_postp=tar_det_disp,
            disp_mask=tar_det_disp_mask
        )

        multi_data = dict(src_disp=src_disp_data,
                    tar_disp=tar_disp_data,
                    tar_det=tar_det_data)

        multi_data_samples = dict(src_disp=source_disp_data_samples,
                    tar_disp=target_disp_data_samples,
                    tar_det=tar_det_data_samples)

        return multi_data, multi_data_samples

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

        det_results, results_list_disp = self.adaptor.predict(data, data_samples)
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
