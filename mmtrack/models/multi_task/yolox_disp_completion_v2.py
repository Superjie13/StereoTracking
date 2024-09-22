import torch
import torch.nn as nn
from torch import Tensor

from mmengine.logging import print_log, MMLogger

from typing import Dict, List, Optional, Tuple, Union
from mmdet.structures import OptSampleList
from mmtrack.registry import MODELS
from mmtrack.utils import ConfigType, OptConfigType, OptMultiConfig, SampleList
from .single_stage import SingleStageDetector_DispCompletion


@MODELS.register_module()
class YOLOX_DISP_Completion_V2(SingleStageDetector_DispCompletion):
    r"""Implementation of `YOLOX: Exceeding YOLO Series in 2021
    <https://arxiv.org/abs/2107.08430>`_ plus a disparity completion head
    =======> train det and depth completion together!

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        disparity_head (:obj:`ConfigDict` or dict): The disparity completion head config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of YOLOX. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of YOLOX. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 disparity_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            disparity_head=disparity_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.train_disp = False  # flag to indicate if train disparity

    def init_weights(self):
        if self.init_cfg is not None:
            if self.init_cfg.get('type', None) == 'ColorPretrained':
                from mmengine.runner.checkpoint import _load_checkpoint, load_state_dict
                logger = MMLogger.get_instance('mmengine')

                pretrained = self.init_cfg.get('checkpoint')
                print_log(f"load model from: {pretrained}", logger=logger)
                checkpoint = _load_checkpoint(pretrained, logger=logger, map_location='cpu')
                state_dict = checkpoint['state_dict']

                print_log(f"update stem conv for disparity: `new stem`", logger=logger)

                disp_branch = dict()
                for name, param in state_dict.items():
                    if 'stem' in name:
                        new_name = name.replace('stem', 'disp_stem')
                        disp_branch.update({new_name: param})
                    if 'stage1' in name:
                        new_name = name.replace('stage1', 'disp_stage1')
                        disp_branch.update({new_name: param})
                state_dict.update(disp_branch)
                load_state_dict(self, state_dict, strict=False, logger=logger)


    def loss(self, batch_inputs: Dict[str, Tensor],
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        losses = dict()
        if self.train_bbox:
            x = self.extract_feat(batch_inputs)
            x = list(x)
            losses_bbox = self.bbox_head.loss(x[:len(self.neck.in_channels)], batch_data_samples)
            losses.update(losses_bbox)
            # parse_det_losses, _ = self.parse_losses(losses_bbox)
            # parse_det_losses.backward(retain_graph=False)

        if self.train_disp:
            self.freeze(self)
            self.unfreeze(self.disp_head)
            # with torch.no_grad():
            #     x = self.extract_feat(batch_inputs)
            x = self.extract_feat(batch_inputs)
            x = list(x)
            for i, x_i in enumerate(x):
                x[i] = x_i.detach()

            gt_disp = batch_inputs.get('disp_gt', None)
            assert gt_disp is not None
            pixels_weight = batch_inputs.get('pixels_weight', None)
            losses_disp = self.disp_head.loss(x, gt_disp, pixels_weight)
            losses.update(losses_disp)
            # parse_disp_losses, _ = self.parse_losses(losses_disp)
            # parse_disp_losses.backward(retain_graph=False)

        return losses

    def extract_feat(self, batch_inputs: Tensor) -> List[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        ext = []
        if self.with_neck:
            if len(x) > len(self.neck.in_channels):
                ext = x[len(self.neck.in_channels):]
            x = self.neck(x[: len(self.neck.in_channels)])
        x = list(x)
        x.extend(ext)
        return x

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs)
        results_bbox = self.bbox_head.forward(x[:len(self.neck.in_channels)])
        results_disp = self.disp_head.forward(x)
        return {'forward_bbox': results_bbox,
                'forward_disp': results_disp}


    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    @staticmethod
    def unfreeze(model: nn.Module):
        """Freeze the model."""
        model.train()
        for param in model.parameters():
            param.requires_grad = True

