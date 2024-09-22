import os
import matplotlib.pyplot as plt
import copy
from typing import Dict, List, Optional, Tuple, Union
from mmtrack.visualization.visualization import subplotimg

import torch
import torch.nn as nn
from torch import Tensor

from mmtrack.registry import MODELS
from mmtrack.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.structures import SampleList
from mmdet.models.utils import (rename_loss_dict, reweight_loss_dict)
from mmdet.models.detectors import BaseDetector


@MODELS.register_module()
class YOLOX_DISP_Adaptation(BaseDetector):
    """Class for Yolox-based detection and disparity completion
    adaptation.

    This class typically consisting of a teacher model updated by exponential
    moving average and a student model updated by gradient descent.

    Args:
        detector (:obj:`ConfigDict` or dict): The detector config.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        da_train_cfg (:obj:`ConfigDict` or dict, optional):
            The domain adaptation training config.
        test_cfg (:obj:`ConfigDict` or dict, optional):
            The domain adaptation testing config.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 detector: ConfigType,
                 discriminator: ConfigType,
                 da_train_cfg: OptConfigType = None,
                 da_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.student = MODELS.build(detector)
        self.teacher = MODELS.build(detector)

        self.discriminator = MODELS.build(discriminator)
        # labels for adversarial training
        self.source_label = 0
        self.target_label = 1

        self.da_train_cfg = da_train_cfg
        self.da_test_cfg = da_test_cfg
        self.local_iter = 0

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

    def loss(self, multi_batch_inputs: Dict[str, dict],
             multi_batch_data_samples: Dict[str, dict]) -> dict:
        """Calculate losses from multi-branch inputs and data samples.

        Args:
            multi_batch_inputs (Dict[str, dict]): The dict of multi-branch
                input images, each value with shape (N, C, H, W).
                Each value should usually be mean centered and std scaled.
            multi_batch_data_samples (Dict[str, List[:obj:`DetDataSample`]]):
                The dict of multi-branch data samples.

        Returns:
            dict: A dictionary of loss components
        """
        src_disp_inputs = multi_batch_inputs['src_disp']
        src_disp_data_samples = multi_batch_data_samples['src_disp']

        tar_disp_inputs = multi_batch_inputs['tar_disp']
        tar_disp_data_samples = multi_batch_data_samples['tar_disp']

        tar_det_inputs = multi_batch_inputs['tar_det']
        tar_det_data_samples = multi_batch_data_samples['tar_det']

        losses = dict()
        # 1. compute losses of gt_det (no gt_disp)
        tar_det_losses = self.loss_by_tar_det(tar_det_inputs, tar_det_data_samples)
        losses.update(tar_det_losses)
        parse_det_losses, _ = self.parse_losses(tar_det_losses)
        parse_det_losses.backward(retain_graph=False)

        # 2. compute losses of gt_disp (no ge_det)
        src_disp_losses, src_feat = self.loss_by_src_disp(src_disp_inputs, src_disp_data_samples)

        src_disp_pred_disp = src_disp_losses.pop('src_disp_pred_disp')
        parse_src_disp_losses, _ = self.parse_losses(src_disp_losses)
        parse_src_disp_losses.backward(retain_graph=False)
        losses.update(src_disp_losses)

        # 3. compute adversarial losses
        # 3.1 Lock Discriminator to train Source Labels
        self.freeze(self.discriminator)

        # 3.2 Target forward pass
        tar_feat = self.student.extract_feat(tar_disp_inputs)
        _, tar_feat = self.student.disp_head.forward(tar_feat, return_feat=True)
        # 3.3 adversarial train
        adv_tar_losses = self.discriminator.loss(tar_feat, label=self.source_label,
                                                 pixel_weight=None)
        adv_tar_losses = rename_loss_dict('adv_tar_', adv_tar_losses)
        parse_adv_tar_loss, _ = self.parse_losses(adv_tar_losses)
        parse_adv_tar_loss.backward(retain_graph=False)

        losses.update(adv_tar_losses)

        # 3.4 Train Discriminator
        self.unfreeze(self.discriminator)

        if isinstance(src_feat, (list, tuple)):
            src_feat = [f.detach() for f in src_feat]
            tar_feat = [f.detach() for f in tar_feat]
        else:
            src_feat = src_feat.detach()
            tar_feat = tar_feat.detach()

        ds_src_losses = self.discriminator.loss(src_feat, label=self.source_label,
                                                pixel_weight=None)
        ds_src_losses = rename_loss_dict('src_', ds_src_losses)
        parse_ds_tar_loss, _ = self.parse_losses(ds_src_losses)
        parse_ds_tar_loss.backward(retain_graph=False)

        losses.update(ds_src_losses)

        ds_tar_losses = self.discriminator.loss(tar_feat, label=self.target_label,
                                                pixel_weight=None)
        ds_tar_losses = rename_loss_dict('tar_', ds_tar_losses)
        parse_ds_tar_loss, _ = self.parse_losses(ds_tar_losses)
        parse_ds_tar_loss.backward(retain_graph=False)

        losses.update(ds_tar_losses)

        if (self.local_iter+1) % 1000 == 0 or self.local_iter==0:
            with torch.no_grad():
                results = self._forward(tar_disp_inputs, tar_disp_data_samples)
            self.vis_disp(src_disp_inputs,
                          src_disp_pred_disp,
                          tar_disp_inputs,
                          results['forward_disp'],
                          outdir='~/Documents/debug')
        self.local_iter += 1

        return losses

    def loss_by_tar_det(self, batch_inputs: Dict[str, Tensor],
                        batch_data_samples: SampleList) -> dict:
        setattr(self.student, 'train_bbox', True)
        setattr(self.student, 'train_disp', False)
        losses = self.student.loss(batch_inputs, batch_data_samples)
        sup_det_weight = self.da_train_cfg.get('tar_det_weight', 1.)

        return rename_loss_dict('tar_det_', reweight_loss_dict(losses, sup_det_weight))

    def loss_by_src_disp(self, batch_inputs: Dict[str, Tensor],
                         batch_data_samples: SampleList,
                         return_feature: bool = True):
        setattr(self.student, 'train_bbox', False)
        setattr(self.student, 'train_disp', True)

        feat = self.student.extract_feat(batch_inputs)

        gt_disp = batch_inputs.get('disp_gt', None)
        assert gt_disp is not None
        pixels_weight = batch_inputs.get('pixels_weight', None)
        losses_disp = self.student.disp_head.loss(feat, gt_disp, pixels_weight, return_feat=True)
        src_disp_weight = self.da_train_cfg.get('src_disp_weight', 1.)

        losses = rename_loss_dict('src_disp_', reweight_loss_dict(losses_disp, src_disp_weight))

        if return_feature:
            return losses, losses_disp.pop('disp_feat')

        return losses

    def predict(self, batch_inputs: Dict[str, Tensor],
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        if self.da_test_cfg.get('predict_on', 'student') == 'student':
            return self.student(
                batch_inputs, batch_data_samples, mode='predict')
        else:
            return self.teacher(
                batch_inputs, batch_data_samples, mode='predict')

    def _forward(self, batch_inputs: Dict[str, Tensor],
                 batch_data_samples: SampleList) -> SampleList:
        if self.da_test_cfg.get('forward_on', 'student') == 'student':
            return self.student(
                batch_inputs, batch_data_samples, mode='tensor')
        else:
            return self.teacher(
                batch_inputs, batch_data_samples, mode='tensor')

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        if self.da_test_cfg.get('extract_feat_on', 'student') == 'student':
            return self.student.extract_feat(batch_inputs)
        else:
            return self.teacher.extract_feat(batch_inputs)

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Add teacher and student prefixes to model parameter names."""
        if not any([
                'student' in key or 'teacher' in key
                for key in state_dict.keys()
        ]):
            keys = list(state_dict.keys())
            state_dict.update({'teacher.' + k: state_dict[k] for k in keys})
            state_dict.update({'student.' + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


    def vis_disp(self, src_inputs: Dict, src_pred_disp: Tensor,
                 tar_inputs: Dict, tar_pred_disp: Tensor,
                 outdir: str):
        out_dir = os.path.join(os.path.expanduser(outdir), 'disp_completion_debug_12')
        os.makedirs(out_dir, exist_ok=True)
        src_img = src_inputs['img']
        src_disp_postp = src_inputs['disp_postp']
        src_disp_gt = src_inputs.get('disp_gt', None)

        src_vis_img = src_img/255.
        src_vis_d = src_disp_postp[:, 0, ...]
        if src_disp_gt is not None:
            src_vis_d_gt = src_disp_gt
        src_vis_d_pred = src_pred_disp

        tar_img = tar_inputs['img']
        tar_disp_postp = tar_inputs['disp_postp']
        tar_disp_gt = tar_inputs.get('disp_gt', None)
        tar_vis_img = tar_img / 255.
        tar_vis_d = tar_disp_postp[:, 0, ...]
        if tar_disp_gt is not None:
            tar_vis_d_gt = tar_disp_gt
        tar_vis_d_pred = tar_pred_disp

        b, c, h, w = src_img.shape

        for j in range(b):
            if j >= 2:
                break
            rows, cols = 2, 4
            fig, axs = plt.subplots(
                rows,
                cols,
                figsize=(3 * cols, 3 * rows),
                gridspec_kw={
                    'hspace': 0.1,
                    'wspace': 0,
                    'top': 0.95,
                    'bottom': 0,
                    'right': 1,
                    'left': 0
                },
            )
            subplotimg(axs[0][0], src_vis_img[j], 'Source Image')
            subplotimg(axs[1][0], src_vis_d[j], 'Source Disp')
            if src_disp_gt is not None:
                subplotimg(axs[0][1], src_vis_d_gt[j], 'Source GT Depth')
            subplotimg(axs[1][1], src_vis_d_pred[j], 'Source Pred Depth')

            subplotimg(axs[0][2], tar_vis_img[j], 'Target Image')
            subplotimg(axs[1][2], tar_vis_d[j], 'Target Disp')
            if tar_disp_gt is not None:
                subplotimg(axs[0][3], tar_vis_d_gt[j], 'Target GT Depth')
            subplotimg(axs[1][3], tar_vis_d_pred[j], 'Target Pred Depth')

            for ax in axs.flat:
                ax.axis('off')
            plt.savefig(
                os.path.join(out_dir,
                             f'{(self.local_iter + 1):06d}_{j}.png'))
            plt.close()

