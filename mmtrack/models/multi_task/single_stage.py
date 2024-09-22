# Support depth completion head
from typing import List, Tuple, Union, Dict

import torch
from torch import Tensor

from mmtrack.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors import SingleStageDetector


@MODELS.register_module()
class SingleStageDetector_DispCompletion(SingleStageDetector):
    """Base class to support auxiliary head for disparity completion.

    Single-stage detectors with disparity completion head directly and densely predict
    bounding boxes and disparity maps on the output features of the backbone+neck.
    """
    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 disparity_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.disp_head = MODELS.build(disparity_head)
        self.train_bbox = True  # flag to indicate if train bbox
        self.train_disp = True  # flag to indicate if train disparity

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
        x = self.extract_feat(batch_inputs)
        if self.train_bbox:
            losses_bbox = self.bbox_head.loss(x, batch_data_samples)
            losses.update(losses_bbox)
        if self.train_disp:
            gt_disp = batch_inputs.get('disp_gt', None)
            assert gt_disp is not None
            pixels_weight = batch_inputs.get('pixels_weight', None)
            losses_disp = self.disp_head.loss(x, gt_disp, pixels_weight)
            losses.update(losses_disp)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> Tuple:
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
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x = self.extract_feat(batch_inputs)
        results_list_bbox = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list_bbox)

        results_list_disp = self.disp_head.predict(
            x, batch_data_samples, rescale=rescale)

        return batch_data_samples, results_list_disp

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
        results_bbox = self.bbox_head.forward(x)
        results_disp = self.disp_head.forward(x)
        return {'forward_bbox': results_bbox,
                'forward_disp': results_disp}
