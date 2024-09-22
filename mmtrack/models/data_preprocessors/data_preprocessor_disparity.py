# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from mmdet.structures.mask import BitmapMasks
from mmengine.model import BaseDataPreprocessor

from mmtrack.registry import MODELS
from mmtrack.structures import TrackDataSample
from mmtrack.utils import stack_batch

from .data_preprocessor import TrackDataPreprocessor


@MODELS.register_module()
class TrackDataPreprocessor_Disparity(TrackDataPreprocessor):

    def forward(self, data: dict, training: bool = False) -> Dict:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``TrackDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Tuple[Dict[str, List[torch.Tensor]], OptSampleList]: Data in the
            same format as the model input.
        """
        batch_pad_shape = self._get_pad_shape(data)
        data = super().forward(data=data, training=training)
        ori_inputs, data_samples = data['inputs'], data['data_samples']

        inputs = dict()
        for imgs_key, imgs in ori_inputs.items():
            # TODO: whether normalize should be after stack_batch
            # imgs is a list contain multiple Tensor of imgs.
            # The shape of imgs[0] is (T, C, H, W).
            channel = imgs[0].size(1)
            if self.channel_conversion and channel == 3:
                imgs = [_img[:, [2, 1, 0], ...] for _img in imgs]
            # change to `float`
            imgs = [_img.float() for _img in imgs]
            if self._enable_normalize:
                imgs = [(_img - self.mean) / self.std for _img in imgs]

            inputs[imgs_key] = stack_batch(imgs, self.pad_size_divisor,
                                           self.pad_value)

        if data_samples is not None:
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            for key, imgs in inputs.items():
                img_shape = tuple(imgs.size()[-2:])
                imgs_shape = [img_shape] * imgs.size(1) if imgs.size(
                    1) > 1 else img_shape
                ref_prefix = key[:-3]
                for data_sample, pad_shapes in zip(data_samples,
                                                   batch_pad_shape[key]):
                    data_sample.set_metainfo({
                        f'{ref_prefix}batch_input_shape':
                        imgs_shape,
                        f'{ref_prefix}pad_shape':
                        pad_shapes
                    })
                if self.pad_mask:
                    self.pad_gt_masks(data_samples, ref_prefix)

        # if training and self.batch_augments is not None:
        #     for batch_aug in self.batch_augments:
        #         # Only yolox need batch_aug, and yolox can only process
        #         # `img` key. Therefore, only img is processed here.
        #         # The shape of `img` is (N, T, C, H, W), hence, we use
        #         # [:, 0] to change the shape to (N, C, H, W).
        #         assert len(inputs) == 1 and 'img' in inputs
        #         aug_inputs, data_samples = batch_aug(inputs['img'][:, 0],
        #                                              data_samples)
        #         inputs['img'] = aug_inputs.unsqueeze(1)

        img = inputs['img']
        disp_postp = inputs['disp_postp']
        inputs['img'] = torch.cat((img, disp_postp), dim=2)

        return dict(inputs=inputs, data_samples=data_samples)

