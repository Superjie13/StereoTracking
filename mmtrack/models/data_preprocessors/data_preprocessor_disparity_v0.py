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

from .data_preprocessor_disparity_v1 import TrackDataPreprocessor_Disparity_V1


@MODELS.register_module()
class TrackDataPreprocessor_Disparity_V0(TrackDataPreprocessor_Disparity_V1):
    """concatenate image and disparity map as input"""

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
        data = super().forward(data=data, training=training)
        new_inputs, new_data_samples = data['inputs'], data['data_samples']

        img = new_inputs['img']
        disp_postp = new_inputs['disp_postp']
        new_inputs['img'] = torch.cat((img, disp_postp), dim=2)

        return dict(inputs=new_inputs, data_samples=new_data_samples)

