# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import Dict, List, Optional, Sequence, Union
from collections import defaultdict

from mmtrack.registry import MODELS

from .data_preprocessor_disparity_v1 import TrackDataPreprocessor_Disparity_V1


@MODELS.register_module()
class TrackDataPreprocessor_Disp_Completion_DA_V1(TrackDataPreprocessor_Disparity_V1):

    def forward(self, data: dict, training: bool = False) -> Dict:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``TrackDataPreprocessor``.
        The input data from 'multi_dataset_collate' of dataloader

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Tuple[Dict[str, List[torch.Tensor]], OptSampleList]: Data in the
            same format as the model input.
        """
        if training:
            inputs = defaultdict(dict)
            data_samples = defaultdict(dict)
            for _field, _values in data.items():
                _inputs = _values['inputs']
                _data_samples = _values['data_samples']
                for k in _inputs.keys():
                    _data = {'inputs': _inputs[k], 'data_samples': _data_samples[k]}
                    _data = super().forward(_data)
                    inputs[_field][k] = _data['inputs']
                    data_samples[_field][k] = _data['data_samples']
        else:
            return super().forward(data, training)

        # inputs = {'src': src_data['inputs'],
        #           'tar': tar_data['inputs']}
        # data_samples = {'src': src_data['data_samples'],
        #                 'tar': tar_data['data_samples']}

        return dict(inputs=inputs, data_samples=data_samples)

