import copy
from abc import ABCMeta, abstractmethod
from inspect import signature
from typing import List, Optional, Tuple

import mmcv
import torch
from mmengine.config import ConfigDict
from mmengine.model import BaseModule, constant_init
from torch import Tensor
from mmtrack.utils import OptMultiConfig, ConfigType, OptConfigType

from mmdet.structures import SampleList
from mmdet.utils import OptMultiConfig
from .utils import resize


class BaseDispHead(BaseModule, metaclass=ABCMeta):
    """Base class for disparity completion heads.

    1. The ``init_weights`` method is used to initialize head's
    model parameters. After detector initialization, ``init_weights``
    is triggered when ``detector.init_weights()`` is called externally.

    2. The ``loss`` method is used to calculate the loss of head,
    which includes two steps: (1) the head model performs forward
    propagation to obtain the disparity maps (2) The ``loss_by_feat`` method
    is called based on the disparity maps to calculate the loss.

    .. code:: text

    loss(): forward() -> loss_by_feat()

    3. The ``predict`` method is used to predict detection results,
    which includes two steps: (1) the head model performs forward
    propagation to obtain the feature maps (2) The ``predict_by_feat`` method
    is called based on the feature maps to predict detection results including
    post-processing.

    .. code:: text

    predict(): forward() -> predict_by_feat()

    Args:
        in_channels (int|Sequence[int]): input channels of previous module.
        channels (int): Channels after first modules.
        out_channels (int): channels of predicted feature maps.
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
    """
    def __init__(self,
                 in_channels,
                 channels,
                 out_channels,
                 in_index=-1,
                 input_transform=None,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001, requires_grad=True),
                 act_cfg: ConfigType = dict(type='ELU'),
                 align_corners: bool = False,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners

    def init_weights(self) -> None:
        """Initialize the weights."""
        super().init_weights()
        # avoid init_cfg overwrite the initialization of `conv_offset`
        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)

    @abstractmethod
    def loss(self, x: Tuple[Tensor], gt_disp: Tensor) -> dict:
        """Perform forward propagation and loss calculation of the disparity completion
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            gt_disp (tuple[Tensor]): The GT data.

        Returns:
            dict: A dictionary of loss components.
        """
        pass

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> List:
        """Perform forward propagation of the head and predict disparity
        completion results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        pred_disp_list = []
        pred_disp = self(x)
        for i in range(pred_disp.shape[0]):
            pred_disp_list.append(pred_disp[i, ...])
        if rescale:
            for i, img_meta in enumerate(batch_img_metas):
                assert img_meta.get('img_shape') is not None
                img_shape = img_meta['img_shape']
                pred_disp_list[i] = resize(
                    input=pred_disp_list[i][None, ...],
                    size=img_shape,
                    mode='bilinear',
                    align_corners=self.align_corners)

        return pred_disp_list


    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs