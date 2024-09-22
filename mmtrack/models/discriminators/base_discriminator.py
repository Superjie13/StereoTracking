from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from mmengine.model import BaseModule, constant_init
from mmtrack.utils import OptMultiConfig, ConfigType, OptConfigType
from mmdet.structures import SampleList
from mmdet.utils import OptMultiConfig


class BaseDiscriminator(BaseModule, metaclass=ABCMeta):
    """Base class for discriminator."""

    def __init__(self,
                 align_corners: bool = False,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
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
    def loss(self, x: Tensor, labels: Tensor, pixel_weight=None) -> dict:
        """Perform forward propagation and loss calculation of the disparity completion
        head on the features of the upstream network.

        Returns:
            dict: A dictionary of loss components.
        """
        pass

    def gen_gt(self, x, label):
        """generate gt map"""
        pass