from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmtrack.registry import MODELS
from mmtrack.utils import OptMultiConfig, ConfigType, OptConfigType

from ..dense_head.utils import resize
from .base_discriminator import BaseDiscriminator

@MODELS.register_module()
class FC_Discriminator(BaseDiscriminator):

    def __init__(self,
                 inplanes,
                 loss_cfg: ConfigDict = dict(type='CrossEntropyLoss',
                                             use_sigmoid=True,
                                             loss_weight=1.0),
                 align_corners: bool = False,
                 init_cfg: OptMultiConfig = None):
        super().__init__(align_corners, init_cfg)

        base_planes = 128

        self.ds = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(inplanes, inplanes * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(inplanes * 2, inplanes * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(inplanes * 4, inplanes * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(inplanes * 8, 1, kernel_size=4, stride=2, padding=1),
        )

        self.loss_func: nn.Module = MODELS.build(loss_cfg)

        self.input_transform = 'resize_concat'
        self.in_index = [0, 1, 2]

    def loss(self, x, label: int, pixel_weight=None) -> dict:
        # forward pass
        # x = self._transform_inputs(x)

        seg_logits = self.ds(x)

        # gen gt map
        gt_map = self.gen_gt(seg_logits, label)

        # compute loss
        losses = dict()
        seg_logit = resize(
            input=seg_logits,
            size=gt_map.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_label = gt_map.squeeze(1)
        seg_logit = seg_logit.squeeze(1)
        losses['loss_discriminator'] = self.loss_func(
            seg_logit,
            seg_label,
            weight=pixel_weight,
            ignore_index=-1)

        return losses

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

    def gen_gt(self, x, label):
        return torch.FloatTensor(x.size()).fill_(label).to(x.device)
