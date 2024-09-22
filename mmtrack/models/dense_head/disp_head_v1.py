import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.config import ConfigDict
from mmengine.model import bias_init_with_prob
from torch import Tensor

from mmtrack.utils import OptMultiConfig, ConfigType, OptConfigType
from mmtrack.registry import MODELS
from mmtrack.models.dense_head.utils import resize
from mmtrack.models.dense_head.base_disp_head import BaseDispHead


@MODELS.register_module()
class DispHeadV1(BaseDispHead):
    """DispHeadV1 used to predict dense disparity map

    """
    def __init__(self,
                 in_channels,
                 channels,
                 out_channels=1,
                 in_index=-1,
                 input_transform=None,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001, requires_grad=True),
                 act_cfg: ConfigType = dict(type='ELU'),
                 loss_cfg: ConfigDict = dict(type='BerHuLoss', loss_weight=1),
                 align_corners: bool = False,
                 init_cfg: OptMultiConfig = dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')
                 ):
        super().__init__(in_channels,
                         channels,
                         out_channels,
                         in_index,
                         input_transform,
                         conv_cfg,
                         norm_cfg,
                         act_cfg,
                         align_corners,
                         init_cfg)
        # 1/8
        self.dconv1_1 = ConvModule(self.in_channels,
                                   self.channels,
                                   3,
                                   padding=1,
                                   conv_cfg=self.conv_cfg,
                                   norm_cfg=self.norm_cfg,
                                   act_cfg=self.act_cfg)

        self.dconv1_2 = ConvModule(self.channels,
                                   self.channels,
                                   3,
                                   padding=1,
                                   conv_cfg=self.conv_cfg,
                                   norm_cfg=self.norm_cfg,
                                   act_cfg=self.act_cfg)
        # 1/4
        self.dconv2_1 = ConvModule(self.channels,
                                   256,
                                   3,
                                   padding=1,
                                   conv_cfg=self.conv_cfg,
                                   norm_cfg=self.norm_cfg,
                                   act_cfg=self.act_cfg)

        self.dconv2_2 = ConvModule(256,
                                   256,
                                   3,
                                   padding=1,
                                   conv_cfg=self.conv_cfg,
                                   norm_cfg=self.norm_cfg,
                                   act_cfg=self.act_cfg)
        # 1/2
        self.dconv3_1 = ConvModule(256,
                                   128,
                                   3,
                                   padding=1,
                                   conv_cfg=self.conv_cfg,
                                   norm_cfg=self.norm_cfg,
                                   act_cfg=self.act_cfg)

        self.dconv3_2 = ConvModule(128,
                                   128,
                                   3,
                                   padding=1,
                                   conv_cfg=self.conv_cfg,
                                   norm_cfg=self.norm_cfg,
                                   act_cfg=self.act_cfg)
        # # 1/1
        # self.dconv4_1 = ConvModule(128,
        #                            64,
        #                            3,
        #                            padding=1,
        #                            conv_cfg=self.conv_cfg,
        #                            norm_cfg=None,
        #                            act_cfg=None)
        #
        # self.dconv4_2 = ConvModule(64,
        #                            64,
        #                            3,
        #                            padding=1,
        #                            conv_cfg=self.conv_cfg,
        #                            norm_cfg=None,
        #                            act_cfg=None)
        self.reg = ConvModule(128,
                              self.out_channels,
                              1,
                              conv_cfg=self.conv_cfg,
                              norm_cfg=None,
                              act_cfg=dict(type='ReLU'))

        self.sigmoid = nn.Sigmoid()
        self.loss_func: nn.Module = MODELS.build(loss_cfg)

    def forward(self, inputs, return_feat=False):
        """Forward function."""
        # outputs = []
        x = self._transform_inputs(inputs)
        out = self.dconv1_1(x)
        out = self.dconv1_2(out)
        # outputs.append(out)

        out = resize(
            out,
            scale_factor=2,
            mode='nearest',
            align_corners=None)
        out = self.dconv2_1(out)
        out = self.dconv2_2(out)
        # outputs.append(out)

        out = resize(
            out,
            scale_factor=2,
            mode='nearest',
            align_corners=None)
        out = self.dconv3_1(out)
        out = self.dconv3_2(out)
        # outputs.append(out)

        # out = resize(
        #     out,
        #     scale_factor=2,
        #     mode='nearest',
        #     align_corners=None)
        # out = self.dconv4_1(out)
        # out = self.dconv4_2(out)

        out_y = self.reg(out)
        # out = self.sigmoid(out)
        # outputs.append(out)

        if return_feat:
            return out_y, out

        return out_y

    def loss(self, x: Tuple[Tensor], gt_disp: Tensor, pixel_weight=None, return_feat=False) -> dict:
        """Perform forward propagation and loss calculation of the disparity completion
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        loss_dict = dict()

        if return_feat:
            pred_disp, feat = self(x, return_feat=return_feat)
            loss_dict['disp_feat'] = feat
        else:
            pred_disp = self(x, return_feat=return_feat)

        pred_disp = resize(
            input=pred_disp,
            size=gt_disp.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        gt_disp = gt_disp.squeeze(1)
        loss_dict['loss_disp'] = self.loss_func(
            pred_disp,
            gt_disp,
            mask=pixel_weight)
        loss_dict['pred_disp'] = pred_disp.detach()

        return loss_dict


if __name__ == '__main__':
    in_channels = [128, 128, 128]
    disp_head = DispHeadV1(
        in_channels=128,
        channels=512,
        out_channels=1,
        in_index=2,
        # input_transform='resize_concat'
    )

    inputs = [torch.randn(2, c, (len(in_channels)-i)*64, (len(in_channels)-i)*128) for i, c in enumerate(in_channels)]
    out = disp_head(inputs)
    print(out.shape)