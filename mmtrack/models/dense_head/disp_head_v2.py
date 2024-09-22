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
class DispHeadV2(BaseDispHead):
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
        self.dconv2_1 = ConvModule(self.channels + 64,  # 64 is channels of depth features
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

        # self.dconv3_2 = ConvModule(128,
        #                            128,
        #                            3,
        #                            padding=1,
        #                            conv_cfg=self.conv_cfg,
        #                            norm_cfg=self.norm_cfg,
        #                            act_cfg=self.act_cfg)
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

        self.cbam = CBAM(64, 4, no_spatial=True)
        # self.c_attn = ChannelAttn(make_divisible(64, self.widen_factor), 4)

    def forward(self, inputs, return_feat=False):
        """Forward function."""
        # outputs = []
        df = inputs[-1]
        x = self._transform_inputs(inputs[: -1])
        out = self.dconv1_1(x)
        out = self.dconv1_2(out)
        # outputs.append(out)

        out = resize(
            out,
            scale_factor=2,
            mode='nearest',
            align_corners=None)

        df = self.cbam(df)
        out = torch.cat((out, df), dim=1)

        out = self.dconv2_1(out)
        out = self.dconv2_2(out)
        # outputs.append(out)

        out = resize(
            out,
            scale_factor=2,
            mode='nearest',
            align_corners=None)

        out = self.dconv3_1(out)
        # out = self.dconv3_2(out)
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

class ChannelAttn(nn.Module):
    def __init__(self, gate_channels, reduction_ration=16, pool_types=['avg', 'max']):
        super(ChannelAttn, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ration),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ration, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = self._logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)

        return scale

    def _logsumexp_2d(self, tensor):
        tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
        s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
        outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
        return outputs

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ration=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ration),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ration, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = self._logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * (1-scale)

    def _logsumexp_2d(self, tensor):
        tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
        s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
        outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
        return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(ConvBnRelu, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        dilation = 2
        self.compress = ChannelPool()
        self.spatial = ConvBnRelu(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * (1 - scale)


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


if __name__ == '__main__':
    in_channels = [128, 128, 128]
    disp_head = DispHeadV2(
        in_channels=128,
        channels=512,
        out_channels=1,
        in_index=2,
        # input_transform='resize_concat'
    )

    inputs = [torch.randn(2, c, (len(in_channels)-i)*64, (len(in_channels)-i)*128) for i, c in enumerate(in_channels)]
    out = disp_head(inputs)
    print(out.shape)