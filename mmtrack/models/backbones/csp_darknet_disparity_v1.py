from typing import List, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.models.backbones.csp_darknet import CSPLayer, Focus
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.models.layers import SPPFBottleneck
from mmyolo.models.utils import make_divisible, make_round

from .det_backbone.base_backbone_disparity_mmyolo import BaseBackbone_Disparity_MMYOLO
from mmtrack.registry import MODELS


@MODELS.register_module()
class YOLOXCSPDarknet_Disparity_V1_MMYOLO(BaseBackbone_Disparity_MMYOLO):
    """CSP-Darknet backbone used in YOLOX.

    Args:
        arch (str): Architecture of CSP-Darknet, from {P5, P6}.
            Defaults to P5.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        input_channels (int): Number of input image channels. Defaults to 3.
        out_indices (Tuple[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        spp_kernal_sizes: (tuple[int]): Sequential of kernel sizes of SPP
            layers. Defaults to (5, 9, 13).
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (Union[dict,list[dict]], optional): Initialization config
            dict. Defaults to None.
    Example:
        >>> from mmyolo.models import YOLOXCSPDarknet
        >>> import torch
        >>> model = YOLOXCSPDarknet()
        >>> model.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = model(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 1024, 3, False, True]],
    }

    def __init__(self,
                 out_fd: bool = False,  # if output depth features
                 arch: str = 'P5',
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 use_depthwise: bool = False,
                 spp_kernal_sizes: Tuple[int] = (5, 9, 13),
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None):
        self.use_depthwise = use_depthwise
        self.spp_kernal_sizes = spp_kernal_sizes
        super().__init__(self.arch_settings[arch], deepen_factor, widen_factor,
                         input_channels, out_indices, frozen_stages, plugins,
                         norm_cfg, act_cfg, norm_eval, init_cfg)

        self.out_fd = out_fd
        self.disp_stem = self.build_stem_layer()
        self.layers.append('disp_stem')

        stage = []
        stage += self.build_stage_layer(0, self.arch_settings[arch][0])
        if plugins is not None:
            stage += self.make_stage_plugins(plugins, 0, self.arch_settings[arch][0])
        self.add_module(f'disp_stage{0 + 1}', nn.Sequential(*stage))
        self.layers.append(f'disp_stage{0 + 1}')

    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer."""
        return Focus(
            self.input_channels,
            make_divisible(64, self.widen_factor),
            kernel_size=3,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        in_channels, out_channels, num_blocks, add_identity, use_spp = setting

        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)
        stage = []
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule
        conv_layer = conv(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(conv_layer)
        if use_spp:
            spp = SPPFBottleneck(
                out_channels,
                out_channels,
                kernel_sizes=self.spp_kernal_sizes,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            stage.append(spp)
        csp_layer = CSPLayer(
            out_channels,
            out_channels,
            num_blocks=num_blocks,
            add_identity=add_identity,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(csp_layer)
        return stage

    def forward(self, x: dict) -> tuple:
        """Forward batch_inputs."""
        img = x['img']
        disp_postp = x['disp_postp']

        outs = []
        layer_stem = getattr(self, 'stem')
        o_stem = layer_stem(img)

        layer_disp_stem = getattr(self, 'disp_stem')
        o_disp_stem = layer_disp_stem(disp_postp)

        # o_disp_stem = self.cbam(o_disp_stem)
        # o_stem_attn = self.c_attn(o_stem)

        # y = (o_stem + o_disp_stem)/2.
        # if 0 in self.out_indices:
        #     outs.append(y)

        layer_stage1 = getattr(self, 'stage1')
        o_stem = layer_stage1(o_stem)
        # if 1 in self.out_indices:
        #     outs.append(o_stem)

        layer_disp_stage1 = getattr(self, 'disp_stage1')
        o_disp_stem = layer_disp_stage1(o_disp_stem)
        # if 1 in self.out_indices:
        #     outs.append(o_disp_stem)

        y = (o_stem + o_disp_stem)/2.
        if 1 in self.out_indices:
            outs.append(y)

        layer_stage2 = getattr(self, 'stage2')
        y = layer_stage2(y)
        if 2 in self.out_indices:
            outs.append(y)

        layer_stage3 = getattr(self, 'stage3')
        y = layer_stage3(y)
        if 3 in self.out_indices:
            outs.append(y)

        layer_stage4 = getattr(self, 'stage4')
        y = layer_stage4(y)
        if 4 in self.out_indices:
            outs.append(y)

        if self.out_fd:
            outs.append(o_disp_stem)

        return tuple(outs)