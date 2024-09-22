from typing import Dict
import torch
from mmdet.models.backbones import CSPDarknet as CSPDarknet_mmdet
from mmtrack.registry import MODELS


@MODELS.register_module()
class CSPDarknet(CSPDarknet_mmdet):
    def forward(self, x: Dict[str, torch.Tensor]):
        """Forward function to adapt to multiple inputs."""
        x = x['img']
        x = super().forward(x)
        return x