# Copyright (c) OpenMMLab. All rights reserved.
from .kl_loss import KLGridLoss, KLMCLoss
from .l2_loss import L2Loss
from .multipos_cross_entropy_loss import MultiPosCrossEntropyLoss
from .triplet_loss import TripletLoss
from .depth_loss import BerHuLoss

__all__ = [
    'TripletLoss', 'MultiPosCrossEntropyLoss', 'L2Loss', 'KLGridLoss',
    'KLMCLoss', 'BerHuLoss'
]
