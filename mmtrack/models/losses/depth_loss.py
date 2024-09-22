import torch
import torch.nn as nn
import torch.nn.functional as F
from mmtrack.registry import MODELS

def loss_calc_depth(pred, label, mask):
    adiff = torch.abs(pred - label)
    adiff = adiff * mask

    batch_max = 0.2 * torch.max(adiff).item()
    t1_mask = adiff.le(batch_max).float()
    t2_mask = adiff.gt(batch_max).float()
    t1 = adiff * t1_mask
    t2 = (adiff * adiff + batch_max * batch_max) / (2 * batch_max)
    t2 = t2 * t2_mask

    return (torch.sum(t1) + torch.sum(t2)) / torch.sum(mask)

def loss_calc_depth_mse(pred, label, mask):
    mse = F.mse_loss(pred, label, reduction='none')
    mse = mse * mask

    return torch.sum(mse) / torch.sum(mask)


@MODELS.register_module()
class BerHuLoss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super(BerHuLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, label, mask):
        """Forward function."""
        n, c, h, w = pred.size()
        assert c == 1
        pred = pred.squeeze()
        if mask is None:
            mask = torch.ones_like(label)
        mask = mask.squeeze()
        if label.dim() == 4:
            assert label.shape[2] == 1
            label = label.squeeze()
        assert label.dim() == 3

        assert label.shape[-2:] == mask.shape[-2:]
        # loss_depth = self.loss_weight * loss_calc_depth(pred, label, mask)
        loss_depth = self.loss_weight * loss_calc_depth(pred, label, mask)

        return loss_depth