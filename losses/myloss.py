import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import losses

@losses.register('mnist_ce')
class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, pred, batch, **kwargs):
        return {
            'loss': self.loss_fn(pred, batch['label'])
        }

@losses.register('liif_l1')
class l1liif(nn.Module):
    def __init__(self, data_norm):
        super().__init__()
        self.loss_fn = nn.L1Loss()
        self._gt_div = data_norm['gt']['div']
        self._gt_sub = data_norm['gt']['sub']

    def forward(self, pred, batch, **kwargs):
        ret = {
            'loss': self.loss_fn(pred, (batch['gt'] - self._gt_sub) / self._gt_div)
        }
        return ret

    def psnr(self, pred, batch, **kwargs):
        pred = pred * self._gt_div + self._gt_sub
        pred.clamp_(0, 1)
        diff = pred - batch['gt']
        mse = diff.pow(2).mean()
        return -10 * torch.log10(mse)