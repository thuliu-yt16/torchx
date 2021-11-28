import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

losses = {}

def register(name):
    def decorator(cls):
        losses[name] = cls
        return cls
    return decorator


def make(loss_spec, args=None):
    loss_args = loss_spec.get('args', {})
    args = args or {}
    loss_args.update(args)

    loss = losses[loss_spec['name']](**loss_args)
    return loss

@register('mnist_ce')
class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, pred, batch, **kwargs):
        return {
            'loss': self.loss_fn(pred, batch['label'])
        }

@register('liif_l1')
class l1liif(nn.Module):
    def __init__(self, data_norm):
        super().__init__()
        self.loss_fn = nn.L1Loss()
        self._gt_div = data_norm['gt']['div']
        self._gt_sub = data_norm['gt']['sub']

    def forward(self, pred, batch, **kwargs):
        return {
            'loss': self.loss_fn(pred, (batch['gt'] - self._gt_sub) / self._gt_div)
        }