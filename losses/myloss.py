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
