import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import models
import losses
from models import register

from torchvision.models import resnet18

@register('resnet18')
class resnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        model = resnet18(num_classes=num_classes)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model = model
    
    def forward(self, batch):
        return self.model(batch['img'])
        


