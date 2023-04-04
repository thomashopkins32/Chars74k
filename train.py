import numpy as np
import pandas as pd
import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision

from .dataset import Chars74k

NUM_CLASSES = 62


class BaselineCNN(nn.Module):
    def __init__(self):
        # 20 x 20 x 3
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1),
        # 20 x 20 x 16
        self.maxp1 = nn.MaxPool2d(2, stride=2),
        # 19 x 19 x 3
        self.bn1 = nn.BatchNorm2d(16),
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1),
        self.maxp2 = nn.MaxPool2d(2, stride=2),
        self.bn2 = nn.BatchNorm2d(32),
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1),
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1, 64)),
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        assert x.shape[1:] == (20, 20, 3)
        x = self.conv1(x)
        assert x.shape[1:] == (20, 20, 16)
        x = self.maxp1(x)
        assert x.shape[1:] == (10, 10, 16)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        assert x.shape[1:] == (10, 10, 32)
        x = self.maxp2(x)
        assert x.shape[1:] == (5, 5, 32)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        assert x.shape[1:] == (5, 5, 64)
        x = self.avg_pool(x)
        assert x.shape[1:] == (1, 1, 64)
        x = F.relu(x)
        x = self.flatten(x)
        assert x.shape[1:] == (64,)
        x = self.linear(x)
        assert(x.shape[1:] == NUM_CLASSES)
        return x

train = Chars74k(test=False)