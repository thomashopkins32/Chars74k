import torch.nn as nn
import torch.nn.functional as F

# Dataset specific configurations
NUM_CLASSES = 62


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 20 x 20 x 3
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # 20 x 20 x 16
        self.maxp1 = nn.MaxPool2d(2, stride=2)
        # 19 x 19 x 3
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.maxp2 = nn.MaxPool2d(2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        assert x.shape[1:] == (3, 20, 20)
        x = self.conv1(x)
        assert x.shape[1:] == (16, 20, 20)
        x = self.maxp1(x)
        assert x.shape[1:] == (16, 10, 10)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        assert x.shape[1:] == (32, 10, 10)
        x = self.maxp2(x)
        assert x.shape[1:] == (32, 5, 5)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        assert x.shape[1:] == (64, 5, 5)
        x = self.avg_pool(x)
        assert x.shape[1:] == (64, 1, 1)
        x = F.relu(x)
        x = self.flatten(x)
        assert x.shape[1:] == (64,)
        x = self.linear(x)
        assert(x.shape[1:] == (NUM_CLASSES,))
        return x