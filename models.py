import torch.nn as nn
import torch.nn.functional as F

# Dataset specific configurations
NUM_CLASSES = 62


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 20 x 20 x 3
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        # 20 x 20 x 16
        self.maxp1 = nn.MaxPool2d(2, stride=2)
        # 19 x 19 x 3
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.maxp2 = nn.MaxPool2d(2, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.avg_pool(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x