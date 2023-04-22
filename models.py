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


class CNNv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2, bias=False)
        self.maxp1 = nn.MaxPool2d(2, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.maxp2 = nn.MaxPool2d(2, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.maxp3 = nn.MaxPool2d(2, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.maxp4 = nn.MaxPool2d(2, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.bn1(self.maxp1(self.conv1(x))))
        x = F.relu(self.bn2(self.maxp2(self.conv2(x))))
        x = F.relu(self.bn3(self.maxp3(self.conv3(x))))
        x = F.relu(self.bn4(self.maxp4(self.conv4(x))))
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x