import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BaseNet


class ResBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1, drop: float = 0) -> None:
        super().__init__()
        self.dropout = nn.Dropout2d(drop)

        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += x
        out = F.relu(out)
        out = self.dropout(out)

        return out


class CustomResNet(BaseNet):
    def __init__(self, drop: float = 0, num_classes: int = 10) -> None:
        super().__init__()

        # perp layer
        self.perlayer = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(drop),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(drop),
            ResBlock(128, 128, drop=drop),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(drop),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(drop),
            ResBlock(512, 512, drop=drop),
        )
        self.pool = nn.MaxPool2d(4)
        self.out = nn.Conv2d(512, num_classes, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.perlayer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = self.out(x)

        return x.view(-1, 10)
