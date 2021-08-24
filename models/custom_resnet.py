import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, resblock=True):
        super(BasicBlock, self).__init__()

        self.has_resblock = resblock

        self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.mp = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(planes)

        if self.has_resblock:
            self.conv1 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = F.relu(self.bn(self.mp(self.conv(x))))
        if self.has_resblock:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            r1 = out + x  # This is the residual block
            return x + r1
        else:
            return x


class MyResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(MyResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=1, resblock=False)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=1)
        self.pool   = nn.MaxPool2d(4)

        self.linear1 = nn.Linear(512 * block.expansion, 1024)
        self.do1 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(1024, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, resblock=True):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, resblock))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)

        out = out.view(out.size(0), -1)
        out = (self.linear2((F.relu(self.linear1(out)))))

        
        return F.softmax(out, dim=-1)


def customResnet():
    return MyResNet(BasicBlock, [1, 1, 1])

