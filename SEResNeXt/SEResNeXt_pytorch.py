from torchsummary import summary
from torch import nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEResNeXtUnit(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, cardinality=32):
        super().__init__()

        self.conv1x1_1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv3x3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                                 stride=stride, groups=cardinality, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv1x1_2 = nn.Conv2d(
            mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.seBlock = SEBlock(out_channels)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, input):
        x = input
        x = self.conv1x1_1(x)
        x = F.relu(self.bn1(x), inplace=True)

        x = self.conv3x3(x)
        x = F.relu(self.bn2(x), inplace=True)

        x = self.conv1x1_2(x)
        x = F.relu(self.bn3(x), inplace=True)

        x = self.seBlock(x)

        return F.relu(self.bn4(x + self.shortcut(input)), inplace=True)


class SEResNeXt(nn.Module):
    def __init__(self, units):
        super().__init__()
        self.conv7x7 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn0 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage_1 = self.stage(64, 128, 256, units[0])
        self.stage_2 = self.stage(256, 256, 512, units[1], 2)
        self.stage_3 = self.stage(512, 512, 1024, units[2], 2)
        self.stage_4 = self.stage(1024, 1024, 2048, units[3], 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Linear(2048, 1000)

    def forward(self, input):
        x = input
        x = self.conv7x7(x)
        x = F.relu(self.bn0(x), inplace=True)
        x = self.maxpool(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.gap(x)
        x = x.view(-1, 2048)
        x = self.dense(x)
        return x

    @staticmethod
    def stage(in_channels, mid_channels, out_channels, units, stride=1):
        layers = [SEResNeXtUnit(
            in_channels, mid_channels, out_channels, stride)]
        for _ in range(1, units):
            layers.append(SEResNeXtUnit(
                out_channels, mid_channels, out_channels, stride=1))
        layers = tuple(layers)
        return nn.Sequential(*layers)


if __name__ == '__main__':
    device = "cpu"
    SEResNeXt50 = SEResNeXt([3, 4, 6, 3])
    summary(SEResNeXt50, (3, 224, 224), device=device)

    SEResNeXt101 = SEResNeXt([3, 4, 23, 3])
    summary(SEResNeXt101, (3, 224, 224), device=device)
