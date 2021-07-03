"""
   Author: Aaron Liu
   Email: tl254@duke.edu
   Created on: July 2 2021
   Code structure reference: https://github.com/milesial/Pytorch-UNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, se=True):
        super().__init__()
        self.se = se
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = nn.ReLU(inplace=True)
        self.skip_connection = (
            self.stride == 1 and self.in_channels == self.out_channels
        )
        self.half = 0
        if self.skip_connection:
            self.half = self.in_channels // 2

        self.conv = nn.Conv2d(
            self.in_channels - self.half,
            4 * (self.in_channels - self.half),
            kernel_size=3,
            bias=False,
            padding=self.kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm2d(4 * (self.in_channels - self.half))
        if self.se:
            squeezed_channels = 4 * (self.in_channels - self.half)
            self.reduce = nn.Conv2d(
                in_channels=4 * (self.in_channels - self.half),
                out_channels=squeezed_channels,
                kernel_size=1,
            )
            self.expand = nn.Conv2d(
                in_channels=squeezed_channels,
                out_channels=4 * (self.in_channels - self.half),
                kernel_size=1,
            )

        self.projection = nn.Conv2d(
            4 * (self.in_channels - self.half),
            self.out_channels - self.half,
            kernel_size=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self.out_channels - self.half)

    def forward(self, x):
        part1, part2 = x[:, : self.half], x[:, self.half :]
        out = self.activation(self.bn1(self.conv(part2)))

        # squeeze and excitation block
        if self.se:
            out_squeezed = F.adaptive_avg_pool2d(out, 1)
            out_squeezed = F.relu(self.reduce(out_squeezed))
            out_squeezed = self.expand(out_squeezed)
            out = torch.sigmoid(out_squeezed) * out

        out = self.bn2(self.projection(out))

        if self.skip_connection:
            out = torch.cat([out + part2, part1], dim=1)
        return out


class MBConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, t=4, kernel_size=3, stride=1, se=True
    ):
        super().__init__()
        self.se = se
        self.t = t
        self.in_channels = in_channels
        self.mid_channels = self.in_channels * self.t
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = nn.ReLU(inplace=True)
        self.skip_connection = (
            self.stride == 1 and self.in_channels == self.out_channels
        )
        self.half = 0
        if self.skip_connection:
            self.half = self.in_channels // 2
            self.expansion = nn.Conv2d(
                self.half, self.mid_channels - self.half, kernel_size=1, bias=False
            )
        else:
            self.expansion = nn.Conv2d(
                self.in_channels, self.mid_channels, kernel_size=1, bias=False
            )
        self.bn1 = nn.BatchNorm2d(self.mid_channels - self.half)
        self.dwise = nn.Conv2d(
            self.mid_channels - self.half,
            self.mid_channels - self.half,
            groups=self.mid_channels - self.half,
            kernel_size=self.kernel_size,
            bias=False,
            stride=self.stride,
            padding=self.kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm2d(self.mid_channels - self.half)
        self.projection = nn.Conv2d(
            self.mid_channels - self.half,
            self.out_channels - self.half,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.out_channels - self.half)
        if self.se:
            se_ratio = 0.25
            squeezed_channels = int((self.mid_channels - self.half) * se_ratio)
            self.reduce = nn.Conv2d(
                in_channels=self.mid_channels - self.half,
                out_channels=squeezed_channels,
                kernel_size=1,
            )
            self.expand = nn.Conv2d(
                in_channels=squeezed_channels,
                out_channels=self.mid_channels - self.half,
                kernel_size=1,
            )

    def forward(self, x):

        part1, part2 = x[:, : self.half], x[:, self.half :]
        out = self.activation(self.bn1(self.expansion(part2)))
        out = self.activation(self.bn2(self.dwise(out)))

        # squeeze and excitation block
        if self.se:
            out_squeezed = F.adaptive_avg_pool2d(out, 1)
            out_squeezed = F.relu(self.reduce(out_squeezed))
            out_squeezed = self.expand(out_squeezed)
            out = torch.sigmoid(out_squeezed) * out

        out = self.bn3(self.projection(out))

        if self.skip_connection:
            out = torch.cat([out + part2, part1], dim=1)
        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out + self.shortcut(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        #         self.half = out_channels
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(2 * out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)