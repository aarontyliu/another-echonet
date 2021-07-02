"""
   Author: Aaron Liu
   Email: tl254@duke.edu
   Created on: July 2 2021
   Code structure reference: https://github.com/milesial/Pytorch-UNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        t=1,
        kernel_size=3,
        stride=1,
    ):
        super().__init__()
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

    def forward(self, x):

        part1, part2 = x[:, : self.half], x[:, self.half :]
        out = self.activation(self.bn1(self.expansion(part2)))
        out = self.activation(self.bn2(self.dwise(out)))
        out = self.bn3(self.projection(out))

        if self.skip_connection:
            out = torch.cat([out, part1], dim=1)
        return out


class TransposeMBConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        t=1,
        kernel_size=3,
        stride=1,
    ):
        super().__init__()
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
        self.dwise = nn.ConvTranspose2d(
            self.mid_channels,
            self.mid_channels,
            groups=self.mid_channels - self.half,
            kernel_size=self.kernel_size,
            bias=False,
            stride=self.stride,
        )
        self.bn2 = nn.BatchNorm2d(self.mid_channels)
        self.projection = nn.Conv2d(
            self.mid_channels,
            self.out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):

        part1, part2 = x[:, : self.half], x[:, self.half :]
        out = self.activation(self.bn1(self.expansion(part2)))
        out = self.activation(self.bn2(self.dwise(out)))
        out = self.bn3(self.projection(out))

        if self.skip_connection:
            out = torch.cat([out, part1], dim=1)
        return out


class UpSamplingConcatenate(nn.Module):
    """Upscaling"""

    def __init__(self, in_channels, out_channels, bottleneck):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bottleneck = bottleneck

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.bottleneck(x)
