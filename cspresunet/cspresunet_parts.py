"""
   Author: Aaron Liu
   Email: tl254@duke.edu
   Created on: July 2 2021
   Code structure reference: https://github.com/milesial/Pytorch-UNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LevelBlock(nn.Module):
    """(BN ==> ReLU ==> Conv) x 2"""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        stride=(1, 1),
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.stride = stride
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                stride=self.stride[0],
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=self.stride[1],
                bias=False,
            ),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, stride=(1,1)):
        super().__init__()
        self.stride = stride
        self.level_block = LevelBlock(in_channels, out_channels, stride=self.stride)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.level_block(x) + self.shortcut(x)
        return  x


class UpSamplingConcatenate(nn.Module):
    """Upscaling"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        return x




class InConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.stem_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.stem_block(x) + self.shortcut(x)
        return x