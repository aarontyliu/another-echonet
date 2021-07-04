"""
   Author: Aaron Liu
   Email: tl254@duke.edu
   Created on: July 2 2021
   Code structure reference: https://github.com/milesial/Pytorch-UNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CSPLevelBlock(nn.Module):
    """(BN ==> ReLU ==> Conv) x 2 + CSP pre-activation shortcut"""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=(1, 1),
        expand_ratio=1.0,
    ):
        super().__init__()
        self.stride = stride
        exp_channels = int(round(out_channels * expand_ratio))
        self.expand_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                exp_channels,
                kernel_size=1,
                bias=False,
                stride=self.stride[0],
            ),
        )
        in_channels = exp_channels // 2
        self.stride = stride
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                stride=self.stride[1],
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                stride=self.stride[1],
                bias=False,
            ),
        )
        self.shortcut = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        self.partial_trans2 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
        )

        self.partial_trans_head = nn.Sequential(
            nn.BatchNorm2d(exp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(exp_channels, out_channels, 1, bias=False),
        )

    def forward(self, x):
        x = self.expand_layer(x)
        half = x.size(1) // 2
        part1, part2 = x[:, :half], x[:, half:]
        part2 = self.partial_trans2((self.double_conv(part2) + self.shortcut(part2)))
        x = self.partial_trans_head(torch.cat([part1, part2], dim=1))
        return x


class LevelBlock(nn.Module):
    """(BN ==> ReLU ==> Conv) x 2 + pre-activation shortcut"""

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
        self.shortcut = nn.Conv2d(
            in_channels, out_channels, 1, stride=self.stride[0], bias=False
        )

    def forward(self, x):
        x = self.double_conv(x) + self.shortcut(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, stride=(1, 1), use_csp=True):
        super().__init__()
        self.stride = stride
        if use_csp:
            self.level_block = CSPLevelBlock(
                in_channels, out_channels, stride=self.stride
            )
        else:
            self.level_block = LevelBlock(in_channels, out_channels, stride=self.stride)

    def forward(self, x):
        x = self.level_block(x)
        return x


class Up(nn.Module):
    """Upscaling"""

    def __init__(self, in_channels, out_channels, stride=(1, 1), use_csp=True):
        super().__init__()
        self.stride = stride
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        if use_csp:
            self.level_block = CSPLevelBlock(
                in_channels, out_channels, stride=self.stride
            )
        else:
            self.level_block = LevelBlock(in_channels, out_channels, stride=self.stride)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.level_block(x)
        return x


class Stem(nn.Module):
    def __init__(self, in_channels, out_channels, use_csp=True, expand_ratio=1.0):
        super().__init__()

        self.use_csp = 0
        if use_csp:
            self.use_csp = 1
            exp_channels = int(round(out_channels * expand_ratio))
            self.expand_layer = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels,
                    exp_channels,
                    kernel_size=1,
                    bias=False,
                ),
            )
            in_channels = exp_channels // 2
            self.stem_block = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            )
            self.shortcut = nn.Conv2d(in_channels, in_channels, 1, bias=False)

            self.partial_trans2 = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 1, bias=False),
            )

            self.partial_trans_head = nn.Sequential(
                nn.BatchNorm2d(exp_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(exp_channels, out_channels, 1, bias=False),
            )
        else:
            self.stem_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            )
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        if self.use_csp:
            x = self.expand_layer(x)
            half = x.size(1) // 2
            part1, part2 = x[:, :half], x[:, half:]
            print(self.stem_block(part2).shape)
            part2 = self.partial_trans2((self.stem_block(part2) + self.shortcut(part2)))
            x = self.partial_trans_head(torch.cat([part1, part2], dim=1))
        else:
            x = self.stem_block(x) + self.shortcut(x)
        return x