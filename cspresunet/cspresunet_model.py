"""
   Author: Aaron Liu
   Email: tl254@duke.edu
   Created on: July 3 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cspresunet_parts import Down, Stem, Up


class CSPResUNet(nn.Module):
    def __init__(self, n_channels, n_classes, use_csp=True, expand_ratio=1.0):
        super(CSPResUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.stem = Stem(n_channels, 64, use_csp, expand_ratio)
        self.down1 = Down(64, 128, (2, 1), use_csp, expand_ratio)
        self.down2 = Down(128, 256, (2, 1), use_csp, expand_ratio)
        self.down3 = Down(256, 512, (2, 1), use_csp, expand_ratio)
        self.up1 = Up(512, 256, (1, 1), use_csp, expand_ratio)
        self.up2 = Up(256, 128, (1, 1), use_csp, expand_ratio)
        self.up3 = Up(128, 64, (1, 1), use_csp, expand_ratio)

        self.outconv = nn.Conv2d(64, n_classes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outconv(x)
        x = self.sigmoid(x)

        return x
