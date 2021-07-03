"""
   Author: Aaron Liu
   Email: tl254@duke.edu
   Created on: July 2 2021
"""

import torch.nn as nn
import torch.nn.functional as F

from .cspresunet_parts import InConv, Down


class CSPResUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(CSPResUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoding
        self.stem = InConv(1, 64)
        self.down1 = Down(64, 128, stride=(2, 1))
        self.down2 = Down(128, 256, stride=(2, 1))
        self.down3 = Down(256, 512, stride=(2, 1))
        
        # Decoding
        self.up1 = UpSamplingConcatenate(512, 256)
        self.shortcut5 = nn.Conv2d(512, 256, 1)
        self.level5 = LevelBlock(512, 256, stride=(1, 1))
        self.up2 = UpSamplingConcatenate(256, 128)
        self.shortcut6 = nn.Conv2d(256, 128, 1)
        self.level6 = LevelBlock(256, 128, stride=(1, 1))
        self.up3 = UpSamplingConcatenate(128, 64)
        self.shortcut7 = nn.Conv2d(128, 64, 1)
        self.level7 = LevelBlock(128, 64, stride=(1, 1))

        self.outconv = nn.Conv2d(64, n_classes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoding
        x1 = self.stem(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # # Decoding
        # x_cat = self.up1(x4, x4_in)
        # x5 = self.level5(x_cat)

        # x_cat = self.up2(x5 + self.shortcut5(x_cat), x3_in)
        # x6 = self.level6(x_cat)

        # x_cat = self.up3(x6 + self.shortcut6(x_cat), x1)
        # x7 = self.level7(x_cat)

        # x = self.outconv(x7 + self.shortcut7(x_cat))
        x = self.sigmoid(x4)

        return x
