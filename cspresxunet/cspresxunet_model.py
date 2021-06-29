"""
   Author: Aaron Liu
   Email: tl254@duke.edu
   Created on: June 29 2021
"""

import torch.nn as nn
import torch.nn.functional as F

from .cspresxunet_parts import LevelBlock, UpSamplingConcatenate

class CSPResXUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(CSPResXUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoding
        self.level1 = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )
        self.level2 = LevelBlock(64, 128, stride=(2, 1))
        self.level3 = LevelBlock(128, 256, stride=(2, 1))
        self.level4 = LevelBlock(256, 512, stride=(2, 1))
        self.level5 = LevelBlock(512, 256, stride=(1, 1))
        self.level6 = LevelBlock(256, 128, stride=(1, 1))
        self.level7 = LevelBlock(128, 64, stride=(1, 1))

        self.up1 = UpSamplingConcatenate(512, 256)
        self.up2 = UpSamplingConcatenate(256, 128)
        self.up3 = UpSamplingConcatenate(128, 64)

        self.shortcut1 = nn.Conv2d(n_channels, 64, kernel_size=1)
        self.shortcut2 = nn.Conv2d(64, 128, kernel_size=1, stride=2)
        self.shortcut3 = nn.Conv2d(128, 256, kernel_size=1, stride=2)
        self.shortcut5 = nn.Conv2d(512, 256, kernel_size=1)
        self.shortcut6 = nn.Conv2d(256, 128, kernel_size=1)
        self.shortcut7 = nn.Conv2d(128, 64, kernel_size=1)

        self.outconv = nn.Conv2d(64, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoding
        x1 = self.level1(x)
        x2_in = x1 + self.shortcut1(x)
        x2 = self.level2(x2_in)
        x3_in = x2 + self.shortcut2(x1)
        x3 = self.level3(x3_in)
        x4_in = x3 + self.shortcut3(x2)

        # Bridge
        x4 = self.level4(x4_in)

        # Decoding
        x_cat = self.up1(x4, x4_in)
        x5 = self.level5(x_cat)
        x_cat = self.up2(x5 + self.shortcut5(x_cat), x3_in)
        x6 = self.level6(x_cat)
        x_cat = self.up3(x6 + self.shortcut6(x_cat), x2_in)
        x7 = self.level7(x_cat)
        x = self.outconv(x7 + self.shortcut7(x_cat))
        x = self.sigmoid(x)

        return x