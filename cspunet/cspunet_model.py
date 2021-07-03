"""
   Author: Aaron Liu
   Email: tl254@duke.edu
   Created on: July 2 2021
"""

import torch.nn as nn
import torch.nn.functional as F

from .cspunet_parts import MBConv, TransposeMBConv, UpSamplingConcatenate



class CSPUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(CSPUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoding
        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, 24, kernel_size=3, padding=1, bias=False, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(*(2 * [FusedMBConv(24, 24, stride=1)]))
        self.down2 = nn.Sequential(
            *(3 * [FusedMBConv(24, 24)] + [FusedMBConv(24, 48, stride=2)])
        )
        self.down3 = nn.Sequential(
            *(3 * [FusedMBConv(48, 48)] + [FusedMBConv(48, 64, stride=2)])
        )
        self.down4 = nn.Sequential(
            *(5 * [MBConv(64, 64, t=4)] + [MBConv(64, 128, stride=2, t=4)])
        )
        self.down5 = nn.Sequential(
            *(8 * [MBConv(128, 128, t=6)] + [MBConv(128, 160, stride=1, t=6)])
        )
        self.down6 = nn.Sequential(
            *(14 * [MBConv(160, 160, t=6)] + [MBConv(160, 256, stride=2, t=6)])
        )

        self.up1 = Up(256, 160)
        self.up2 = Up(160, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 48)
        self.up5 = Up(48, 24)
        self.up6 = Up(24, 24)

        self.outconv = nn.ConvTranspose2d(24, n_classes, kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoding
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)

        # Decoding
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        x = self.up6(x, x0)

        # Head
        x = self.sigmoid(self.outconv(x))

        return x
