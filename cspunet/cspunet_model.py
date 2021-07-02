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
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.down1 = MBConv(16, 16, t=1, kernel_size=3, stride=1)
        self.down2 = nn.Sequential(
            MBConv(16, 24, t=6, kernel_size=3, stride=2),
            MBConv(24, 24, t=6, kernel_size=3, stride=1),
        )
        self.down3 = nn.Sequential(
            MBConv(24, 40, t=6, kernel_size=5, stride=2),
            MBConv(40, 40, t=6, kernel_size=5, stride=1),
        )
        self.down4 = nn.Sequential(
            MBConv(40, 80, t=6, kernel_size=3, stride=2),
            MBConv(80, 80, t=6, kernel_size=3, stride=1),
            MBConv(80, 80, t=6, kernel_size=3, stride=1),
        )
        self.down5 = nn.Sequential(
            MBConv(80, 112, t=6, kernel_size=5, stride=1),
            MBConv(112, 112, t=6, kernel_size=5, stride=1),
            MBConv(112, 112, t=6, kernel_size=5, stride=1),
        )
        self.down6 = nn.Sequential(
            MBConv(112, 192, t=6, kernel_size=5, stride=2),
            MBConv(192, 192, t=6, kernel_size=5, stride=1),
            MBConv(192, 192, t=6, kernel_size=5, stride=1),
            MBConv(192, 192, t=6, kernel_size=5, stride=1),
        )
        self.down7 = MBConv(192, 320)

        self.up1 = UpSamplingConcatenate(320, 192, bottleneck=MBConv(384, 192))
        self.up2 = UpSamplingConcatenate(
            192,
            112,
            bottleneck=nn.Sequential(
                MBConv(224, 112, t=6, kernel_size=5, stride=2),
                MBConv(112, 112, t=6, kernel_size=5, stride=1),
                MBConv(112, 112, t=6, kernel_size=5, stride=1),
                MBConv(112, 112, t=6, kernel_size=5, stride=1),
            ),
        )
        self.up3 = UpSamplingConcatenate(
            112,
            80,
            bottleneck=nn.Sequential(
                MBConv(160, 80, t=6, kernel_size=5, stride=1),
                MBConv(80, 80, t=6, kernel_size=5, stride=1),
                MBConv(80, 80, t=6, kernel_size=5, stride=1),
            ),
        )

        self.up4 = UpSamplingConcatenate(
            80,
            40,
            bottleneck=nn.Sequential(
                MBConv(80, 40, t=6, kernel_size=3, stride=2),
                MBConv(40, 40, t=6, kernel_size=3, stride=1),
                MBConv(40, 40, t=6, kernel_size=3, stride=1),
            ),
        )

        self.up5 = UpSamplingConcatenate(
            40,
            24,
            bottleneck=nn.Sequential(
                MBConv(48, 24, t=6, kernel_size=5, stride=2),
                MBConv(24, 24, t=6, kernel_size=5, stride=1),
            ),
        )

        self.up6 = UpSamplingConcatenate(
            24,
            16,
            bottleneck=nn.Sequential(
                MBConv(32, 16, t=6, kernel_size=3, stride=2),
                MBConv(16, 16, t=6, kernel_size=3, stride=1),
            ),
        )
        self.up7 = TransposeMBConv(16, 16, stride=2, kernel_size=2)
        

        # Final
        self.outconv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.001),
            nn.Conv2d(16, n_classes, kernel_size=3, padding=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoding
        x = self.conv1(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)

        # Decoding
        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        x = self.up7(x)
        x = self.outconv(x)

        # Sigmoid
        x = self.sigmoid(x)

        return x
