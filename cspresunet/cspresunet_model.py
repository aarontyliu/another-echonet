"""
    Author: Aaron Liu
    Email: tl254@duke.edu
    Created on: July 3 2021
    Reference papers:
        - U-Net: https://arxiv.org/pdf/1505.04597.pdf
        - Deep Residual U-Net: https://arxiv.org/pdf/1711.10684.pdf
        - CSPNet: https://arxiv.org/pdf/1911.11929.pdf
        - Squeeze-and-Excitation Networks: https://arxiv.org/pdf/1709.01507.pdf
        - Dropout: https://dl.acm.org/doi/10.5555/2627435.2670313
        - Rethinking the Usage of Batch Normalization and Dropout: https://arxiv.org/pdf/1905.05928.pdf
        - Batch Normalization: https://arxiv.org/pdf/1502.03167.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cspresunet_parts import IC, CSPLevelBlock, Down, Up


class CSPResUNet(nn.Module):
    def __init__(self, n_channels, n_classes, expand_ratio=1.0):
        super(CSPResUNet, self).__init__()
        self.stem = CSPLevelBlock(n_channels, 64, expand_ratio)
        self.down1 = Down(64, 128, expand_ratio)
        self.down2 = Down(128, 256, expand_ratio)
        self.down3 = Down(256, 512, expand_ratio)

        # implicit data augmentation (Ronneberger et at., 2015).
        self.up1 = Up(512, 256, expand_ratio, dropout=0.1)
        self.up2 = Up(256, 128, expand_ratio)
        self.up3 = Up(128, 64, expand_ratio)
        self.outconv = nn.Sequential(
            IC(64, 0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1),
        )

    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outconv(x)

        return x
