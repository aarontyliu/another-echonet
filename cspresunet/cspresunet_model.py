"""
    Author: Aaron Liu
    Email: tl254@duke.edu
    Created on: July 3 2021
    Reference papers:
        - U-Net: https://arxiv.org/pdf/1505.04597.pdf
        - Deep Residual U-Net: https://arxiv.org/pdf/1711.10684.pdf
        - CSPNet: https://arxiv.org/pdf/1911.11929.pdf
        - Squeeze-and-Excitation Networks: https://arxiv.org/pdf/1709.01507.pdf
        - Batch Normalization: https://arxiv.org/pdf/1502.03167.pdf
"""
from torch import nn

from .cspresunet_parts import Down, Stem, Up


class CSPResUNet(nn.Module):
    def __init__(self, n_channels, n_classes, expand_ratio=4.0):
        super(CSPResUNet, self).__init__()
        self.stem = Stem(n_channels, 16, expand_ratio)
        self.down1 = Down(16, 32, expand_ratio)
        self.down2 = Down(32, 64, expand_ratio)
        self.down3 = Down(64, 128, expand_ratio)

        self.up1 = Up(128, 64, expand_ratio)
        self.up2 = Up(64, 32, expand_ratio)
        self.up3 = Up(32, 16, expand_ratio)
        self.outconv = nn.Conv2d(16, n_classes, 1)

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
