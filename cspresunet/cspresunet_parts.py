"""
    Author: Aaron Liu
    Email: tl254@duke.edu
    Created on: July 2 2021
    Code structure reference: https://github.com/milesial/Pytorch-UNet
"""

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce


class CSPLevelBlock(nn.Module):
    """
          Input (x)
             |
         Expanding (for cross stage partial design)
             |
        Part1, Part2
         /       |
        /        |__________________________
       /         |                          |
      /      BatchNorm2d                    |
     /          ReLU                        |
    |          Conv2d                       |
    |            |                          |
    |        BatchNorm2d              Pre-activation Shortcut
    |           ReLU                        |
    |          Conv2d                       |
    |            |                          |
    |   Squeeze and Excite (SE)             |
    |            |__________________________|
    |            |
    |        Transition
    |            |
    |______ ______|
           |
           |
       Transition
           |
         Output

    """

    def __init__(self, in_channels, out_channels, expand_ratio=1.0):
        super(CSPLevelBlock, self).__init__()
        assert expand_ratio > 0
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
        self.se = SE(in_channels)
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )

        self.transition_pt2 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
        )

        self.transition = nn.Sequential(
            nn.BatchNorm2d(exp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(exp_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        x = self.expand_layer(x)
        half = x.size(1) // 2
        part1, part2 = x[:, :half], x[:, half:]
        part2 = self.transition_pt2(
            (self.se(self.double_conv(part2)) + part2)
        ).contiguous()
        x = self.transition(torch.cat([part1, part2], dim=1))

        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=1.0):
        super(Down, self).__init__()
        self.mix_pool = MixPooling()
        self.level_block = CSPLevelBlock(in_channels, out_channels, expand_ratio)

    def forward(self, x):
        x = self.mix_pool(x)
        x = self.level_block(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=1.0, dropout=0):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            IC(in_channels, dropout),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2, bias=False
            ),
        )
        self.level_block = CSPLevelBlock(in_channels, out_channels, expand_ratio)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.level_block(x)
        return x


class MixPooling(nn.Module):
    def __init__(self, kernel_size=2):
        super(MixPooling, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size)
        self.avg_pool2d = nn.AvgPool2d(kernel_size)
        self.gamma = nn.Parameter(torch.Tensor([0.5]))

    def forward(self, x):
        gamma = torch.sigmoid(self.gamma)
        return gamma * self.max_pool2d(x) + (1 - gamma) * self.avg_pool2d(x)


class SE(nn.Module):
    """Squeeze and Excitation on means * variances"""

    def __init__(self, in_channels, se_ratio=0.5):
        super(SE, self).__init__()
        num_squeezed_channels = max(1, int(in_channels * se_ratio))
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.reduce_expand = nn.Sequential(
            nn.Conv2d(in_channels, num_squeezed_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_squeezed_channels, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        means = self.squeeze(x)
        variances = self.squeeze((x - means) ** 2)
        attn = self.reduce_expand(means * variances)
        return x * attn


class IC(nn.Module):
    """Independent component"""

    def __init__(self, in_channels, p=0.05):
        super(IC, self).__init__()
        self.ic = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.Dropout(p, inplace=True)
        )

    def forward(self, x):
        return self.ic(x)


# Experimenting
class SelfAttenion(nn.Module):
    def __init__(self, in_channels, embed_dim=None, num_heads=1):
        super().__init__()
        if not embed_dim:
            embed_dim = in_channels // 32
        self.transpose = Rearrange("b c n -> b n c")
        self.flatten = Rearrange("b c v1 v2 -> b c (v1 v2)")

        self.f = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 1), self.flatten, self.transpose
        )
        self.g = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 1), self.flatten, self.transpose
        )
        self.h = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 1), self.flatten, self.transpose
        )
        self.v = nn.Conv2d(embed_dim, in_channels, 1)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        _, _, h, w = x.size()
        query, key, value = self.f(x), self.g(x), self.h(x)
        attn_output, _ = self.multihead_attn(query, key, value)
        o = self.v(
            einops.rearrange(
                self.transpose(attn_output), "b l (h w) -> b l h w", h=h, w=w
            )
        )
        output = x + self.gamma * o

        return F.avg_pool2d(output, 2)
