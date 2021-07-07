"""
    Author: Aaron Liu
    Email: tl254@duke.edu
    Created on: July 2 2021
    Code structure reference: https://github.com/milesial/Pytorch-UNet
"""
import torch
import torch.nn as nn


class Stem(nn.Module):
    """
          Input (x)
             |
    Conv2d only expanding
    (for cross stage partial design)
             |
        Part1, Part2
         /       |
        /        |__________________________
       /         |                          |
      /      BatchNorm2d                    |
     /          ReLU                        |
    |          Conv2d                       |
    |            |                       Shortcut
    |            |                          |
    |   Squeeze and Excite (SE)             |
    |            |__________________________|
    |            |
    |        Transition
    |            |
    |______ _____|
           |
           |
       Transition
           |
         Output

    """

    def __init__(self, in_channels, out_channels, expand_ratio=1.0):
        super(Stem, self).__init__()
        assert expand_ratio > 0.0
        exp_channels = int(round(out_channels * expand_ratio))
        self.expand_layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                exp_channels,
                kernel_size=1,
                bias=False,
            ),
        )
        in_channels = exp_channels // 2
        self.se = SE(in_channels)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                groups=in_channels,
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
            (self.se(self.conv(part2)) + part2)
        ).contiguous()
        x = self.transition(torch.cat([part1, part2], dim=1))

        return x


class CSPLevelBlock(nn.Module):
    """
          Input (x)
             |
         Expanding
         (for cross stage partial design)
             |
        Part1, Part2
         /       |
        /        |__________________________
       /         |                          |
      /      BatchNorm2d                    |
     /          ReLU                        |
    |          Conv2d                       |
    |            |                       Shortcut
    |            |                          |
    |   Squeeze and Excite (SE)             |
    |            |__________________________|
    |            |
    |        Transition
    |            |
    |______ _____|
           |
           |
       Transition
           |
         Output

    """

    def __init__(self, in_channels, out_channels, expand_ratio=1.0):
        super(CSPLevelBlock, self).__init__()
        assert expand_ratio > 0.0
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
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                groups=in_channels,
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
            (self.se(self.conv(part2)) + part2)
        ).contiguous()
        x = self.transition(torch.cat([part1, part2], dim=1))

        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=1.0):
        super(Down, self).__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.level_block = CSPLevelBlock(in_channels, out_channels, expand_ratio)

    def forward(self, x):
        x = self.max_pool(x)
        x = self.level_block(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=1.0):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.BatchNorm2d(in_channels),
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


class SE(nn.Module):
    """Squeeze and Excitation on means * variances"""

    def __init__(self, in_channels, se_ratio=0.5):
        super(SE, self).__init__()
        num_squeezed_channels = max(1, int(in_channels * se_ratio))
        self.hs = nn.Hardsigmoid(inplace=True)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.reduce_expand = nn.Sequential(
            nn.Conv2d(in_channels, num_squeezed_channels, 1),
            nn.Mish(inplace=True),
            nn.Conv2d(num_squeezed_channels, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.hs(x)
        means = self.squeeze(x)
        variances = self.squeeze((x - means) ** 2)
        attn = self.reduce_expand(means * variances)
        return x * attn
