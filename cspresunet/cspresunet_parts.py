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
    |______ _____|
           |
           |
       Transition
           |
    Squeeze and Excite (SE)
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
            SE(out_channels),
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
    |______ _____|
           |
           |
       Transition
           |
    Squeeze and Excite (SE)
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
            SE(out_channels),
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
            nn.Mish(inplace=True),
            nn.Conv2d(num_squeezed_channels, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        means = self.squeeze(x)
        variances = self.squeeze((x - means) ** 2)
        attn = self.reduce_expand(means * variances)
        return x * attn






class SE3(nn.Module):
    """Squeeze and Excitation on means * variances"""

    def __init__(self, in_channels, se_ratio=0.5):
        super(SE3, self).__init__()
        num_squeezed_channels = max(1, int(in_channels * se_ratio))
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.view4w = nn.Sequential(Rearrange('b c h w -> b w c h'))
        self.viewbk4w = Rearrange('b w c h -> b c h w')
        self.view4h = nn.Sequential(Rearrange('b c h w -> b h c w'))
        self.viewbk4h = Rearrange('b h c w -> b c h w')
        self.sigmoid = nn.Sigmoid()
#         self.reduce_expand = nn.Sequential(
#             nn.Conv2d(in_channels, num_squeezed_channels, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_squeezed_channels, in_channels, 1),
#             nn.Sigmoid(),
#         )

    def forward(self, x):
        mean_w = self.squeeze(self.view4w(x))
        var_w = self.squeeze((self.view4w(x) - mean_w) ** 2)
        weight_w = self.sigmoid(self.viewbk4w(mean_w * var_w))
        
        mean_h = self.squeeze(self.view4h(x))
        var_h = self.squeeze((self.view4h(x) - mean_h) ** 2)
        weight_h = self.sigmoid(self.viewbk4h(mean_h * var_h))
        
        mean_c = self.squeeze(x)
        var_c = self.squeeze((x - mean_c) ** 2)
        weight_c = self.sigmoid(mean_c * var_c)
#         print(weight_h * weight_w * weight_c)
        weighted_x = self.sigmoid(weight_h * weight_w * weight_c) * x
#         mean_c = self.squeeze(weighted_x)
#         var_c = self.squeeze((x - mean_c) ** 2)
#         attn = self.reduce_expand(mean_c * var_c)
#         return x * attn
        return weighted_x