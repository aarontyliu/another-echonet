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
from einops.layers.torch import Rearrange


class CSPLevelBlock(nn.Module):
    """(BN ==> ReLU ==> Conv) x 2 + CSP pre-activation shortcut"""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=(1, 1),
        expand_ratio=1.0,
    ):
        super().__init__()
        self.stride = stride
        self.channeling = MixPooling() if stride[0] == 2 else nn.Identity()
        exp_channels = int(round(out_channels * expand_ratio))
        self.expand_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Mish(inplace=True),
            nn.Conv2d(
                in_channels,
                exp_channels,
                kernel_size=1,
                stride=self.stride[1],
                bias=False,
            ),
        )
        in_channels = exp_channels // 2
        self.stride = stride
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Mish(inplace=True),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                stride=self.stride[1],
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.Mish(inplace=True),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                stride=self.stride[1],
                bias=False,
            ),
        )

        self.partial_trans2 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Mish(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
        )

        self.partial_trans_head = nn.Sequential(
            nn.BatchNorm2d(exp_channels),
            nn.Mish(inplace=True),
            nn.Conv2d(exp_channels, out_channels, 1, bias=False),
        )

    def forward(self, x):
        x = self.channeling(x)
        x = self.expand_layer(x)
        half = x.size(1) // 2
        part1, part2 = x[:, :half], x[:, half:]
        part2 = self.partial_trans2((self.double_conv(part2) + part2)).contiguous()
        x = self.partial_trans_head(torch.cat([part1, part2], dim=1))

        return x


class LevelBlock(nn.Module):
    """(BN ==> ReLU ==> Conv) x 2 + pre-activation shortcut"""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        stride=(1, 1),
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.stride = stride
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                stride=self.stride[0],
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=self.stride[1],
                bias=False,
            ),
        )
        self.shortcut = nn.Conv2d(
            in_channels, out_channels, 1, stride=self.stride[0], bias=False
        )

    def forward(self, x):
        x = self.double_conv(x) + self.shortcut(x)
        return x


class Down(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride=(2, 1), use_csp=True, expand_ratio=1.0
    ):
        super().__init__()
        self.stride = stride
        if use_csp:
            self.level_block = CSPLevelBlock(
                in_channels, out_channels, self.stride, expand_ratio
            )
        else:
            self.level_block = LevelBlock(in_channels, out_channels, stride=self.stride)

    def forward(self, x):
        x = self.level_block(x)
        return x


class Up(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride=(1, 1), use_csp=True, expand_ratio=1.0
    ):
        super().__init__()
        self.stride = stride
        self.up = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Mish(inplace=True),
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2, bias=False
            ),
        )
        if use_csp:
            self.level_block = CSPLevelBlock(
                in_channels, out_channels, self.stride, expand_ratio
            )
        else:
            self.level_block = LevelBlock(in_channels, out_channels, stride=self.stride)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.level_block(x)
        return x


class Stem(nn.Module):
    def __init__(self, in_channels, out_channels, use_csp=True, expand_ratio=1.0):
        super().__init__()

        self.use_csp = 0
        if use_csp:
            self.use_csp = 1
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
            self.stem_block = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.Mish(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Mish(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            )

            self.partial_trans2 = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.Mish(inplace=True),
                nn.Conv2d(in_channels, in_channels, 1, bias=False),
            )

            self.partial_trans_head = nn.Sequential(
                nn.BatchNorm2d(exp_channels),
                nn.Mish(inplace=True),
                nn.Conv2d(exp_channels, out_channels, 1, bias=False),
            )
        else:
            self.stem_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Mish(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            )
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        if self.use_csp:
            x = self.expand_layer(x)
            half = x.size(1) // 2
            part1, part2 = x[:, :half], x[:, half:]
            part2 = self.partial_trans2((self.stem_block(part2) + part2)).contiguous()
            x = self.partial_trans_head(torch.cat([part1, part2], dim=1))
        else:
            x = self.stem_block(x) + self.shortcut(x)
        return x


class MixPooling(nn.Module):
    def __init__(self, kernel_size=2):
        super().__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size)
        self.avg_pool2d = nn.AvgPool2d(kernel_size)
        self.gamma = nn.Parameter(torch.Tensor([0.5]))

    def forward(self, x):
        gamma = torch.sigmoid(self.gamma)
        return gamma * self.max_pool2d(x) + (1 - gamma) * self.avg_pool2d(x)


class DetailPreservegPool(nn.Module):
    """Detailed-Preserving Pooling"""

    def __init__(self, kernel_size=2, stride=None, padding=0, variant="sym"):
        super().__init__()
        self.filter = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, bias=False
        )
        self.k = _pair(kernel_size)
        if not stride:
            self.stride = _pair(kernel_size)
        else:
            self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b

    def _padding(self, x):
        padding = self.padding
        return padding

    def _pho(self, x, beta=0.5, variant="sym"):
        if variant == "sym":
            return torch.sqrt(x ** 2 + 1e-6) ** beta

    def forward(self, I):
        I_ = self.filter(I)

        x = F.pad(x, self._padding(x), mode="reflect")
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


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
