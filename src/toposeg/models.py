from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class CompactUNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 7, base_channels: int = 24, skeleton_head: bool = True) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, int(round(base_channels * 32 / 3))]
        self.channels = channels
        self.encoders = nn.ModuleList([
            ConvBlock(in_channels, channels[0]),
            ConvBlock(channels[0], channels[1]),
            ConvBlock(channels[1], channels[2]),
            ConvBlock(channels[2], channels[3]),
        ])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(channels[3], channels[4])
        self.decoders = nn.ModuleList([
            UpBlock(channels[4], channels[3], channels[3]),
            UpBlock(channels[3], channels[2], channels[2]),
            UpBlock(channels[2], channels[1], channels[1]),
            UpBlock(channels[1], channels[0], channels[0]),
        ])
        self.region_head = nn.Conv2d(channels[0], num_classes, 1)
        self.centerline_head = nn.Conv2d(channels[0], 4, 1) if skeleton_head else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
        return {
            "region": self.region_head(x),
            "centerline": self.centerline_head(x) if self.centerline_head is not None else None,
        }
