from __future__ import annotations

import torch
from torch import nn

from .xception2d import SeparableConv2d, XceptionBlock, ensure_pseudo_image_batch


class DualResXception2D(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        base_channels: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        base_channels = int(base_channels)
        mid_channels = int(base_channels * 2)
        high_channels = int(base_channels * 4)
        final_channels = int(base_channels * 6)

        self.num_channels = int(num_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=False),
        )
        self.block1 = XceptionBlock(base_channels, mid_channels, repeats=2, stride=(2, 1), dropout=dropout)
        self.block2 = XceptionBlock(mid_channels, high_channels, repeats=2, stride=(2, 2), dropout=dropout)
        self.block3 = XceptionBlock(high_channels, high_channels, repeats=2, stride=(1, 1), dropout=dropout)
        self.skip1 = nn.Sequential(
            nn.Conv2d(mid_channels, high_channels, kernel_size=1, stride=(2, 2), bias=False),
            nn.BatchNorm2d(high_channels),
        )
        self.exit = nn.Sequential(
            SeparableConv2d(high_channels, final_channels, kernel_size=3, stride=(1, 1), padding=1),
            nn.BatchNorm2d(final_channels),
            nn.ReLU(inplace=False),
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(high_channels, final_channels, kernel_size=1, stride=(1, 1), bias=False),
            nn.BatchNorm2d(final_channels),
        )
        self.post_residual = nn.ReLU(inplace=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = int(final_channels)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.output_dim, int(num_classes)),
        )

    def extract_features(self, window_batch: torch.Tensor) -> torch.Tensor:
        features = ensure_pseudo_image_batch(window_batch, expected_channels=self.num_channels)
        stem_features = self.stem(features)
        block1_features = self.block1(stem_features)
        block2_features = self.block2(block1_features)
        block3_features = self.block3(block2_features)
        block3_features = self.post_residual(block3_features + self.skip1(block1_features))
        exit_features = self.exit(block3_features)
        exit_features = self.post_residual(exit_features + self.skip2(block2_features))
        return self.pool(exit_features).contiguous()

    def forward(self, window_batch: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(window_batch)
        return self.classifier(features)
