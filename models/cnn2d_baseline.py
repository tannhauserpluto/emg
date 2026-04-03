from __future__ import annotations

import torch
from torch import nn

from .xception2d import ensure_pseudo_image_batch


class CNN2DBaseline(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        base_channels: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        base_channels = int(base_channels)
        hidden_1 = int(base_channels)
        hidden_2 = int(base_channels * 2)
        hidden_3 = int(base_channels * 4)

        self.num_channels = int(num_channels)
        self.features = nn.Sequential(
            nn.Conv2d(1, hidden_1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(hidden_1, hidden_2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(hidden_2, hidden_3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_3),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.output_dim = int(hidden_3)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.output_dim, int(num_classes)),
        )

    def extract_features(self, window_batch: torch.Tensor) -> torch.Tensor:
        pseudo_image = ensure_pseudo_image_batch(window_batch, expected_channels=self.num_channels)
        return self.features(pseudo_image).contiguous()

    def forward(self, window_batch: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(window_batch)
        return self.classifier(features)
