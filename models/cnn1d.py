from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        base_filters: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        hidden_1 = int(base_filters)
        hidden_2 = int(base_filters * 2)
        hidden_3 = int(base_filters * 4)

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(num_channels, hidden_1, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(hidden_1, hidden_2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(hidden_2, hidden_3, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_3, num_classes),
        )

    def forward(self, window_batch: torch.Tensor) -> torch.Tensor:
        features = window_batch.transpose(1, 2)
        features = self.feature_extractor(features)
        return self.classifier(features)
