from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .feature_mlp import FeatureMLPEncoder
from .tcn import TCNEncoder


class FusionTCNMLP(nn.Module):
    def __init__(
        self,
        num_channels: int,
        feature_dim: int,
        num_classes: int,
        tcn_channels: Sequence[int] | None = None,
        tcn_hidden_channels: int = 64,
        tcn_num_blocks: int = 4,
        tcn_kernel_size: int = 5,
        feature_hidden_size: int = 256,
        feature_embedding_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.raw_encoder = TCNEncoder(
            num_channels=num_channels,
            channels=tcn_channels,
            hidden_channels=tcn_hidden_channels,
            num_blocks=tcn_num_blocks,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
        )
        self.feature_encoder = FeatureMLPEncoder(
            input_dim=feature_dim,
            hidden_size=feature_hidden_size,
            embedding_dim=feature_embedding_dim,
            dropout=dropout,
        )
        fusion_dim = int(self.raw_encoder.output_dim + self.feature_encoder.output_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(fusion_dim, int(num_classes)),
        )

    def forward(self, window_batch: torch.Tensor, feature_batch: torch.Tensor) -> torch.Tensor:
        raw_embedding = self.raw_encoder(window_batch)
        feature_embedding = self.feature_encoder(feature_batch)
        fused_embedding = torch.cat([raw_embedding, feature_embedding], dim=1)
        return self.classifier(fused_embedding)
