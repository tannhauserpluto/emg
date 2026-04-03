from __future__ import annotations

import torch
from torch import nn


class FeatureMLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        embedding_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_size)),
            nn.BatchNorm1d(int(hidden_size)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), int(embedding_dim)),
            nn.GELU(),
        )
        self.output_dim = int(embedding_dim)

    def forward(self, feature_batch: torch.Tensor) -> torch.Tensor:
        return self.network(feature_batch)
