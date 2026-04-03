from __future__ import annotations

import torch
from torch import nn


class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        conv_channels: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        hidden_channels = int(conv_channels * 2)
        self.frontend = nn.Sequential(
            nn.Conv1d(num_channels, conv_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(conv_channels, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.temporal_model = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=float(dropout) if lstm_layers > 1 else 0.0,
        )
        lstm_output_dim = int(lstm_hidden * (2 if bidirectional else 1))
        self.classifier = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(lstm_output_dim, num_classes),
        )

    def forward(self, window_batch: torch.Tensor) -> torch.Tensor:
        features = window_batch.transpose(1, 2)
        features = self.frontend(features)
        features = features.transpose(1, 2)
        temporal_output, _ = self.temporal_model(features)
        pooled_output = temporal_output[:, -1, :]
        return self.classifier(pooled_output)
