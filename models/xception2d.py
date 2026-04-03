from __future__ import annotations

import torch
from torch import nn


def ensure_pseudo_image_batch(window_batch: torch.Tensor, expected_channels: int | None = None) -> torch.Tensor:
    if window_batch.ndim == 3:
        window_batch = window_batch.unsqueeze(1)
    if window_batch.ndim != 4:
        raise ValueError(f"Expected input to have shape [B, 1, T, C] or [B, T, C], got {tuple(window_batch.shape)}")
    if int(window_batch.shape[1]) != 1:
        raise ValueError(f"Expected pseudo-image input channel dimension to be 1, got {tuple(window_batch.shape)}")
    if expected_channels is not None and int(window_batch.shape[-1]) != int(expected_channels):
        raise ValueError(
            f"Expected pseudo-image width to match num_channels={expected_channels}, got {tuple(window_batch.shape)}"
        )
    return window_batch.contiguous()


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: tuple[int, int] = (1, 1),
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            int(in_channels),
            int(in_channels),
            kernel_size=int(kernel_size),
            stride=tuple(stride),
            padding=int(padding),
            groups=int(in_channels),
            bias=False,
        )
        self.pointwise = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.depthwise(inputs)
        outputs = self.pointwise(outputs)
        return outputs.contiguous()


class XceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        repeats: int = 2,
        stride: tuple[int, int] = (1, 1),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_channels = int(in_channels)
        for repeat_index in range(int(repeats)):
            block_stride = tuple(stride) if repeat_index == int(repeats) - 1 else (1, 1)
            layers.extend(
                [
                    SeparableConv2d(
                        in_channels=current_channels,
                        out_channels=int(out_channels),
                        kernel_size=3,
                        stride=block_stride,
                        padding=1,
                    ),
                    nn.BatchNorm2d(int(out_channels)),
                    nn.ReLU(inplace=False),
                ]
            )
            if float(dropout) > 0:
                layers.append(nn.Dropout2d(float(dropout)))
            current_channels = int(out_channels)
        self.main = nn.Sequential(*layers)

        if int(in_channels) == int(out_channels) and tuple(stride) == (1, 1):
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, stride=tuple(stride), bias=False),
                nn.BatchNorm2d(int(out_channels)),
            )
        self.activation = nn.ReLU(inplace=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.main(inputs)
        residual = self.residual(inputs)
        outputs = outputs + residual
        return self.activation(outputs)


class Xception2D(nn.Module):
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
        self.exit = nn.Sequential(
            SeparableConv2d(high_channels, final_channels, kernel_size=3, stride=(1, 1), padding=1),
            nn.BatchNorm2d(final_channels),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.output_dim = int(final_channels)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.output_dim, int(num_classes)),
        )

    def extract_features(self, window_batch: torch.Tensor) -> torch.Tensor:
        features = ensure_pseudo_image_batch(window_batch, expected_channels=self.num_channels)
        features = self.stem(features)
        features = self.block1(features)
        features = self.block2(features)
        features = self.block3(features)
        features = self.exit(features)
        return features.contiguous()

    def forward(self, window_batch: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(window_batch)
        return self.classifier(features)
