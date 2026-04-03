from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _same_padding(kernel_size: int, dilation: int) -> tuple[int, int]:
    kernel_size = int(kernel_size)
    dilation = int(dilation)
    if kernel_size <= 0:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}")
    if dilation <= 0:
        raise ValueError(f"dilation must be positive, got {dilation}")
    total_padding = dilation * (kernel_size - 1)
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding
    return left_padding, right_padding


def _shape_as_tuple(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _ensure_finite(tensor: torch.Tensor, name: str, block_index: int) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite values detected in {name} at TCN block {block_index}.")


class SamePadConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.left_padding, self.right_padding = _same_padding(kernel_size=kernel_size, dilation=dilation)
        self.conv = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            stride=1,
            padding=0,
            dilation=int(dilation),
            bias=bool(bias),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(f"SamePadConv1d expects [B, C, T], got {_shape_as_tuple(inputs)}")
        if int(inputs.shape[-1]) <= 0:
            raise RuntimeError(f"Conv input temporal length must stay positive, got {_shape_as_tuple(inputs)}")
        features = inputs.contiguous()
        if self.left_padding > 0 or self.right_padding > 0:
            features = F.pad(features, (self.left_padding, self.right_padding))
        outputs = self.conv(features)
        if int(outputs.shape[-1]) <= 0:
            raise RuntimeError(f"Conv output temporal length must stay positive, got {_shape_as_tuple(outputs)}")
        return outputs.contiguous()


class TCNResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        block_index: int,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.block_index = int(block_index)

        self.conv1 = SamePadConv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=int(kernel_size),
            dilation=int(dilation),
        )
        self.norm1 = nn.BatchNorm1d(self.out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(float(dropout))

        self.conv2 = SamePadConv1d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=int(kernel_size),
            dilation=int(dilation),
        )
        self.norm2 = nn.BatchNorm1d(self.out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(float(dropout))

        self.residual = nn.Identity()
        if self.in_channels != self.out_channels:
            self.residual = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor, debug_shapes: bool = False) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                f"TCNResidualBlock expects [B, C, T], got {_shape_as_tuple(inputs)} at block {self.block_index}"
            )
        if int(inputs.shape[1]) != self.in_channels:
            raise RuntimeError(
                f"TCN block {self.block_index} expected {self.in_channels} input channels, got {int(inputs.shape[1])}"
            )
        if int(inputs.shape[-1]) <= 0:
            raise RuntimeError(
                f"TCN block {self.block_index} received non-positive temporal length: {_shape_as_tuple(inputs)}"
            )

        if debug_shapes:
            print(f"[TCN][block {self.block_index}] input shape={_shape_as_tuple(inputs)}")
            _ensure_finite(inputs, "block input", self.block_index)

        residual = self.residual(inputs.contiguous()).contiguous()

        outputs = self.conv1(inputs.contiguous())
        outputs = self.norm1(outputs)
        outputs = self.act1(outputs)
        outputs = self.drop1(outputs)
        outputs = self.conv2(outputs.contiguous())
        outputs = self.norm2(outputs)
        outputs = self.act2(outputs)
        outputs = self.drop2(outputs)
        outputs = outputs.contiguous()

        if int(outputs.shape[-1]) <= 0:
            raise RuntimeError(
                f"TCN block {self.block_index} produced non-positive temporal length: {_shape_as_tuple(outputs)}"
            )
        if int(residual.shape[-1]) <= 0:
            raise RuntimeError(
                f"TCN residual {self.block_index} produced non-positive temporal length: {_shape_as_tuple(residual)}"
            )
        if int(outputs.shape[1]) != int(residual.shape[1]):
            raise RuntimeError(
                f"TCN block {self.block_index} channel mismatch before residual add: output={_shape_as_tuple(outputs)}, residual={_shape_as_tuple(residual)}"
            )
        if int(outputs.shape[-1]) != int(residual.shape[-1]):
            raise RuntimeError(
                f"TCN block {self.block_index} temporal mismatch before residual add: output={_shape_as_tuple(outputs)}, residual={_shape_as_tuple(residual)}"
            )

        summed = outputs + residual
        if int(summed.shape[-1]) <= 0:
            raise RuntimeError(
                f"TCN block {self.block_index} produced non-positive summed temporal length: {_shape_as_tuple(summed)}"
            )
        final = self.activation(summed).contiguous()

        if debug_shapes:
            print(f"[TCN][block {self.block_index}] block output shape={_shape_as_tuple(outputs)}")
            print(f"[TCN][block {self.block_index}] residual shape={_shape_as_tuple(residual)}")
            print(f"[TCN][block {self.block_index}] summed shape={_shape_as_tuple(summed)}")
            print(f"[TCN][block {self.block_index}] final shape={_shape_as_tuple(final)}")
            _ensure_finite(final, "block output", self.block_index)

        return final


class TCNEncoder(nn.Module):
    def __init__(
        self,
        num_channels: int,
        channels: Sequence[int] | None = None,
        hidden_channels: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.3,
        debug_shapes: bool = False,
    ) -> None:
        super().__init__()

        if channels is None:
            channel_sequence = [int(hidden_channels)] * int(num_blocks)
        else:
            channel_sequence = [int(value) for value in channels]
            if not channel_sequence:
                raise ValueError("channels must be non-empty when provided.")

        self.blocks = nn.ModuleList()
        in_channels = int(num_channels)
        for block_index, out_channels in enumerate(channel_sequence):
            dilation = 2 ** block_index
            self.blocks.append(
                TCNResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=int(kernel_size),
                    dilation=dilation,
                    dropout=float(dropout),
                    block_index=block_index,
                )
            )
            in_channels = out_channels

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = int(in_channels)
        self.debug_shapes = bool(debug_shapes)
        self._debug_batches_remaining = 1 if self.debug_shapes else 0

    def forward(self, window_batch: torch.Tensor) -> torch.Tensor:
        if window_batch.ndim != 3:
            raise ValueError(f"TCNEncoder expects [B, T, C], got {_shape_as_tuple(window_batch)}")
        if int(window_batch.shape[1]) <= 0:
            raise RuntimeError(f"TCNEncoder received non-positive temporal length: {_shape_as_tuple(window_batch)}")

        debug_this_batch = bool(self.debug_shapes and self._debug_batches_remaining > 0)
        if debug_this_batch:
            print(f"[TCN] window batch shape={_shape_as_tuple(window_batch)}")

        features = window_batch.transpose(1, 2).contiguous()
        if debug_this_batch:
            print(f"[TCN] transposed features shape={_shape_as_tuple(features)}")
            _ensure_finite(features, "encoder input", -1)

        for block in self.blocks:
            if debug_this_batch:
                _ensure_finite(features, "pre-block activation", block.block_index)
            features = block(features, debug_shapes=debug_this_batch)
            if debug_this_batch:
                _ensure_finite(features, "post-block activation", block.block_index)

        pooled = self.pool(features.contiguous()).squeeze(-1).contiguous()
        if debug_this_batch:
            print(f"[TCN] pooled embedding shape={_shape_as_tuple(pooled)}")
            _ensure_finite(pooled, "pooled embedding", len(self.blocks))
            self._debug_batches_remaining -= 1
        return pooled


class TCNClassifier(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        channels: Sequence[int] | None = None,
        hidden_channels: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.3,
        debug_shapes: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = TCNEncoder(
            num_channels=num_channels,
            channels=channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
            debug_shapes=debug_shapes,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(self.encoder.output_dim, int(num_classes)),
        )

    def forward(self, window_batch: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(window_batch)
        return self.classifier(embedding)
