from __future__ import annotations

"""Run a tiny one-step TCN smoke test on CPU and CUDA (if available).

Examples
--------
python smoke_test_tcn.py
python smoke_test_tcn.py --amp --debug-shapes
"""

import argparse

import torch
from torch import nn

from models.tcn import TCNClassifier


def autocast_context(device: torch.device, amp_enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=bool(amp_enabled and device.type == "cuda"))


def run_single_step(device: torch.device, amp_enabled: bool, debug_shapes: bool) -> None:
    batch_size = 4
    time_steps = 80
    num_channels = 16
    num_classes = 10

    model = TCNClassifier(
        num_channels=num_channels,
        num_classes=num_classes,
        hidden_channels=32,
        num_blocks=3,
        kernel_size=5,
        dropout=0.2,
        debug_shapes=debug_shapes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=amp_enabled)

    inputs = torch.randn(batch_size, time_steps, num_channels, device=device)
    targets = torch.randint(low=0, high=num_classes, size=(batch_size,), device=device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, amp_enabled):
        logits = model(inputs)
        loss = criterion(logits, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    print(
        f"[SMOKE] success on device={device.type} amp={amp_enabled} "
        f"loss={float(loss.detach().cpu().item()):.6f} logits_shape={tuple(logits.shape)}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a tiny TCN smoke test on CPU and CUDA if available.")
    parser.add_argument("--amp", action="store_true", help="Also run the CUDA step with AMP enabled")
    parser.add_argument("--debug-shapes", action="store_true", help="Print TCN tensor shapes during the first step")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    print(f"[SMOKE] torch.__version__={torch.__version__}")
    print(f"[SMOKE] torch.version.cuda={torch.version.cuda}")
    print(f"[SMOKE] torch.cuda.is_available()={torch.cuda.is_available()}")

    run_single_step(torch.device("cpu"), amp_enabled=False, debug_shapes=bool(args.debug_shapes))

    if torch.cuda.is_available():
        run_single_step(torch.device("cuda"), amp_enabled=False, debug_shapes=bool(args.debug_shapes))
        if args.amp:
            run_single_step(torch.device("cuda"), amp_enabled=True, debug_shapes=bool(args.debug_shapes))
    else:
        print("[SMOKE] CUDA not available; skipped CUDA smoke step.")


if __name__ == "__main__":
    main()
