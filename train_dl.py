from __future__ import annotations

"""Train deep learning models on clean DB5 raw-window datasets.

Examples
--------
python db5_windows.py --subjects 1-3 --gestures 1-10 --window-ms 200 --step-ms 50; python train_dl.py --dataset data/ninapro_db5/windows/db5_s1-3_g1-10_win200_step50.npz --model cnn1d --split-mode repetition_holdout --train-reps 1,3,4 --val-reps 6 --test-reps 2,5 --run-dir artifacts/dl/db5_s1-3_rep_holdout

python db5_windows.py --subjects 1-10 --gestures 1-10 --window-ms 200 --step-ms 50 --out data/ninapro_db5/windows/db5_s1-10_g1-10_win200_step50.npz; python train_dl.py --dataset data/ninapro_db5/windows/db5_s1-10_g1-10_win200_step50.npz --model cnn_lstm --split-mode subject_holdout --train-subjects 1-7 --val-subjects 8 --test-subjects 9-10 --run-dir artifacts/dl/db5_s1-10_subject_holdout

Window-size experiment prep
---------------------------
python db5_windows.py --subjects 1-3 --gestures 1-10 --window-ms 200 --step-ms 50
python db5_windows.py --subjects 1-3 --gestures 1-10 --window-ms 300 --step-ms 50
python db5_windows.py --subjects 1-3 --gestures 1-10 --window-ms 400 --step-ms 50

TCN and fusion experiments
--------------------------
python train_dl.py --dataset data/ninapro_db5/windows/db5_s1-3_g1-10_win400_step50.npz --model tcn --split-mode repetition_holdout --train-reps 1,3,4 --val-reps 6 --test-reps 2,5 --run-dir artifacts/dl/db5_s1-3_win400_tcn

python db5_window_features.py --dataset data/ninapro_db5/windows/db5_s1-3_g1-10_win400_step50.npz; python train_dl.py --dataset data/ninapro_db5/windows/db5_s1-3_g1-10_win400_step50.npz --feature-path data/ninapro_db5/window_features/db5_s1-3_g1-10_win400_step50_features.npz --model fusion_tcn_mlp --split-mode repetition_holdout --train-reps 1,3,4 --val-reps 6 --test-reps 2,5 --run-dir artifacts/dl/db5_s1-3_win400_fusion

2D reconstruction + Xception family
-----------------------------------
python train_dl.py --dataset data/ninapro_db5/windows/db5_s1-3_g1-10_win400_step50.npz --model cnn2d_baseline --split-mode repetition_holdout --train-reps 1,3,4 --val-reps 6 --test-reps 2,5 --run-dir artifacts/dl/db5_s1-3_win400_cnn2d

python train_dl.py --dataset data/ninapro_db5/windows/db5_s1-3_g1-10_win400_step50.npz --model xception2d --split-mode repetition_holdout --train-reps 1,3,4 --val-reps 6 --test-reps 2,5 --run-dir artifacts/dl/db5_s1-3_win400_xception2d

python train_dl.py --dataset data/ninapro_db5/windows/db5_s1-3_g1-10_win400_step50.npz --model dualres_xception2d --split-mode repetition_holdout --train-reps 1,3,4 --val-reps 6 --test-reps 2,5 --run-dir artifacts/dl/db5_s1-3_win400_dualres_xception2d

python train_dl.py --dataset data/ninapro_db5/windows/db5_s1-3_g1-10_win400_step50.npz --model dualres_xception2d --pretrained-checkpoint artifacts/dl/db5_s1-3_win400_dualres_xception2d/best.pt --freeze-backbone --finetune-lr 1e-4 --split-mode repetition_holdout --train-reps 1,3,4 --val-reps 6 --test-reps 2,5 --run-dir artifacts/dl/db5_s1-3_win400_dualres_xception2d_finetune

Larger S1-S10 repetition-holdout setup
--------------------------------------
python db5_windows.py --subjects 1-10 --gestures 1-10 --window-ms 400 --step-ms 50 --out data/ninapro_db5/windows/db5_s1-10_g1-10_win400_step50.npz

python train_dl.py --dataset data/ninapro_db5/windows/db5_s1-10_g1-10_win400_step50.npz --model cnn_lstm --split-mode repetition_holdout --train-reps 1,3,4 --val-reps 6 --test-reps 2,5 --run-dir artifacts/dl/db5_s1-10_win400_cnn_lstm

python train_dl.py --dataset data/ninapro_db5/windows/db5_s1-10_g1-10_win400_step50.npz --model xception2d --split-mode repetition_holdout --train-reps 1,3,4 --val-reps 6 --test-reps 2,5 --run-dir artifacts/dl/db5_s1-10_win400_xception2d

python train_dl.py --dataset data/ninapro_db5/windows/db5_s1-10_g1-10_win400_step50.npz --model dualres_xception2d --split-mode repetition_holdout --train-reps 1,3,4 --val-reps 6 --test-reps 2,5 --run-dir artifacts/dl/db5_s1-10_win400_dualres_xception2d
"""

import argparse
import csv
import json
import os
import pickle
import random
import re
import time
import warnings
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader

from dl_dataset import (
    ChannelwiseNormalizer,
    EMGAugmenter,
    VectorNormalizer,
    WindowedEMGDataset,
    assert_feature_alignment,
    assert_no_overlap,
    build_split_indices,
    load_feature_dataset,
    load_window_dataset,
    make_split_summary,
    parse_int_list,
    print_split_summary,
)
from models.cnn1d import CNN1D
from models.cnn2d_baseline import CNN2DBaseline
from models.cnn_lstm import CNNLSTM
from models.dualres_xception2d import DualResXception2D
from models.fusion_tcn_mlp import FusionTCNMLP
from models.tcn import TCNClassifier
from models.xception2d import Xception2D


DATASET_SUBJECT_PATTERN = re.compile(r"_s(?P<subjects>[0-9_-]+)_g")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is False.")
    if device_name not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device option: {device_name}")
    return torch.device(device_name)


def print_runtime_info(device: torch.device) -> None:
    print(f"[INFO] torch.__version__: {torch.__version__}")
    print(f"[INFO] torch.version.cuda: {torch.version.cuda}")
    print(f"[INFO] torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"[INFO] selected device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")


def autocast_context(device: torch.device, amp_enabled: bool):
    device_type = "cuda" if device.type == "cuda" else "cpu"
    return torch.amp.autocast(device_type=device_type, enabled=bool(amp_enabled and device.type == "cuda"))


def safe_torch_load(checkpoint_path: str, map_location: torch.device | str):
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using `torch.load` with `weights_only=False`",
                category=FutureWarning,
            )
            return torch.load(checkpoint_path, map_location=map_location)
    except (pickle.UnpicklingError, RuntimeError, ValueError):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using `torch.load` with `weights_only=False`",
                category=FutureWarning,
            )
            return torch.load(checkpoint_path, map_location=map_location)


def infer_subjects_from_dataset_path(dataset_path: str) -> list[int] | None:
    file_name = os.path.basename(dataset_path)
    match = DATASET_SUBJECT_PATTERN.search(file_name)
    if match is None:
        return None

    subject_block = match.group("subjects").replace("_", ",")
    return parse_int_list(subject_block)


def run_pre_training_integrity_checks(
    dataset: Dict[str, np.ndarray],
    dataset_path: str,
    split_mode: str,
    split_config: Dict[str, object],
) -> None:
    dataset_subjects = sorted(np.unique(dataset["subject_id"]).astype(np.int64).tolist())
    dataset_repetitions = sorted(np.unique(dataset["repetition_id"]).astype(np.int64).tolist())

    print(f"[INFO] Unique subjects in dataset: {dataset_subjects}")
    print(f"[INFO] Unique repetitions in dataset: {dataset_repetitions}")

    subjects_from_path = infer_subjects_from_dataset_path(dataset_path)
    if subjects_from_path is not None:
        subjects_from_path = sorted({int(value) for value in subjects_from_path})
        print(f"[INFO] Subjects inferred from dataset filename: {subjects_from_path}")
        if subjects_from_path != dataset_subjects:
            raise AssertionError(
                f"Dataset subject_id metadata {dataset_subjects} does not match dataset file name subjects {subjects_from_path}"
            )
    else:
        print("[INFO] No subject list could be inferred from dataset filename.")

    if split_mode == "repetition_holdout":
        train_reps = {int(value) for value in split_config.get("train_reps", [])}
        val_reps = {int(value) for value in split_config.get("val_reps", [])}
        test_reps = {int(value) for value in split_config.get("test_reps", [])}
        print(f"[INFO] Requested repetition split: train={sorted(train_reps)}, val={sorted(val_reps)}, test={sorted(test_reps)}")
        if train_reps & val_reps or train_reps & test_reps or val_reps & test_reps:
            raise AssertionError("Train/val/test repetition IDs overlap.")

    if split_mode == "subject_holdout":
        requested_subjects = sorted(
            {
                int(value)
                for key in ("train_subjects", "val_subjects", "test_subjects")
                for value in split_config.get(key, [])
            }
        )
        print(f"[INFO] Requested CLI subjects from split config: {requested_subjects}")
        if requested_subjects != dataset_subjects:
            raise AssertionError(
                f"Requested CLI subjects {requested_subjects} do not match dataset subject metadata {dataset_subjects}"
            )


def parse_optional_int_list(text: str) -> list[int] | None:
    values = parse_int_list(text)
    return values or None


def requires_pseudo_image_input(model_name: str) -> bool:
    return model_name in {"cnn2d_baseline", "xception2d", "dualres_xception2d"}


def freeze_backbone_parameters(model: nn.Module) -> None:
    classifier_prefixes = ("classifier",)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(classifier_prefixes)


def maybe_load_pretrained_checkpoint(
    model: nn.Module,
    model_name: str,
    checkpoint_path: str,
    device: torch.device,
    freeze_backbone: bool,
) -> dict[str, object] | None:
    if not checkpoint_path:
        return None

    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    checkpoint_model_name = str(checkpoint.get("model_name", ""))
    if checkpoint_model_name and checkpoint_model_name != model_name:
        raise ValueError(
            f"Pretrained checkpoint model_name={checkpoint_model_name} does not match requested model={model_name}."
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print(f"[INFO] Loaded pretrained checkpoint from {checkpoint_path}")
    if freeze_backbone:
        freeze_backbone_parameters(model)
        print("[INFO] Fine-tuning mode: backbone frozen, classifier trainable.")
    else:
        print("[INFO] Fine-tuning mode: full model trainable.")
    return checkpoint


def build_model_from_config(model_name: str, model_config: Dict[str, object]) -> nn.Module:
    if model_name == "cnn1d":
        return CNN1D(**model_config)
    if model_name == "cnn2d_baseline":
        return CNN2DBaseline(**model_config)
    if model_name == "cnn_lstm":
        return CNNLSTM(**model_config)
    if model_name == "xception2d":
        return Xception2D(**model_config)
    if model_name == "dualres_xception2d":
        return DualResXception2D(**model_config)
    if model_name == "tcn":
        return TCNClassifier(**model_config)
    if model_name == "fusion_tcn_mlp":
        return FusionTCNMLP(**model_config)
    raise ValueError(f"Unknown model: {model_name}")


def resolve_model_config(
    model_name: str,
    args: argparse.Namespace,
    inferred_num_channels: int,
    inferred_num_classes: int,
    feature_dim: int | None = None,
) -> Dict[str, object]:
    num_channels = inferred_num_channels
    if args.num_channels is not None and int(args.num_channels) != inferred_num_channels:
        raise ValueError(
            f"Provided num_channels={args.num_channels} does not match dataset channels={inferred_num_channels}."
        )

    num_classes = inferred_num_classes
    if args.num_classes is not None and int(args.num_classes) != inferred_num_classes:
        raise ValueError(
            f"Provided num_classes={args.num_classes} does not match dataset classes={inferred_num_classes}."
        )

    tcn_channels = parse_optional_int_list(args.tcn_channels)

    if model_name == "cnn1d":
        return {
            "num_channels": num_channels,
            "num_classes": num_classes,
            "base_filters": int(args.base_filters),
            "dropout": float(args.dropout),
        }

    if model_name == "cnn2d_baseline":
        return {
            "num_channels": num_channels,
            "num_classes": num_classes,
            "base_channels": int(args.image_base_channels),
            "dropout": float(args.dropout),
        }

    if model_name == "cnn_lstm":
        return {
            "num_channels": num_channels,
            "num_classes": num_classes,
            "conv_channels": int(args.base_filters),
            "lstm_hidden": int(args.lstm_hidden),
            "lstm_layers": int(args.lstm_layers),
            "bidirectional": bool(args.bidirectional),
            "dropout": float(args.dropout),
        }

    if model_name == "xception2d":
        return {
            "num_channels": num_channels,
            "num_classes": num_classes,
            "base_channels": int(args.image_base_channels),
            "dropout": float(args.dropout),
        }

    if model_name == "dualres_xception2d":
        return {
            "num_channels": num_channels,
            "num_classes": num_classes,
            "base_channels": int(args.image_base_channels),
            "dropout": float(args.dropout),
        }

    if model_name == "tcn":
        return {
            "num_channels": num_channels,
            "num_classes": num_classes,
            "channels": tcn_channels,
            "hidden_channels": int(args.tcn_hidden_channels),
            "num_blocks": int(args.tcn_num_blocks),
            "kernel_size": int(args.tcn_kernel_size),
            "dropout": float(args.dropout),
            "debug_shapes": bool(args.debug_shapes),
        }

    if model_name == "fusion_tcn_mlp":
        if feature_dim is None:
            raise ValueError("feature_dim is required for fusion_tcn_mlp.")
        return {
            "num_channels": num_channels,
            "feature_dim": int(feature_dim),
            "num_classes": num_classes,
            "tcn_channels": tcn_channels,
            "tcn_hidden_channels": int(args.tcn_hidden_channels),
            "tcn_num_blocks": int(args.tcn_num_blocks),
            "tcn_kernel_size": int(args.tcn_kernel_size),
            "feature_hidden_size": int(args.feature_hidden_size),
            "feature_embedding_dim": int(args.feature_embedding_dim),
            "dropout": float(args.dropout),
        }

    raise ValueError(f"Unknown model: {model_name}")


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def move_batch_to_device(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    use_non_blocking = device.type == "cuda"
    if len(batch) == 2:
        window_batch, labels = batch
        return (
            window_batch.to(device, non_blocking=use_non_blocking),
            None,
            labels.to(device, non_blocking=use_non_blocking),
        )
    if len(batch) == 3:
        window_batch, feature_batch, labels = batch
        return (
            window_batch.to(device, non_blocking=use_non_blocking),
            feature_batch.to(device, non_blocking=use_non_blocking),
            labels.to(device, non_blocking=use_non_blocking),
        )
    raise ValueError(f"Unexpected batch structure with {len(batch)} items.")


def forward_model(
    model: nn.Module,
    window_batch: torch.Tensor,
    feature_batch: torch.Tensor | None,
) -> torch.Tensor:
    if feature_batch is None:
        return model(window_batch)
    return model(window_batch, feature_batch)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in data_loader:
        window_batch, feature_batch, labels = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            logits = forward_model(model, window_batch, feature_batch)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(window_batch.shape[0])
        running_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    amp_enabled: bool = False,
) -> Dict[str, object]:
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for batch in data_loader:
            window_batch, feature_batch, labels = move_batch_to_device(batch, device)

            with autocast_context(device, amp_enabled):
                logits = forward_model(model, window_batch, feature_batch)
                loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)

            batch_size = int(window_batch.shape[0])
            running_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            all_targets.extend(labels.cpu().numpy().astype(int).tolist())
            all_predictions.extend(predictions.cpu().numpy().astype(int).tolist())

    targets = np.asarray(all_targets, dtype=np.int64)
    predictions = np.asarray(all_predictions, dtype=np.int64)
    labels = list(range(int(num_classes)))
    matrix = confusion_matrix(targets, predictions, labels=labels)

    return {
        "loss": running_loss / max(total_samples, 1),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": matrix,
        "targets": targets,
        "predictions": predictions,
    }


def save_history_csv(history_rows: Sequence[Dict[str, object]], output_path: str) -> None:
    if not history_rows:
        return
    field_names = list(history_rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(history_rows)


def save_confusion_matrix_outputs(
    confusion: np.ndarray,
    class_gesture_ids: Sequence[int],
    output_dir: str,
    prefix: str,
) -> tuple[str, str]:
    labels = [f"G{int(gesture_id)}" for gesture_id in class_gesture_ids]
    csv_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.csv")
    png_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.png")

    frame = pd.DataFrame(confusion, index=labels, columns=labels)
    frame.to_csv(csv_path)

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(confusion, cmap="Blues")
    axis.set_xticks(np.arange(len(labels)))
    axis.set_yticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(f"{prefix.replace('_', ' ').title()} Confusion Matrix")

    for row_index in range(confusion.shape[0]):
        for col_index in range(confusion.shape[1]):
            axis.text(
                col_index,
                row_index,
                int(confusion[row_index, col_index]),
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(png_path, dpi=220)
    plt.close(figure)
    return csv_path, png_path


def save_per_class_accuracy_outputs(
    confusion: np.ndarray,
    class_gesture_ids: Sequence[int],
    output_dir: str,
    prefix: str,
) -> tuple[str, str]:
    supports = confusion.sum(axis=1).astype(np.int64)
    correct = np.diag(confusion).astype(np.int64)
    per_class_accuracy = np.divide(
        correct,
        supports,
        out=np.zeros_like(correct, dtype=np.float64),
        where=supports > 0,
    )

    rows = []
    for class_index, gesture_id in enumerate(class_gesture_ids):
        rows.append(
            {
                "class_index": int(class_index),
                "gesture_id": int(gesture_id),
                "support": int(supports[class_index]),
                "correct": int(correct[class_index]),
                "accuracy": float(per_class_accuracy[class_index]),
            }
        )

    csv_path = os.path.join(output_dir, f"{prefix}_per_class_accuracy.csv")
    png_path = os.path.join(output_dir, f"{prefix}_per_class_accuracy.png")
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)

    figure, axis = plt.subplots(figsize=(9, 5))
    labels = [f"G{int(gesture_id)}" for gesture_id in class_gesture_ids]
    axis.bar(labels, per_class_accuracy, color="#4C78A8", edgecolor="black")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Accuracy")
    axis.set_title(f"{prefix.replace('_', ' ').title()} Per-Class Accuracy")

    for class_index, value in enumerate(per_class_accuracy):
        axis.text(class_index, float(value) + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    figure.tight_layout()
    figure.savefig(png_path, dpi=220)
    plt.close(figure)
    return csv_path, png_path


def save_json(data: Dict[str, object], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train deep learning models on clean DB5 raw-window datasets.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the .npz dataset built by db5_windows.py")
    parser.add_argument("--feature-path", type=str, default="", help="Optional handcrafted feature dataset for fusion")
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "cnn1d",
            "cnn_lstm",
            "cnn2d_baseline",
            "xception2d",
            "dualres_xception2d",
            "tcn",
            "fusion_tcn_mlp",
        ],
        default="cnn1d",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["repetition_holdout", "subject_holdout"],
        default="repetition_holdout",
    )
    parser.add_argument("--train-reps", type=str, default="1,3,4")
    parser.add_argument("--val-reps", type=str, default="6")
    parser.add_argument("--test-reps", type=str, default="2,5")
    parser.add_argument("--train-subjects", type=str, default="1-7")
    parser.add_argument("--val-subjects", type=str, default="8")
    parser.add_argument("--test-subjects", type=str, default="9-10")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--num-channels", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--base-filters", type=int, default=64)
    parser.add_argument("--image-base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--tcn-channels", type=str, default="", help="Optional comma-separated TCN channel widths")
    parser.add_argument("--tcn-hidden-channels", type=int, default=64)
    parser.add_argument("--tcn-num-blocks", type=int, default=4)
    parser.add_argument("--tcn-kernel-size", type=int, default=5)
    parser.add_argument("--debug-shapes", action="store_true", help="Print TCN tensor shapes for the first batch only")
    parser.add_argument("--feature-hidden-size", type=int, default=256)
    parser.add_argument("--feature-embedding-dim", type=int, default=128)
    parser.add_argument("--per-channel-zscore", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment-scale", type=float, default=0.0)
    parser.add_argument("--augment-noise", type=float, default=0.0)
    parser.add_argument("--augment-shift", type=int, default=0)
    parser.add_argument("--augment-channel-dropout", type=float, default=0.0)
    parser.add_argument("--pretrained-checkpoint", type=str, default="")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--run-dir", type=str, default="")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print_runtime_info(device)
    amp_enabled = device.type == "cuda" if args.amp is None else bool(args.amp)
    if amp_enabled and device.type != "cuda":
        raise RuntimeError("AMP was requested, but the selected device is not CUDA. Use --device cuda/auto or pass --no-amp.")
    print(f"[INFO] AMP enabled: {amp_enabled}")
    if args.freeze_backbone and not args.pretrained_checkpoint:
        raise ValueError("--freeze-backbone requires --pretrained-checkpoint.")

    dataset = load_window_dataset(args.dataset)
    feature_dataset = None
    if args.model == "fusion_tcn_mlp":
        if not args.feature_path:
            raise ValueError("--feature-path is required when --model fusion_tcn_mlp is used.")
        feature_dataset = load_feature_dataset(args.feature_path)
        assert_feature_alignment(dataset, feature_dataset)
        print(
            f"[INFO] Loaded handcrafted feature dataset: {args.feature_path} "
            f"shape={tuple(feature_dataset['features'].shape)}"
        )
    elif args.feature_path:
        print(f"[INFO] Ignoring --feature-path for model {args.model}.")

    if args.debug_shapes and args.model != "tcn":
        print(f"[INFO] Ignoring --debug-shapes for model {args.model}; it currently applies to model=tcn only.")

    split_indices, split_config = build_split_indices(
        dataset,
        split_mode=args.split_mode,
        train_reps=parse_int_list(args.train_reps),
        val_reps=parse_int_list(args.val_reps),
        test_reps=parse_int_list(args.test_reps),
        train_subjects=parse_int_list(args.train_subjects),
        val_subjects=parse_int_list(args.val_subjects),
        test_subjects=parse_int_list(args.test_subjects),
    )
    run_pre_training_integrity_checks(
        dataset=dataset,
        dataset_path=args.dataset,
        split_mode=args.split_mode,
        split_config=split_config,
    )
    assert_no_overlap(dataset, split_indices)
    print("[INFO] Split overlap check passed for subject_id/repetition_id groups.")

    split_summary = make_split_summary(dataset, split_indices, split_config)
    print_split_summary(split_summary)

    if args.run_dir:
        run_dir = args.run_dir
    else:
        dataset_stem = os.path.splitext(os.path.basename(args.dataset))[0]
        run_dir = os.path.join("artifacts", "dl", f"{dataset_stem}_{args.model}_{args.split_mode}")
    os.makedirs(run_dir, exist_ok=True)

    save_json(vars(args), os.path.join(run_dir, "train_args.json"))
    save_json(split_summary, os.path.join(run_dir, "split_summary.json"))

    inferred_num_channels = int(dataset["x"].shape[2])
    inferred_num_classes = int(len(dataset["class_gesture_ids"]))
    feature_matrix = feature_dataset["features"] if feature_dataset is not None else None
    feature_dim = int(feature_matrix.shape[1]) if feature_matrix is not None else None
    use_image_input = requires_pseudo_image_input(args.model)
    print(f"[INFO] Pseudo-image reconstruction enabled: {use_image_input}")

    normalizer = ChannelwiseNormalizer.fit(
        dataset["x"][split_indices.train],
        per_channel=bool(args.per_channel_zscore),
    )
    print(f"[INFO] Normalization fit on training split only: {len(split_indices.train)} windows")
    print(f"[INFO] Per-channel z-score normalization: {normalizer.per_channel}")
    print(
        f"[INFO] Normalization mean shape={tuple(normalizer.mean.shape)}, "
        f"range=[{float(normalizer.mean.min()):.6f}, {float(normalizer.mean.max()):.6f}]"
    )
    print(
        f"[INFO] Normalization std shape={tuple(normalizer.std.shape)}, "
        f"range=[{float(normalizer.std.min()):.6f}, {float(normalizer.std.max()):.6f}]"
    )
    if bool(args.per_channel_zscore) and tuple(normalizer.mean.shape) != (inferred_num_channels,):
        raise AssertionError(
            f"Expected per-channel normalization statistics with shape ({inferred_num_channels},), got {tuple(normalizer.mean.shape)}"
        )

    feature_normalizer = None
    if feature_matrix is not None:
        feature_normalizer = VectorNormalizer.fit(feature_matrix[split_indices.train])
        print(f"[INFO] Feature normalization fit on training split only: shape={tuple(feature_normalizer.mean.shape)}")
        print(
            f"[INFO] Feature mean range=[{float(feature_normalizer.mean.min()):.6f}, {float(feature_normalizer.mean.max()):.6f}] "
            f"std range=[{float(feature_normalizer.std.min()):.6f}, {float(feature_normalizer.std.max()):.6f}]"
        )

    augmenter = EMGAugmenter(
        scale=float(args.augment_scale),
        noise=float(args.augment_noise),
        shift=int(args.augment_shift),
        channel_dropout=float(args.augment_channel_dropout),
    )
    print(
        "[INFO] Training augmentations (train split only): "
        f"scale={args.augment_scale}, noise={args.augment_noise}, "
        f"shift={args.augment_shift}, channel_dropout={args.augment_channel_dropout}"
    )

    train_dataset = WindowedEMGDataset(
        dataset["x"],
        dataset["y"],
        split_indices.train,
        normalizer=normalizer,
        feature_matrix=feature_matrix,
        feature_normalizer=feature_normalizer,
        augmenter=augmenter,
        image_input=use_image_input,
    )
    val_dataset = WindowedEMGDataset(
        dataset["x"],
        dataset["y"],
        split_indices.val,
        normalizer=normalizer,
        feature_matrix=feature_matrix,
        feature_normalizer=feature_normalizer,
        augmenter=None,
        image_input=use_image_input,
    )
    test_dataset = WindowedEMGDataset(
        dataset["x"],
        dataset["y"],
        split_indices.test,
        normalizer=normalizer,
        feature_matrix=feature_matrix,
        feature_normalizer=feature_normalizer,
        augmenter=None,
        image_input=use_image_input,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model_config = resolve_model_config(
        args.model,
        args,
        inferred_num_channels,
        inferred_num_classes,
        feature_dim=feature_dim,
    )
    model = build_model_from_config(args.model, model_config).to(device)
    pretrained_checkpoint = maybe_load_pretrained_checkpoint(
        model=model,
        model_name=args.model,
        checkpoint_path=args.pretrained_checkpoint,
        device=device,
        freeze_backbone=bool(args.freeze_backbone),
    )

    effective_lr = float(args.finetune_lr if args.pretrained_checkpoint else args.lr)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters remain after applying fine-tuning freeze settings.")

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Trainable parameters: {parameter_count(model)}")
    print(
        "[INFO] Run config: "
        f"lr={effective_lr}, base_lr={args.lr}, finetune_lr={args.finetune_lr}, weight_decay={args.weight_decay}, dropout={args.dropout}, "
        f"label_smoothing={args.label_smoothing}, per_channel_zscore={args.per_channel_zscore}, "
        f"image_base_channels={args.image_base_channels}, tcn_channels={parse_optional_int_list(args.tcn_channels)}, "
        f"tcn_hidden_channels={args.tcn_hidden_channels}, tcn_num_blocks={args.tcn_num_blocks}, "
        f"tcn_kernel_size={args.tcn_kernel_size}, debug_shapes={args.debug_shapes}, "
        f"feature_path={args.feature_path or 'None'}, pretrained_checkpoint={args.pretrained_checkpoint or 'None'}, "
        f"freeze_backbone={args.freeze_backbone}, amp_enabled={amp_enabled}"
    )
    if pretrained_checkpoint is not None:
        print(
            f"[INFO] Loaded fine-tuning start point: best_epoch={pretrained_checkpoint.get('best_epoch', 'unknown')}, "
            f"best_val_macro_f1={pretrained_checkpoint.get('best_val_macro_f1', 'unknown')}"
        )

    criterion = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    optimizer = torch.optim.Adam(trainable_parameters, lr=effective_lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=amp_enabled)

    history_rows: list[Dict[str, object]] = []
    history_path = os.path.join(run_dir, "history.csv")
    checkpoint_path = os.path.join(run_dir, "best.pt")

    best_val_macro_f1 = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, int(args.epochs) + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, amp_enabled)
        val_metrics = evaluate_model(model, val_loader, criterion, device, inferred_num_classes, amp_enabled=amp_enabled)
        elapsed_seconds = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "elapsed_seconds": float(elapsed_seconds),
        }
        history_rows.append(row)
        save_history_csv(history_rows, history_path)

        print(
            f"[EPOCH {epoch:03d}] train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if float(val_metrics["macro_f1"]) > best_val_macro_f1 + 1e-8:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            checkpoint = {
                "model_name": args.model,
                "model_config": model_config,
                "model_state_dict": model.state_dict(),
                "normalizer": normalizer.to_state(),
                "feature_normalizer": feature_normalizer.to_state() if feature_normalizer is not None else None,
                "class_gesture_ids": dataset["class_gesture_ids"].astype(np.int64).tolist(),
                "dataset_path": args.dataset,
                "feature_path": args.feature_path,
                "pretrained_checkpoint": args.pretrained_checkpoint,
                "freeze_backbone": bool(args.freeze_backbone),
                "split_config": split_config,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_val_macro_f1,
                "args": vars(args),
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"[INFO] Saved new best checkpoint to {checkpoint_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(args.patience):
            print(f"[INFO] Early stopping triggered at epoch {epoch}")
            break

    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    best_model = build_model_from_config(str(checkpoint["model_name"]), dict(checkpoint["model_config"])).to(device)
    best_model.load_state_dict(checkpoint["model_state_dict"])
    best_normalizer = ChannelwiseNormalizer.from_state(dict(checkpoint["normalizer"]))
    best_feature_normalizer = None
    best_use_image_input = requires_pseudo_image_input(str(checkpoint["model_name"]))
    if checkpoint.get("feature_normalizer") is not None:
        best_feature_normalizer = VectorNormalizer.from_state(dict(checkpoint["feature_normalizer"]))

    best_val_dataset = WindowedEMGDataset(
        dataset["x"],
        dataset["y"],
        split_indices.val,
        normalizer=best_normalizer,
        feature_matrix=feature_matrix,
        feature_normalizer=best_feature_normalizer,
        image_input=best_use_image_input,
    )
    best_test_dataset = WindowedEMGDataset(
        dataset["x"],
        dataset["y"],
        split_indices.test,
        normalizer=best_normalizer,
        feature_matrix=feature_matrix,
        feature_normalizer=best_feature_normalizer,
        image_input=best_use_image_input,
    )
    best_val_loader = DataLoader(
        best_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    best_test_loader = DataLoader(
        best_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_metrics = evaluate_model(best_model, best_val_loader, criterion, device, inferred_num_classes, amp_enabled=amp_enabled)
    test_metrics = evaluate_model(best_model, best_test_loader, criterion, device, inferred_num_classes, amp_enabled=amp_enabled)

    class_gesture_ids = [int(value) for value in checkpoint["class_gesture_ids"]]
    save_confusion_matrix_outputs(val_metrics["confusion_matrix"], class_gesture_ids, run_dir, "val")
    save_confusion_matrix_outputs(test_metrics["confusion_matrix"], class_gesture_ids, run_dir, "test")
    save_per_class_accuracy_outputs(val_metrics["confusion_matrix"], class_gesture_ids, run_dir, "val")
    save_per_class_accuracy_outputs(test_metrics["confusion_matrix"], class_gesture_ids, run_dir, "test")

    metrics_summary = {
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_macro_f1),
        "val": {
            "loss": float(val_metrics["loss"]),
            "accuracy": float(val_metrics["accuracy"]),
            "macro_f1": float(val_metrics["macro_f1"]),
        },
        "test": {
            "loss": float(test_metrics["loss"]),
            "accuracy": float(test_metrics["accuracy"]),
            "macro_f1": float(test_metrics["macro_f1"]),
        },
    }
    save_json(metrics_summary, os.path.join(run_dir, "metrics_summary.json"))

    print("[INFO] Final validation metrics:")
    print(json.dumps(metrics_summary["val"], ensure_ascii=False, indent=2))
    print("[INFO] Final test metrics:")
    print(json.dumps(metrics_summary["test"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
