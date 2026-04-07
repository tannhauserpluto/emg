# -*- coding: utf-8 -*-
"""Evaluate deep learning robustness under the ML noise protocol.

This script aligns DL noise evaluation with the existing ML protocol in
run_robust_eval.py. It applies the same noise types, levels, and denoise
methods to DB5 window datasets, then evaluates a saved DL checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dl_dataset import (
    ChannelwiseNormalizer,
    WindowedEMGDataset,
    assert_no_overlap,
    build_split_indices,
    filter_dataset_by_subject,
    load_fold_ids,
    load_window_dataset,
    make_split_summary,
    parse_int_list,
    print_split_summary,
)
from eval_dl import (
    REPETITION_PRED_FIELDS,
    WINDOW_PRED_FIELDS,
    aggregate_repetition_predictions,
    build_window_prediction_rows,
    collect_window_outputs,
    compute_metrics,
    save_prediction_csv,
)
from train_dl import (
    build_model_from_config,
    requires_pseudo_image_input,
    resolve_device,
    safe_torch_load,
    save_confusion_matrix_outputs,
    save_json,
    save_per_class_accuracy_outputs,
)
from src.denoise import kalman_denoise, notch_filter, pca_denoise, wavelet_denoise
from src.noise import apply_noises


@dataclass(frozen=True)
class NoiseCondition:
    noise_type: str
    level: float
    denoise_method: str
    noise_cfg_list: list[dict[str, object]] | None


NOISE_TYPES = {
    "wgn": lambda val: [{"kind": "wgn", "snr_db": float(val)}],
    "pink": lambda val: [{"kind": "pink", "snr_db": float(val), "beta": 1.0}],
    "hum": lambda val: [{"kind": "hum", "amp_ratio": float(val)}],
    "drift": lambda val: [{"kind": "drift", "amp_ratio": float(val)}],
    "motion": lambda val: [{"kind": "motion", "amp_ratio": float(val)}],
    "spikes": lambda val: [{"kind": "spikes", "amp_ratio": float(val)}],
}

DEFAULT_SNR_LEVELS = [20.0, 10.0, 0.0, -5.0]
DEFAULT_AMP_LEVELS = [0.1, 0.3, 0.6, 1.0]
DEFAULT_DENOISE_METHODS = ["none", "notch", "wavelet", "kalman", "pca"]


def parse_list(values: str) -> list[str]:
    if not values:
        return []
    return [item.strip() for item in values.split(",") if item.strip()]


def parse_float_list(values: str) -> list[float]:
    return [float(item) for item in parse_list(values)]


def format_level(level: float) -> str:
    return str(float(level))


def level_slug(level: float) -> str:
    text = format_level(level)
    if text.startswith("-"):
        text = f"neg{text[1:]}"
    return text.replace(".", "p")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_noise_conditions(
    noise_types: Sequence[str],
    snr_levels: Sequence[float],
    amp_levels: Sequence[float],
    denoise_methods: Sequence[str],
    include_clean: bool = True,
) -> list[NoiseCondition]:
    conditions: list[NoiseCondition] = []
    if include_clean:
        conditions.append(
            NoiseCondition(
                noise_type="clean",
                level=0.0,
                denoise_method="none",
                noise_cfg_list=None,
            )
        )

    for noise_name in noise_types:
        if noise_name not in NOISE_TYPES:
            raise ValueError(f"Unknown noise type: {noise_name}")
        if noise_name in {"wgn", "pink"}:
            levels = snr_levels
        else:
            levels = amp_levels

        for level in levels:
            for denoise in denoise_methods:
                conditions.append(
                    NoiseCondition(
                        noise_type=noise_name,
                        level=float(level),
                        denoise_method=str(denoise),
                        noise_cfg_list=NOISE_TYPES[noise_name](level),
                    )
                )
    return conditions


def _match_length(signal: np.ndarray, target_len: int) -> np.ndarray:
    if signal.shape[0] == target_len:
        return signal
    if signal.shape[0] > target_len:
        return signal[:target_len]
    pad = target_len - signal.shape[0]
    return np.pad(signal, (0, pad), mode="edge")


def apply_denoise_window(window: np.ndarray, fs: float, method: str) -> np.ndarray:
    method = method.lower()
    if method == "none":
        return window
    if method == "pca":
        return pca_denoise(window, k_remove=1)

    out = np.zeros_like(window)
    for ch in range(window.shape[1]):
        if method == "notch":
            cleaned = notch_filter(window[:, ch], fs=fs)
        elif method == "wavelet":
            cleaned = wavelet_denoise(window[:, ch])
        elif method == "kalman":
            cleaned = kalman_denoise(window[:, ch])
        else:
            raise ValueError(f"Unknown denoise method: {method}")
        out[:, ch] = _match_length(np.asarray(cleaned, dtype=np.float32), window.shape[0])
    return out


def apply_noise_window(window: np.ndarray, fs: float, noise_cfg_list: list[dict[str, object]] | None) -> np.ndarray:
    if noise_cfg_list is None:
        return window.copy()
    out = np.zeros_like(window)
    for ch in range(window.shape[1]):
        out[:, ch] = apply_noises(window[:, ch], fs=fs, noise_cfg_list=noise_cfg_list)
    return out


def apply_noise_batch(
    windows: np.ndarray,
    fs: float,
    noise_cfg_list: list[dict[str, object]] | None,
    denoise_method: str,
    seed: int | None = None,
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    noisy = np.empty_like(windows, dtype=np.float32)
    for idx in tqdm(range(int(windows.shape[0])), desc="Noise injection", leave=False):
        window = windows[idx]
        window = apply_noise_window(window, fs, noise_cfg_list)
        window = apply_denoise_window(window, fs, denoise_method)
        noisy[idx] = window.astype(np.float32)
    return noisy


def build_subset_dataset(dataset: Dict[str, np.ndarray], indices: np.ndarray, windows: np.ndarray) -> Dict[str, np.ndarray]:
    subset = {
        "x": windows,
        "y": dataset["y"][indices],
        "subject_id": dataset["subject_id"][indices],
        "exercise_id": dataset["exercise_id"][indices],
        "gesture_id": dataset["gesture_id"][indices],
        "repetition_id": dataset["repetition_id"][indices],
        "window_start": dataset["window_start"][indices],
        "window_end": dataset["window_end"][indices],
        "class_gesture_ids": dataset["class_gesture_ids"],
    }
    return subset


def evaluate_condition(
    condition: NoiseCondition,
    base_windows: np.ndarray,
    dataset: Dict[str, np.ndarray],
    indices: np.ndarray,
    normalizer: ChannelwiseNormalizer,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    num_classes: int,
    aggregation: str,
    image_input: bool,
    output_dir: str,
    seed: int | None = None,
) -> dict[str, object]:
    fs = float(dataset.get("sampling_rate_hz", 0.0))
    noisy_windows = apply_noise_batch(
        base_windows,
        fs=fs,
        noise_cfg_list=condition.noise_cfg_list,
        denoise_method=condition.denoise_method,
        seed=seed,
    )

    subset = build_subset_dataset(dataset, indices, noisy_windows)
    subset_indices = np.arange(int(noisy_windows.shape[0]), dtype=np.int64)

    eval_dataset = WindowedEMGDataset(
        subset["x"],
        subset["y"],
        subset_indices,
        normalizer=normalizer,
        augmenter=None,
        image_input=image_input,
    )
    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    criterion = nn.CrossEntropyLoss()
    outputs = collect_window_outputs(model, loader, criterion, device, amp_enabled=False)
    window_metrics = compute_metrics(outputs["logits"], outputs["targets"], num_classes)
    window_metrics["loss"] = float(outputs["loss"])

    window_rows = build_window_prediction_rows(
        dataset=subset,
        indices=subset_indices,
        logits=outputs["logits"],
        targets=outputs["targets"],
    )
    save_prediction_csv(window_rows, os.path.join(output_dir, "window_predictions.csv"), WINDOW_PRED_FIELDS)

    subjects = subset["subject_id"][subset_indices].astype(np.int64)
    gestures = subset["gesture_id"][subset_indices].astype(np.int64)
    repetitions = subset["repetition_id"][subset_indices].astype(np.int64)
    repetition_rows, rep_targets, rep_predictions = aggregate_repetition_predictions(
        logits=outputs["logits"],
        targets=outputs["targets"],
        subjects=subjects,
        gestures=gestures,
        repetitions=repetitions,
        num_classes=num_classes,
        aggregation=aggregation,
    )
    save_prediction_csv(
        repetition_rows,
        os.path.join(output_dir, "repetition_predictions.csv"),
        REPETITION_PRED_FIELDS,
    )

    labels = list(range(int(num_classes)))
    rep_confusion = confusion_matrix(rep_targets, rep_predictions, labels=labels)
    repetition_metrics = {
        "accuracy": float(accuracy_score(rep_targets, rep_predictions)),
        "macro_f1": float(f1_score(rep_targets, rep_predictions, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": rep_confusion,
    }

    class_gesture_ids = [int(value) for value in dataset["class_gesture_ids"]]
    save_confusion_matrix_outputs(window_metrics["confusion_matrix"], class_gesture_ids, output_dir, "window")
    save_confusion_matrix_outputs(rep_confusion, class_gesture_ids, output_dir, "repetition")
    save_per_class_accuracy_outputs(window_metrics["confusion_matrix"], class_gesture_ids, output_dir, "window")
    save_per_class_accuracy_outputs(rep_confusion, class_gesture_ids, output_dir, "repetition")

    summary = {
        "noise_type": condition.noise_type,
        "noise_level": float(condition.level),
        "denoise_method": condition.denoise_method,
        "window": {
            "loss": float(window_metrics["loss"]),
            "accuracy": float(window_metrics["accuracy"]),
            "macro_f1": float(window_metrics["macro_f1"]),
            "num_windows": int(subset_indices.shape[0]),
        },
        "repetition": {
            "accuracy": float(repetition_metrics["accuracy"]),
            "macro_f1": float(repetition_metrics["macro_f1"]),
            "num_repetitions": int(len(repetition_rows)),
        },
    }
    save_json(summary, os.path.join(output_dir, "metrics_summary.json"))
    return summary


def update_noise_summary(
    summary_path: str,
    results: Iterable[dict[str, object]],
    model_name: str,
    dataset: str,
    subject_setting: str,
    split_mode: str,
    evaluation_level: str,
) -> None:
    if not summary_path:
        return

    rows: list[dict[str, str]] = []
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            rows = [dict(row) for row in reader]

    index: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            row.get("method_family", ""),
            row.get("model", ""),
            row.get("noise_type", ""),
            row.get("noise_level", ""),
            row.get("denoise_method", ""),
        )
        index[key] = row

    for result in results:
        noise_type = str(result["noise_type"])
        noise_level = format_level(float(result["noise_level"]))
        denoise_method = str(result["denoise_method"])
        key = ("DL", model_name, noise_type, noise_level, denoise_method)
        row = index.get(key)
        if row is None:
            row = {
                "experiment_id": f"dl_noise_{noise_type}_lvl{level_slug(float(result['noise_level']))}_{denoise_method}",
                "category": "noise_robustness",
                "method_family": "DL",
                "model": model_name,
                "dataset": dataset,
                "subject_setting": subject_setting,
                "split_mode": split_mode,
                "evaluation_level": evaluation_level,
                "noise_type": noise_type,
                "noise_level": noise_level,
                "denoise_method": denoise_method,
                "accuracy": "",
                "macro_f1": "",
                "std_accuracy": "",
                "std_macro_f1": "",
                "source_metrics_file": "",
                "notes": "DL noise eval",
            }
            rows.append(row)
            index[key] = row
        else:
            row["category"] = "noise_robustness"
            row["method_family"] = "DL"
            row["model"] = model_name
            row["dataset"] = dataset
            row["subject_setting"] = subject_setting
            row["split_mode"] = split_mode
            row["evaluation_level"] = evaluation_level
            row["noise_type"] = noise_type
            row["noise_level"] = noise_level
            row["denoise_method"] = denoise_method

        row["accuracy"] = f"{result['window']['accuracy']:.6f}"
        row["macro_f1"] = f"{result['window']['macro_f1']:.6f}"
        row["source_metrics_file"] = str(result.get("metrics_path", row.get("source_metrics_file", "")))
        row["notes"] = "DL noise eval (window-level)"

    fieldnames = [
        "experiment_id",
        "category",
        "method_family",
        "model",
        "dataset",
        "subject_setting",
        "split_mode",
        "evaluation_level",
        "noise_type",
        "noise_level",
        "denoise_method",
        "accuracy",
        "macro_f1",
        "std_accuracy",
        "std_macro_f1",
        "source_metrics_file",
        "notes",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DL robustness under the ML noise protocol.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the DB5 window dataset (.npz)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained DL checkpoint (best.pt)")
    parser.add_argument("--subject", type=int, default=None, help="Optional subject ID for subject-dependent evaluation")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["repetition_holdout", "subject_holdout", "kfold"],
        default="",
    )
    parser.add_argument("--train-reps", type=str, default="")
    parser.add_argument("--val-reps", type=str, default="")
    parser.add_argument("--test-reps", type=str, default="")
    parser.add_argument("--train-subjects", type=str, default="")
    parser.add_argument("--val-subjects", type=str, default="")
    parser.add_argument("--test-subjects", type=str, default="")
    parser.add_argument("--fold-file", type=str, default="")
    parser.add_argument("--test-fold", type=int, default=None)
    parser.add_argument("--val-fold", type=int, default=None)
    parser.add_argument("--aggregation", type=str, choices=["logits_mean", "majority_vote"], default="logits_mean")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-types", type=str, default="")
    parser.add_argument("--denoise-methods", type=str, default="")
    parser.add_argument("--snr-levels", type=str, default="")
    parser.add_argument("--amp-levels", type=str, default="")
    parser.add_argument("--include-clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=str, default="outputs/dl_noise_eval")
    parser.add_argument("--summary-csv", type=str, default="outputs/summaries/dl_noise_results.csv")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = safe_torch_load(args.checkpoint, map_location=device)
    split_mode = args.split_mode or str(checkpoint.get("split_config", {}).get("split_mode", "repetition_holdout"))

    dataset = load_window_dataset(args.dataset)
    if args.subject is not None:
        if split_mode == "subject_holdout":
            raise ValueError("--subject is not compatible with split-mode subject_holdout.")
        dataset = filter_dataset_by_subject(dataset, int(args.subject))
        print(f"[INFO] Filtered dataset to subject {int(args.subject)}.")

    fold_ids = None
    if split_mode == "kfold":
        fold_ids = load_fold_ids(args.fold_file, int(dataset["x"].shape[0]))

    split_indices, split_config = build_split_indices(
        dataset,
        split_mode=split_mode,
        train_reps=parse_int_list(args.train_reps) or checkpoint.get("split_config", {}).get("train_reps", []),
        val_reps=parse_int_list(args.val_reps) or checkpoint.get("split_config", {}).get("val_reps", []),
        test_reps=parse_int_list(args.test_reps) or checkpoint.get("split_config", {}).get("test_reps", []),
        train_subjects=parse_int_list(args.train_subjects) or checkpoint.get("split_config", {}).get("train_subjects", []),
        val_subjects=parse_int_list(args.val_subjects) or checkpoint.get("split_config", {}).get("val_subjects", []),
        test_subjects=parse_int_list(args.test_subjects) or checkpoint.get("split_config", {}).get("test_subjects", []),
        fold_ids=fold_ids,
        test_fold=args.test_fold,
        val_fold=args.val_fold,
    )
    assert_no_overlap(dataset, split_indices, check_group_overlap=split_mode != "kfold")
    split_summary = make_split_summary(dataset, split_indices, split_config)
    print_split_summary(split_summary)

    selected_indices = getattr(split_indices, args.split)
    base_windows = dataset["x"][selected_indices]

    noise_types = parse_list(args.noise_types) or list(NOISE_TYPES.keys())
    denoise_methods = parse_list(args.denoise_methods) or DEFAULT_DENOISE_METHODS
    snr_levels = parse_float_list(args.snr_levels) or DEFAULT_SNR_LEVELS
    amp_levels = parse_float_list(args.amp_levels) or DEFAULT_AMP_LEVELS
    conditions = build_noise_conditions(
        noise_types=noise_types,
        snr_levels=snr_levels,
        amp_levels=amp_levels,
        denoise_methods=denoise_methods,
        include_clean=bool(args.include_clean),
    )

    model_name = str(checkpoint["model_name"])
    model = build_model_from_config(model_name, dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    normalizer = ChannelwiseNormalizer.from_state(dict(checkpoint["normalizer"]))
    image_input = requires_pseudo_image_input(model_name)
    num_classes = int(len(checkpoint["class_gesture_ids"]))

    summary_rows = []
    for condition in conditions:
        if condition.noise_type == "clean":
            condition_name = "clean_lvl0_none"
        else:
            condition_name = f"{condition.noise_type}_lvl{level_slug(condition.level)}_{condition.denoise_method}"
        condition_dir = os.path.join(args.out_dir, model_name, condition_name)
        ensure_dir(condition_dir)

        summary = evaluate_condition(
            condition=condition,
            base_windows=base_windows,
            dataset=dataset,
            indices=selected_indices,
            normalizer=normalizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_classes=num_classes,
            aggregation=args.aggregation,
            image_input=image_input,
            output_dir=condition_dir,
            seed=int(args.seed),
        )
        summary["metrics_path"] = os.path.join(condition_dir, "metrics_summary.json")
        summary_rows.append(summary)

    subject_setting = "subject-dependent (S1-3)" if args.subject is None else f"subject-dependent (S{args.subject})"
    update_noise_summary(
        summary_path=args.summary_csv,
        results=summary_rows,
        model_name=model_name,
        dataset="db5",
        subject_setting=subject_setting,
        split_mode=str(split_mode),
        evaluation_level="window",
    )

    summary_csv = os.path.join(args.out_dir, model_name, "dl_noise_summary.csv")
    ensure_dir(os.path.dirname(summary_csv))
    with open(summary_csv, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "noise_type",
            "noise_level",
            "denoise_method",
            "window_accuracy",
            "window_macro_f1",
            "repetition_accuracy",
            "repetition_macro_f1",
            "metrics_path",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    "noise_type": row["noise_type"],
                    "noise_level": format_level(row["noise_level"]),
                    "denoise_method": row["denoise_method"],
                    "window_accuracy": f"{row['window']['accuracy']:.6f}",
                    "window_macro_f1": f"{row['window']['macro_f1']:.6f}",
                    "repetition_accuracy": f"{row['repetition']['accuracy']:.6f}",
                    "repetition_macro_f1": f"{row['repetition']['macro_f1']:.6f}",
                    "metrics_path": row["metrics_path"],
                }
            )


if __name__ == "__main__":
    main()
