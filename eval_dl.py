from __future__ import annotations

"""Evaluate a saved deep learning checkpoint on a selected DB5 split.

Examples
--------
python eval_dl.py --dataset data/ninapro_db5/windows/db5_s1-3_g1-10_win400_step50.npz --checkpoint outputs/dl_subject_dependent/dualres_xception2d/s1_rep1234_val5_test6/best.pt --subject 1 --split-mode repetition_holdout --train-reps 1,2,3,4 --val-reps 5 --test-reps 6 --aggregation logits_mean --out-dir outputs/dl_subject_dependent/dualres_xception2d/s1_rep1234_val5_test6
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader

from dl_dataset import (
    ChannelwiseNormalizer,
    VectorNormalizer,
    WindowedEMGDataset,
    assert_feature_alignment,
    assert_no_overlap,
    build_split_indices,
    filter_dataset_by_subject,
    load_fold_ids,
    load_feature_dataset,
    load_window_dataset,
    make_split_summary,
    parse_int_list,
    print_split_summary,
)
from train_dl import (
    autocast_context,
    build_model_from_config,
    forward_model,
    move_batch_to_device,
    requires_pseudo_image_input,
    resolve_device,
    safe_torch_load,
    save_confusion_matrix_outputs,
    save_json,
    save_per_class_accuracy_outputs,
)


WINDOW_PRED_FIELDS = [
    "subject",
    "gesture",
    "repetition",
    "sample_index",
    "y_true",
    "y_pred",
    "correct",
    "prob_max",
]

REPETITION_PRED_FIELDS = [
    "subject",
    "gesture",
    "repetition",
    "n_windows",
    "y_true",
    "y_pred",
    "correct",
    "prob_max",
    "aggregation_method",
]


def softmax_logits(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def collect_window_outputs(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    logits_batches: list[torch.Tensor] = []
    targets_batches: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in data_loader:
            window_batch, feature_batch, labels = move_batch_to_device(batch, device)
            with autocast_context(device, amp_enabled):
                logits = forward_model(model, window_batch, feature_batch)
                loss = criterion(logits, labels)

            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            logits_batches.append(logits.detach().cpu())
            targets_batches.append(labels.detach().cpu())

    logits_array = torch.cat(logits_batches, dim=0).numpy()
    targets_array = torch.cat(targets_batches, dim=0).numpy().astype(np.int64)
    return {
        "loss": total_loss / max(total_samples, 1),
        "logits": logits_array,
        "targets": targets_array,
    }


def compute_metrics(logits: np.ndarray, targets: np.ndarray, num_classes: int) -> dict[str, object]:
    predictions = logits.argmax(axis=1).astype(np.int64)
    labels = list(range(int(num_classes)))
    matrix = confusion_matrix(targets, predictions, labels=labels)
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": matrix,
        "predictions": predictions,
    }


def save_prediction_csv(rows: list[dict[str, object]], output_path: str, fieldnames: list[str]) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_window_prediction_rows(
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    logits: np.ndarray,
    targets: np.ndarray,
) -> list[dict[str, object]]:
    predictions = logits.argmax(axis=1).astype(np.int64)
    probs = softmax_logits(logits)
    prob_max = probs.max(axis=1)

    rows: list[dict[str, object]] = []
    for local_index, sample_index in enumerate(indices.tolist()):
        rows.append(
            {
                "subject": int(dataset["subject_id"][sample_index]),
                "gesture": int(dataset["gesture_id"][sample_index]),
                "repetition": int(dataset["repetition_id"][sample_index]),
                "sample_index": int(sample_index),
                "y_true": int(targets[local_index]),
                "y_pred": int(predictions[local_index]),
                "correct": int(predictions[local_index] == targets[local_index]),
                "prob_max": float(prob_max[local_index]),
            }
        )
    return rows


def aggregate_repetition_predictions(
    logits: np.ndarray,
    targets: np.ndarray,
    subjects: np.ndarray,
    gestures: np.ndarray,
    repetitions: np.ndarray,
    num_classes: int,
    aggregation: str,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    group_map: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index in range(int(targets.shape[0])):
        key = (int(subjects[index]), int(gestures[index]), int(repetitions[index]))
        group_map[key].append(int(index))

    rows: list[dict[str, object]] = []
    rep_targets: list[int] = []
    rep_predictions: list[int] = []
    for key in sorted(group_map.keys()):
        subject_id, gesture_id, repetition_id = key
        group_indices = group_map[key]
        group_targets = np.unique(targets[group_indices])
        if group_targets.size != 1:
            raise RuntimeError(
                f"Inconsistent targets within subject={subject_id}, gesture={gesture_id}, repetition={repetition_id}: {group_targets.tolist()}"
            )

        if aggregation == "logits_mean":
            aggregated_logits = logits[group_indices].mean(axis=0)
            pred = int(np.argmax(aggregated_logits))
            prob_max = float(softmax_logits(aggregated_logits[None, :]).max())
        elif aggregation == "majority_vote":
            window_preds = logits[group_indices].argmax(axis=1).astype(np.int64)
            counts = np.bincount(window_preds, minlength=int(num_classes))
            pred = int(np.argmax(counts))
            prob_max = float(counts.max() / max(counts.sum(), 1))
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")

        true_label = int(group_targets[0])
        rep_targets.append(true_label)
        rep_predictions.append(pred)
        rows.append(
            {
                "subject": int(subject_id),
                "gesture": int(gesture_id),
                "repetition": int(repetition_id),
                "n_windows": int(len(group_indices)),
                "y_true": int(true_label),
                "y_pred": int(pred),
                "correct": int(pred == true_label),
                "prob_max": float(prob_max),
                "aggregation_method": str(aggregation),
            }
        )

    return rows, np.asarray(rep_targets, dtype=np.int64), np.asarray(rep_predictions, dtype=np.int64)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved deep learning checkpoint on DB5 splits.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the .npz dataset built by db5_windows.py")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt produced by train_dl.py")
    parser.add_argument("--feature-path", type=str, default="", help="Optional handcrafted feature dataset for fusion checkpoints")
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
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["logits_mean", "majority_vote"],
        default="logits_mean",
        help="Repetition-level aggregation method",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = safe_torch_load(args.checkpoint, map_location=device)
    checkpoint_split_config = dict(checkpoint.get("split_config", {}))
    split_mode = args.split_mode or str(checkpoint_split_config.get("split_mode", "repetition_holdout"))

    dataset = load_window_dataset(args.dataset)
    feature_path = args.feature_path or str(checkpoint.get("feature_path", ""))
    feature_dataset = None
    feature_matrix = None
    feature_normalizer = None
    if str(checkpoint.get("model_name")) == "fusion_tcn_mlp":
        if not feature_path:
            raise ValueError("Fusion checkpoint requires --feature-path or a stored feature_path in the checkpoint.")
        feature_dataset = load_feature_dataset(feature_path)
        assert_feature_alignment(dataset, feature_dataset)
        feature_matrix = feature_dataset["features"]
        if checkpoint.get("feature_normalizer") is None:
            raise ValueError("Fusion checkpoint is missing feature_normalizer state.")
        feature_normalizer = VectorNormalizer.from_state(dict(checkpoint["feature_normalizer"]))

    if args.subject is not None:
        if split_mode == "subject_holdout":
            raise ValueError("--subject is not compatible with split-mode subject_holdout.")
        dataset = filter_dataset_by_subject(dataset, int(args.subject))
        if feature_dataset is not None:
            feature_dataset = filter_dataset_by_subject(feature_dataset, int(args.subject))
            feature_matrix = feature_dataset["features"]
        print(f"[INFO] Filtered dataset to subject {int(args.subject)}.")

    fold_ids = None
    if split_mode == "kfold":
        fold_ids = load_fold_ids(args.fold_file, int(dataset["x"].shape[0]))

    split_indices, split_config = build_split_indices(
        dataset,
        split_mode=split_mode,
        train_reps=parse_int_list(args.train_reps) or checkpoint_split_config.get("train_reps", []),
        val_reps=parse_int_list(args.val_reps) or checkpoint_split_config.get("val_reps", []),
        test_reps=parse_int_list(args.test_reps) or checkpoint_split_config.get("test_reps", []),
        train_subjects=parse_int_list(args.train_subjects) or checkpoint_split_config.get("train_subjects", []),
        val_subjects=parse_int_list(args.val_subjects) or checkpoint_split_config.get("val_subjects", []),
        test_subjects=parse_int_list(args.test_subjects) or checkpoint_split_config.get("test_subjects", []),
        fold_ids=fold_ids,
        test_fold=args.test_fold,
        val_fold=args.val_fold,
    )
    assert_no_overlap(dataset, split_indices, check_group_overlap=split_mode != "kfold")

    split_summary = make_split_summary(dataset, split_indices, split_config)
    print_split_summary(split_summary)

    if args.out_dir:
        output_dir = args.out_dir
    else:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        output_dir = os.path.join(checkpoint_dir, f"eval_{args.split}")
    os.makedirs(output_dir, exist_ok=True)
    save_json(split_summary, os.path.join(output_dir, "split_summary.json"))

    selected_indices = getattr(split_indices, args.split)
    normalizer = ChannelwiseNormalizer.from_state(dict(checkpoint["normalizer"]))
    image_input = requires_pseudo_image_input(str(checkpoint["model_name"]))
    evaluation_dataset = WindowedEMGDataset(
        dataset["x"],
        dataset["y"],
        selected_indices,
        normalizer=normalizer,
        feature_matrix=feature_matrix,
        feature_normalizer=feature_normalizer,
        image_input=image_input,
    )
    data_loader = DataLoader(
        evaluation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model_from_config(str(checkpoint["model_name"]), dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    num_classes = int(len(checkpoint["class_gesture_ids"]))
    criterion = nn.CrossEntropyLoss()
    window_outputs = collect_window_outputs(
        model,
        data_loader,
        criterion,
        device,
        amp_enabled=False,
    )
    expected_targets = dataset["y"][selected_indices].astype(np.int64)
    if not np.array_equal(expected_targets, window_outputs["targets"]):
        raise AssertionError("Window targets collected during evaluation do not match dataset labels.")

    window_metrics = compute_metrics(window_outputs["logits"], window_outputs["targets"], num_classes)
    window_metrics["loss"] = float(window_outputs["loss"])

    window_rows = build_window_prediction_rows(
        dataset=dataset,
        indices=selected_indices,
        logits=window_outputs["logits"],
        targets=window_outputs["targets"],
    )
    save_prediction_csv(window_rows, os.path.join(output_dir, "window_predictions.csv"), WINDOW_PRED_FIELDS)

    subjects = dataset["subject_id"][selected_indices].astype(np.int64)
    gestures = dataset["gesture_id"][selected_indices].astype(np.int64)
    repetitions = dataset["repetition_id"][selected_indices].astype(np.int64)
    repetition_rows, rep_targets, rep_predictions = aggregate_repetition_predictions(
        logits=window_outputs["logits"],
        targets=window_outputs["targets"],
        subjects=subjects,
        gestures=gestures,
        repetitions=repetitions,
        num_classes=num_classes,
        aggregation=args.aggregation,
    )
    save_prediction_csv(repetition_rows, os.path.join(output_dir, "repetition_predictions.csv"), REPETITION_PRED_FIELDS)

    labels = list(range(int(num_classes)))
    rep_confusion = confusion_matrix(rep_targets, rep_predictions, labels=labels)
    repetition_metrics = {
        "accuracy": float(accuracy_score(rep_targets, rep_predictions)),
        "macro_f1": float(f1_score(rep_targets, rep_predictions, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": rep_confusion,
    }

    class_gesture_ids = [int(value) for value in checkpoint["class_gesture_ids"]]
    save_confusion_matrix_outputs(window_metrics["confusion_matrix"], class_gesture_ids, output_dir, "window")
    save_confusion_matrix_outputs(rep_confusion, class_gesture_ids, output_dir, "repetition")
    save_per_class_accuracy_outputs(window_metrics["confusion_matrix"], class_gesture_ids, output_dir, "window")
    save_per_class_accuracy_outputs(rep_confusion, class_gesture_ids, output_dir, "repetition")

    summary = {
        "split": args.split,
        "aggregation": args.aggregation,
        "window": {
            "loss": float(window_metrics["loss"]),
            "accuracy": float(window_metrics["accuracy"]),
            "macro_f1": float(window_metrics["macro_f1"]),
            "num_windows": int(selected_indices.shape[0]),
        },
        "repetition": {
            "accuracy": float(repetition_metrics["accuracy"]),
            "macro_f1": float(repetition_metrics["macro_f1"]),
            "num_repetitions": int(len(repetition_rows)),
        },
        "checkpoint": args.checkpoint,
    }
    if args.subject is not None:
        summary["subject"] = int(args.subject)
    if checkpoint.get("best_epoch") is not None:
        summary["best_epoch"] = int(checkpoint["best_epoch"])
    if checkpoint.get("best_val_macro_f1") is not None:
        summary["best_val_macro_f1"] = float(checkpoint["best_val_macro_f1"])
    save_json(summary, os.path.join(output_dir, "metrics_summary.json"))
    save_json(summary, os.path.join(output_dir, "metrics.json"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
