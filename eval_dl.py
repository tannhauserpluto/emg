from __future__ import annotations

"""Evaluate a saved deep learning checkpoint on a selected DB5 split."""

import argparse
import json
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from dl_dataset import (
    ChannelwiseNormalizer,
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
from train_dl import (
    build_model_from_config,
    evaluate_model,
    resolve_device,
    safe_torch_load,
    save_confusion_matrix_outputs,
    save_json,
    save_per_class_accuracy_outputs,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved deep learning checkpoint on DB5 splits.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the .npz dataset built by db5_windows.py")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt produced by train_dl.py")
    parser.add_argument("--feature-path", type=str, default="", help="Optional handcrafted feature dataset for fusion checkpoints")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["repetition_holdout", "subject_holdout"],
        default="",
    )
    parser.add_argument("--train-reps", type=str, default="")
    parser.add_argument("--val-reps", type=str, default="")
    parser.add_argument("--test-reps", type=str, default="")
    parser.add_argument("--train-subjects", type=str, default="")
    parser.add_argument("--val-subjects", type=str, default="")
    parser.add_argument("--test-subjects", type=str, default="")
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

    split_indices, split_config = build_split_indices(
        dataset,
        split_mode=split_mode,
        train_reps=parse_int_list(args.train_reps) or checkpoint_split_config.get("train_reps", []),
        val_reps=parse_int_list(args.val_reps) or checkpoint_split_config.get("val_reps", []),
        test_reps=parse_int_list(args.test_reps) or checkpoint_split_config.get("test_reps", []),
        train_subjects=parse_int_list(args.train_subjects) or checkpoint_split_config.get("train_subjects", []),
        val_subjects=parse_int_list(args.val_subjects) or checkpoint_split_config.get("val_subjects", []),
        test_subjects=parse_int_list(args.test_subjects) or checkpoint_split_config.get("test_subjects", []),
    )
    assert_no_overlap(dataset, split_indices)

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
    evaluation_dataset = WindowedEMGDataset(
        dataset["x"],
        dataset["y"],
        selected_indices,
        normalizer=normalizer,
        feature_matrix=feature_matrix,
        feature_normalizer=feature_normalizer,
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
    metrics = evaluate_model(model, data_loader, nn.CrossEntropyLoss(), device, num_classes)
    class_gesture_ids = [int(value) for value in checkpoint["class_gesture_ids"]]
    save_confusion_matrix_outputs(metrics["confusion_matrix"], class_gesture_ids, output_dir, args.split)
    save_per_class_accuracy_outputs(metrics["confusion_matrix"], class_gesture_ids, output_dir, args.split)

    summary = {
        "split": args.split,
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "checkpoint": args.checkpoint,
    }
    save_json(summary, os.path.join(output_dir, "metrics_summary.json"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
