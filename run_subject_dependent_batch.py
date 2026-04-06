from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Sequence

import numpy as np

from dl_dataset import parse_int_list


SUMMARY_FIELDS = [
    "subject",
    "window_accuracy",
    "window_macro_f1",
    "repetition_accuracy",
    "repetition_macro_f1",
    "best_epoch",
    "n_train_windows",
    "n_val_windows",
    "n_test_windows",
]


def format_rep_tag(text: str) -> str:
    reps = parse_int_list(text)
    if not reps:
        raise ValueError("Repetition list cannot be empty.")
    return "".join(str(value) for value in reps)


def run_command(command: Sequence[str]) -> None:
    print("[INFO] Running:", " ".join(command))
    subprocess.run(list(command), check=True)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_summary_csv(rows: list[dict[str, object]], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def append_mean_std(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return rows
    metric_fields = [field for field in SUMMARY_FIELDS if field != "subject"]
    values = {field: np.asarray([row[field] for row in rows], dtype=np.float64) for field in metric_fields}
    mean_row = {"subject": "mean"}
    std_row = {"subject": "std"}
    for field, series in values.items():
        mean_row[field] = float(np.mean(series))
        std_row[field] = float(np.std(series))
    return rows + [mean_row, std_row]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch train/evaluate subject-dependent repetition-holdout runs.")
    parser.add_argument("--dataset", type=str, required=True, help="DB5 raw-window dataset path")
    parser.add_argument("--subjects", type=str, default="1-3", help="Subject IDs to run, e.g. 1-3")
    parser.add_argument("--model", type=str, default="dualres_xception2d")
    parser.add_argument("--train-reps", type=str, default="1,2,3,4")
    parser.add_argument("--val-reps", type=str, default="5")
    parser.add_argument("--test-reps", type=str, default="6")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--augment-scale", type=float, default=0.0)
    parser.add_argument("--augment-noise", type=float, default=0.0)
    parser.add_argument("--augment-shift", type=int, default=0)
    parser.add_argument("--augment-channel-dropout", type=float, default=0.0)
    parser.add_argument("--aggregation", type=str, choices=["logits_mean", "majority_vote"], default="logits_mean")
    parser.add_argument("--output-root", type=str, default=os.path.join("outputs", "dl_subject_dependent"))
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    subjects = parse_int_list(args.subjects)
    if not subjects:
        raise ValueError("--subjects parsed to an empty list.")

    train_tag = format_rep_tag(args.train_reps)
    val_tag = format_rep_tag(args.val_reps)
    test_tag = format_rep_tag(args.test_reps)
    model_root = os.path.join(args.output_root, args.model)
    os.makedirs(model_root, exist_ok=True)

    summary_rows: list[dict[str, object]] = []

    for subject_id in subjects:
        run_dir = os.path.join(model_root, f"s{subject_id}_rep{train_tag}_val{val_tag}_test{test_tag}")
        os.makedirs(run_dir, exist_ok=True)

        train_cmd = [
            sys.executable,
            "train_dl.py",
            "--dataset",
            args.dataset,
            "--model",
            args.model,
            "--subject",
            str(subject_id),
            "--split-mode",
            "repetition_holdout",
            "--train-reps",
            args.train_reps,
            "--val-reps",
            args.val_reps,
            "--test-reps",
            args.test_reps,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--dropout",
            str(args.dropout),
            "--augment-scale",
            str(args.augment_scale),
            "--augment-noise",
            str(args.augment_noise),
            "--augment-shift",
            str(args.augment_shift),
            "--augment-channel-dropout",
            str(args.augment_channel_dropout),
            "--run-dir",
            run_dir,
        ]
        run_command(train_cmd)

        checkpoint_path = os.path.join(run_dir, "best.pt")
        eval_cmd = [
            sys.executable,
            "eval_dl.py",
            "--dataset",
            args.dataset,
            "--checkpoint",
            checkpoint_path,
            "--subject",
            str(subject_id),
            "--split",
            "test",
            "--split-mode",
            "repetition_holdout",
            "--train-reps",
            args.train_reps,
            "--val-reps",
            args.val_reps,
            "--test-reps",
            args.test_reps,
            "--aggregation",
            args.aggregation,
            "--out-dir",
            run_dir,
        ]
        run_command(eval_cmd)

        metrics = load_json(os.path.join(run_dir, "metrics_summary.json"))
        split_summary = load_json(os.path.join(run_dir, "split_summary.json"))
        summary_rows.append(
            {
                "subject": int(subject_id),
                "window_accuracy": float(metrics["window"]["accuracy"]),
                "window_macro_f1": float(metrics["window"]["macro_f1"]),
                "repetition_accuracy": float(metrics["repetition"]["accuracy"]),
                "repetition_macro_f1": float(metrics["repetition"]["macro_f1"]),
                "best_epoch": int(metrics.get("best_epoch", 0)),
                "n_train_windows": int(split_summary["splits"]["train"]["num_windows"]),
                "n_val_windows": int(split_summary["splits"]["val"]["num_windows"]),
                "n_test_windows": int(split_summary["splits"]["test"]["num_windows"]),
            }
        )

    summary_rows = append_mean_std(summary_rows)
    summary_path = os.path.join(model_root, "summary.csv")
    save_summary_csv(summary_rows, summary_path)
    print(f"[INFO] Saved subject summary to {summary_path}")


if __name__ == "__main__":
    main()
