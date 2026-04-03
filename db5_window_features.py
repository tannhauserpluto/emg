from __future__ import annotations

"""Export handcrafted features aligned 1-to-1 with a DB5 raw-window dataset.

Examples
--------
python db5_window_features.py --dataset data/ninapro_db5/windows/db5_s1-3_g1-10_win400_step50.npz

python db5_window_features.py --dataset data/ninapro_db5/windows/db5_s1-3_g1-10_win400_step50.npz \
    --out data/ninapro_db5/window_features/db5_s1-3_g1-10_win400_step50_features.npz

python train_dl.py --dataset data/ninapro_db5/windows/db5_s1-3_g1-10_win400_step50.npz \
    --feature-path data/ninapro_db5/window_features/db5_s1-3_g1-10_win400_step50_features.npz \
    --model fusion_tcn_mlp --split-mode repetition_holdout --train-reps 1,3,4 --val-reps 6 --test-reps 2,5
"""

import argparse
import json
import os

import numpy as np

from dl_dataset import load_window_dataset
from src.features import compute_td_features


def default_output_path(dataset_path: str) -> str:
    dataset_stem = os.path.splitext(os.path.basename(dataset_path))[0]
    return os.path.join("data", "ninapro_db5", "window_features", f"{dataset_stem}_features.npz")


def build_window_feature_dataset(window_dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    windows = window_dataset["x"]
    sampling_rate_hz = int(np.asarray(window_dataset.get("sampling_rate_hz", np.asarray([200]))).item())

    feature_rows = []
    for index in range(int(windows.shape[0])):
        feature_vector = compute_td_features(windows[index], fs=float(sampling_rate_hz)).astype(np.float32)
        feature_rows.append(feature_vector)

    feature_matrix = np.stack(feature_rows).astype(np.float32)
    output = {
        "features": feature_matrix,
        "feature_names": np.asarray([f"f{index}" for index in range(int(feature_matrix.shape[1]))]),
        "feature_dim": np.asarray(int(feature_matrix.shape[1]), dtype=np.int64),
        "source_sampling_rate_hz": np.asarray(sampling_rate_hz, dtype=np.int64),
    }

    for key, value in window_dataset.items():
        if key == "x":
            continue
        output[key] = value

    return output


def save_feature_dataset(feature_dataset: dict[str, np.ndarray], output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(output_path, **feature_dataset)

    summary = {
        "num_samples": int(feature_dataset["features"].shape[0]),
        "feature_dim": int(feature_dataset["features"].shape[1]),
        "subjects": [int(value) for value in np.unique(feature_dataset["subject_id"]).tolist()],
        "repetitions": [int(value) for value in np.unique(feature_dataset["repetition_id"]).tolist()],
        "gestures": [int(value) for value in np.unique(feature_dataset["gesture_id"]).tolist()],
    }
    summary_path = os.path.splitext(output_path)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved window feature dataset to {output_path}")
    print(f"[INFO] Saved summary to {summary_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export handcrafted features aligned to a DB5 raw-window dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to a raw DB5 window dataset .npz")
    parser.add_argument("--out", type=str, default="", help="Output feature dataset path")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    window_dataset = load_window_dataset(args.dataset)
    output_path = args.out or default_output_path(args.dataset)
    feature_dataset = build_window_feature_dataset(window_dataset)

    for key in ("y", "subject_id", "exercise_id", "gesture_id", "repetition_id", "window_start", "window_end"):
        if not np.array_equal(feature_dataset[key], window_dataset[key]):
            raise AssertionError(f"Feature dataset key {key} is not aligned with the raw window dataset.")

    save_feature_dataset(feature_dataset, output_path)


if __name__ == "__main__":
    main()
