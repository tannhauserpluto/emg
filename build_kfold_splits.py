# -*- coding: utf-8 -*-
"""Build stratified k-fold assignments for DB5 window datasets."""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold

from dl_dataset import load_window_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate stratified k-fold assignments for a DB5 window dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the window .npz dataset.")
    parser.add_argument("--out", type=str, required=True, help="Output .npz file for fold assignments.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    data = load_window_dataset(args.dataset)
    labels = data["y"].astype(np.int64)
    num_samples = int(labels.shape[0])

    skf = StratifiedKFold(n_splits=int(args.n_splits), shuffle=True, random_state=int(args.seed))
    fold_ids = np.empty(num_samples, dtype=np.int64)

    for fold_index, (_, test_indices) in enumerate(skf.split(np.zeros(num_samples), labels)):
        fold_ids[test_indices] = int(fold_index + 1)

    if int(fold_ids.min()) < 1 or int(fold_ids.max()) > int(args.n_splits):
        raise RuntimeError("Fold assignment contains out-of-range values.")

    np.savez(
        args.out,
        fold_id=fold_ids,
        n_splits=int(args.n_splits),
        seed=int(args.seed),
        dataset_path=str(args.dataset),
    )
    unique_folds = sorted(int(value) for value in np.unique(fold_ids).tolist())
    print(f"[INFO] Saved k-fold assignments to {args.out}")
    print(f"[INFO] Unique folds: {unique_folds}, total samples: {num_samples}")


if __name__ == "__main__":
    main()
