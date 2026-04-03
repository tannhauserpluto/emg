# -*- coding: utf-8 -*-
# run_ml_baseline.py
"""
从现有 pipeline 输出的：
  - data/intermediate/cleaned_gX.csv
  - data/intermediate/gX_segments.json

构建特征数据集 -> 训练 LDA / SVM / RF -> 输出结果到 report/ml_baseline.csv
"""

import os
import pandas as pd

from src.features import extract_features_for_gesture
from src.ml_models import cross_val_evaluate


def build_dataset(intermediate_dir: str = "data/intermediate",
                  n_gestures: int = 6):
    """
    汇总 1..n_gestures 的特征，返回 X, y, df_all。
    gesture_label 就用 1..n_gestures 的编号。
    """
    all_dfs = []

    for i in range(1, n_gestures + 1):
        clean_csv = os.path.join(intermediate_dir, f"cleaned_g{i}.csv")
        seg_json = os.path.join(intermediate_dir, f"g{i}_segments.json")

        if not (os.path.exists(clean_csv) and os.path.exists(seg_json)):
            print(f"[WARN] Missing files for gesture {i}, skip.")
            continue

        df_feat = extract_features_for_gesture(
            clean_csv_path=clean_csv,
            seg_json_path=seg_json,
            gesture_label=i,
        )
        all_dfs.append(df_feat)

    if not all_dfs:
        raise RuntimeError("No feature data built. Please check intermediate_dir.")

    df_all = pd.concat(all_dfs, ignore_index=True)

    feature_cols = [c for c in df_all.columns if c.startswith("f")]
    X = df_all[feature_cols].values
    y = df_all["gesture"].values

    return X, y, df_all


def main():
    os.makedirs("report", exist_ok=True)

    X, y, df_all = build_dataset(intermediate_dir="data/intermediate", n_gestures=6)

    print(f"Total segments: {len(df_all)}, feature_dim = {X.shape[1]}")

    results = cross_val_evaluate(X, y, n_splits=5, random_state=42)
    print("\nCross-validation results:")
    print(results)

    out_csv = os.path.join("report", "ml_baseline.csv")
    results.to_csv(out_csv, index=False)
    print(f"\nSaved ML baseline results to: {out_csv}")


if __name__ == "__main__":
    main()
