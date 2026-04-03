# -*- coding: utf-8 -*-
# run_db5_ml.py
"""
NinaPro DB5:
- 从 data/ninapro_db5/zip/sX.zip 解压 & 解析
- 构建 TD 特征数据集
- 用之前的 ml_models 做交叉验证
"""

import os
import pandas as pd

from src.ninapro_db5 import build_features_db5
from src.ml_models import cross_val_evaluate


def main():
    # 你可以先只用 1~3 号受试者，减小规模
    subjects = [1, 2, 3]

    # 选前 10 个手势做实验（DB5 gesture label 1..52 中的子集）
    # 后面你可以根据论文或需要，改成特定的一组手势。
    gestures = list(range(1, 11))

    df = build_features_db5(
        subjects=subjects,
        gestures=gestures,
        out_name="db5_s1-3_g1-10.csv",
        max_reps=6,
        win_sec=0.200,
        step_sec=0.050,
    )

    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].values
    y = df["gesture"].values

    print(f"\n[INFO] Final dataset: {X.shape[0]} samples, "
          f"feature_dim = {X.shape[1]}, "
          f"gestures = {sorted(df['gesture'].unique())}")

    results = cross_val_evaluate(X, y, n_splits=5, random_state=42)

    print("\n[ML baseline on NinaPro DB5]")
    print(results)

    os.makedirs("report", exist_ok=True)
    out_csv = os.path.join("report", "ml_db5_baseline.csv")
    results.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved ML baseline results to: {out_csv}")


if __name__ == "__main__":
    main()
