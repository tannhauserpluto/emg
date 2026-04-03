# -*- coding: utf-8 -*-
# run_feature_ablation.py
"""
NinaPro DB5 特征消融实验脚本

对同一份 DB5 特征数据集，比较：
  - TD        （时域 5 特征）
  - TD+FD     （时域 5 + 频域 3）
  - TD+FD+AR  （时域 5 + 频域 3 + AR(4)）

每种特征组合上分别训练：
  - LDA
  - SVM_RBF
  - RandomForest

并做 5 折交叉验证，结果保存到：
  report/ml_db5_feature_ablation.csv
"""

import os
import pandas as pd

from src.ninapro_db5 import build_features_db5, FEATURE_DIR
from src.ml_models import cross_val_evaluate


def load_or_build_db5_features(
    subjects,
    gestures,
    out_name="db5_s1-3_g1-10.csv",
    **kwargs,
) -> pd.DataFrame:
    """
    如果已经存在特征 CSV，则直接读取；
    否则调用 build_features_db5 重新生成。
    """
    os.makedirs(FEATURE_DIR, exist_ok=True)
    feat_path = os.path.join(FEATURE_DIR, out_name)

    if os.path.exists(feat_path):
        print(f"[INFO] Loading existing features: {feat_path}")
        df = pd.read_csv(feat_path)
    else:
        print(f"[INFO] Building features to: {feat_path}")
        df = build_features_db5(
            subjects=subjects,
            gestures=gestures,
            out_name=out_name,
            **kwargs,
        )
    return df


def get_feature_index_groups(feature_cols):
    """
    根据特征索引划分:
      每个通道 12 个特征： 0..4 = TD, 5..7 = FD, 8..11 = AR
    输入: feature_cols 形如 ["f0","f1",...]
    返回: 三个列表（里面是整数索引）:
      idx_td_only, idx_td_fd, idx_td_fd_ar
    """
    # 将 "f0","f1"... 按索引排序
    idx_all = sorted([int(c[1:]) for c in feature_cols])
    n_feats = len(idx_all)

    if n_feats % 12 != 0:
        print(f"[WARN] feature_dim={n_feats} 不能被 12 整除，"
              f"可能不是 5(TD)+3(FD)+4(AR) 模式，请检查。")
    n_channels = n_feats // 12
    print(f"[INFO] 推断通道数 n_channels = {n_channels}")

    idx_td = []
    idx_td_fd = []
    idx_all_feats = []

    for ch in range(n_channels):
        base = ch * 12
        td = list(range(base + 0, base + 5))   # 0..4
        fd = list(range(base + 5, base + 8))   # 5..7
        ar = list(range(base + 8, base + 12))  # 8..11

        idx_td.extend(td)
        idx_td_fd.extend(td + fd)
        idx_all_feats.extend(td + fd + ar)

    return idx_td, idx_td_fd, idx_all_feats


def subset_X_by_indices(df: pd.DataFrame, indices):
    """
    根据索引列表 [0,1,2,...] 选取对应的 f{idx} 特征列，返回 X。
    """
    cols = [f"f{k}" for k in indices if f"f{k}" in df.columns]
    X = df[cols].values
    return X, cols


def main():
    # 与 run_db5_ml.py 保持一致
    subjects = [1, 2, 3]
    gestures = list(range(1, 11))
    out_name = "db5_s1-3_g1-10.csv"

    # 1. 读取或生成 DB5 特征
    df = load_or_build_db5_features(
        subjects=subjects,
        gestures=gestures,
        out_name=out_name,
        max_reps=6,
        win_sec=0.200,
        step_sec=0.050,
    )

    feature_cols = sorted([c for c in df.columns if c.startswith("f")],
                          key=lambda s: int(s[1:]))

    X_all = df[feature_cols].values
    y = df["gesture"].values

    print(f"[INFO] Dataset: {X_all.shape[0]} samples, "
          f"feature_dim = {X_all.shape[1]}, "
          f"gestures = {sorted(df['gesture'].unique())}")

    # 2. 按通道拆分特征索引
    idx_td, idx_td_fd, idx_td_fd_ar = get_feature_index_groups(feature_cols)

    setting_defs = [
        ("TD", idx_td),
        ("TD+FD", idx_td_fd),
        ("TD+FD+AR", idx_td_fd_ar),
    ]

    all_results = []

    # 3. 对每种特征组合跑一遍 ML baseline
    for name, idx_list in setting_defs:
        print(f"\n[ABLT] Running setting: {name}")
        X_sub, used_cols = subset_X_by_indices(df, idx_list)

        print(f"[ABLT]  feature_dim = {X_sub.shape[1]} "
              f"(selected {len(used_cols)} cols)")

        res = cross_val_evaluate(X_sub, y, n_splits=5, random_state=42)
        res.insert(0, "feature_set", name)  # 在最前面加一列

        print(res)

        all_results.append(res)

    # 4. 汇总 & 保存
    results_all = pd.concat(all_results, ignore_index=True)

    os.makedirs("report", exist_ok=True)
    out_csv = os.path.join("report", "ml_db5_feature_ablation.csv")
    results_all.to_csv(out_csv, index=False)

    print(f"\n[INFO] Saved feature ablation results to: {out_csv}")
    print("\n[INFO] 完成：你现在有一张表，可以直接放进论文：")
    print(results_all)


if __name__ == "__main__":
    main()
