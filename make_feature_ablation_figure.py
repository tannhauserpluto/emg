# -*- coding: utf-8 -*-
# make_feature_ablation_figure.py

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


FEATURE_ORDER = ["TD", "TD+FD", "TD+FD+AR"]


def load_and_validate(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = {"feature_set", "model", "acc_mean"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")

    # Keep only needed columns + optional std
    keep_cols = ["feature_set", "model", "acc_mean"]
    if "acc_std" in df.columns:
        keep_cols.append("acc_std")
    df = df[keep_cols].copy()

    # Enforce category order
    df["feature_set"] = pd.Categorical(df["feature_set"], categories=FEATURE_ORDER, ordered=True)

    # Basic sanity checks
    if df["feature_set"].isna().any():
        unknown = df[df["feature_set"].isna()]["feature_set"].unique()
        raise ValueError(f"Unknown feature_set values (not in {FEATURE_ORDER}): {unknown}")

    return df


def pivot_for_plot(df: pd.DataFrame):
    """
    Returns:
      acc_table: index=feature_set, columns=model, values=acc_mean
      std_table: same (if acc_std exists) else None
    """
    acc_table = df.pivot_table(index="feature_set", columns="model", values="acc_mean", aggfunc="mean")
    std_table = None
    if "acc_std" in df.columns:
        std_table = df.pivot_table(index="feature_set", columns="model", values="acc_std", aggfunc="mean")
    return acc_table, std_table


def plot_all_models(acc_table: pd.DataFrame, std_table: pd.DataFrame | None, out_path: str):
    x_labels = FEATURE_ORDER
    x = list(range(len(x_labels)))

    plt.figure(figsize=(7.5, 4.5))

    for model in acc_table.columns:
        y = [acc_table.loc[fs, model] for fs in x_labels]
        if std_table is not None and model in std_table.columns:
            yerr = [std_table.loc[fs, model] for fs in x_labels]
            plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=1.5, label=model)
        else:
            plt.plot(x, y, marker="o", linewidth=1.5, label=model)

    plt.xticks(x, x_labels)
    plt.xlabel("Feature Set")
    plt.ylabel("Accuracy (mean)")
    plt.title("Feature Ablation Study (Accuracy vs Feature Set)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_rf_only(acc_table: pd.DataFrame, std_table: pd.DataFrame | None, out_path: str):
    x_labels = FEATURE_ORDER
    x = list(range(len(x_labels)))

    # Find RF column name robustly
    rf_col = None
    for c in acc_table.columns:
        if str(c).upper() == "RF" or "RANDOM" in str(c).upper():
            rf_col = c
            break
    if rf_col is None:
        raise ValueError(f"Cannot find RF model in columns: {list(acc_table.columns)}")

    y = [acc_table.loc[fs, rf_col] for fs in x_labels]
    plt.figure(figsize=(7.0, 4.3))

    if std_table is not None and rf_col in std_table.columns:
        yerr = [std_table.loc[fs, rf_col] for fs in x_labels]
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=1.8, label="RF")
    else:
        plt.plot(x, y, marker="o", linewidth=1.8, label="RF")

    plt.xticks(x, x_labels)
    plt.xlabel("Feature Set")
    plt.ylabel("Accuracy (mean)")
    plt.title("Feature Ablation (Random Forest)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="report/ml_db5_feature_ablation.csv",
                    help="Path to ml_db5_feature_ablation.csv")
    ap.add_argument("--out_dir", type=str, default="figures",
                    help="Output directory for figures")
    args = ap.parse_args()

    df = load_and_validate(args.csv)
    acc_table, std_table = pivot_for_plot(df)

    out_all = os.path.join(args.out_dir, "Figure_3-Feature_Ablation_AllModels.png")
    out_rf = os.path.join(args.out_dir, "Figure_3-Feature_Ablation_RF.png")

    plot_all_models(acc_table, std_table, out_all)
    plot_rf_only(acc_table, std_table, out_rf)

    print("[DONE] Saved:")
    print(" -", out_all)
    print(" -", out_rf)


if __name__ == "__main__":
    main()
