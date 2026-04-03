# -*- coding: utf-8 -*-
# make_clean_vs_noisy_summary_rf.py

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


NOISE_ORDER = ["wgn", "pink", "hum", "drift", "motion", "spikes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="report/robust_db5_noise.csv",
                    help="Path to robust_db5_noise.csv")
    ap.add_argument("--out", type=str, default="figures/Figure_3-Clean_vs_Noisy_Summary_RF.png",
                    help="Output figure path")
    ap.add_argument("--mode", choices=["none_only", "avg_all_denoise"], default="none_only",
                    help="How to compute noisy average: "
                         "none_only = only denoise='none' (recommended); "
                         "avg_all_denoise = average across all denoise methods")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Basic column checks
    required = {"noise", "model", "acc_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")

    # Keep RF
    df_rf = df[df["model"].astype(str).str.upper().isin(["RF", "RANDOMFOREST", "RANDOM_FOREST"])].copy()
    if df_rf.empty:
        # Sometimes model might be 'RandomForest'
        df_rf = df[df["model"].astype(str).str.contains("RF", case=False, na=False)].copy()
    if df_rf.empty:
        raise ValueError("No RF rows found in CSV (model column).")

    # Determine clean baseline (prefer denoise='none' if present)
    df_clean = df_rf[df_rf["noise"].astype(str).str.lower() == "clean"].copy()
    if df_clean.empty:
        raise ValueError("No clean baseline found (noise == 'clean').")

    if "denoise" in df_clean.columns:
        df_clean_none = df_clean[df_clean["denoise"].astype(str).str.lower() == "none"]
        if not df_clean_none.empty:
            clean_acc = float(df_clean_none["acc_mean"].mean())
        else:
            clean_acc = float(df_clean["acc_mean"].mean())
    else:
        clean_acc = float(df_clean["acc_mean"].mean())

    # Compute noisy average per noise type
    df_noisy = df_rf[df_rf["noise"].astype(str).str.lower() != "clean"].copy()
    df_noisy["noise"] = df_noisy["noise"].astype(str).str.lower()

    if args.mode == "none_only":
        if "denoise" not in df_noisy.columns:
            raise ValueError("mode=none_only requires 'denoise' column in CSV, but it's missing.")
        df_noisy = df_noisy[df_noisy["denoise"].astype(str).str.lower() == "none"].copy()

    # Keep only known noise types (avoid unexpected)
    df_noisy = df_noisy[df_noisy["noise"].isin(NOISE_ORDER)].copy()
    if df_noisy.empty:
        raise ValueError("No noisy rows found after filtering. Check noise names / denoise names.")

    # Aggregate: mean across levels (and optionally across denoise methods)
    grouped = df_noisy.groupby("noise")["acc_mean"].agg(["mean", "std", "count"]).reset_index()

    # Ensure order
    grouped["noise"] = pd.Categorical(grouped["noise"], categories=NOISE_ORDER, ordered=True)
    grouped = grouped.sort_values("noise")

    # Plot
    plt.figure(figsize=(8.2, 4.6))
    x = np.arange(len(grouped))
    y = grouped["mean"].values
    yerr = grouped["std"].fillna(0.0).values  # std over levels (and denoise if avg_all_denoise)

    # clean baseline line
    plt.axhline(clean_acc, linestyle="--", linewidth=1.5, label=f"Clean RF (acc={clean_acc:.4f})")

    # points with error bars
    plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=4, linewidth=1.2, label="Noisy average (RF)")

    plt.xticks(x, grouped["noise"].tolist())
    plt.xlabel("Noise Type")
    plt.ylabel("Accuracy (mean)")
    title_mode = "denoise=none" if args.mode == "none_only" else "avg over denoise methods"
    plt.title(f"Clean vs Noisy Performance Summary (RF) [{title_mode}]")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=300)
    plt.close()

    print("[DONE] Saved:", args.out)
    print("\n[INFO] Summary table:")
    print(grouped)


if __name__ == "__main__":
    main()
