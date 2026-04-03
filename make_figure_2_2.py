# -*- coding: utf-8 -*-
# make_figure_2_2_real.py
"""
Generate Figure 2-2: Example of EMG Segmentation and Windowing (REAL DATA)

Input CSV columns:
  emg1..emg12, label

Assumptions:
  - label is an integer class id per sample (0 for rest is common but not required)
  - sampling rate fs is known (default 500 Hz)
  - window=200 ms, step=50 ms (default)

Output:
  figures/Figure_2-2.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_one_gesture_interval(df: pd.DataFrame, label_col: str = "label"):
    """
    Find a continuous interval [start_idx, end_idx] where label is constant
    and not equal to the most frequent label (often rest).
    Fallback: longest constant-label interval of any non-rest label.
    """
    labels = df[label_col].values
    # Most frequent label (often rest)
    uniq, counts = np.unique(labels, return_counts=True)
    rest_label = uniq[np.argmax(counts)]

    # Identify runs
    runs = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            runs.append((start, i - 1, labels[i - 1]))
            start = i
    runs.append((start, len(labels) - 1, labels[-1]))

    # Prefer non-rest runs
    non_rest = [r for r in runs if r[2] != rest_label]
    if non_rest:
        # choose the longest non-rest run
        s, e, lab = max(non_rest, key=lambda r: r[1] - r[0])
        return s, e, lab, rest_label

    # If everything is the same label (e.g., single gesture file), use the middle 60% as "gesture"
    s = int(0.2 * len(labels))
    e = int(0.8 * len(labels)) - 1
    return s, e, labels[s], rest_label


def make_figure(csv_path: str, fs: float, channel: str,
                win_len_s: float, step_s: float, out_path: str):
    df = pd.read_csv(csv_path)

    # Basic checks
    if channel not in df.columns:
        raise ValueError(f"Channel '{channel}' not found. Available: {list(df.columns)}")
    if "label" not in df.columns:
        raise ValueError("Column 'label' not found.")

    x = df[channel].astype(float).values
    t = np.arange(len(x)) / fs

    # Find gesture interval from labels
    s_idx, e_idx, gesture_label, rest_label = find_one_gesture_interval(df, "label")
    seg_start = t[s_idx]
    seg_end = t[e_idx]

    # Sliding windows inside the gesture interval
    starts = np.arange(seg_start, seg_end - win_len_s + 1e-12, step_s)

    # Plot
    plt.figure(figsize=(11, 6))

    # Top: full signal with onset/offset
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(t, x, linewidth=0.8)
    ax1.axvline(seg_start, linestyle="--", linewidth=1.0)
    ax1.axvline(seg_end, linestyle="--", linewidth=1.0)
    ax1.set_title(f"Figure 2-2. Example of EMG Segmentation and Windowing ({channel})")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.text(seg_start, ax1.get_ylim()[1] * 0.85, "Gesture onset", rotation=90, va="top")
    ax1.text(seg_end, ax1.get_ylim()[1] * 0.85, "Gesture offset", rotation=90, va="top")
    ax1.text(0.01, 0.95,
             f"Selected gesture label={gesture_label} (rest label≈{rest_label})",
             transform=ax1.transAxes, va="top")

    # Bottom: zoom + windows
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(t, x, linewidth=0.8)
    ax2.set_xlim(seg_start - 0.05, seg_end + 0.05)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title(f"Zoom-in: window = {int(win_len_s*1000)} ms, step = {int(step_s*1000)} ms")

    # Draw every other window to keep readable
    for i, s in enumerate(starts):
        if i % 2 == 0:
            ax2.axvspan(s, s + win_len_s, alpha=0.15)

    # Annotate first two windows
    if len(starts) >= 2:
        y_top = ax2.get_ylim()[1]
        ax2.annotate("Window 1",
                     xy=(starts[0] + win_len_s / 2, y_top * 0.6),
                     xytext=(starts[0] + win_len_s / 2, y_top * 0.9),
                     arrowprops=dict(arrowstyle="->"))
        ax2.annotate("Window 2",
                     xy=(starts[1] + win_len_s / 2, y_top * 0.6),
                     xytext=(starts[1] + win_len_s / 2, y_top * 0.85),
                     arrowprops=dict(arrowstyle="->"))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/intermediate/cleaned_g1.csv",
                    help="Path to cleaned_gX.csv")
    ap.add_argument("--fs", type=float, default=500.0, help="Sampling rate (Hz)")
    ap.add_argument("--channel", type=str, default="emg1", help="Channel to plot (emg1..emg12)")
    ap.add_argument("--win", type=float, default=0.200, help="Window length (s)")
    ap.add_argument("--step", type=float, default=0.050, help="Step size (s)")
    ap.add_argument("--out", type=str, default="figures/Figure_2-2.png",
                    help="Output path")
    args = ap.parse_args()

    make_figure(args.csv, args.fs, args.channel, args.win, args.step, args.out)
    print(f"[DONE] Saved Figure 2-2 to: {args.out}")


if __name__ == "__main__":
    main()
