# -*- coding: utf-8 -*-
"""Generate thesis-ready figures from standardized summaries."""
from __future__ import annotations

import csv
import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


FIG_DIR = os.path.join("outputs", "figures")


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def float_or_none(value: str) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
        }
    )


def build_ml_dl_key_condition_plot(
    rows: list[dict[str, str]],
    value_key: str,
    std_key: str,
    output_path: str,
    title: str,
    y_label: str,
) -> None:
    conditions = [
        "clean@0.0 none",
        "wgn@20.0 none",
        "wgn@10.0 none",
        "motion@0.3 none",
        "drift@0.3 none",
    ]
    condition_labels = ["clean", "wgn@20", "wgn@10", "motion@0.3", "drift@0.3"]
    model_order = [
        ("ML", "LDA", "LDA"),
        ("ML", "SVM_RBF", "SVM"),
        ("ML", "RF", "RF"),
        ("DL", "dualres_xception2d", "DL"),
    ]
    color_map = {
        "LDA": "#4C78A8",
        "SVM": "#F58518",
        "RF": "#54A24B",
        "DL": "#B279A2",
    }

    value_map: dict[tuple[str, str, str], dict[str, float | None]] = {}
    for row in rows:
        key = (row["condition"], row["method_family"], row["model"])
        value_map[key] = {
            "value": float_or_none(row.get(value_key, "")),
            "std": float_or_none(row.get(std_key, "")),
        }

    x = np.arange(len(conditions))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(model_order))

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for offset, (family, model, label) in zip(offsets, model_order):
        values = []
        errors = []
        for condition in conditions:
            entry = value_map.get((condition, family, model), {})
            values.append(entry.get("value"))
            errors.append(entry.get("std"))
        ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=color_map[label],
            yerr=errors,
            capsize=3,
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_main_loro_plot(rows: list[dict[str, str]], output_path: str) -> None:
    overall = next((row for row in rows if row.get("evaluation_level") == "overall"), None)
    fold_rows = [row for row in rows if row.get("evaluation_level") == "fold"]

    if overall is None or not fold_rows:
        raise RuntimeError("Missing overall or fold-level rows in main_results.csv.")

    window_values = [float(row["window_accuracy"]) for row in fold_rows]
    repetition_values = [float(row["repetition_accuracy"]) for row in fold_rows]
    overall_window = float(overall["window_accuracy"])
    overall_rep = float(overall["repetition_accuracy"])

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    box = ax.boxplot(
        [window_values, repetition_values],
        tick_labels=["Window-level", "Repetition-level"],
        patch_artist=True,
        showfliers=False,
    )
    colors = ["#4C78A8", "#F58518"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.scatter([1, 2], [overall_window, overall_rep], color="black", marker="*", s=90, zorder=3, label="Overall")
    ax.set_ylabel("Accuracy")
    ax.set_title("Subject-dependent LORO: Window vs Repetition Accuracy")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_ml_baseline_plot(rows: list[dict[str, str]], output_path: str) -> None:
    ml_rows = [row for row in rows if row.get("category") == "ml_baseline"]
    if not ml_rows:
        raise RuntimeError("No ML baseline rows found in foundation_results.csv.")

    order = ["LDA", "SVM_RBF", "RF"]
    labels = ["LDA", "SVM", "RF"]
    data = {row["model"]: row for row in ml_rows}

    accs = [float(data[key]["accuracy"]) for key in order]
    acc_std = [float(data[key]["std_accuracy"]) for key in order]
    f1s = [float(data[key]["macro_f1"]) for key in order]
    f1_std = [float(data[key]["std_macro_f1"]) for key in order]

    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=False)

    axes[0].bar(x, accs, yerr=acc_std, capsize=3, color="#4C78A8", edgecolor="black", linewidth=0.3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("ML Baseline Accuracy")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    axes[1].bar(x, f1s, yerr=f1_std, capsize=3, color="#F58518", edgecolor="black", linewidth=0.3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_title("ML Baseline Macro-F1")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _build_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    title: str,
) -> None:
    cmap = plt.cm.viridis
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title(title)
    ax.set_xlabel("Noise level / SNR")
    ax.set_ylabel("Noise type")
    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data[row_idx, col_idx]
            if np.isnan(value):
                continue
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    return im


def build_dl_full_matrix_heatmap(rows: list[dict[str, str]], output_path: str) -> None:
    none_rows = [row for row in rows if row.get("denoise_method") == "none" and row.get("noise_type") != "clean"]
    if not none_rows:
        raise RuntimeError("No denoise=none rows found for DL full matrix heatmap.")

    snr_types = ["wgn", "pink"]
    amp_types = ["hum", "drift", "motion", "spikes"]
    snr_levels = ["20.0", "10.0", "0.0", "-5.0"]
    amp_levels = ["0.1", "0.3", "0.6", "1.0"]

    snr_matrix = np.full((len(snr_types), len(snr_levels)), np.nan, dtype=float)
    amp_matrix = np.full((len(amp_types), len(amp_levels)), np.nan, dtype=float)

    for row in none_rows:
        noise_type = row["noise_type"]
        level = row["noise_level"]
        value = float(row["mean_accuracy"])
        if noise_type in snr_types and level in snr_levels:
            snr_matrix[snr_types.index(noise_type), snr_levels.index(level)] = value
        if noise_type in amp_types and level in amp_levels:
            amp_matrix[amp_types.index(noise_type), amp_levels.index(level)] = value

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    im1 = _build_heatmap(axes[0], snr_matrix, snr_levels, snr_types, "SNR-based noise")
    im2 = _build_heatmap(axes[1], amp_matrix, amp_levels, amp_types, "Amplitude-based noise")

    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("Accuracy")
    fig.suptitle("DL Full Noise Matrix (denoise=none)")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_best_denoise_comparison(
    full_rows: list[dict[str, str]],
    best_rows: list[dict[str, str]],
    output_path: str,
) -> None:
    best_map = {(row["noise_type"], row["noise_level"]): row for row in best_rows}
    none_map = {
        (row["noise_type"], row["noise_level"]): row
        for row in full_rows
        if row.get("denoise_method") == "none"
    }
    noise_types = ["wgn", "pink", "hum", "drift", "motion", "spikes"]
    level_order = {
        "wgn": ["20.0", "10.0", "0.0", "-5.0"],
        "pink": ["20.0", "10.0", "0.0", "-5.0"],
        "hum": ["0.1", "0.3", "0.6", "1.0"],
        "drift": ["0.1", "0.3", "0.6", "1.0"],
        "motion": ["0.1", "0.3", "0.6", "1.0"],
        "spikes": ["0.1", "0.3", "0.6", "1.0"],
    }

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.6), sharey=True, constrained_layout=True)
    axes = axes.ravel()

    for ax, noise_type in zip(axes, noise_types):
        levels = level_order[noise_type]
        x = np.arange(len(levels))
        none_vals = []
        best_vals = []
        for level in levels:
            none_entry = none_map.get((noise_type, level))
            best_entry = best_map.get((noise_type, level))
            none_vals.append(float(none_entry["mean_accuracy"]) if none_entry else np.nan)
            best_vals.append(float(best_entry["best_accuracy"]) if best_entry else np.nan)

        ax.plot(x, none_vals, marker="o", label="none", color="#4C78A8")
        ax.plot(x, best_vals, marker="o", label="best", color="#F58518")
        ax.set_xticks(x)
        ax.set_xticklabels(levels)
        ax.set_title(noise_type)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(0.0, 1.0)

    axes[0].set_ylabel("Accuracy")
    axes[3].set_ylabel("Accuracy")
    fig.suptitle("DL Denoise Comparison (none vs best)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_cross_subject_vs_subject_dependent(
    cross_rows: list[dict[str, str]],
    main_rows: list[dict[str, str]],
    output_path: str,
) -> list[str]:
    overall = next((row for row in main_rows if row.get("evaluation_level") == "overall"), None)
    if overall is None:
        raise RuntimeError("Missing overall row in main_results.csv for subject-dependent result.")

    def find_cross(model_name: str) -> dict[str, str] | None:
        for row in cross_rows:
            if row.get("model") == model_name:
                return row
        return None

    items: list[tuple[str, float]] = []
    labels: list[str] = []

    cnn_row = find_cross("cnn_lstm")
    xcep_row = find_cross("xception2d")
    if cnn_row:
        items.append(("Cross-subject CNN-LSTM\n(window)", float(cnn_row["accuracy"])))
        labels.append("cross_cnn_lstm")
    if xcep_row:
        items.append(("Cross-subject Xception2D\n(window)", float(xcep_row["accuracy"])))
        labels.append("cross_xception2d")

    overall_window = float(overall["window_accuracy"])
    overall_rep = float(overall["repetition_accuracy"])
    items.append(("Subject-dependent Overall\n(window)", overall_window))
    labels.append("subject_window")
    items.append(("Subject-dependent Overall\n(repetition)", overall_rep))
    labels.append("subject_repetition")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    values = [item[1] for item in items]
    x = np.arange(len(items))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"][: len(items)]
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([item[0] for item in items], rotation=10, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cross-subject vs Subject-dependent Performance")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return labels


def main() -> None:
    setup_style()
    ensure_dir(FIG_DIR)

    key_rows = read_csv(os.path.join("outputs", "tables", "table_ml_vs_dl_noise_key_conditions.csv"))
    build_ml_dl_key_condition_plot(
        key_rows,
        value_key="accuracy",
        std_key="std_accuracy",
        output_path=os.path.join(FIG_DIR, "fig_ml_vs_dl_noise_key_conditions_accuracy.png"),
        title="ML vs DL (Key Noise Conditions) - Accuracy",
        y_label="Accuracy",
    )
    build_ml_dl_key_condition_plot(
        key_rows,
        value_key="macro_f1",
        std_key="std_macro_f1",
        output_path=os.path.join(FIG_DIR, "fig_ml_vs_dl_noise_key_conditions_macro_f1.png"),
        title="ML vs DL (Key Noise Conditions) - Macro-F1",
        y_label="Macro-F1",
    )

    main_rows = read_csv(os.path.join("outputs", "summaries", "main_results.csv"))
    build_main_loro_plot(
        main_rows,
        output_path=os.path.join(FIG_DIR, "fig_main_loro_window_vs_repetition.png"),
    )

    foundation_rows = read_csv(os.path.join("outputs", "summaries", "foundation_results.csv"))
    build_ml_baseline_plot(
        foundation_rows,
        output_path=os.path.join(FIG_DIR, "fig_ml_baseline_comparison.png"),
    )

    full_matrix_rows = read_csv(os.path.join("outputs", "tables", "table_dl_noise_full_matrix.csv"))
    build_dl_full_matrix_heatmap(
        full_matrix_rows,
        output_path=os.path.join(FIG_DIR, "fig_dl_noise_full_matrix_heatmap.png"),
    )

    best_rows = read_csv(os.path.join("outputs", "tables", "table_dl_best_denoise_by_condition.csv"))
    build_best_denoise_comparison(
        full_matrix_rows,
        best_rows,
        output_path=os.path.join(FIG_DIR, "fig_dl_best_denoise_comparison.png"),
    )

    cross_rows = read_csv(os.path.join("outputs", "tables", "table_cross_subject_results.csv"))
    build_cross_subject_vs_subject_dependent(
        cross_rows,
        main_rows,
        output_path=os.path.join(FIG_DIR, "fig_cross_subject_vs_subject_dependent.png"),
    )


if __name__ == "__main__":
    main()
