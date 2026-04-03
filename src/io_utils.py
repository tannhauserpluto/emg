# -*- coding: utf-8 -*-
# src/io_utils.py

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


def ensure_dirs(figures_dir: str, report_dir: str, intermediate_dir: str):
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)


def load_mat_main_array(path):
    """
    从 .mat 文件中自动寻找主 EMG 矩阵和可选的 label。
    逻辑与原 run_all.py 一致。
    """
    d = loadmat(path)

    # 优先找 "常见通道数的 2D 矩阵"
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] in (8, 12, 16):
            return v, d.get("label", None)

    # 否则找任意 2D 数值矩阵
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.dtype.kind in "fiu":
            return v, d.get("label", None)

    raise RuntimeError(f"No valid 2-D EMG matrix found in {path}")


def mat_to_csv_if_needed(stem: str, raw_dir: str, intermediate_dir: str) -> str:
    """
    兼容 .mat / .csv：
    - 如果 intermediate_dir 中已有 csv，则直接返回；
    - 否则尝试从 raw_dir 中的 mat 转换成 csv 保存到 intermediate_dir。
    """
    csv_path = os.path.join(intermediate_dir, f"{stem}.csv")
    mat_path = os.path.join(raw_dir, f"{stem}.mat")

    if os.path.exists(csv_path):
        return csv_path

    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Neither '{csv_path}' nor '{mat_path}' found.")

    X, label = load_mat_main_array(mat_path)
    n, c = X.shape
    df = pd.DataFrame(X, columns=[f"emg{i+1}" for i in range(c)])

    if isinstance(label, np.ndarray) and label.size in (n,):
        df["label"] = label.reshape(-1)

    os.makedirs(intermediate_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path


def plot_before_after(raw_csv, clean_csv, out_png, ylim=None, xlim=None):
    """
    画滤波前后对比图，使用第一个 emg 通道。
    """
    r = pd.read_csv(raw_csv)
    c = pd.read_csv(clean_csv)
    e = [col for col in r.columns if col.lower().startswith("emg")][0]

    plt.figure(figsize=(12, 4))
    plt.plot(r[e].values, label="Raw")
    plt.plot(c[e].values, label="Filtered")

    if ylim:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)

    plt.title("Filter Before/After")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_segments(csv_path, segs, out_png):
    """
    画出 EMG 波形，并用半透明区域标记分段结果。
    segs: [(start_idx, end_idx), ...]
    """
    df = pd.read_csv(csv_path)
    e = [c for c in df.columns if c.lower().startswith("emg")][0]
    y = df[e].values

    plt.figure(figsize=(12, 4))
    plt.plot(y, label="EMG1 (filtered)")

    for s, e_idx in segs:
        plt.axvspan(s, e_idx, alpha=0.15)

    plt.legend()
    plt.title("Segments (shaded)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def save_segments_json(segments, out_json):
    """
    将分段结果保存为 JSON，格式与原脚本一致：
    {
      "segments": [
        {"start": int, "end": int},
        ...
      ]
    }
    """
    data = {"segments": [{"start": int(s), "end": int(e)} for s, e in segments]}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
