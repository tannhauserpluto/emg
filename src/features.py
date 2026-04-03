# -*- coding: utf-8 -*-
# src/features.py
"""
特征提取模块：

每个通道提取三类特征：

1. 时域（TD）5 个：
   - RMS: Root Mean Square
   - MAV: Mean Absolute Value
   - WL : Waveform Length
   - ZC : Zero Crossing
   - SSC: Slope Sign Changes

2. 频域（FD）3 个（基于归一化功率谱）：
   - MNF: Mean Frequency
   - MDF: Median Frequency
   - SE : Spectral Entropy

3. AR 模型系数（4 阶）：
   - AR1, AR2, AR3, AR4

总计：每通道 5 + 3 + 4 = 12 维。
例如 NinaPro DB5 有 16 通道 -> 16 * 12 = 192 维特征。
"""

import json
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import toeplitz


# -----------------------
# 时域特征
# -----------------------

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def _mav(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x)))


def _wl(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(x))))


def _zc(x: np.ndarray, threshold: float = 1e-6) -> int:
    """
    零交叉次数，排除幅度很小的抖动。
    """
    x = x - np.mean(x)
    s1 = x[:-1]
    s2 = x[1:]
    sign_change = (s1 * s2) < 0
    amp_ok = np.abs(s1 - s2) > threshold
    return int(np.sum(sign_change & amp_ok))


def _ssc(x: np.ndarray, threshold: float = 1e-6) -> int:
    """
    Slope Sign Changes，一阶差分符号变化次数。
    """
    diff = np.diff(x)
    d1 = diff[:-1]
    d2 = diff[1:]
    sign_change = (d1 * d2) < 0
    amp_ok = (np.abs(d1) > threshold) | (np.abs(d2) > threshold)
    return int(np.sum(sign_change & amp_ok))


# -----------------------
# 频域特征
# -----------------------

def _power_spectrum(x: np.ndarray, fs: float, nfft: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    简单功率谱估计：FFT -> |X|^2，做概率归一化，作为离散频率分布。
    """
    x = x - np.mean(x)
    if len(x) < 2:
        # 太短就直接填零谱
        freqs = np.linspace(0, fs / 2, nfft // 2 + 1)
        P = np.ones_like(freqs) / len(freqs)
        return freqs, P

    X = np.fft.rfft(x, n=nfft)
    P = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    # 归一化为概率分布
    P_sum = np.sum(P)
    if P_sum <= 0:
        P = np.ones_like(P) / len(P)
    else:
        P = P / P_sum

    return freqs, P


def _mnf(x: np.ndarray, fs: float) -> float:
    """
    Mean Frequency: sum(f * P(f))
    """
    freqs, P = _power_spectrum(x, fs)
    return float(np.sum(freqs * P))


def _mdf(x: np.ndarray, fs: float) -> float:
    """
    Median Frequency: 使累积功率达到 50% 的频率。
    """
    freqs, P = _power_spectrum(x, fs)
    cumsum = np.cumsum(P)
    idx = np.searchsorted(cumsum, 0.5)
    if idx >= len(freqs):
        idx = len(freqs) - 1
    return float(freqs[idx])


def _spec_entropy(x: np.ndarray, fs: float) -> float:
    """
    归一化谱熵：-sum(P log P) / log(N)
    """
    _, P = _power_spectrum(x, fs)
    eps = 1e-12
    H = -np.sum(P * np.log(P + eps))
    H_norm = H / np.log(len(P) + eps)
    return float(H_norm)


# -----------------------
# AR 特征（Yule-Walker）
# -----------------------

def _ar_yule_walker(x: np.ndarray, order: int = 4) -> List[float]:
    """
    使用 Yule-Walker 方程估计 AR 系数。
    如果长度太短，返回全 0。
    """
    x = x - np.mean(x)
    N = len(x)
    if N <= order:
        return [0.0] * order

    # 自相关 r[0..order]
    r_full = np.correlate(x, x, mode="full")
    mid = len(r_full) // 2
    r = r_full[mid : mid + order + 1] / N  # r[0]..r[order]

    # 构造 Toeplitz 矩阵 R 和右端 r[1..order]
    R = toeplitz(r[:-1])   # (order, order)
    r_right = r[1:]

    try:
        a = np.linalg.solve(R, r_right)
    except np.linalg.LinAlgError:
        a = np.zeros(order, dtype=float)

    return [float(v) for v in a]


# -----------------------
# 汇总：对一个 segment 提取全部特征
# -----------------------

def compute_td_features(seg_arr: np.ndarray,
                        fs: float = 200.0) -> np.ndarray:
    """
    对单个 segment 做特征提取。
    seg_arr: shape (n_samples, n_channels)
    fs: 采样率（NinaPro DB5 是 200 Hz）

    返回：
      feature_vec: shape (n_channels * n_feats_per_channel,)
    """
    if seg_arr.ndim == 1:
        seg_arr = seg_arr[:, None]  # 单通道也转成 (N, 1)

    n_samples, n_channels = seg_arr.shape
    feats = []

    for ch in range(n_channels):
        x = seg_arr[:, ch]

        # 时域
        td = [
            _rms(x),
            _mav(x),
            _wl(x),
            _zc(x),
            _ssc(x),
        ]

        # 频域
        fd = [
            _mnf(x, fs),
            _mdf(x, fs),
            _spec_entropy(x, fs),
        ]

        # AR(4) 系数
        ar = _ar_yule_walker(x, order=4)

        feats.extend(td + fd + ar)

    return np.array(feats, dtype=float)


# -----------------------
# JSON + CSV glue
# -----------------------

def load_segments_json(path: str):
    """
    读取 gX_segments.json，返回 [(start, end), ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segs = data.get("segments", [])
    return [(int(s["start"]), int(s["end"])) for s in segs]


def extract_features_for_gesture(clean_csv_path: str,
                                 seg_json_path: str,
                                 gesture_label: int,
                                 fs: float = 500.0) -> pd.DataFrame:
    """
    读取某个手势的 cleaned_gX.csv 和 gX_segments.json，
    对每个 segment 提取特征，生成 DataFrame。
    """
    df = pd.read_csv(clean_csv_path)
    emg_cols = [c for c in df.columns if c.lower().startswith("emg")]
    X = df[emg_cols].values

    segs = load_segments_json(seg_json_path)

    rows = []
    for idx, (s, e) in enumerate(segs):
        seg_arr = X[s:e]           # (n_samples, n_channels)
        feat_vec = compute_td_features(seg_arr, fs=fs)

        row = {
            "gesture": gesture_label,
            "segment_index": idx,
        }
        # 展开为 f0, f1, ...
        for k, v in enumerate(feat_vec):
            row[f"f{k}"] = v

        rows.append(row)

    return pd.DataFrame(rows)
