# -*- coding: utf-8 -*-
# run_robust_eval.py
"""
NinaPro DB5 鲁棒性 + 去噪对比 实验（完整版）

对 DB5 的所有 segment：

- baseline：干净信号
- 噪声类型：
    - wgn      （按 SNR 控制）
    - pink     （1/f 粉噪）
    - hum      （工频干扰）
    - drift    （低频基线漂移）
    - motion   （运动伪迹）
    - spikes   （尖峰）

- 噪声强度：
    - 对 wgn/pink: SNR ∈ {20, 10, 0, -5} dB
    - 对 hum/drift/motion/spikes: amp_ratio ∈ {0.1, 0.3, 0.6, 1.0}

- 去噪方法：
    - none      （不做去噪）
    - notch     （50Hz 陷波）
    - wavelet   （小波软阈值）
    - kalman    （一阶卡尔曼滤波）
    - pca       （去掉前 1 个主成分）

对每种 (noise, level, denoise) 组合：
    - 在原始 EMG 上逐段加噪
    - 可选进行去噪
    - 用 compute_td_features 提取特征 (TD+FD+AR)
    - 用 LDA / SVM_RBF / RF 做 5 折交叉验证

输出：
  report/robust_db5_noise.csv
  figures/robust/acc_vs_level_<noise>.png
"""

import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import loadmat

from src.ninapro_db5 import RAW_DIR, unzip_subject
from src.features import compute_td_features
from src.noise import apply_noises
from src.denoise import (
    notch_filter,
    wavelet_denoise,
    pca_denoise,
    kalman_denoise,
)
from src.ml_models import cross_val_evaluate


# --------------------------------------------------
# 工具
# --------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_db5_mats(
    subjects: List[int],
    exercises: Tuple[int, ...] = (1, 2, 3),
) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    """
    一次性将需要的 DB5 .mat 文件全部加载到内存，避免重复 IO。

    返回字典：
      mats[(subject_id, exercise)] = {
          "emg": emg,
          "restimulus": restimulus,
          "rerepetition": rerepetition,
          "fs": fs
      }
    """
    mats = {}

    for sid in subjects:
        unzip_subject(sid)  # 确保已解压

        subj_dir = os.path.join(RAW_DIR, f"s{sid}")
        if not os.path.isdir(subj_dir):
            raise FileNotFoundError(f"RAW 目录 {subj_dir} 不存在，请检查。")

        for ex in exercises:
            mat_name = f"S{sid}_E{ex}_A1.mat"
            mat_path = os.path.join(subj_dir, mat_name)
            if not os.path.exists(mat_path):
                print(f"[WARN] {mat_path} not found, skip.")
                continue

            print(f"[INFO] Loading {mat_path} ...")
            d = loadmat(mat_path)

            emg = d["emg"]
            restimulus = d["restimulus"].ravel().astype(int)
            rerepetition = d["rerepetition"].ravel().astype(int)
            fs = int(d["frequency"].ravel()[0])

            mats[(sid, ex)] = {
                "emg": emg,
                "restimulus": restimulus,
                "rerepetition": rerepetition,
                "fs": fs,
            }

    if not mats:
        raise RuntimeError("没有成功加载任何 DB5 .mat 文件，请检查路径。")

    return mats


def apply_denoise(seg: np.ndarray, fs: float, method: str) -> np.ndarray:
    """
    seg: (N, C)
    method: none / notch / wavelet / kalman / pca
    """
    method = method.lower()
    if method == "none":
        return seg

    if method == "notch":
        out = np.zeros_like(seg)
        for ch in range(seg.shape[1]):
            out[:, ch] = notch_filter(seg[:, ch], fs=fs)
        return out

    if method == "wavelet":
        out = np.zeros_like(seg)
        for ch in range(seg.shape[1]):
            out[:, ch] = wavelet_denoise(seg[:, ch])
        return out

    if method == "kalman":
        out = np.zeros_like(seg)
        for ch in range(seg.shape[1]):
            out[:, ch] = kalman_denoise(seg[:, ch])
        return out

    if method == "pca":
        return pca_denoise(seg, k_remove=1)

    raise ValueError(f"Unknown denoise method: {method}")


def build_features_db5_condition(
    mats: Dict[Tuple[int, int], Dict[str, np.ndarray]],
    gestures: List[int],
    max_reps: int = 6,
    win_sec: float = 0.200,
    step_sec: float = 0.050,
    noise_cfg_list=None,
    denoise_method: str = "none",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    针对给定的 DB5 原始数据 (mats)，构建一个特征数据集：

    - 遍历 (subject, exercise, gesture, repetition)
    - 在每个 (gesture, rep) 区间内做滑动窗口
    - 对每个窗口：
        - 若 noise_cfg_list 为 None：直接作为 clean
        - 若不为 None：对每个通道加噪 (apply_noises)
        - 然后根据 denoise_method 选择是否做去噪
        - 最后用 compute_td_features 提取 TD+FD+AR 特征

    返回：
      X: 特征矩阵 (n_samples, n_features)
      y: 标签向量 (n_samples,)  (gesture id)
    """
    rows = []

    for (sid, ex), d in mats.items():
        emg = d["emg"]
        restimulus = d["restimulus"]
        rerepetition = d["rerepetition"]
        fs = d["fs"]

        win_len = int(win_sec * fs)
        step = int(step_sec * fs)

        for g in gestures:
            for rep in range(1, max_reps + 1):
                mask = (restimulus == g) & (rerepetition == rep)
                idx = np.where(mask)[0]
                if idx.size == 0:
                    continue

                start = idx[0]
                end = idx[-1] + 1

                for s in range(start, end - win_len + 1, step):
                    e = s + win_len
                    seg = emg[s:e, :]  # (win_len, C)

                    # 加噪
                    if noise_cfg_list is not None:
                        seg_noisy = np.zeros_like(seg)
                        for ch in range(seg.shape[1]):
                            seg_noisy[:, ch] = apply_noises(
                                seg[:, ch],
                                fs=fs,
                                noise_cfg_list=noise_cfg_list,
                            )
                        seg = seg_noisy

                    # 去噪
                    seg = apply_denoise(seg, fs=fs, method=denoise_method)

                    feat_vec = compute_td_features(seg, fs=fs)
                    rows.append((feat_vec, g))

    if not rows:
        raise RuntimeError("没有构建出任何窗口特征，请检查 gestures / max_reps / 窗长配置。")

    feat_list, label_list = zip(*rows)
    X = np.vstack(feat_list).astype(float)
    y = np.asarray(label_list, dtype=int)

    return X, y


# --------------------------------------------------
# 主函数：鲁棒性 + 去噪实验
# --------------------------------------------------

def main():
    ensure_dir("report")
    ensure_dir("figures")
    robust_fig_dir = os.path.join("figures", "robust")
    ensure_dir(robust_fig_dir)

    subjects = [1, 2, 3]
    gestures = list(range(1, 11))
    max_reps = 6
    win_sec = 0.200
    step_sec = 0.050

    print("[INFO] Loading DB5 .mat files into memory...")
    mats = load_db5_mats(subjects=subjects, exercises=(1, 2, 3))

    # ----------------------------------------------
    # 1. baseline (clean, 无噪声 & 无去噪)
    # ----------------------------------------------
    print("\n[BASELINE] Building clean features (no noise, no denoise)...")
    X_clean, y_clean = build_features_db5_condition(
        mats=mats,
        gestures=gestures,
        max_reps=max_reps,
        win_sec=win_sec,
        step_sec=step_sec,
        noise_cfg_list=None,
        denoise_method="none",
    )
    print(f"[BASELINE] Dataset: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")

    print("[BASELINE] Evaluating ML models on clean data...")
    baseline_res = cross_val_evaluate(X_clean, y_clean, n_splits=5, random_state=42)
    baseline_res["noise"] = "clean"
    baseline_res["level"] = 0.0
    baseline_res["denoise"] = "none"
    print(baseline_res)

    all_results = [baseline_res]

    # ----------------------------------------------
    # 2. 定义噪声类型与强度列表 & 去噪方法
    # ----------------------------------------------

    noise_types = {
        "wgn":   lambda val: [{"kind": "wgn", "snr_db": val}],
        "pink":  lambda val: [{"kind": "pink", "snr_db": val, "beta": 1.0}],
        "hum":   lambda val: [{"kind": "hum", "amp_ratio": val}],
        "drift": lambda val: [{"kind": "drift", "amp_ratio": val}],
        "motion": lambda val: [{"kind": "motion", "amp_ratio": val}],
        "spikes": lambda val: [{"kind": "spikes", "amp_ratio": val}],
    }

    snr_list = [20.0, 10.0, 0.0, -5.0]      # dB
    amp_list = [0.1, 0.3, 0.6, 1.0]        # 相对幅度

    denoise_methods = ["none", "notch", "wavelet", "kalman", "pca"]

    # ----------------------------------------------
    # 3. 循环：噪声 × level × 去噪
    # ----------------------------------------------

    for noise_name, cfg_func in noise_types.items():
        print(f"\n[ROBUST] Noise type: {noise_name}")

        if noise_name in ["wgn", "pink"]:
            levels = snr_list
        else:
            levels = amp_list

        for val in levels:
            for denoise in denoise_methods:
                desc = f"{noise_name}, level={val}, denoise={denoise}"
                print(f"[ROBUST]   - {desc}")
                noise_cfg = cfg_func(val)

                X_noisy, y_noisy = build_features_db5_condition(
                    mats=mats,
                    gestures=gestures,
                    max_reps=max_reps,
                    win_sec=win_sec,
                    step_sec=step_sec,
                    noise_cfg_list=noise_cfg,
                    denoise_method=denoise,
                )

                # 标签一致性
                assert np.array_equal(
                    np.unique(y_noisy), np.unique(y_clean)
                ), "Noisy 数据的手势标签集合与 clean 不一致。"

                res = cross_val_evaluate(X_noisy, y_noisy, n_splits=5, random_state=42)
                res["noise"] = noise_name
                res["level"] = val
                res["denoise"] = denoise

                print(res)
                all_results.append(res)

    # ----------------------------------------------
    # 4. 汇总 & 保存 CSV
    # ----------------------------------------------

    final_df = pd.concat(all_results, ignore_index=True)
    out_csv = os.path.join("report", "robust_db5_noise.csv")
    final_df.to_csv(out_csv, index=False)

    print(f"\n[INFO] Saved robustness + denoise result table to {out_csv}")
    print(final_df.head())

    # ----------------------------------------------
    # 5. 画 Accuracy vs level 曲线（每种噪声一张图，只画 RF，用不同去噪法的曲线）
    # ----------------------------------------------

    for noise_name in ["wgn", "pink", "hum", "drift", "motion", "spikes"]:
        df_n = final_df[(final_df["noise"] == noise_name) & (final_df["model"] == "RF")].copy()
        if df_n.empty:
            continue

        plt.figure(figsize=(6, 4))

        for denoise in sorted(df_n["denoise"].unique()):
            df_d = df_n[df_n["denoise"] == denoise].copy()
            df_d = df_d.sort_values("level")
            plt.plot(df_d["level"], df_d["acc_mean"], marker="o", label=denoise)

        # clean baseline（RF, no noise, no denoise）
        df_clean_rf = final_df[
            (final_df["noise"] == "clean")
            & (final_df["model"] == "RF")
            & (final_df["denoise"] == "none")
        ]
        if not df_clean_rf.empty:
            clean_acc = df_clean_rf["acc_mean"].values[0]
            plt.axhline(clean_acc, linestyle="--", linewidth=1.0, label="clean RF")

        plt.xlabel("SNR (dB) or amplitude ratio")
        plt.ylabel("Accuracy (RF)")
        plt.title(f"RF Accuracy vs level - {noise_name}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        fig_path = os.path.join(robust_fig_dir, f"acc_vs_level_{noise_name}.png")
        plt.savefig(fig_path, dpi=200)
        plt.close()

        print(f"[INFO] Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
