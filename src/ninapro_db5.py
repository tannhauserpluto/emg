# -*- coding: utf-8 -*-
# src/ninapro_db5.py
"""
NinaPro DB5 helper
- unzip sX.zip
- parse Sx_E*_A1.mat
- slice EMG into overlapping windows per (gesture, repetition)
- extract TD features using src.features.compute_td_features
"""

import os
import zipfile
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .features import compute_td_features  # 你前面写过的


# -----------------------
# 路径 & 参数配置
# -----------------------

DATA_ROOT = os.path.join("data", "ninapro_db5")
ZIP_DIR = os.path.join(DATA_ROOT, "zip")
RAW_DIR = os.path.join(DATA_ROOT, "raw")
FEATURE_DIR = os.path.join(DATA_ROOT, "features")


def ensure_dirs():
    os.makedirs(ZIP_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(FEATURE_DIR, exist_ok=True)


# -----------------------
# 解压
# -----------------------

def unzip_subject(subject_id: int):
    """
    解压 data/ninapro_db5/zip/s{subject_id}.zip -> data/ninapro_db5/raw/s{subject_id}/
    """
    ensure_dirs()
    zip_name = f"s{subject_id}.zip"
    zip_path = os.path.join(ZIP_DIR, zip_name)

    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"找不到 {zip_path}，请先从 Zenodo 下载 s{subject_id}.zip 放到 zip 目录。"
        )

    out_dir = os.path.join(RAW_DIR, f"s{subject_id}")
    os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    print(f"[INFO] Unzipped {zip_path} -> {out_dir}")


# -----------------------
# 解析 .mat 并切窗口
# -----------------------

def _extract_windows_from_mat(
    mat_path: str,
    gestures: List[int],
    max_reps: int = 6,
    win_sec: float = 0.200,
    step_sec: float = 0.050,
) -> pd.DataFrame:
    """
    从一个 NinaPro DB5 的 .mat 文件中抽取 (gesture, repetition) 对应片段，
    对每个片段用滑动窗口切分，并计算 TD 特征。

    返回 DataFrame，每行是一个窗口的特征。
    """
    d = loadmat(mat_path)

    emg = d["emg"]           # shape (T, 16)
    restimulus = d["restimulus"].ravel().astype(int)  # corrected labels
    rerepetition = d["rerepetition"].ravel().astype(int)
    fs = int(d["frequency"].ravel()[0])               # 200 Hz
    subject = int(d["subject"].ravel()[0])
    exercise = int(d["exercise"].ravel()[0])

    win_len = int(win_sec * fs)
    step = int(step_sec * fs)

    rows = []

    for g in gestures:
        for rep in range(1, max_reps + 1):
            mask = (restimulus == g) & (rerepetition == rep)
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue

            start = idx[0]
            end = idx[-1] + 1  # slice 上界

            # 滑动窗口
            for s in range(start, end - win_len + 1, step):
                e = s + win_len
                seg = emg[s:e, :]  # (win_len, 16)
                feat_vec = compute_td_features(seg)

                row = {
                    "subject": subject,
                    "exercise": exercise,
                    "gesture": g,
                    "rep": rep,
                }
                for k, v in enumerate(feat_vec):
                    row[f"f{k}"] = v

                rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def build_features_for_subject(
    subject_id: int,
    gestures: List[int],
    exercises: Tuple[int, ...] = (1, 2, 3),
    max_reps: int = 6,
    win_sec: float = 0.200,
    step_sec: float = 0.050,
) -> pd.DataFrame:
    """
    为某个 subject 汇总指定手势+练习的所有特征。
    """
    subject_dir = os.path.join(RAW_DIR, f"s{subject_id}")
    if not os.path.isdir(subject_dir):
        raise FileNotFoundError(
            f"RAW 目录 {subject_dir} 不存在，请先 unzip_subject({subject_id})。"
        )

    all_dfs = []

    for ex in exercises:
        # DB5 中常见文件名：S1_E1_A1.mat, S1_E2_A1.mat, S1_E3_A1.mat
        mat_name = f"S{subject_id}_E{ex}_A1.mat"
        mat_path = os.path.join(subject_dir, mat_name)

        if not os.path.exists(mat_path):
            print(f"[WARN] {mat_path} not found, skip.")
            continue

        print(f"[INFO] Parsing {mat_path} ...")
        df_ex = _extract_windows_from_mat(
            mat_path=mat_path,
            gestures=gestures,
            max_reps=max_reps,
            win_sec=win_sec,
            step_sec=step_sec,
        )
        if not df_ex.empty:
            all_dfs.append(df_ex)

    if not all_dfs:
        return pd.DataFrame()

    df_subj = pd.concat(all_dfs, ignore_index=True)
    return df_subj


def build_features_db5(
    subjects: List[int],
    gestures: List[int],
    out_name: str = "db5_s1-3_g1-10.csv",
    **kwargs,
) -> pd.DataFrame:
    """
    为若干 subject + 指定手势，构建一个大的特征表，并保存到 CSV。
    """
    ensure_dirs()

    all_dfs = []
    for sid in subjects:
        # 先解压
        unzip_subject(sid)
        # 再解析
        df_subj = build_features_for_subject(sid, gestures, **kwargs)
        if df_subj.empty:
            print(f"[WARN] Subject {sid} produced no data.")
            continue
        all_dfs.append(df_subj)

    if not all_dfs:
        raise RuntimeError("No data built from DB5. Check zip files & gesture list.")

    df_all = pd.concat(all_dfs, ignore_index=True)

    out_path = os.path.join(FEATURE_DIR, out_name)
    df_all.to_csv(out_path, index=False)
    print(f"[INFO] Saved features to {out_path}")
    print(f"[INFO] Total samples: {len(df_all)}, feature_dim: "
          f"{len([c for c in df_all.columns if c.startswith('f')])}")

    return df_all

# -----------------------
# 额外工具：取一段原始 EMG 片段用于噪声演示
# -----------------------

def get_demo_segment(
    subject_id: int = 1,
    gesture: int = 1,
    repetition: int = 1,
    exercise: int = 1,
):
    """
    从 DB5 中取一段原始 EMG 片段，用于噪声注入演示。

    返回:
      fs : 采样率 (Hz)
      t  : 相对时间轴 (秒)，shape (N,)
      emg_seg : shape (N, n_channels)
    """
    # 与 build_features_for_subject 中一样的目录选择逻辑
    base_dir = os.path.join(RAW_DIR, f"s{subject_id}")
    nested_dir = os.path.join(base_dir, f"s{subject_id}")

    candidate_dirs = []
    if os.path.isdir(base_dir):
        candidate_dirs.append(base_dir)
    if os.path.isdir(nested_dir):
        candidate_dirs.append(nested_dir)

    subject_dir = None
    for d in candidate_dirs:
        mats = [f for f in os.listdir(d) if f.lower().endswith(".mat")]
        if mats:
            subject_dir = d
            break

    if subject_dir is None:
        raise FileNotFoundError(
            f"找不到含 .mat 文件的目录，检查 {base_dir} 或 {nested_dir}。"
        )

    mat_name = f"S{subject_id}_E{exercise}_A1.mat"
    mat_path = os.path.join(subject_dir, mat_name)
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"{mat_path} 不存在。")

    d = loadmat(mat_path)
    emg = d["emg"]                    # (T, 16)
    restimulus = d["restimulus"].ravel().astype(int)
    rerepetition = d["rerepetition"].ravel().astype(int)
    fs = int(d["frequency"].ravel()[0])

    mask = (restimulus == gesture) & (rerepetition == repetition)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise RuntimeError(
            f"在 subject {subject_id}, gesture {gesture}, repetition {repetition}, "
            f"exercise {exercise} 中没有找到对应片段。"
        )

    start = idx[0]
    end = idx[-1] + 1
    emg_seg = emg[start:end, :]
    t = (np.arange(start, end) - start) / fs  # 相对时间，从 0 开始

    return fs, t, emg_seg
