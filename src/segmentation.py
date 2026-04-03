# -*- coding: utf-8 -*-
# src/segmentation.py

import numpy as np


def segment_by_energy(
    emg,
    we=200,
    overlap=0.2,
    A=0.28,
    B=0.20,
    N=9,
):
    """
    基于 EMG 能量阈值的分段算法（简化为只用 EMG 能量）。
    emg: 预处理后的多通道 EMG，shape (n_samples, n_channels)
    返回: [(start_idx, end_idx), ...]
    """

    def win_energy(X, w):
        step = max(1, int(w * (1 - overlap)))
        E = []
        for s in range(0, X.shape[0] - w + 1, step):
            seg = X[s : s + w]
            E.append((s, s + w, float(np.mean(seg**2))))
        return np.array(E, dtype=object)

    Ee = win_energy(emg, we)
    if Ee.size == 0:
        return []

    E = np.array([v for _, _, v in Ee])
    Emax = E.max() if E.size else 1.0

    thrA, thrB = A * Emax, B * Emax  # B 目前未用到，但保留以便后续扩展

    segs = []
    cnt, start = 0, None

    for i, e in enumerate(E):
        if e >= thrA:
            cnt += 1
            if cnt == 1:
                start = i
        else:
            if cnt >= N and start is not None:
                s0 = int(Ee[start][0])
                e0 = int(Ee[i - 1][1])
                segs.append((s0, e0))
            cnt, start = 0, None

    return segs


def get_segmentation_params(gesture_index: int):
    """
    统一管理不同手势的分段参数。
    目前：
      - 默认 A=0.28, B=0.20, N=9
      - 手势4: A=0.20, B=0.15, N=6
    """
    A, B, N = 0.28, 0.20, 9
    if gesture_index == 4:
        A, B, N = 0.20, 0.15, 6
    return A, B, N
