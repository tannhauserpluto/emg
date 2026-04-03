# -*- coding: utf-8 -*-
# src/denoise.py
"""
去噪方法集合：
  - notch_filter
  - wavelet_denoise
  - pca_denoise
  - kalman_denoise

所有函数接受 1D numpy 数组，返回 1D numpy 数组
"""

import numpy as np
import pywt
from scipy.signal import iirnotch, filtfilt


# ----------------------------------------
# 1. Notch Filter（50Hz 工频）
# ----------------------------------------
def notch_filter(x, fs=200, freq=50.0, Q=30.0):
    w0 = freq / (fs / 2)
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, x)


# ----------------------------------------
# 2. Wavelet 去噪（软阈值）
# ----------------------------------------
def wavelet_denoise(x, wavelet="db4", level=3):
    coeffs = pywt.wavedec(x, wavelet, mode="per")
    sigma_est = np.median(np.abs(coeffs[-1])) / 0.6745
    uth = sigma_est * np.sqrt(2 * np.log(len(x)))
    coeffs_th = [coeffs[0]]  # cA 不做阈值
    for cD in coeffs[1:]:
        coeffs_th.append(pywt.threshold(cD, uth, mode="soft"))
    return pywt.waverec(coeffs_th, wavelet, mode="per")


# ----------------------------------------
# 3. PCA 去伪迹（去掉前 k 个主成分）
# ----------------------------------------
def pca_denoise(seg, k_remove=1):
    """
    seg: (N, C) 多通道 EMG
    返回同样 shape
    """
    X = seg.copy()
    X = X - X.mean(axis=0, keepdims=True)

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S[k_remove:] = S[k_remove:]  # 保留后面的成分
    S[:k_remove] = 0             # 把伪迹成分置 0

    X_hat = U @ np.diag(S) @ Vt
    return X_hat


# ----------------------------------------
# 4. Kalman Filter（简单一阶状态模型）
# ----------------------------------------
def kalman_denoise(x, R=0.01, Q=0.001):
    """
    R: 观测噪声
    Q: 过程噪声
    """
    x = np.asarray(x)
    n = len(x)
    x_hat = np.zeros(n)
    P = 1.0
    x_hat[0] = x[0]

    for t in range(1, n):
        # 预测
        x_pred = x_hat[t-1]
        P_pred = P + Q

        # 更新
        K = P_pred / (P_pred + R)
        x_hat[t] = x_pred + K * (x[t] - x_pred)
        P = (1 - K) * P_pred

    return x_hat
