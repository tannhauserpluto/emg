# -*- coding: utf-8 -*-
# src/filters.py

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def butter_bandpass(x, fs, low=20.0, high=200.0, order=4):
    """
    多通道带通滤波（axis=0），适用于 EMG。
    x: shape (n_samples, n_channels)
    """
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


def butter_highpass(x, fs, fc=15.9, order=6):
    """
    多通道高通滤波（axis=0），适用于去除低频漂移。
    """
    b, a = butter(order, fc / (fs / 2), btype="highpass")
    return filtfilt(b, a, x, axis=0)


def notch_50hz(x, fs, f0=50.0, Q=30.0):
    """
    50 Hz 工频陷波（iirnotch 实现），多通道。
    """
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, x, axis=0)


def preprocess_emg(
    X,
    fs,
    mode="bandpass",
    bp_low=20.0,
    bp_high=200.0,
    hp_fc=15.9,
    order=4,
    use_notch=False,
):
    """
    统一的 EMG 预处理入口：
    - mode="bandpass"：带通 [bp_low, bp_high]
    - mode="highpass"：高通 fc=hp_fc
    - 可选工频陷波
    """
    if mode == "bandpass":
        Xf = butter_bandpass(X, fs, low=bp_low, high=bp_high, order=order)
    else:
        Xf = butter_highpass(X, fs, fc=hp_fc, order=max(3, order))

    if use_notch:
        Xf = notch_50hz(Xf, fs)

    return Xf
