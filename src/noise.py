# -*- coding: utf-8 -*-
# src/noise.py
"""
EMG 噪声注入模块

支持的噪声类型 (kind):
  - "wgn"      : 白噪声 (按目标 SNR 控制)
  - "hum"      : 工频干扰 (50 Hz ± Δ + 二次谐波)
  - "drift"    : 低频基线漂移
  - "motion"   : 运动伪迹 / burst
  - "spikes"   : 尖峰噪声
  - "pink"     : 1/f 粉噪

核心接口:
  add_noise_*(x, fs, ...)
  apply_noises(x, fs, noise_cfg_list)
"""

from typing import List, Dict, Any, Sequence
import numpy as np


# -----------------------
# 工具函数
# -----------------------

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def _ensure_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).astype(float)
    if x.ndim != 1:
        raise ValueError("noise functions 目前假定输入为 1D 信号 (单通道)，请先选定一个通道。")
    return x


# -----------------------
# 1. 白噪声 (WGN) - 按 SNR 控制
# -----------------------

def add_wgn_snr(x: np.ndarray, snr_db: float) -> np.ndarray:
    """
    按目标 SNR(dB) 添加白噪声:
      SNR = 10 * log10(P_signal / P_noise)
    """
    x = _ensure_1d(x)
    Ps = _rms(x) ** 2
    if Ps <= 0:
        return x.copy()

    snr_lin = 10 ** (snr_db / 10.0)
    Pn = Ps / snr_lin
    sigma = np.sqrt(Pn)

    n = np.random.randn(len(x)) * sigma
    return x + n


# -----------------------
# 2. 工频干扰 (hum / power-line)
# -----------------------

def add_powerline_hum(
    x: np.ndarray,
    fs: float,
    f0: float = 50.0,
    delta_f: float = 0.5,
    amp_ratio: float = 0.3,
) -> np.ndarray:
    """
    添加工频干扰 (50 Hz ± delta_f) 及其二次谐波。
    amp_ratio: 干扰的幅度 ~ amp_ratio * RMS(signal)
    """
    x = _ensure_1d(x)
    N = len(x)
    t = np.arange(N) / fs

    sig_rms = _rms(x)
    A = amp_ratio * sig_rms

    f1 = f0 + delta_f
    f2 = 2 * f1

    hum = (A * np.sin(2 * np.pi * f1 * t) +
           0.5 * A * np.sin(2 * np.pi * f2 * t))

    return x + hum


# -----------------------
# 3. 低频基线漂移 (drift)
# -----------------------

def add_baseline_drift(
    x: np.ndarray,
    fs: float,
    max_freq: float = 0.5,
    amp_ratio: float = 0.3,
) -> np.ndarray:
    """
    添加低频 (< max_freq Hz) 基线漂移。
    用几根低频正弦叠加模拟姿势缓变 / 电极压紧变化。
    """
    x = _ensure_1d(x)
    N = len(x)
    t = np.arange(N) / fs
    sig_rms = _rms(x)

    drift = np.zeros_like(x)
    # 多个随机频率 & 相位
    freqs = np.random.uniform(0.05, max_freq, size=3)
    phases = np.random.uniform(0, 2 * np.pi, size=3)
    for f, ph in zip(freqs, phases):
        drift += np.sin(2 * np.pi * f * t + ph)

    drift = drift / _rms(drift) * (amp_ratio * sig_rms)
    return x + drift


# -----------------------
# 4. 运动伪迹 / burst (motion artifact)
# -----------------------

def add_motion_artifact(
    x: np.ndarray,
    fs: float,
    n_bursts: int = 3,
    burst_duration: float = 0.1,
    amp_ratio: float = 2.0,
) -> np.ndarray:
    """
    添加短时大幅度 burst 噪声，模拟运动伪迹。
    """
    x = _ensure_1d(x)
    N = len(x)
    y = x.copy()
    sig_rms = _rms(x)

    burst_len = max(1, int(burst_duration * fs))

    for _ in range(n_bursts):
        start = np.random.randint(0, max(1, N - burst_len))
        end = start + burst_len

        burst = np.random.randn(burst_len)
        burst = burst / _rms(burst) * (amp_ratio * sig_rms)

        y[start:end] += burst

    return y


# -----------------------
# 5. 尖峰噪声 (spikes)
# -----------------------

def add_spikes(
    x: np.ndarray,
    n_spikes: int = 20,
    amp_ratio: float = 3.0,
) -> np.ndarray:
    """
    添加稀疏尖峰噪声。
    """
    x = _ensure_1d(x)
    N = len(x)
    y = x.copy()
    sig_rms = _rms(x)

    for _ in range(n_spikes):
        idx = np.random.randint(0, N)
        sign = np.random.choice([-1.0, 1.0])
        y[idx] += sign * amp_ratio * sig_rms

    return y


# -----------------------
# 6. 1/f 粉噪声 (pink)
# -----------------------

def add_pink_noise(
    x: np.ndarray,
    snr_db: float = 10.0,
    beta: float = 1.0,
) -> np.ndarray:
    """
    生成近似 1/f^beta 频谱的噪声，并按目标 SNR 缩放。
    简单实现：在频域对随机相位 + 1/sqrt(f^beta) 权重。
    """
    x = _ensure_1d(x)
    N = len(x)
    Ps = _rms(x) ** 2
    if Ps <= 0:
        return x.copy()

    # 频域构造
    freqs = np.fft.rfftfreq(N, d=1.0)  # 采样间隔设为1即可，相对频率
    mag = np.ones_like(freqs)
    # 防止除零
    mag[1:] = 1.0 / (freqs[1:] ** (beta / 2.0))

    # 随机相位
    phase = np.exp(1j * 2 * np.pi * np.random.rand(len(freqs)))
    Xn = mag * phase

    # 变回时域
    n = np.fft.irfft(Xn, n=N)
    # 归一化 RMS 后再按目标 SNR 调整
    Pn_raw = _rms(n) ** 2
    if Pn_raw <= 0:
        return x.copy()

    snr_lin = 10 ** (snr_db / 10.0)
    Pn_target = Ps / snr_lin
    scale = np.sqrt(Pn_target / Pn_raw)
    n = n * scale

    return x + n


# -----------------------
# 7. 总入口：apply_noises
# -----------------------

def apply_noises(
    x: np.ndarray,
    fs: float,
    noise_cfg_list: Sequence[Dict[str, Any]],
) -> np.ndarray:
    """
    按顺序依次叠加多种噪声。

    noise_cfg_list: 形如
      [
        {"kind": "wgn", "snr_db": 10},
        {"kind": "hum", "amp_ratio": 0.2},
        {"kind": "drift", "amp_ratio": 0.3},
      ]
    """
    y = _ensure_1d(x).copy()

    for cfg in noise_cfg_list:
        kind = cfg.get("kind", "").lower()

        if kind == "wgn":
            snr_db = cfg.get("snr_db", 10.0)
            y = add_wgn_snr(y, snr_db=snr_db)

        elif kind == "hum":
            f0 = cfg.get("f0", 50.0)
            delta_f = cfg.get("delta_f", 0.5)
            amp_ratio = cfg.get("amp_ratio", 0.3)
            y = add_powerline_hum(y, fs=fs, f0=f0, delta_f=delta_f, amp_ratio=amp_ratio)

        elif kind == "drift":
            max_freq = cfg.get("max_freq", 0.5)
            amp_ratio = cfg.get("amp_ratio", 0.3)
            y = add_baseline_drift(y, fs=fs, max_freq=max_freq, amp_ratio=amp_ratio)

        elif kind == "motion":
            n_bursts = cfg.get("n_bursts", 3)
            burst_duration = cfg.get("burst_duration", 0.1)
            amp_ratio = cfg.get("amp_ratio", 2.0)
            y = add_motion_artifact(
                y,
                fs=fs,
                n_bursts=n_bursts,
                burst_duration=burst_duration,
                amp_ratio=amp_ratio,
            )

        elif kind == "spikes":
            n_spikes = cfg.get("n_spikes", 20)
            amp_ratio = cfg.get("amp_ratio", 3.0)
            y = add_spikes(y, n_spikes=n_spikes, amp_ratio=amp_ratio)

        elif kind == "pink":
            snr_db = cfg.get("snr_db", 10.0)
            beta = cfg.get("beta", 1.0)
            y = add_pink_noise(y, snr_db=snr_db, beta=beta)

        else:
            raise ValueError(f"Unknown noise kind: {kind}")

    return y
