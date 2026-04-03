# -*- coding: utf-8 -*-
# run_gen_noisy.py
"""
NinaPro DB5 噪声演示脚本：

- 从 subject 1, gesture 1, repetition 1 取一段原始 EMG
- 选定一个通道（例如 ch=0）
- 生成多种噪声版本：
    - WGN 不同 SNR
    - 工频 hum
    - drift
    - motion artifact
    - spikes
    - pink noise
- 对每一种，画：
    - time-domain 波形 (clean vs noisy)
    - magnitude spectrum 频谱 (clean vs noisy)
- 图片保存在 figures/noise_demo/
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.ninapro_db5 import unzip_subject, get_demo_segment
from src.noise import apply_noises


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_time_and_spectrum(
    fs: float,
    t: np.ndarray,
    x_clean: np.ndarray,
    x_noisy: np.ndarray,
    title: str,
    out_prefix: str,
):
    """
    画时域 & 频域对比图，分别保存为:
      out_prefix + "_time.png"
      out_prefix + "_spec.png"
    """
    # 时域
    plt.figure(figsize=(10, 4))
    plt.plot(t, x_clean, label="Clean", linewidth=1.0)
    plt.plot(t, x_noisy, label="Noisy", linewidth=0.8, alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Time Domain - {title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + "_time.png", dpi=200)
    plt.close()

    # 频谱
    N = len(x_clean)
    # 使用相同窗长 & N 做对比
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    Xc = np.fft.rfft(x_clean * np.hanning(N))
    Xn = np.fft.rfft(x_noisy * np.hanning(N))

    mag_c = 20 * np.log10(np.abs(Xc) + 1e-12)
    mag_n = 20 * np.log10(np.abs(Xn) + 1e-12)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, mag_c, label="Clean", linewidth=1.0)
    plt.plot(freqs, mag_n, label="Noisy", linewidth=0.8, alpha=0.8)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"Spectrum - {title}")
    plt.xlim(0, fs / 2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + "_spec.png", dpi=200)
    plt.close()


def main():
    out_dir = os.path.join("figures", "noise_demo")
    ensure_dir(out_dir)

    # 1. 确保解压 DB5 subject 1
    unzip_subject(1)

    # 2. 取一段演示用 EMG 片段
    fs, t, emg_seg = get_demo_segment(
        subject_id=1,
        gesture=1,
        repetition=1,
        exercise=1,
    )

    # 选一个通道，比如第一个通道
    ch = 0
    x_clean = emg_seg[:, ch]

    print(f"[INFO] Demo segment: length={len(x_clean)}, fs={fs} Hz, channel={ch}")

    # 3. 定义一组噪声配置
    noise_settings = [
        # 纯 WGN，不同 SNR
        ("wgn_20dB", [{"kind": "wgn", "snr_db": 20.0}]),
        ("wgn_10dB", [{"kind": "wgn", "snr_db": 10.0}]),
        ("wgn_0dB",  [{"kind": "wgn", "snr_db": 0.0}]),

        # 单个噪声类型
        ("hum",      [{"kind": "hum", "amp_ratio": 0.3}]),
        ("drift",    [{"kind": "drift", "amp_ratio": 0.3}]),
        ("motion",   [{"kind": "motion", "n_bursts": 3, "burst_duration": 0.1, "amp_ratio": 2.0}]),
        ("spikes",   [{"kind": "spikes", "n_spikes": 30, "amp_ratio": 3.0}]),
        ("pink_10dB", [{"kind": "pink", "snr_db": 10.0, "beta": 1.0}]),

        # 组合噪声示例
        ("wgn10_hum_drift", [
            {"kind": "wgn", "snr_db": 10.0},
            {"kind": "hum", "amp_ratio": 0.2},
            {"kind": "drift", "amp_ratio": 0.2},
        ]),
    ]

    # 4. 为每种噪声配置生成波形 & 频谱图
    for name, cfg in noise_settings:
        print(f"[INFO] Generating noisy signal: {name} ...")
        x_noisy = apply_noises(x_clean, fs=fs, noise_cfg_list=cfg)

        out_prefix = os.path.join(out_dir, f"demo_{name}")
        plot_time_and_spectrum(
            fs=fs,
            t=t,
            x_clean=x_clean,
            x_noisy=x_noisy,
            title=name,
            out_prefix=out_prefix,
        )

    print(f"\n[INFO] All noise demo figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
