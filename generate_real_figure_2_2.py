# -*- coding: utf-8 -*-
# generate_real_figure_2_2.py

import os
import glob
import numpy as np
import pandas as pd  # <--- 补上了这个关键的 import
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch

# 导入你 src 里的滤波函数（确保 src 文件夹在旁边）
from src.filters import notch_50hz

def butter_bandpass_local(data, fs, low=20.0, high=200.0, order=4):
    """
    本地定义带通滤波，增加对采样率的自动适应防止报错。
    """
    nyq = 0.5 * fs
    # 如果高频截止超过了奈奎斯特频率，强制截断到 0.95 * Nyquist
    if high >= nyq:
        high = nyq * 0.95
        print(f"[WARN] Sample rate is {fs}Hz. Adjusting High-cut to {high:.1f}Hz.")
    
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

def find_active_window(signal, window_size=1000):
    """
    自动寻找信号中波动最大的区间，确保画出来的图有动作。
    """
    # 计算滚动标准差
    s = pd.Series(signal)
    rolling_std = s.rolling(window_size).std()
    # 找到波动最大的位置中心，fillna(0) 防止开头 NaN 报错
    max_idx = np.argmax(rolling_std.fillna(0).values)
    
    start = max(0, max_idx - window_size)
    end = min(len(signal), max_idx + window_size)
    return start, end

def main():
    # 1. 设定数据路径 (根据你的 ninapro_db5.py 逻辑)
    # 尝试寻找 s1 的第一个练习文件
    search_path = os.path.join("data", "ninapro_db5", "raw", "s1", "*.mat")
    files = glob.glob(search_path)
    
    if not files:
        print(f"[ERROR] No .mat files found in {search_path}")
        print("Please ensure your data structure is: data/ninapro_db5/raw/s1/S1_E1_A1.mat")
        return

    # 默认取第一个文件，通常是 E1 (基本手势)
    mat_path = files[0]
    print(f"[INFO] Loading: {mat_path}")
    
    # 2. 读取数据
    d = loadmat(mat_path)
    emg = d['emg']  # shape (Time, Channels)
    
    # --- 修复 DeprecationWarning ---
    # 使用 .flat[0] 安全地提取标量，不管它是 [[200]] 还是 [200]
    freq_data = d.get('frequency', np.array([[200]]))
    fs = int(freq_data.flat[0])
    print(f"[INFO] Sampling Frequency: {fs} Hz")

    # 选取第1个通道 (Channel 0)
    raw_signal = emg[:, 0]

    # 3. 寻找一个活跃片段 (Active Segment)
    # 我们不画整段，只画 3~4 秒的动作片段，这样细节更清晰
    # window_size 为半个窗口长
    win_len = int(2.0 * fs) 
    start, end = find_active_window(raw_signal, window_size=win_len)
    
    # 截取片段
    t = np.arange(0, end-start) / fs
    segment_raw = raw_signal[start:end]

    # 4. 执行滤波 (Pipeline)
    # 步骤 A: 带通 20-200Hz (或适应 fs)
    segment_bp = butter_bandpass_local(segment_raw, fs, low=20.0, high=200.0)
    
    # 步骤 B: 陷波 50Hz (去除工频)
    segment_filtered = notch_50hz(segment_bp, fs, f0=50.0)

    # 5. 绘图 (仿照 Figure 2-2 风格)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # 子图 1: Raw
    ax1.plot(t, segment_raw, color='#1f77b4', linewidth=1.0, label='Raw Signal')
    ax1.set_title('(a) Raw sEMG Signal (Real Data)', loc='left', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Amplitude (mV)', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 子图 2: Filtered
    ax2.plot(t, segment_filtered, color='#ff7f0e', linewidth=1.0, label='Filtered (Bandpass + Notch)')
    ax2.set_title('(b) Processed Signal', loc='left', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Time (s)', fontweight='bold')
    ax2.set_ylabel('Amplitude (mV)', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 动态获取文件名做标题
    file_name = os.path.basename(mat_path)
    plt.suptitle(f'Figure 2-2: Comparison of sEMG Signals Before and After Hybrid Filtering\n(Source: {file_name}, Ch-1)', 
                 fontsize=13, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    out_file = 'figure_2_2_real.png'
    plt.savefig(out_file, dpi=300)
    print(f"[SUCCESS] Real figure generated: {out_file}")

if __name__ == "__main__":
    main()