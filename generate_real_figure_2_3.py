# -*- coding: utf-8 -*-
# generate_real_figure_2_3.py

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch

# --- 1. 复用滤波逻辑 (确保信号干净) ---
def butter_bandpass_local(data, fs, low=20.0, high=200.0, order=4):
    nyq = 0.5 * fs
    if high >= nyq: high = nyq * 0.95
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

def notch_50hz_local(data, fs, f0=50.0, Q=30.0):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, data, axis=0)

# --- 2. 复用并可视化分段逻辑 ---
def calculate_energy_profile(x, fs, window_size=200, overlap=0.2):
    """
    计算用于绘图的能量曲线 (对应 segmentation.py 中的 win_energy)
    """
    step = max(1, int(window_size * (1 - overlap)))
    energy_values = []
    time_indices = []
    
    # 简单的滑动窗口
    for s in range(0, len(x) - window_size + 1, step):
        seg = x[s : s + window_size]
        # 计算均方能量 (Mean Square)
        e = np.mean(seg**2)
        energy_values.append(e)
        # 记录时间点（取窗口中心）
        time_indices.append(s + window_size//2)
        
    return np.array(time_indices), np.array(energy_values)

def get_segments_from_energy(energy_arr, time_indices, A=0.28, B=0.20):
    """
    简单的双阈值逻辑演示
    """
    Emax = np.max(energy_arr)
    Tha = A * Emax
    Thb = B * Emax
    
    segments = []
    in_segment = False
    start_t = 0
    
    for t, e in zip(time_indices, energy_arr):
        if not in_segment:
            if e > Tha:
                in_segment = True
                start_t = t
        else:
            if e < Thb:
                in_segment = False
                end_t = t
                segments.append((start_t, end_t))
                
    # 处理结尾
    if in_segment:
        segments.append((start_t, time_indices[-1]))
        
    return segments, Tha, Thb

# --- 3. 主程序 ---
def main():
    # A. 读取数据
    search_path = os.path.join("data", "ninapro_db5", "raw", "s1", "*.mat")
    files = glob.glob(search_path)
    if not files:
        print("[ERROR] No .mat files found. Check path.")
        return
    
    mat_path = files[0]
    print(f"[INFO] Loading: {mat_path}")
    d = loadmat(mat_path)
    
    # 获取采样率和数据
    freq_data = d.get('frequency', np.array([[200]]))
    fs = int(freq_data.flat[0])
    raw_signal = d['emg'][:, 0] # 取通道 0
    
    # B. 截取一段明显的动作区域 (比如 5秒)
    # 为了图表好看，我们找波动最大的地方
    window_len = int(5.0 * fs)
    s_series = pd.Series(raw_signal)
    roll_std = s_series.rolling(fs).std().fillna(0)
    center_idx = np.argmax(roll_std.values)
    
    start_idx = max(0, center_idx - window_len//2)
    end_idx = min(len(raw_signal), center_idx + window_len//2)
    
    # 提取并预处理
    segment_raw = raw_signal[start_idx:end_idx]
    segment_clean = butter_bandpass_local(segment_raw, fs)
    segment_clean = notch_50hz_local(segment_clean, fs)
    
    # 时间轴 (从 0 开始)
    t_axis = np.arange(len(segment_clean)) / fs
    
    # C. 计算能量和阈值
    win_samples = int(0.200 * fs) # 200ms 窗口
    t_energy_idx, energy_vals = calculate_energy_profile(segment_clean, fs, window_size=win_samples)
    t_energy_sec = t_energy_idx / fs # 转换为秒
    
    # 获取分段点
    active_segs, Tha, Thb = get_segments_from_energy(energy_vals, t_energy_idx)
    
    # D. 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    # --- 子图 1: EMG 波形 + 标记区域 ---
    ax1.plot(t_axis, segment_clean, color='#333333', linewidth=0.8, label='Filtered sEMG')
    
    # 画出检测到的活动区域 (绿色阴影)
    for i, (s, e) in enumerate(active_segs):
        # 转换为当前片段的相对时间
        s_sec = s / fs
        e_sec = e / fs
        label = "Active Segment" if i == 0 else None
        ax1.axvspan(s_sec, e_sec, color='#2ca02c', alpha=0.3, label=label)
        
    ax1.set_ylabel('Amplitude (mV)', fontweight='bold')
    ax1.set_title('(a) Filtered Signal with Detected Activity', loc='left', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # --- 子图 2: 能量曲线 + 阈值线 ---
    ax2.plot(t_energy_sec, energy_vals, color='#ff7f0e', linewidth=1.5, label='Short-time Energy')
    
    # 画阈值线
    ax2.axhline(Tha, color='red', linestyle='--', linewidth=1.2, label=f'Onset Thr (A={0.28})')
    ax2.axhline(Thb, color='green', linestyle='--', linewidth=1.2, label=f'Offset Thr (B={0.20})')
    
    # 填充能量超过阈值的区域
    # 为了视觉效果，稍微填充一下
    ax2.fill_between(t_energy_sec, 0, energy_vals, where=(energy_vals > Thb), color='#ff7f0e', alpha=0.1)

    ax2.set_ylabel('Energy (a.u.)', fontweight='bold')
    ax2.set_xlabel('Time (s)', fontweight='bold')
    ax2.set_title('(b) Energy Profile & Dual Thresholds', loc='left', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # 总标题
    file_name = os.path.basename(mat_path)
    plt.suptitle(f'Figure 2-3: Visualization of Adaptive Activation Segmentation Results\n(Source: {file_name}, Window=200ms)', 
                 fontsize=13, fontweight='bold', y=0.96)
    
    plt.tight_layout()
    out_file = 'figure_2_3_real.png'
    plt.savefig(out_file, dpi=300)
    print(f"[SUCCESS] Generated: {out_file}")

if __name__ == "__main__":
    main()