# -*- coding: utf-8 -*-
# generate_presentation_figure.py

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, welch
import matplotlib.gridspec as gridspec

# ==========================================
# 1. 核心算法实现 (与您的描述完全一致)
# ==========================================

def butter_bandpass(data, fs, low=20.0, high=200.0, order=4):
    nyq = 0.5 * fs
    if high >= nyq: high = nyq * 0.95
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data, fs, f0=50.0, Q=30.0):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, data)

def calculate_energy(x, window_size):
    # 使用 Pandas 的 rolling mean square 计算能量
    return pd.Series(x).pow(2).rolling(window_size).mean().fillna(0).values

# ==========================================
# 2. 绘图主程序
# ==========================================

def main():
    # --- A. 加载真实数据 ---
    # 自动寻找 S1_E1_A1.mat
    search_path = os.path.join("data", "ninapro_db5", "raw", "s1", "*.mat")
    files = glob.glob(search_path)
    if not files:
        print("[Error] No .mat files found in data/ninapro_db5/raw/s1/")
        return
    
    mat_path = files[0] # 默认取第一个
    data_dict = loadmat(mat_path)
    
    # 获取采样率
    try:
        fs = int(data_dict.get('frequency', [200]).flat[0])
    except:
        fs = 200
        
    # 获取单通道信号 (选取动作比较明显的一段)
    raw_full = data_dict['emg'][:, 0]
    
    # 为了图表美观，我们自动搜索一段 "有静息也有动作" 的区间 (约 4秒)
    win_len = 4 * fs 
    # 简单策略：找方差最大的位置作为动作中心
    center_idx = np.argmax(pd.Series(raw_full).rolling(fs).std().fillna(0))
    start_idx = max(0, center_idx - win_len // 2)
    end_idx = min(len(raw_full), center_idx + win_len // 2)
    
    t = np.arange(0, end_idx - start_idx) / fs
    raw = raw_full[start_idx:end_idx]

    # --- B. 信号处理 (完全对应演讲稿) ---
    # 1. 混合滤波
    bp_filtered = butter_bandpass(raw, fs, low=20, high=200) # 20-200Hz
    final_filtered = notch_filter(bp_filtered, fs, f0=50)    # 50Hz Notch
    
    # 2. 能量计算
    energy_win = int(0.2 * fs) # 200ms 窗口
    energy = calculate_energy(final_filtered, energy_win)
    
    # 3. 阈值计算
    E_max = np.max(energy)
    T_onset = 0.28 * E_max
    T_offset = 0.20 * E_max
    
    # --- C. 计算 PSD (用于展示频域滤波效果) ---
    f_raw, Pxx_raw = welch(raw, fs, nperseg=1024)
    f_filt, Pxx_filt = welch(final_filtered, fs, nperseg=1024)

    # ==========================================
    # 3. 生成“学术级”组合图
    # ==========================================
    fig = plt.figure(figsize=(12, 10))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 使用 GridSpec 布局：上(时域)、中(频域)、下(分段)
    gs = gridspec.GridSpec(3, 2, figure=fig)
    
    # --- 子图 1: 时域滤波对比 (占据第一行) ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, raw, color="#686862", label='Raw Signal (Noisy)', linewidth=1, alpha=0.7)
    ax1.plot(t, final_filtered, color='#d62728', label='Filtered (Bandpass + Notch)', linewidth=1.2)
    ax1.set_title('(A) Signal Conditioning: Time Domain Comparison', fontweight='bold', loc='left', fontsize=12)
    ax1.set_ylabel('Amplitude (mV)', fontweight='bold')
    ax1.legend(loc='upper right', frameon=True)
    ax1.margins(x=0)
    
    # 标注：指向毛刺
    ax1.annotate('Noise Reduced', xy=(t[len(t)//2], np.max(final_filtered)*0.8), 
                 xytext=(t[len(t)//2]+0.5, np.max(raw)*0.8),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)

    # --- 子图 2: 频域 PSD 分析 (占据第二行) ---
    # 这个图非常高级，能直接证明你用了 Notch 和 Bandpass
    ax2 = fig.add_subplot(gs[1, :])
    ax2.semilogy(f_raw, Pxx_raw, color="#686862", label='Raw Spectrum', linewidth=1)
    ax2.semilogy(f_filt, Pxx_filt, color='#1f77b4', label='Filtered Spectrum', linewidth=1.5)
    
    # 重点标注 50Hz 陷波
    ax2.annotate('50Hz Notch Filter', xy=(50, np.min(Pxx_filt[f_filt>45])), 
                 xytext=(60, np.min(Pxx_filt)*100),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontweight='bold', color='#1f77b4')
    
    # 重点标注 20Hz 高通
    ax2.axvspan(0, 20, color='gray', alpha=0.1)
    ax2.text(5, np.max(Pxx_raw)/10, 'Motion Artifacts\n(<20Hz)', color='gray', fontsize=9)

    ax2.set_title('(B) Spectral Analysis: Validation of Hybrid Filtering', fontweight='bold', loc='left', fontsize=12)
    ax2.set_ylabel('PSD (V**2/Hz)', fontweight='bold')
    ax2.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax2.set_xlim(0, 150) # 只展示到 150Hz 即可，看清楚低频
    ax2.legend(loc='upper right', frameon=True)

    # --- 子图 3: 自适应分段逻辑 (占据第三行) ---
    ax3 = fig.add_subplot(gs[2, :])
    # 归一化能量方便展示
    ax3.plot(t, energy, color='#2ca02c', label='Short-Time Energy', linewidth=2)
    
    # 画阈值线
    ax3.axhline(T_onset, color='red', linestyle='--', linewidth=1.5, label=f'Onset Threshold ($0.28 \cdot E_{{max}}$)')
    ax3.axhline(T_offset, color='orange', linestyle='--', linewidth=1.5, label=f'Offset Threshold ($0.20 \cdot E_{{max}}$)')
    
    # 填充激活区域 (简单的逻辑演示)
    # 找到大于 Offset 的区域作为大致的激活区演示
    active_mask = energy > T_offset
    ax3.fill_between(t, 0, np.max(energy), where=active_mask, color='#2ca02c', alpha=0.1, label='Active Segment')

    ax3.set_title('(C) Adaptive Segmentation: Dual-Threshold Energy Logic', fontweight='bold', loc='left', fontsize=12)
    ax3.set_ylabel('Energy (a.u.)', fontweight='bold')
    ax3.set_xlabel('Time (s)', fontweight='bold')
    ax3.legend(loc='upper right', frameon=True)
    ax3.margins(x=0)

    plt.tight_layout()
    
    # 保存
    save_path = 'presentation_core_algorithm.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Generated presentation figure: {save_path}")

if __name__ == "__main__":
    main()