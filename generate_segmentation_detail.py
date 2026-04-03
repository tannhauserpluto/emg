# -*- coding: utf-8 -*-
# generate_segmentation_detail.py

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch

# ==========================================
# 1. 基础信号处理函数
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
    # 均方根能量 (RMS Energy) 或 均方能量 (Mean Square)
    # 这里用 Mean Square 与报告描述一致
    return pd.Series(x).pow(2).rolling(window_size).mean().fillna(0).values

# ==========================================
# 2. 绘图主程序
# ==========================================
def main():
    # --- 加载数据 ---
    search_path = os.path.join("data", "ninapro_db5", "raw", "s1", "*.mat")
    files = glob.glob(search_path)
    if not files:
        print("[Error] No files found.")
        return
    
    # 读取数据
    d = loadmat(files[0])
    try: fs = int(d.get('frequency', [200]).flat[0])
    except: fs = 200
    
    # 取单通道并预处理
    raw = d['emg'][:, 0]
    clean = notch_filter(butter_bandpass(raw, fs), fs)
    
    # --- 寻找一个完美的“单个动作”片段进行放大展示 ---
    # 计算全局能量以定位动作
    full_energy = calculate_energy(clean, int(0.1*fs))
    # 找最大峰值
    peak_idx = np.argmax(full_energy)
    
    # 截取峰值前后 1.5秒，总共 3秒，这样细节看得清
    win_sec = 3.0
    half_win = int((win_sec * fs) / 2)
    start = max(0, peak_idx - half_win)
    end = min(len(clean), peak_idx + half_win)
    
    t = np.arange(0, end-start) / fs
    sig_segment = clean[start:end]
    
    # --- 重新计算该片段的能量 (模拟实时计算) ---
    win_samples = int(0.200 * fs) # 200ms 窗口
    energy_segment = calculate_energy(sig_segment, win_samples)
    
    # --- 设定阈值 ---
    E_max = np.max(energy_segment)
    T_onset = 0.28 * E_max
    T_offset = 0.20 * E_max
    
    # --- 模拟双阈值判定逻辑 (找到触发点) ---
    state = 0 # 0:Rest, 1:Active
    trigger_idx = -1
    release_idx = -1
    
    # 简单的模拟循环找切点
    for i, e in enumerate(energy_segment):
        if state == 0:
            if e > T_onset:
                state = 1
                if trigger_idx == -1: trigger_idx = i # 记录第一次触发
        elif state == 1:
            if e < T_offset:
                state = 0
                if release_idx == -1 and trigger_idx != -1: # 记录第一次释放
                    release_idx = i
                    break # 只需要演示这一个动作即可

    # 如果没找到完整的，手动兜底（防止画图报错）
    if trigger_idx == -1: trigger_idx = int(len(t)*0.3)
    if release_idx == -1: release_idx = int(len(t)*0.7)

    # ==========================================
    # 3. 绘制学术风格图表
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- 上图：信号波形与激活区 ---
    ax1.plot(t, sig_segment, color="#191818", linewidth=1, label='Conditioned sEMG')
    
    # 绘制激活阴影
    ax1.axvspan(t[trigger_idx], t[release_idx], color='#2ca02c', alpha=0.2, label='Identified Gesture Segment')
    
    ax1.set_ylabel('Amplitude (mV)', fontweight='bold', fontsize=12)
    ax1.set_title('(A) Segmented Signal Output', loc='left', fontweight='bold', fontsize=14)
    ax1.legend(loc='upper right', frameon=True)
    ax1.margins(x=0)

    # --- 下图：能量与双阈值逻辑 (核心) ---
    ax2.plot(t, energy_segment, color='#d62728', linewidth=2.5, label='Short-Time Energy')
    
    # 阈值线
    ax2.axhline(T_onset, color='#1f77b4', linestyle='--', linewidth=2, label=f'Onset ($T_{{high}}$) = 28%')
    ax2.axhline(T_offset, color='#ff7f0e', linestyle='--', linewidth=2, label=f'Offset ($T_{{low}}$) = 20%')
    
    # --- 关键：添加学术标注 (Arrows & Text) ---
    
    # 1. 触发点标注
    ax2.annotate('Action Triggered', 
                 xy=(t[trigger_idx], T_onset), 
                 xytext=(t[trigger_idx]-0.6, T_onset + E_max*0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                 fontsize=11, fontweight='bold', color='#1f77b4')
    
    # 2. 释放点标注
    ax2.annotate('Action Released', 
                 xy=(t[release_idx], T_offset), 
                 xytext=(t[release_idx]+0.2, T_offset + E_max*0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                 fontsize=11, fontweight='bold', color='#ff7f0e')

    # 3. 迟滞区间标注 (Hysteresis)
    # 在两个阈值中间画一个双向箭头
    mid_time = t[len(t)//2]
    ax2.annotate('', xy=(mid_time, T_offset), xytext=(mid_time, T_onset),
                 arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax2.text(mid_time + 0.05, (T_onset+T_offset)/2, 'Hysteresis Zone\n(Prevents Jitter)', 
             va='center', fontsize=10, color='gray', style='italic')

    # 4. 填充阴影
    ax2.axvspan(t[trigger_idx], t[release_idx], color='#2ca02c', alpha=0.1)

    ax2.set_ylabel('Energy Amplitude', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Time (s)', fontweight='bold', fontsize=12)
    ax2.set_title('(B) Dual-Threshold Energy Logic', loc='left', fontweight='bold', fontsize=14)
    ax2.legend(loc='upper right', frameon=True)
    ax2.margins(x=0)
    
    plt.tight_layout()
    save_path = 'presentation_segmentation_detail.png'
    plt.savefig(save_path, dpi=300)
    print(f"[SUCCESS] Generated: {save_path}")

if __name__ == "__main__":
    main()