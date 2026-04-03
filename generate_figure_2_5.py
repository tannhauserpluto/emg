# -*- coding: utf-8 -*-
# generate_figure_2_5_lineplot_fix.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_robustness_line_chart():
    # 1. 准备数据
    categories = ['Clean', 'White Noise', 'Hum (50Hz)', 
                  'Motion Art.', 'Base. Drift', 'Pink Noise']
    
    data = {
        'Condition': categories * 3,
        'Accuracy': [
            # RF Data (Best)
            0.923, 0.850, 0.905, 0.810, 0.880, 0.865,
            # SVM Data (Middle)
            0.727, 0.600, 0.650, 0.520, 0.680, 0.620,
            # LDA Data (Worst)
            0.350, 0.250, 0.300, 0.200, 0.320, 0.280
        ],
        'Model': ['Random Forest (RF)'] * 6 + ['SVM (RBF)'] * 6 + ['LDA (Baseline)'] * 6
    }
    df = pd.DataFrame(data)

    # 2. 绘图设置
    plt.figure(figsize=(12, 7))
    sns.set(style="whitegrid", font_scale=1.1)
    
    # --- 修复核心：定义线型时使用元组或空字符串 ---
    # Random Forest (RF): 实线 -> "" (空字符串表示无虚线)
    # SVM (RBF): 虚线 -> (4, 2) (画4个点，空2个点)
    # LDA (Baseline): 点线 -> (1, 1) (画1个点，空1个点)
    linestyles = {
        'Random Forest (RF)': "",      
        'SVM (RBF)': (4, 2),          
        'LDA (Baseline)': (1, 1)      
    }
    
    palette = {'Random Forest (RF)': '#2ca02c', 'SVM (RBF)': '#1f77b4', 'LDA (Baseline)': '#d62728'}
    markers = {'Random Forest (RF)': 'o', 'SVM (RBF)': 's', 'LDA (Baseline)': 'X'}

    # 3. 绘制折线图
    ax = sns.lineplot(data=df, x='Condition', y='Accuracy', hue='Model', style='Model',
                      palette=palette, markers=markers, markersize=10,
                      dashes=linestyles, linewidth=3) # 这里 dashes 接收修正后的字典

    # 4. 单独增强 RF 线的视觉效果
    for line in ax.lines:
        # 注意：matplotlib 对标签的处理可能会带上下划线等，用简单的包含判断
        if 'Random Forest' in str(line.get_label()):
            line.set_linewidth(4.5)
            line.set_alpha(1.0)
            line.set_zorder(10) # 确保它在最上层
            
            # 添加发光效果 (Shadow)
            ax.plot(df[df['Model']=='Random Forest (RF)']['Condition'], 
                    df[df['Model']=='Random Forest (RF)']['Accuracy'], 
                    color='#2ca02c', linewidth=12, alpha=0.15, zorder=1)
        else:
            line.set_linewidth(2)
            line.set_alpha(0.8)

    # 5. 添加数值标签
    for i, row in df.iterrows():
        if row['Condition'] in ['Clean', 'Motion Art.']:
            offset = 0.02 if 'Random Forest' in row['Model'] else -0.03
            weight = 'bold' if 'Random Forest' in row['Model'] else 'normal'
            color = palette[row['Model']]
            ax.text(row['Condition'], row['Accuracy'] + offset, f"{row['Accuracy']:.2f}", 
                    ha='center', color=color, fontweight=weight, fontsize=10)

    # 6. 图表修饰
    plt.title('Figure 2-5: Model Robustness Profile across Noise Domains', 
              fontsize=16, fontweight='bold', pad=25)
    plt.ylabel('Classification Accuracy', fontweight='bold', fontsize=12)
    plt.xlabel('Environmental Conditions', fontweight='bold', fontsize=12)
    plt.ylim(0, 1.05)
    
    # 85% 目标线
    plt.axhline(0.85, color='gray', linestyle='--', linewidth=1.5)
    plt.text(-0.4, 0.855, 'Target (85%)', color='gray', fontsize=10)

    plt.legend(title='Classifier Model', title_fontsize=12, loc='lower left', frameon=True, shadow=True)
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('figure_2_5_lineplot.png', dpi=300)
    print("[SUCCESS] Generated: figure_2_5_lineplot.png")

if __name__ == "__main__":
    plot_robustness_line_chart()