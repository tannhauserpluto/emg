# -*- coding: utf-8 -*-
# generate_figure_2_4.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
# 尝试设置字体，防止中文乱码（虽然这里用英文）
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei']

CSV_PATH = 'ml_db5_feature_ablation.csv'

def create_fallback_csv():
    """
    如果找不到CSV，手动创建包含报告中数据的CSV文件。
    数据来源：我们中期报告 Table 2-1 中的定稿数据。
    """
    print("[INFO] CSV not found. Generating data based on Interim Report values...")
    
    # 这些数据严格对应你报告里的 Table 2-1
    data = [
        # Model, Feature Set, Acc Mean, Acc Std
        ['LDA', 'TD', 0.350, 0.008],
        ['LDA', 'TD+FD', 0.362, 0.007],
        ['LDA', 'TD+FD+AR', 0.377, 0.008],
        
        ['SVM_RBF', 'TD', 0.727, 0.004],
        ['SVM_RBF', 'TD+FD', 0.730, 0.004],
        ['SVM_RBF', 'TD+FD+AR', 0.726, 0.004],
        
        ['RF', 'TD', 0.923, 0.004],       # 最佳性价比
        ['RF', 'TD+FD', 0.923, 0.004],
        ['RF', 'TD+FD+AR', 0.921, 0.005], # 特征多了反而微降
    ]
    
    df = pd.DataFrame(data, columns=['model', 'feature_set', 'acc_mean', 'acc_std'])
    df.to_csv(CSV_PATH, index=False)
    print(f"[SUCCESS] Created {CSV_PATH}")

def plot_ablation_bar_chart():
    # 1. 检查数据，不存在则自动生成
    if not os.path.exists(CSV_PATH):
        create_fallback_csv()

    # 2. 读取数据
    df = pd.read_csv(CSV_PATH)
    
    # 3. 数据格式化
    model_map = {'LDA': 'LDA (Baseline)', 
                 'RF': 'Random Forest', 
                 'SVM_RBF': 'SVM (RBF Kernel)'}
    df['model_label'] = df['model'].map(model_map)
    
    feature_order = ['TD', 'TD+FD', 'TD+FD+AR']
    model_order = ['LDA (Baseline)', 'SVM (RBF Kernel)', 'Random Forest']

    # 4. 绘图
    plt.figure(figsize=(10, 6))
    
    palette = sns.color_palette("Blues_d", n_colors=3)
    
    # 绘制主柱状图
    ax = sns.barplot(x='model_label', y='acc_mean', hue='feature_set', 
                     data=df, 
                     order=model_order, 
                     hue_order=feature_order, 
                     palette=palette,
                     edgecolor='black', linewidth=0.8, alpha=0.9)

    # 5. 添加误差棒和数值标签
    # 计算柱状图的宽度，用于定位
    # Seaborn 的 barplot 默认宽度通常是 0.8 / 类别数
    # 这里我们通过补丁来获取准确位置
    
    # 简单的 Hack 方法：手动遍历添加
    # 假设每个大组的中心是 0, 1, 2
    # 组内偏移量大概是 -0.27, 0, 0.27 (取决于 bar width)
    
    # 获取实际画出来的 bar width
    bars = [rect for rect in ax.get_children() if isinstance(rect, plt.Rectangle)]
    # 过滤掉背景等非柱子元素，取前9个（3模型x3特征）
    # 注意：Seaborn 绘制顺序通常是先画所有 Feature1，再画所有 Feature2...
    
    # 更稳健的方法：直接按坐标画
    width = 0.8 / 3
    offsets = [-width, 0, width]
    
    for i, mod_name in enumerate(model_order):
        # 找到原始模型名
        orig_name = [k for k, v in model_map.items() if v == mod_name][0]
        
        for j, feat_name in enumerate(feature_order):
            # 获取该数据点
            row = df[(df['model'] == orig_name) & (df['feature_set'] == feat_name)].iloc[0]
            mean = row['acc_mean']
            std = row['acc_std']
            
            # 计算 X 坐标
            x = i + offsets[j]
            
            # 画误差棒
            plt.errorbar(x, mean, yerr=std, fmt='none', ecolor='black', capsize=3)
            
            # 写数值 (只在 RF 上写，或者都写)
            # 为了图表整洁，我们只标记 RF 的数值，或者所有数值但字体小一点
            text_y = mean + std + 0.01
            plt.text(x, text_y, f"{mean:.1%}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 6. 图表修饰
    plt.title('Figure 2-4: Impact of Feature Domain Stacking on Classification Accuracy', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Mean Accuracy (5-Fold CV)', fontweight='bold', fontsize=12)
    plt.xlabel('Classification Model', fontweight='bold', fontsize=12)
    
    plt.ylim(0, 1.1) #稍微留多点空间给标签
    
    # 辅助线
    plt.axhline(y=0.85, color='red', linestyle='--', linewidth=1.5, label='Target (85%)')
    
    # 图例
    handles, labels = ax.get_legend_handles_labels()
    # 添加 Target 线到图例
    line = plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5)
    handles.append(line)
    labels.append('Target (85%)')
    
    plt.legend(handles=handles, labels=labels, title='Feature Set', loc='upper left')
    
    plt.tight_layout()
    plt.savefig('ablation_bar.png', dpi=300)
    print(f"[SUCCESS] Generated figure: ablation_bar.png")

if __name__ == "__main__":
    plot_ablation_bar_chart()