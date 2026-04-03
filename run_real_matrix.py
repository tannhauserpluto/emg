# -*- coding: utf-8 -*-
# run_real_matrix.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# 修正导入：只导入库里有的函数
from src.ninapro_db5 import build_features_db5, FEATURE_DIR

# ---------------------------------------------------------
# 补上缺失的辅助函数：检查文件是否存在，存在则读取，不存在则生成
# ---------------------------------------------------------
def load_or_build_db5_features(
    subjects,
    gestures,
    out_name="db5_s1-3_g1-10.csv",
    **kwargs,
) -> pd.DataFrame:
    """
    如果已经存在特征 CSV，则直接读取；
    否则调用 build_features_db5 重新生成。
    """
    os.makedirs(FEATURE_DIR, exist_ok=True)
    feat_path = os.path.join(FEATURE_DIR, out_name)

    if os.path.exists(feat_path):
        print(f"[INFO] Loading existing features: {feat_path}")
        df = pd.read_csv(feat_path)
    else:
        print(f"[INFO] Building features to: {feat_path}")
        df = build_features_db5(
            subjects=subjects,
            gestures=gestures,
            out_name=out_name,
            **kwargs,
        )
    return df

# ---------------------------------------------------------
# 主程序
# ---------------------------------------------------------
def main():
    # 1. 准备数据
    subjects = [1, 2, 3]
    gestures = list(range(1, 11))
    out_name = "db5_s1-3_g1-10.csv"
    
    print("[INFO] Loading Dataset...")
    
    # 现在直接调用本地定义的这个函数，就不会报错了
    df = load_or_build_db5_features(
        subjects=subjects,
        gestures=gestures,
        out_name=out_name,
        max_reps=6,
        win_sec=0.200,
        step_sec=0.050,
    )

    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].values
    y = df["gesture"].values

    print(f"[INFO] Data loaded. Shape: {X.shape}")

    # 2. 设置 RF 模型
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )

    # 3. 获取真实的预测结果 (5折交叉验证)
    print("[INFO] Running Cross-Validation to get predictions... (This may take a minute)")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=skf, n_jobs=-1)

    # 4. 计算混淆矩阵
    cm = confusion_matrix(y, y_pred)
    # 归一化 (按行求百分比)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    acc = accuracy_score(y, y_pred)
    print(f"[RESULT] Real Accuracy: {acc:.4f}")

    # 5. 画图 - 混淆矩阵
    gestures_labels = [f"G{i}" for i in sorted(list(set(y)))]
    
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.1)
    ax = sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                     xticklabels=gestures_labels, yticklabels=gestures_labels)
    ax.set_title(f'Figure 2-6: Confusion Matrix (Real Data) - Acc: {acc:.2%}', fontweight='bold')
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    plt.tight_layout()
    plt.savefig('real_confusion_matrix.png', dpi=300)
    print("[INFO] Saved real_confusion_matrix.png")

    # 6. 画图 - 单手势准确率
    per_class_acc = cm_norm.diagonal()
    
    plt.figure(figsize=(10, 5))
    colors = ['#2ca02c' if x >= 0.85 else '#ff7f0e' for x in per_class_acc] # 绿色达标，橙色未达标
    bars = plt.bar(gestures_labels, per_class_acc, color=colors, edgecolor='black', alpha=0.8)
    plt.axhline(0.85, color='red', linestyle='--', linewidth=2, label='Target 85%')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.ylim(0, 1.1)
    plt.title('Figure 2-7: Per-Gesture Recognition Accuracy (Real Data)', fontweight='bold')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('real_per_class_accuracy.png', dpi=300)
    print("[INFO] Saved real_per_class_accuracy.png")

if __name__ == "__main__":
    main()