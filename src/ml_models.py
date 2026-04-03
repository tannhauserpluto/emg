# -*- coding: utf-8 -*-
# src/ml_models.py
"""
传统机器学习基线：
  - LDA
  - SVM (RBF)
  - Random Forest

使用 StratifiedKFold 做交叉验证，输出平均 Accuracy 和 Macro-F1。
"""

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score


def _get_models():
    models = {
        "LDA": LinearDiscriminantAnalysis(),
        "SVM_RBF": SVC(kernel="rbf", C=10.0, gamma="scale"),
        "RF": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        ),
    }
    return models


def cross_val_evaluate(X: np.ndarray,
                       y: np.ndarray,
                       n_splits: int = 5,
                       random_state: int = 42) -> pd.DataFrame:
    """
    对多种模型做 k 折交叉验证。
    返回 DataFrame，每行一个模型，包含均值和标准差。
    """
    models = _get_models()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    records = []

    for name, model in models.items():
        acc_list = []
        f1_list = []

        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            acc = accuracy_score(y_te, y_pred)
            f1 = f1_score(y_te, y_pred, average="macro")

            acc_list.append(acc)
            f1_list.append(f1)

        records.append({
            "model": name,
            "acc_mean": np.mean(acc_list),
            "acc_std": np.std(acc_list),
            "macro_f1_mean": np.mean(f1_list),
            "macro_f1_std": np.std(f1_list),
        })

    return pd.DataFrame(records)
