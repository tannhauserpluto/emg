# NOTE: ASCII header to keep patch tool stable; content below is authoritative.
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

# Machine Learning Baselines (事实型上下文)

## 核心事实
- 传统 ML 基线是论文基础证据层的重要组成。
- 使用 TD/FD/AR 特征与 LDA/SVM/RF 分类器进行对比。

## 关键脚本/路径
- 特征工程: `src/features.py`
- ML 模型: `src/ml_models.py`
- 运行脚本: `run_ml_baseline.py`, `run_db5_ml.py`
- 特征数据: `data/ninapro_db5/features/db5_s1-3_g1-10.csv`
- 报告结果: `report/ml_db5_baseline.csv`

## 已确认结论
- ML 基线用于与深度学习方法对比，不可省略。

## 注意事项
- 论文写作需引用 ML baseline 结果作为对比基线。
