# NOTE: ASCII header to keep patch tool stable; content below is authoritative.
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

# Problem Statement (事实型上下文)

## 核心事实
- 课题目标是基于 EMG 信号实现手势识别软件，要求覆盖十个以上手势且准确率 ≥ 85%。
- 论文必须同时覆盖噪声分析与滤波、分割与活动段提取、机器学习基线、深度学习方法与对比分析。
- DB5 数据集是当前统一主线数据来源。

## 应读取的代码/结果路径
- 任务约束文件: `docs/project_requirements.md`
- 数据与窗口构建: `db5_windows.py`, `data/ninapro_db5/windows/*.npz`
- 传统 ML 基线: `run_ml_baseline.py`, `report/ml_db5_baseline.csv`
- 深度学习主线: `train_dl.py`, `eval_dl.py`, `outputs/dl_subject_dependent/dualres_xception2d/`

## 已确认关键结论
- subject-dependent + LORO + repetition-level 聚合满足最终准确率目标。

## 注意事项
- 论文叙述不得仅聚焦深度学习，需覆盖 foundation 层实验。
