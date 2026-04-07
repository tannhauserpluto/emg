# NOTE: ASCII header to keep patch tool stable; content below is authoritative.
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

# Repository Map (thesis-ready view)

## 主要目录
- `data/ninapro_db5/`: DB5 原始数据、窗口数据、特征与 calibration 子集。
- `models/`: 深度学习模型实现 (cnn1d/cnn_lstm/tcn/xception/dualres 等)。
- `train_dl.py` / `eval_dl.py`: 深度学习训练与评估主线。
- `db5_windows.py` / `dl_dataset.py`: DB5 窗口构建与 split/归一化工具。
- `calibrate_dl.py` / `build_calibration_dataset.py`: calibration 数据与微调流程。
- `src/`: 传统 ML、噪声、滤波、分割与特征工程实现。
- `report/`、`figures/`: 传统 ML 与鲁棒性等实验产物。
- `outputs/`: subject-dependent 主结果候选输出 (当前不移动)。
- `artifacts/`: 历史/补充实验输出 (包含 cross-subject 与调试 run)。

## 主线代码入口
- 数据构建: `db5_windows.py`
- 数据集/拆分/归一化: `dl_dataset.py`
- 训练: `train_dl.py`
- 评估 (含 repetition-level 聚合): `eval_dl.py`
- calibration: `build_calibration_dataset.py`, `calibrate_dl.py`, `calibration_policy.py`

## 论文层级定位 (只读标记，不做移动)
- foundation experiments: 传统 ML、噪声/鲁棒性、feature ablation、分割/预处理 (`src/` + `run_*.py` + `report/` + `figures/`)。
- main results: subject-dependent + LORO + repetition-level 聚合，主要在 `outputs/dl_subject_dependent/dualres_xception2d/`。
- supplementary results: cross-subject (多在 `artifacts/dl/`) 与 calibration (见 `data/ninapro_db5/calibration/` 与 calibration 运行目录)。
- archive candidates: `artifacts/_smoke_*`、`outputs/dl_subject_dependent/*_smoke*` 等调试输出。
