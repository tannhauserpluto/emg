# NOTE: ASCII header to keep patch tool stable; content below is authoritative.
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

# Supplementary Results (补充实验层)

## Cross-subject (困难设定)
- 目的: 对比 subject-dependent 主结果，展示跨被试的难度与差距。
- 主要代码: `train_dl.py`, `eval_dl.py`
- 可能输出路径: `artifacts/dl/db5_s1-10_win400_*`
- 结论定位: 仅用于补充讨论，不作为主结论。

### Quantified cross-subject baselines (DB5 S1-10, window-level)
- Standardized table: `outputs/tables/table_cross_subject_results.csv`
- cnn_lstm: accuracy 0.406443, macro-F1 0.402281 (`artifacts/dl/db5_s1-10_win400_cnn_lstm/metrics_summary.json`)
- xception2d: accuracy 0.509469, macro-F1 0.510178 (`artifacts/dl/db5_s1-10_win400_xception2d/metrics_summary.json`)
- Setting: repetition-holdout with subject split train 1-7 / val 8 / test 9-10.
- Interpretation: cross-subject remains a difficult setting; results are supplementary and do not override subject-dependent main results.

## Calibration (个体化微调)
- 目的: 评估 calibration 的收益与局限。
- 核心脚本: `build_calibration_dataset.py`, `calibrate_dl.py`, `calibration_utils.py`, `calibration_policy.py`
- 数据路径: `data/ninapro_db5/calibration/*.json`
- 已确认结论: subject 10 上有效，subject 9 收益有限。
- 结论定位: 系统补充结论，不替代主结果。

## 其他补充
- 模型演进对比 (cnn1d → cnn_lstm → xception/dualres) 的历史 run 需保留为补充证据。
- Noise robustness and clean/noisy comparisons belong to foundation experiments; see `docs/foundation_experiments.md`.
