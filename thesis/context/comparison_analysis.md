# NOTE: ASCII header to keep patch tool stable; content below is authoritative.
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

# Comparison Analysis (事实型上下文)

## 核心事实
- 论文需对传统 ML 与深度学习方法进行对比。
- 论文需对 clean vs noisy 与鲁棒性结果进行对比。
- 主结果指标必须满足准确率 ≥ 85% 目标。

## 关键脚本/路径
- ML baseline: `report/ml_db5_baseline.csv`, `run_ml_baseline.py`
- Noise/robustness: `report/robust_db5_noise.csv`, `figures/robustness_heatmap.png`
- DL noise summary: `outputs/summaries/dl_noise_results.csv`
- DL noise eval script: `run_dl_noise_eval.py`
- 深度学习主结果: `outputs/summaries/main_results.csv`
- 汇总索引: `outputs/summaries/foundation_results.csv`, `outputs/summaries/supplementary_results.csv`

## 注意事项
- 对比分析需明确哪些是 foundation、哪些是主结果、哪些是补充结果。
