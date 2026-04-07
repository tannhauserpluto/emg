# NOTE: ASCII header to keep patch tool stable; content below is authoritative.
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

# Foundation Experiments (论文基础证据层)

## 必须保留的基础实验范围
1. 预处理与分割 (activity segment)。
2. 特征工程 (TD/FD/AR) 与传统分类器 (LDA/SVM/RF)。
3. 噪声注入、降噪、鲁棒性评估。
4. feature ablation 与 clean vs noisy 对比。

## 对应代码与结果路径
- 预处理/分割: `src/segmentation.py`, `src/pipeline.py`, `generate_segmentation_detail.py`, `report/segment_summary.csv`
- 噪声/降噪: `src/noise.py`, `src/denoise.py`, `src/filters.py`, `run_gen_noisy.py`, `run_robust_eval.py`
- DL noise eval: `run_dl_noise_eval.py`, `outputs/summaries/dl_noise_results.csv`, `outputs/dl_noise_eval/`
- 特征工程: `src/features.py`, `data/ninapro_db5/features/db5_s1-3_g1-10.csv`
- ML baseline: `src/ml_models.py`, `run_ml_baseline.py`, `run_db5_ml.py`, `report/ml_db5_baseline.csv`
- feature ablation: `run_feature_ablation.py`, `make_feature_ablation_figure.py`, `report/ml_db5_feature_ablation.csv`
- 图表证据: `figures/robustness_heatmap.png`, `figures/Figure_3-Feature_Ablation_RF.png`, `figures/Figure_3-Clean_vs_Noisy_Summary_RF.png`

## 保留规则
- 以上内容不得误归档，属于论文基础证据层。
- 若后续整理 outputs/，应为这些实验建立 `outputs/foundation_experiments/` 的索引或镜像，但不删除原始报告文件。
