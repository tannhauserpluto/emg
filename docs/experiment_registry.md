# NOTE: ASCII header to keep patch tool stable; content below is authoritative.
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

# Experiment Registry (论文分层清单)

## 分层规则 (必须遵守)
- foundation experiments: 中期与基础证据层 (preprocessing/segmentation/ML baseline/noise/robustness/feature ablation)。
- main results: subject-dependent + LORO + repetition-level aggregation 的最终结论。
- supplementary results: cross-subject、calibration、模型演进等补充讨论。
- archive candidates: smoke run、调试输出、重复或无论文引用价值的产物 (本轮不移动)。

## 核心实验清单

| Experiment ID | Category | Description | Scripts | Outputs / Evidence | Status |
| --- | --- | --- | --- | --- | --- |
| foundation_preprocessing_segmentation | foundation | 分割与活动段提取证据 | `src/segmentation.py`, `generate_segmentation_detail.py` | `report/segment_summary.csv`, `presentation_segmentation_detail.png` | confirmed |
| foundation_ml_baselines | foundation | TD/FD/AR + LDA/SVM/RF 基线 | `run_ml_baseline.py`, `run_db5_ml.py`, `src/ml_models.py` | `report/ml_db5_baseline.csv` | confirmed |
| foundation_noise_robustness | foundation | 噪声注入与鲁棒性 | `run_gen_noisy.py`, `run_robust_eval.py`, `src/noise.py`, `src/denoise.py` | `report/robust_db5_noise.csv`, `figures/robustness_heatmap.png` | confirmed |
| foundation_feature_ablation | foundation | feature ablation 对比 | `run_feature_ablation.py` | `report/ml_db5_feature_ablation.csv`, `figures/Figure_3-Feature_Ablation_RF.png` | confirmed |
| main_subject_dependent_loro | main | S1-S3 LORO repetition-level 主结果 | `train_dl.py`, `eval_dl.py` | `outputs/dl_subject_dependent/dualres_xception2d/` | active |
| supp_cross_subject_dl | supplementary | S1-S10 cross-subject 深度学习 | `train_dl.py`, `eval_dl.py` | `artifacts/dl/db5_s1-10_win400_*` | active |
| supp_calibration_subject10 | supplementary | calibration 正例 (subject10) | `build_calibration_dataset.py`, `calibrate_dl.py` | `data/ninapro_db5/calibration/*subject10*` | active |
| supp_calibration_subject9 | supplementary | calibration 负例 (subject9) | `build_calibration_dataset.py`, `calibrate_dl.py` | `data/ninapro_db5/calibration/*subject9*` | active |
| archive_smoke_runs | archive_candidate | smoke/debug run | `smoke_test_tcn.py` | `artifacts/_smoke_*` | candidate |
