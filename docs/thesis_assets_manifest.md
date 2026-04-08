# NOTE: ASCII header to keep patch tool stable; content below is authoritative.
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

# Thesis Assets Manifest (占位骨架)

| Asset | Type | Path | Purpose | Status |
| --- | --- | --- | --- | --- |
| Main results summary | CSV | `outputs/summaries/main_results.csv` | 主结果指标汇总 | active |
| Foundation results summary | CSV | `outputs/summaries/foundation_results.csv` | 传统 ML / 噪声 / 鲁棒性汇总 | active |
| Supplementary results summary | CSV | `outputs/summaries/supplementary_results.csv` | cross-subject / calibration 汇总 | active |
| Error analysis | CSV | `outputs/summaries/error_analysis.csv` | 关键失败案例 | placeholder |
| Robustness heatmap | Figure | `figures/robustness_heatmap.png` | 噪声鲁棒性可视化 | existing |
| Feature ablation (RF) | Figure | `figures/Figure_3-Feature_Ablation_RF.png` | 特征消融对比 | existing |
| Clean vs noisy (RF) | Figure | `figures/Figure_3-Clean_vs_Noisy_Summary_RF.png` | clean vs noisy 对比 | existing |
| Segmentation detail | Figure | `presentation_segmentation_detail.png` | 分割流程展示 | existing |
| ML vs DL key noise (Accuracy) | Figure | `outputs/figures/fig_ml_vs_dl_noise_key_conditions_accuracy.png` | ML vs DL 关键噪声条件准确率对比 | generated |
| ML vs DL key noise (Macro-F1) | Figure | `outputs/figures/fig_ml_vs_dl_noise_key_conditions_macro_f1.png` | ML vs DL 关键噪声条件 Macro-F1 对比 | generated |
| Main LORO window vs repetition | Figure | `outputs/figures/fig_main_loro_window_vs_repetition.png` | 主结果 LORO window vs repetition 准确率对比 | generated |
| ML baseline comparison | Figure | `outputs/figures/fig_ml_baseline_comparison.png` | ML baseline accuracy/macro-F1 对比 | generated |
| DL full matrix heatmap | Figure | `outputs/figures/fig_dl_noise_full_matrix_heatmap.png` | DL 噪声矩阵热力图（denoise=none） | generated |
| DL best denoise comparison | Figure | `outputs/figures/fig_dl_best_denoise_comparison.png` | DL 去噪效果对比（none vs best） | generated |
| Cross-subject vs subject-dependent | Figure | `outputs/figures/fig_cross_subject_vs_subject_dependent.png` | 不同设定与决策层策略的性能差异 | generated |

> 说明：该清单为论文消费层入口，后续整理时应补全每个条目的来源 run/脚本与指标。
