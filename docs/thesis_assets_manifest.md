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

> 说明：该清单为论文消费层入口，后续整理时应补全每个条目的来源 run/脚本与指标。
