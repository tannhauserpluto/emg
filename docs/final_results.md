# NOTE: ASCII header to keep patch tool stable; content below is authoritative.
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

## Evidence Path Update (2026-04-06)
- Anchor (main): `outputs/summaries/main_results.csv`
- Supporting runs: `outputs/dl_subject_dependent/dualres_xception2d/*loro*_eval`
- Auxiliary only: `outputs/dl_subject_dependent/dualres_xception2d/summary_fixed_test6.csv`
- Note: this update supersedes the evidence-path list below.

# Final Main Results (主结果说明)

## 主结果选择规则
- 主结果必须来自 subject-dependent + LORO + repetition-level aggregation。
- 主指标为 repetition-level accuracy / macro-F1，window-level 指标仅作为附加参考。
- cross-subject 结果属于补充实验，不得覆盖主结果。
- calibration 属于系统补充结论，不得替代主结果。
- foundation experiments 必须保留并进入论文基础证据层。

## 主结果配置 (当前已确认)
- 数据集: NinaPro DB5
- Subjects: S1-S3
- Gestures: 1-10
- 设定: subject-dependent
- Split: leave-one-repetition-out (LORO)
- 聚合: repetition-level logits mean
- 模型: DualRes-Xception2D + augmentation

## 主指标 (已确认汇总值)
- repetition accuracy ~ 98.33%
- repetition macro-F1 ~ 98.33%
- window accuracy ~ 72.01%
- window macro-F1 ~ 71.97%

## 证据路径 (当前存在但不移动)
- LORO 运行目录: `outputs/dl_subject_dependent/dualres_xception2d/`
- Auxiliary summary: `outputs/dl_subject_dependent/dualres_xception2d/summary_fixed_test6.csv` (non-main)
- Main standardized summary: `outputs/summaries/main_results.csv` (main anchor)

## 注意事项
- 不允许将单次固定 split 当作唯一主结果。
- Main standardized summary: `outputs/summaries/main_results.csv` (main anchor)
