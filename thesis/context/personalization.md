# NOTE: ASCII header to keep patch tool stable; content below is authoritative.
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

# Personalization / Subject-dependent (事实型上下文)

## 核心事实
- 主结果属于 subject-dependent 实验，按 repetition 进行 LORO 评估。
- 结果以 repetition-level 聚合为主指标。

## 关键脚本/路径
- 训练与评估: `train_dl.py`, `eval_dl.py`, `run_subject_dependent_batch.py`
- 主结果目录: `outputs/dl_subject_dependent/dualres_xception2d/`
- 汇总表: `outputs/summaries/main_results.csv`

## 已确认关键指标
- repetition accuracy ≈ 98.33%
- repetition macro-F1 ≈ 97.78%
- window accuracy ≈ 72.18%
- window macro-F1 ≈ 72.16%

## 注意事项
- 该结果是论文主结论，不得被 cross-subject 结果覆盖。
