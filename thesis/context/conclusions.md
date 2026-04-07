# NOTE: ASCII header to keep patch tool stable; content below is authoritative.
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

# Conclusions (事实型上下文)

## 核心事实
- subject-dependent + LORO + repetition-level 聚合提供当前最高精度主结果。
- cross-subject 结果体现更高难度，应作为补充讨论。
- calibration 结论为“有条件有效”，属于系统补充结论。
- foundation 实验 (ML、噪声、鲁棒性、feature ablation) 是论文基础证据层，必须保留。

## 关键路径
- 主结果: `outputs/summaries/main_results.csv`
- 基础实验: `outputs/summaries/foundation_results.csv`
- 补充实验: `outputs/summaries/supplementary_results.csv`

## 注意事项
- 结论章节必须体现任务目标与准确率要求。
