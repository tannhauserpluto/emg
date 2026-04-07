# NOTE: ASCII header to keep patch tool stable; content below is authoritative.
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

# Deep Learning (事实型上下文)

## 核心事实
- 深度学习主线使用 DB5 窗口数据与多模型对比 (cnn1d/cnn_lstm/xception/dualres)。
- subject-dependent + LORO + repetition-level 聚合为主结果来源。
- cross-subject 结果仅作为困难设定补充。

## 关键脚本/路径
- 数据构建: `db5_windows.py`
- 训练与评估: `train_dl.py`, `eval_dl.py`
- 模型: `models/cnn1d.py`, `models/cnn_lstm.py`, `models/xception2d.py`, `models/dualres_xception2d.py`
- 主结果输出: `outputs/dl_subject_dependent/dualres_xception2d/`
- 跨被试输出: `artifacts/dl/db5_s1-10_win400_*`

## 已确认关键指标 (主结果)
- repetition accuracy ≈ 98.33%
- repetition macro-F1 ≈ 97.78%
- window accuracy ≈ 72.18%
- window macro-F1 ≈ 72.16%

## 注意事项
- 不允许用单次固定 split 代替 LORO 汇总结果。
