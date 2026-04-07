# NOTE: ASCII header to keep patch tool stable; content below is authoritative.
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

# Calibration (事实型上下文)

## 核心事实
- calibration 旨在对目标被试进行小样本微调。
- 结论为“有条件有效”：subject 10 上有效，subject 9 收益有限。
- calibration 属于补充结论层，不替代主结果。

## 关键脚本/路径
- 数据构建: `build_calibration_dataset.py`
- 微调与评估: `calibrate_dl.py`
- 支持函数: `calibration_utils.py`, `calibration_policy.py`
- 数据路径: `data/ninapro_db5/calibration/*.json`

## 注意事项
- 不应将 calibration 结果作为主指标结论。
