# NOTE: ASCII header to keep patch tool stable; content below is authoritative.
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

# Dataset (事实型上下文)

## 核心事实
- 使用 NinaPro DB5 数据集，当前具备 S1-S10 原始数据。
- 深度学习使用滑窗数据集，窗口数据包含 `subject_id`, `gesture_id`, `repetition_id` 等元信息。
- 主结果使用 S1-S3，cross-subject 补充实验使用 S1-S10。

## 关键脚本/路径
- 原始数据: `data/ninapro_db5/raw/`, `data/ninapro_db5/zip/`
- 窗口构建: `db5_windows.py`
- 窗口数据: `data/ninapro_db5/windows/db5_s1-3_g1-10_win400_step50.npz`
- 跨被试窗口数据: `data/ninapro_db5/windows/db5_s1-10_g1-10_win400_step50.npz`

## 已确认结论
- 不允许随机拆分窗口，必须按 repetition 分组。
- 归一化仅在训练 split 上 fit。

## 注意事项
- LORO 主结果依赖 repetition-level 聚合，不能用固定 split 代替。
