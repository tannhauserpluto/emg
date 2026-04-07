# NOTE: ASCII header to keep patch tool stable; content below is authoritative.
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

# Preprocessing & Segmentation (事实型上下文)

## 核心事实
- 课题要求包含噪声分析、滤波降噪与活动段分割。
- 这些内容属于论文 foundation 层，不可误归档。

## 关键脚本/路径
- 滤波/降噪: `src/filters.py`, `src/denoise.py`
- 噪声建模: `src/noise.py`
- 分割/活动段: `src/segmentation.py`, `src/pipeline.py`
- 分割可视化: `generate_segmentation_detail.py`, `presentation_segmentation_detail.png`
- 分割统计: `report/segment_summary.csv`

## 已确认结论
- segmentation 与 preprocessing 作为传统 ML 与 DL 的共同基础。

## 注意事项
- 论文中应明确 preprocessing/segmentation 的作用与对后续识别的影响。
