# -*- coding: utf-8 -*-
# src/pipeline.py

import os
import pandas as pd

from .filters import preprocess_emg
from .segmentation import segment_by_energy, get_segmentation_params
from .io_utils import (
    ensure_dirs,
    mat_to_csv_if_needed,
    plot_before_after,
    plot_segments,
    save_segments_json,
)


def run_full_pipeline(
    fs: float = 500.0,
    mode: str = "bandpass",
    bp_low: float = 20.0,
    bp_high: float = 200.0,
    hp_fc: float = 15.9,
    order: int = 4,
    use_notch: bool = False,
    raw_dir: str = "data/raw",
    intermediate_dir: str = "data/intermediate",
    figures_dir: str = "figures",
    report_dir: str = "report",
    ylim=None,
):
    """
    对 6 个手势数据依次执行：
      - mat/csv 读取
      - EMG 预处理
      - 分段
      - 画图（滤波前后 + 分段）
      - 汇总 csv + markdown 片段
    """

    ensure_dirs(figures_dir, report_dir, intermediate_dir)

    summary_rows = []

    for i in range(1, 7):
        # -----------------------
        # 1. 读取数据 / mat -> csv
        # -----------------------
        stem = f"gesture{i}_1"
        csv_path = mat_to_csv_if_needed(stem, raw_dir=raw_dir, intermediate_dir=intermediate_dir)
        raw_csv = csv_path  # 对比图用

        df = pd.read_csv(csv_path)
        emg_cols = [c for c in df.columns if c.lower().startswith("emg")]
        X = df[emg_cols].values

        # -----------------------
        # 2. 滤波预处理
        # -----------------------
        Xf = preprocess_emg(
            X,
            fs=fs,
            mode=mode,
            bp_low=bp_low,
            bp_high=bp_high,
            hp_fc=hp_fc,
            order=order,
            use_notch=use_notch,
        )

        # 保存滤波后的 csv
        clean_csv_name = f"cleaned_g{i}.csv"
        clean_csv_path = os.path.join(intermediate_dir, clean_csv_name)
        out_df = df.copy()
        out_df[emg_cols] = Xf
        out_df.to_csv(clean_csv_path, index=False)

        # 滤波前后图
        before_after_png = os.path.join(figures_dir, f"g{i}_before_after.png")
        plot_before_after(
            raw_csv,
            clean_csv_path,
            before_after_png,
            ylim=tuple(ylim) if ylim else None,
        )

        # -----------------------
        # 3. 分段
        # -----------------------
        A, B, N = get_segmentation_params(i)
        segs = segment_by_energy(Xf, A=A, B=B, N=N)

        seg_json_name = f"g{i}_segments.json"
        seg_json_path = os.path.join(intermediate_dir, seg_json_name)
        save_segments_json(segs, seg_json_path)

        seg_png = os.path.join(figures_dir, f"g{i}_segments.png")
        plot_segments(clean_csv_path, segs, seg_png)

        # -----------------------
        # 4. 汇总信息
        # -----------------------
        total_samples = len(df)
        nseg = len(segs)
        dur_s = sum((e - s) / fs for s, e in segs)

        summary_rows.append(
            {
                "gesture": f"g{i}",
                "samples": total_samples,
                "segments_detected": nseg,
                "total_active_seconds": round(dur_s, 3),
                "params_A": A,
                "params_B": B,
                "params_N": N,
            }
        )

    # ---------------------------
    # 5. 写出 summary csv + md
    # ---------------------------
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(report_dir, "segment_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    md_lines = [
        "### 阶段实验分割结果汇总",
        "",
        "| 手势 | 样本点数 | 活动段数 | 活动总时长(s) | A | B | N |",
        "|:---:|---:|---:|---:|:--:|:--:|:--:|",
    ]
    for r in summary_rows:
        md_lines.append(
            f"| {r['gesture']} | {r['samples']} | {r['segments_detected']} | "
            f"{r['total_active_seconds']} | {r['params_A']} | {r['params_B']} | {r['params_N']} |"
        )

    md_path = os.path.join(report_dir, "分割结果_可粘贴到报告.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return {
        "summary_csv": summary_csv_path,
        "summary_md": md_path,
        "figures_dir": figures_dir,
        "intermediate_dir": intermediate_dir,
    }
