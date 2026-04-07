# NOTE: ASCII header to keep patch tool stable; content below is authoritative.
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

# Noise & Robustness (Fact Context)

## Core Facts
- Noise injection, denoise, and robustness analysis are foundation evidence.
- Clean vs noisy gaps must be explained in the thesis comparison section.

## Existing Noise Protocol
- noise types: wgn/pink (SNR=20,10,0,-5), hum/drift/motion/spikes (amp=0.1,0.3,0.6,1.0)
- denoise methods: none, notch, wavelet, kalman, pca

## Key Scripts/Paths
- noise injection: `run_gen_noisy.py`, `src/noise.py`
- robustness eval (ML): `run_robust_eval.py`
- DL noise eval: `run_dl_noise_eval.py`
- standardized summary: `outputs/summaries/dl_noise_results.csv`
- report results: `report/robust_db5_noise.csv`
- figures: `figures/robustness_heatmap.png`, `figures/Figure_3-Clean_vs_Noisy_Summary_RF.png`

## Confirmed Findings
- clean vs noisy performance gaps are key evidence for comparison analysis.
- DL noise results must be generated via `run_dl_noise_eval.py` and written to `outputs/summaries/dl_noise_results.csv`.

## Notes
- Do not archive noise/robustness experiments.
