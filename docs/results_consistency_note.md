# NOTE: ASCII header to keep patch tool stable; content below is authoritative.
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

# Results Consistency Note

## Overall computation rule
- Overall LORO metrics are recomputed by concatenating all fold-level predictions
  from `*loro*_eval/window_predictions.csv` and `*loro*_eval/repetition_predictions.csv`.
- Accuracy is computed from total correct / total samples.
- Macro-F1 is computed from aggregated y_true/y_pred across all folds.

## Overall vs fold-level
- Fold-level rows provide per subject / test repetition metrics.
- Overall row is an aggregate of all folds and is the only source for main results.

## Authoritative summary files
- `outputs/summaries/main_results.csv`
- `outputs/summaries/foundation_results.csv`
- `outputs/summaries/supplementary_results.csv`
- `outputs/summaries/error_analysis.csv`
