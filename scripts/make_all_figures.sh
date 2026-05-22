#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-results/figures}"
mkdir -p "$OUT_DIR"

# Figure inputs depend on which pipelines have produced outputs. The script
# below runs figures only where the necessary input CSV is present and exits
# cleanly otherwise. It does not generate placeholder figures from missing
# inputs.

run_if_exists() {
  local module="$1"
  local input_csv="$2"
  if [ -f "$input_csv" ]; then
    python -m "$module" --input_csv "$input_csv" --output_dir "$OUT_DIR"
  else
    echo "skip: $module (missing $input_csv)"
  fi
}

run_if_exists src.figures.design_space            results/synthetic/nonvacuity_check.csv
run_if_exists src.figures.certificate_vs_reference results/synthetic/certificate_check.csv
run_if_exists src.figures.nonvacuity              results/synthetic/certificate_check.csv
run_if_exists src.figures.design_space            results/panel_a_pilot/pilot_metrics.csv
run_if_exists src.figures.certificate_vs_reference results/panel_a_pilot/pilot_metrics.csv
run_if_exists src.figures.nonvacuity              results/panel_a_pilot/pilot_metrics.csv
run_if_exists src.figures.baseline_comparison     results/panel_a_full/panel_a_baseline_comparison.csv
run_if_exists src.figures.cka_alignment           results/panel_b/panel_b_pairwise_stats.csv
