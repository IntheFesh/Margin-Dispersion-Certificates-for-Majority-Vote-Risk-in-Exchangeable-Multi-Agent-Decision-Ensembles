#!/usr/bin/env bash
set -euo pipefail
# Panel B CKA requires HF model resources; running this script with no real
# probe representations available will raise an explicit error rather than
# fabricate a fallback. To compute CKA programmatically use
# src.panel_b.cka.compute_pairwise_cka / cka_layer_sweep.
CONFIG_PATH="${1:-configs/panel_b.yaml}"
OUT_DIR="${2:-results/panel_b}"
python -c "from src.panel_b.cka import compute_pairwise_cka, cka_layer_sweep; print('CKA helpers loaded; pass real representations to compute_pairwise_cka / cka_layer_sweep with config $CONFIG_PATH, output dir $OUT_DIR')"
