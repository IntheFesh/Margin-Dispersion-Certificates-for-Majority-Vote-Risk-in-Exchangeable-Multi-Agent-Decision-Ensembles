#!/usr/bin/env bash
set -euo pipefail
CONFIG_PATH="${1:-configs/panel_b.yaml}"
OUT_DIR="${2:-results/panel_b}"
python -m src.panel_b.analyze_alignment --config "$CONFIG_PATH" --output_dir "$OUT_DIR"
