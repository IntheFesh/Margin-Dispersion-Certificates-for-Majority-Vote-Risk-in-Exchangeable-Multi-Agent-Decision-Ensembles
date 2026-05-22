#!/usr/bin/env bash
set -euo pipefail
CONFIG_PATH="${1:-configs/panel_c.yaml}"
OUT_DIR="${2:-results/panel_c}"
python -m src.panel_c.analyze_leaderboard --config "$CONFIG_PATH" --output_dir "$OUT_DIR"
