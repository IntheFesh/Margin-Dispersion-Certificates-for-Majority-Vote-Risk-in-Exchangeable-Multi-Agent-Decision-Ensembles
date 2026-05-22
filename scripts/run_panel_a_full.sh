#!/usr/bin/env bash
set -euo pipefail
python -m src.panel_a.run_full --config "${1:-configs/panel_a_full.yaml}" --output_dir "${2:-results/panel_a_full}"
