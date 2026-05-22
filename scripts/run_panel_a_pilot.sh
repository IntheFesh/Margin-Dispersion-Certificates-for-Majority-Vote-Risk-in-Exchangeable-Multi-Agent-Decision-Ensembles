#!/usr/bin/env bash
set -euo pipefail
python -m src.panel_a.run_pilot --config "${1:-configs/panel_a_pilot.yaml}" --output_dir "${2:-results/panel_a_pilot}"
