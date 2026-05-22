#!/usr/bin/env bash
set -euo pipefail
python -m src.synthetic.run_synthetic_checks --config "${1:-configs/synthetic_tests.yaml}" --output_dir "${2:-results/synthetic}"
