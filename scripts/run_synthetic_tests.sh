#!/usr/bin/env bash
set -euo pipefail
CONFIG_PATH="${1:-configs/synthetic_tests.yaml}"
OUT_DIR="${2:-results}"
python -c "print('Use Python entrypoints with config:', 'run_synthetic_tests', '$CONFIG_PATH', '$OUT_DIR')"
