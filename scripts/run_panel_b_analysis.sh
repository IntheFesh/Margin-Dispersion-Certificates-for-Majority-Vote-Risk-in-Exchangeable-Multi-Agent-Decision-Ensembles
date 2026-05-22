#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
print('Use src.panel_b.analyze_alignment.run(...) with generated CSV paths')
PY
