# Reproduction guide

Step-by-step commands to reproduce Panel A end to end, with the expected
outputs of each step. All commands are run from the repository root with
`PYTHONPATH=.` so `src` is importable. Phases are labelled **(no GPU)** or
**(GPU)**; the no-GPU phases need no network beyond the dataset download that
the GPU phases trigger.

Every orchestration script accepts `--config` and `--output_dir`, copies the
config it used to `<output_dir>/config_used.yaml`, captures provenance (git
commit, code version, seed, hardware, vLLM/torch versions, UTC timestamp), and
appends structured events to `outputs/logs/<script>.jsonl`.

## 0. Install (no GPU for everything except the inference engine)

```bash
python -m pip install -r requirements.txt
# GPU host only, for the generation phases:
python -m pip install "vllm>=0.6"
```

Requires Python 3.10+. Plotting is matplotlib-only; figures are written as both
300-DPI PNG and PDF.

## 1. Unit tests (no GPU)

```bash
PYTHONPATH=. pytest tests
```

Expected: all tests pass. They cover the closed-form certificates
(`test_theorem1`, `test_refinement1`, `test_theorem3`), the empirical
certificate and Bonferroni budget (`test_empirical`, `test_bonferroni`), the
hierarchy and Proposition 1 (`test_hierarchy`, `test_proposition1`), the
unbiased dispersion estimator (`test_unbiased_F`), answer extraction/labelling
(`test_extraction`, `test_labeling`), and pool disjointness
(`test_pools_disjoint`).

## 2. Phase 0 — numerical proof verification (no GPU)

```bash
PYTHONPATH=. python scripts/00_verify_proofs.py --output_dir outputs/logs
```

Expected: prints four gaps and writes `outputs/logs/phase_0.jsonl`. The
hierarchy gap is `< 1e-6`, the Theorem 3 primal-dual gap `< 1e-8` on the coarse
proof grid, the Refinement 1 closed-form-vs-LP gap `< 1e-4`, and the `F_hat`
bias `|.| < 1e-3`. Any hierarchy or primal-dual violation raises and aborts.

## 3. Phase 1.2 — design-space pre-check / go-no-go (no GPU)

```bash
PYTHONPATH=. python scripts/01_design_space_precheck.py \
  --config configs/design_grid.yaml \
  --output_dir outputs/precheck \
  --figures_dir outputs/figures
```

Sweeps the `(alpha_bar, F_relative, N, M)` grid, computes `B_N^CH`, `B_N^CH'`,
`B_N^star` (+dual) and the empirical-certificate informativeness proxy, runs the
hierarchy and primal-dual bug detectors, and emits:

- `outputs/precheck/informativeness.csv` — one row per grid point;
- `outputs/precheck/go_no_go.json` — the `GO`/`NO-GO` decision plus the max
  hierarchy / primal-dual gaps and the tolerances used;
- `outputs/figures/design_space.{png,pdf}` — the certificate over the
  (mean-margin, dispersion) plane for reference sizes `N = 15` and `N = 63`.

Expected: `decision: GO` (the design admits informative, issuable cells at the
largest `M`). The primal-dual detector uses `--primal_dual_tol 1e-6` by default
(the dense LP has a `~1e-7` floating-point floor; a real bug would be far
larger). This script is fully runnable with no GPU/network.

## 4. Phase 1.1 — engineering pilot, M=50 smoke test (GPU)

```bash
PYTHONPATH=. python scripts/02_engineering_pilot.py \
  --config configs/panel_a_protocol_a1.yaml \
  --benchmark arc_challenge \
  --output_dir outputs/pilots
```

Requires a live vLLM server for the model under test (see `configs/models.yaml`
for ports). Exercises the full pipeline on a 50-instance slice and writes
`outputs/pilots/engineering_pilot_summary.json` (tagged `"smoke_test": true`,
`"reportable": false`) plus the per-pool JSONL records.

**Expected and important**: certificate values from `M < 200` are plumbing
diagnostics only and are **never** reported as findings. This step exists to
catch wiring bugs.

## 5. Phase 1.1 — statistical pilot, M=500 single cell (GPU)

```bash
PYTHONPATH=. python scripts/03_statistical_pilot.py \
  --config configs/panel_a_protocol_a1.yaml \
  --benchmark arc_challenge \
  --output_dir outputs/pilots
```

Runs one `(protocol=A1, benchmark, model)` cell at the reporting scale `M=500`,
issues the certificate over the odd `N` grid, and runs the Analysis-1
instance-bootstrap of coverage. Writes
`outputs/pilots/statistical_pilot_summary.json` (`"reportable": true`).

Expected: this is the **smallest scale at which certificate numbers are
reported**. The per-`N` rows carry `R_N^cert`, `Q_N^cert`, the refusal mode, and
the bootstrap coverage rate vs the pre-registered `0.95` level.

## 6. Phase 1.3 — full Panel A (GPU)

```bash
PYTHONPATH=. python scripts/04_panel_a_full.py \
  --config    configs/panel_a_protocol_a1.yaml \
  --config_a2 configs/panel_a_protocol_a2.yaml \
  --output_dir outputs/panel_a
```

Runs Protocol A1 over `benchmarks × models` and Protocol A2 (serial model-swap)
over its benchmarks, then issues per-cell certificates over the odd `N` grid.
The number of `(protocol, benchmark, K_est)` cells `C` is computed at run time;
both `C` and the resulting `delta_cell = delta_global / C` are written to
`outputs/panel_a/panel_a_summary.json` together with the per-cell certificate
rows and provenance. Per-cell raw records land in
`outputs/panel_a/<protocol>_<benchmark>/{estimation,oracle}.jsonl`.

Expected: `panel_a_summary.json` reports `C` (with the A1 config's three
benchmarks × three models plus the A2 config's two benchmarks, `C = 11`) and the
matching `delta_cell`.

## 7. Phase 2 — analyses, figures, tables (no GPU)

```bash
PYTHONPATH=. python scripts/05_run_analyses.py \
  --panel_a_dir outputs/panel_a --output_dir outputs/analyses
PYTHONPATH=. python scripts/06_render_figures.py \
  --analyses_dir outputs/analyses --panel_a_dir outputs/panel_a --output_dir outputs/figures
PYTHONPATH=. python scripts/07_build_tables.py \
  --analyses_dir outputs/analyses --output_dir outputs/analyses/tables
```

- **`05_run_analyses.py`** runs Analyses 1-7 over the Panel A outputs and writes
  `outputs/analyses/analysis_1_bootstrap.csv` … `analysis_7_conservativeness.csv`.
- **`06_render_figures.py`** renders, into `outputs/figures/` (each PNG + PDF):
  the `design_space` figure (with empirical operating points overlaid from the
  Panel A cells), the **Analysis 7 principal** four-component conservativeness
  `analysis7_stacked_bar`, and the `budget_curves` (`R_N^cert` vs `N` per cell).
- **`07_build_tables.py`** writes paired LaTeX (`.tex`) and markdown (`.md`)
  tables under `outputs/analyses/tables/`.

Expected: all three steps are pure post-processing and need no GPU/network. A
**missing required input CSV raises `FileNotFoundError`** — no figure or table
is ever produced from fabricated or empty data.

## Error-handling and provenance contract

- Missing input files raise `FileNotFoundError`; there are no silent fallbacks.
- Numerical hierarchy (`B_N^star <= B_N^CH' <= B_N^CH`) or primal-dual
  violations abort the pipeline.
- All randomness is seeded deterministically from the global seed via
  `src.utils.seeds`; runs are reproducible across machines.
- Each script copies its config to `<output_dir>/config_used.yaml` and records
  provenance, so every output directory is self-describing.
