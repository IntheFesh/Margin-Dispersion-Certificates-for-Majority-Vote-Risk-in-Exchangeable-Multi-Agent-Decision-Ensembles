# Margin-Dispersion Certificates for Majority-Vote Risk

Reference implementation accompanying the paper
"Margin-Dispersion Certificates for Majority-Vote Risk in Exchangeable
Multi-Agent Decision Ensembles".

## 1. Research goal

When does majority voting help, and when does it fail, in exchangeable
multi-agent decision ensembles?  We instantiate the question with
multi-agent LLM ensembles. The repository implements:

1. Core certificate theory and numerical routines.
2. Synthetic validation and unit tests.
3. Panel A: controlled exchangeable validation (pilot + full).
4. Panel B: heterogeneous LLM panel and CKA alignment (external grounding).
5. Panel C: large-scale supplementary correctness-correlation sanity check.
6. Figures and Markdown reports.
7. Strict reproducibility through YAML configs and saved outputs.

## 2. This is not MARL

This repository studies **non-interactive multi-agent decision
aggregation** (a.k.a. multi-agent LLM ensemble majority voting). It does
not implement multi-agent reinforcement learning. No agents act in an
environment and there is no policy training loop.

## 3. Panel B is external grounding, not theorem validation

Panel B correlates per-instance correctness across heterogeneous LLMs with
representational similarity (linear CKA on prompt embeddings). It is an
observational, non-causal check that the dispersion/dependence statistic
behaves the way the theory expects on a real model panel. Nothing in
Panel B validates Theorem 2.

## 4. No fake data / no silent fallback policy

- No fake benchmark data, fake model outputs, fake correctness vectors,
  fake CKA values, or fake leaderboard predictions are ever produced.
- `tests/` use small hand-crafted numerical examples for unit tests
  only; these never appear under `data/` or `results/`.
- Missing files, schema mismatches, model load failures, hidden-state
  extraction failures, CKA failures, dimension errors and device errors
  raise explicit errors. The only documented exception is the inner
  `optimize_eta` fallback from `scipy.minimize_scalar` to a dense grid,
  used to keep the certificate numerically robust.
- Parsing failures are recorded as experimental data with
  `invalid_parse=True`, `parsed_answer=None`, `correct=0`, and the raw
  output is preserved. They are never dropped.
- No automatic re-tuning of protocols / prompts / model lists /
  thresholds / `delta` / `K_ref` / `K_est` / `N_values` / benchmark
  subsets based on observed results.
- Negative results (`R_cert=1`, `m_L<=0`, certificate not issued, low
  non-vacuity rate, weak CKA alignment, high invalid_parse, unmet
  interpretive targets) are reported, not hidden.

## 5. Repository structure

```
configs/                          # YAML configs for every experiment
data/{raw,processed,cache}/       # raw/processed datasets and caches
results/
  synthetic/                      # synthetic check outputs
  panel_a_pilot/                  # Panel A pilot outputs
  panel_a_full/                   # Panel A full outputs
  panel_b/                        # Panel B outputs
  panel_c/                        # Panel C outputs
  figures/                        # generated PNG/PDF figures
  reports/                        # consolidated reports
scripts/                          # bash entrypoints
src/
  theory/                         # estimators, majority risk, certificate, insufficiency
  synthetic/                      # synthetic data + checks
  panel_a/                        # protocols, run_pilot, run_full, analysis
  panel_b/                        # eval, pairwise stats, CKA, perm nulls, alignment
  panel_c/                        # leaderboard supplementary
  data_adapters/                  # ARC-Challenge, GSM8K, MMLU, HellaSwag, leaderboard
  inference/                      # hf_generate, answer extractors, correctness
  figures/                        # matplotlib figure scripts (no seaborn)
  io/                             # paths, save_load, report
  config/                         # load_config
  tests/                          # pytest tests
```

## 6. Installation

Requires Python 3.10+.

```
pip install -r requirements.txt
```

Required libraries: numpy, pandas, scipy, scikit-learn, matplotlib,
pyyaml, tqdm, pytest, torch, transformers, datasets. Do not use seaborn.

## 7. Running tests

```
python -m compileall src
python -m pytest -q
```

All unit tests must pass before running real experiments.

## 8. Running the synthetic checks

```
bash scripts/run_synthetic_tests.sh configs/synthetic_tests.yaml results/synthetic
```

Outputs:

- `results/synthetic/estimator_check.csv`
- `results/synthetic/certificate_check.csv`
- `results/synthetic/nonvacuity_check.csv`
- `results/synthetic/insufficiency_check.csv`
- `results/synthetic/synthetic_summary.json`
- `results/synthetic/synthetic_report.md`

## 9. Running Panel A pilot

```
bash scripts/run_panel_a_pilot.sh configs/panel_a_pilot.yaml results/panel_a_pilot
```

The pilot requires real per-(instance, sample) correctness CSV at
`pilot_correctness_csv` in the config; no fake inference is generated.
Pass `--validate_only` to write a placeholder report when resources are
unavailable.

Pilot outputs:

- `results/panel_a_pilot/pilot_metrics.csv`
- `results/panel_a_pilot/pilot_summary.json`
- `results/panel_a_pilot/pilot_report.md`

If fewer than 20% of cells satisfy `R_cert<1` the pilot writes a
warning. It does not automatically modify any config.

## 10. Running Panel A full

```
bash scripts/run_panel_a_full.sh configs/panel_a_full.yaml results/panel_a_full
```

Outputs:

- `panel_a_cell_metrics.csv`
- `panel_a_bootstrap_metrics.csv`
- `panel_a_baseline_comparison.csv`
- `panel_a_nonvacuity_summary.csv`
- `panel_a_baseline_summary.csv`
- `panel_a_summary.json`
- `panel_a_report.md`

## 11. Running Panel B evaluation

```
bash scripts/run_panel_b_eval.sh configs/panel_b.yaml results/panel_b
```

The full path requires real per-(instance, model) prediction CSV at
`predictions_csv` in the config; greedy decoding (`temperature=0`).

CKA helpers (`src.panel_b.cka.compute_pairwise_cka`,
`src.panel_b.cka.cka_layer_sweep`) consume real probe-set
representations. The probe set must be disjoint from the
correctness-evaluation set; the loader checks this and raises
`ValueError` on overlap.

## 12. Running Panel B alignment analysis

```
bash scripts/run_panel_b_analysis.sh configs/panel_b.yaml results/panel_b
```

Outputs (when real inputs available):

- `panel_b_evaluation.csv`
- `panel_b_pairwise_stats.csv`
- `panel_b_alignment_summary.json`
- `panel_b_permutation_nulls.csv`
- `panel_b_report.md`

## 13. Running Panel C

```
bash scripts/run_panel_c.sh configs/panel_c.yaml results/panel_c
```

Panel C is appendix-only. It consumes per-instance leaderboard
predictions and computes within/cross-family correctness correlations
and the family gap. No CKA. No representation-alignment claims.

Outputs:

- `panel_c_pairwise_correctness.csv`
- `panel_c_family_summary.csv`
- `panel_c_summary.json`
- `panel_c_report.md`

## 14. Output locations

All outputs land under `results/`. Every major script copies the
resolved config to `results/<panel>/config_used.yaml` and every report
records seed, UTC datetime, and (when available) git commit hash.

## 15. Reproducibility notes

- Every script accepts `--config <yaml>`, `--output_dir <dir>`, and
  `--seed <int>`. CLI seed/output_dir override the YAML values.
- `--validate_only` runs a config validation pass and writes a
  resource-limited report instead of fabricating outputs.
- Random number generators use `numpy.random.default_rng(seed)`; child
  seeds are derived deterministically from the parent seed.
- Estimation and reference column subsets are disjoint and validated
  via `src.panel_a.split_reference_estimation.split_estimation_reference`.
- Panel B CKA probe set must be disjoint from correctness-evaluation
  set; `src.panel_b.cka.assert_probe_correctness_disjoint` enforces it.

## 16. Error handling policy

Any of the following raises an explicit error (no silent fallback):

- missing input files,
- schema mismatch (missing columns, NaN in required columns, `correct`
  outside `{0,1}`, duplicate `(instance_id, sample_id)`),
- model / tokenizer / hidden-state extraction failures,
- CKA inputs with degenerate feature variance,
- shape mismatches between observed and comparator pairwise matrices,
- estimation/reference or probe/correctness leakage,
- invalid `delta`, `M`, `K`, `N` values.

The documented numerical fallback for `optimize_eta` (a dense
2000-point grid when `scipy.optimize.minimize_scalar` fails) is the only
place a `try/except` wraps a primary computation; the resulting `eta`
is recorded in every certificate output.
