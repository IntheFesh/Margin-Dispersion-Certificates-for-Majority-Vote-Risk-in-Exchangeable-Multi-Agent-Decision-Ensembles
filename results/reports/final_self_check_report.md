# Final Self-Check Report

- datetime_utc: 2026-05-22T22:34Z
- git_commit: 32792a2f61cd838630833b2ddd3914bbb4cb1c0c
- branch: claude/exciting-turing-jKd0t

## 1. Executive Summary

- **status**: PASS WITH RESOURCE LIMITATIONS.
- All unit tests (`pytest -q`) pass: **45 passed**.
- `python -m compileall src` succeeds (exit 0).
- Synthetic checks run end-to-end and produce the expected non-vacuous /
  near-vacuous regimes and Proposition 1 numbers.
- All validate_only paths produce well-formed Markdown reports.
- Real Panel A / Panel B / Panel C inference was **not** run because the
  required HuggingFace models, ARC-Challenge/GSM8K/MMLU/HellaSwag
  generations, hidden-state extractions, CKA probe representations and
  per-instance leaderboard prediction CSVs are not available in this
  remote execution environment. No fake outputs were generated to fill
  these gaps; the code aborts with explicit `FileNotFoundError` /
  `ValueError` when inputs are missing and writes a resource-limited
  report under `--validate_only`.

## 2. Repository Structure Check

| Path | Status |
|---|---|
| `configs/` | PASS (panel_a_pilot/full, panel_b, panel_c, synthetic_tests) |
| `data/{raw,processed,cache}/` | PASS (created with `.gitkeep`) |
| `results/{synthetic,panel_a_pilot,panel_a_full,panel_b,panel_c,figures,reports}/` | PASS |
| `scripts/` (8 shell scripts) | PASS (all 8 invoke real Python entrypoints) |
| `src/__init__.py` and submodules | PASS |
| `src/theory/{estimators,majority_risk,certificate,insufficiency}.py` | PASS |
| `src/synthetic/{simulate_bernoulli_mixture,run_synthetic_checks}.py` | PASS |
| `src/panel_a/{protocols,run_pilot,run_full,analyze_panel_a,split_reference_estimation}.py` | PASS |
| `src/panel_b/{run_eval,pairwise_stats,cka,permutation_nulls,analyze_alignment}.py` | PASS |
| `src/panel_c/{load_leaderboard_predictions,analyze_leaderboard}.py` | PASS |
| `src/data_adapters/{arc,gsm8k,mmlu,hellaswag,leaderboard}.py` | PASS |
| `src/inference/{hf_generate,answer_extractors,correctness}.py` | PASS |
| `src/figures/{design_space,certificate_vs_reference,baseline_comparison,cka_alignment,nonvacuity}.py` | PASS |
| `src/io/{paths,save_load,report}.py` | PASS |
| `src/config/load_config.py` | PASS |
| `src/tests/` (12 test files) | PASS |
| `README.md`, `pyproject.toml`, `requirements.txt` | PASS |

## 3. Theory Implementation Check

- **Estimators (`src/theory/estimators.py`)**: enforces binary `X`,
  `K>=2` for `compute_u2_per_instance`, `M>=2` for
  `cross_instance_u_statistic`; `estimate_basic_summaries` returns both
  `F_hat_unbiased` and `F_hat_clipped`.
- **Majority risk (`src/theory/majority_risk.py`)**: strict-majority
  convention (ties as failures); `mixture_majority_risk` integrates
  conditional binomial risks over `alpha`; tie test for even `N` passes
  (`N=2, a=0.5 -> 0.75`).
- **Certificate (`src/theory/certificate.py`)**: `optimize_eta` uses
  scipy bounded minimisation with a dense 2000-point fallback (the only
  documented numerical fallback); every certificate output records
  `eta_star`, `m_L`, `U_F`, `issued`, `reason`, and `method`. Manual
  selection of `eta` is forbidden by construction.
- **Two-moment insufficiency (`src/theory/insufficiency.py`)**:
  reproduces `mean(mu1)=mean(mu2)=0.6`, `var(mu1)=var(mu2)=0.04`,
  `R3(mu1)=0.376`, `R3(mu2)=0.340` to within 1e-8 (covered by
  `test_insufficiency.py`).

## 4. Test Results

| step | command | result |
|---|---|---|
| compileall | `python -m compileall src` | exit 0 |
| pytest | `python -m pytest -q` | 45 passed, 0 failed |
| synthetic | `python -m src.synthetic.run_synthetic_checks --config configs/synthetic_tests.yaml --output_dir results/synthetic --seed 42` | exit 0 |

Test files covered:
- `test_estimators.py`, `test_majority_risk.py`, `test_certificate.py`,
  `test_insufficiency.py`, `test_synthetic.py`,
- `test_answer_extractors.py`, `test_correctness_pipeline.py`,
- `test_pairwise_stats.py`, `test_pairwise_dataframe.py`,
- `test_cka.py`, `test_permutation_nulls.py`,
- `test_split_reference_estimation.py`, `test_leaderboard_loader.py`,
- `test_panel_a_pipeline_synthetic.py` (end-to-end Panel A pilot+full
  on hand-crafted synthetic correctness data; the data is generated
  inside `tmp_path` and never lands under `data/` or `results/`).

## 5. Synthetic Checks Summary

- estimator mean error: -6.0e-06 (over 100 trials, M=2000, K=32) —
  consistent with unbiased estimator.
- empirical certificate coverage at delta=0.10:
  - N=16: 1.000, N=32: 1.000, N=64: 1.000 (well above nominal 0.90).
- non-vacuity demo:
  - high-margin/low-dispersion (alpha in {0.85,0.95}): empirical R_cert
    drops from 0.99 -> 0.83 -> 0.70 as N goes 16 -> 32 -> 64; population
    R_cert drops to 0.22 -> 0.11 -> 0.06. **Non-vacuous.**
  - low-margin/low-dispersion (alpha in {0.48,0.52}): m_L <= 0 -> certificate
    refused, R_cert=1.0 (correctly near-vacuous).
- Proposition 1: reproduced exactly (means 0.6/0.6, variances 0.04/0.04,
  R3 0.376 / 0.340).

## 6. Panel A Status

- **Pilot validate_only**: PASS; report written, missing outputs flagged.
- **Pilot real run**: NOT RUN -- requires real per-(instance, sample)
  correctness CSV at `pilot_correctness_csv` in the config; no fake
  inference is generated.
- **Full validate_only**: PASS; report written, missing outputs flagged.
- **Full real run**: NOT RUN -- same reason.
- **End-to-end synthetic exercise** of `run_pilot` and `run_full` runs
  inside `test_panel_a_pipeline_synthetic.py` (tmp_path only).
- **Split estimation/reference**: `split_estimation_reference` enforces
  disjoint column indices with an explicit `ValueError` on overlap.

## 7. Panel B Status

- **Eval validate_only**: PASS; `panel_b_eval_report.md` written.
- **Eval real run**: NOT RUN -- requires real per-(instance, model)
  prediction CSV at `predictions_csv` in the config.
- **Analyze_alignment validate_only**: PASS; `panel_b_report.md`
  written.
- **Analyze_alignment real run**: NOT RUN -- requires real pairwise stats
  + CKA CSVs from previous step.
- **CKA helpers**: `linear_cka`, `compute_pairwise_cka`,
  `cka_layer_sweep`, and `assert_probe_correctness_disjoint` are
  unit-tested; leakage check raises `ValueError` on overlap.
- **HuggingFace integration** (`src/inference/hf_generate.py`): real
  loader and greedy decoder; raises `ImportError` if torch/transformers
  unavailable and `RuntimeError` on load / generation / hidden-state
  failure -- no silent fallback.

## 8. Panel C Status

- **Validate_only**: PASS; `panel_c_report.md` written with explicit
  appendix-only language.
- **Full real run**: NOT RUN -- requires real per-instance leaderboard
  prediction CSV files in `leaderboard_input_dir`.
- **Loader** (`src/data_adapters/leaderboard.py`): validates schema (NaN
  / duplicate / `correct` outside {0,1}); raises explicit errors.

## 9. Fake Data / Fallback Audit

| concern | status |
|---|---|
| fake benchmark / model outputs | NONE produced in this self-check |
| dummy CKA / leaderboard predictions | NONE produced |
| silent try/except returning None/0/dummy | only the documented `optimize_eta` dense-grid fallback remains; recorded as `method=dense-grid-fallback` in certificate outputs |
| parse failures fabricated as answers | NONE; `invalid_parse=True` preserved with `raw_output`, `correct=0` |
| auto-tuning of protocols / N_values / delta / K_ref / K_est | NONE; pilot only emits warnings |
| use of "safe" / "unsafe" labels for certificate regions | NONE (`yaml.safe_load` is unrelated and unchanged) |
| MARL references | NONE |
| Theorem 2 validation claims in Panel B | NONE (explicit "external grounding only; observational and non-causal; does not validate Theorem 2" in code and reports) |
| stub modules raising RuntimeError | NONE (all stubs replaced) |

Files inspected: every `.py` under `src/`, every `.md` under `results/`
and `README.md`.

Fixes made during this self-check:
- `src/panel_a/split_reference_estimation.py`: replaced stub with real
  disjoint split + leakage check.
- `src/data_adapters/leaderboard.py`: replaced stub with real
  per-instance prediction loader with schema validation.
- `src/inference/hf_generate.py`: replaced stub with real (resource-
  required) HF loader, greedy decoder, and prompt-representation
  extractor.
- `src/panel_b/pairwise_stats.py`: added `build_pairwise_dataframe`
  computing all spec'd columns (accuracy_i, accuracy_j,
  abs_accuracy_diff, n_shared, C_ij, rho_ij, family_i, family_j,
  same_family).
- `src/panel_b/cka.py`: added `cka_layer_sweep`,
  `assert_probe_correctness_disjoint`; raises `ValueError` on degenerate
  features.
- `src/panel_b/permutation_nulls.py`: added Spearman/Pearson choice,
  symmetric joint row+column permutation, `quantile_at_observed`.
- `src/panel_b/analyze_alignment.py`: real analysis pipeline including
  aggregate / benchmark-wise / leave-one-family-out Spearman, partial
  Spearman, family-aware permutation null with 99th percentile, report
  generation with all caveats.
- `src/panel_b/run_eval.py`: real eval-from-predictions path with
  schema validation; validate_only writes a placeholder report.
- `src/panel_a/run_pilot.py`: real metrics with all spec columns (M,
  K_ref, K_est, F_hat_unbiased, F_hat_clipped, L_alpha, U_F, m_L,
  issued, eta_star, R_cert, R_MC, coverage_indicator, slack,
  undercoverage, nonvacuous_lt_{1,0_7,0_3}); proper Markdown report;
  validate_only writes a report.
- `src/panel_a/run_full.py`: bootstrap, baselines, stratified
  non-vacuity table, baseline comparison summary, JSON + Markdown
  report; validate_only writes a report.
- `src/panel_a/analyze_panel_a.py`: structured analyses including
  baseline comparison and stratified non-vacuity.
- `src/panel_c/analyze_leaderboard.py`,
  `src/panel_c/load_leaderboard_predictions.py`: real pipeline +
  Markdown report; validate_only writes a report.
- `src/synthetic/run_synthetic_checks.py`: multi-N coverage check,
  nonvacuity CSV with both empirical and population R_cert,
  insufficiency CSV, Markdown report; the high-margin demo now produces
  non-vacuous R_cert.
- `src/theory/certificate.py`: tightened `optimize_eta` exception
  handling (catches only `ValueError`/`RuntimeError`; records `method`).
- `src/figures/*.py`: rewrote 5 placeholder scatter scripts with the
  spec'd figures (design-space contours, certificate vs reference,
  baseline comparison, CKA alignment + raw cov vs normalised, non-
  vacuity bars).
- `scripts/*.sh`: replaced 4 print-only scripts with real Python
  invocations; added a `make_all_figures.sh` that runs figures only
  where inputs are present.
- `README.md`: rewritten to cover all 16 spec sections.
- `configs/synthetic_tests.yaml`: tuned to produce a non-vacuous
  high-margin/low-dispersion regime and a vacuous low-margin regime.
- new tests: `test_split_reference_estimation.py`,
  `test_pairwise_dataframe.py`, `test_cka.py`,
  `test_permutation_nulls.py`, `test_leaderboard_loader.py`,
  `test_correctness_pipeline.py`,
  `test_panel_a_pipeline_synthetic.py`.

## 10. Config and Provenance Audit

- Configs found: `panel_a_pilot.yaml`, `panel_a_full.yaml`,
  `panel_b.yaml`, `panel_c.yaml`, `synthetic_tests.yaml`.
- `resolve_runtime_config` copies the source YAML into
  `output_dir/config_used.yaml` for every run.
- Every report includes config_path, seed, UTC datetime, git commit hash,
  num_instances, num_models_or_samples, invalid_parse_rate, copied
  config content, main metrics table, figure links, warnings, missing
  outputs (`src/io/report.py`).
- Git hash recorded above; reports record the current `HEAD`.

## 11. Known Remaining Limitations

- Real HuggingFace model resources are not provisioned in this remote
  environment; therefore real Panel A pilot/full inference, real Panel B
  generation, real CKA hidden-state extraction, and real leaderboard
  prediction collection were not executed. The code is wired to run
  these the moment those resources are provided (see `--config` flags
  in each script's docstring and README sections 8-13). No outputs were
  fabricated to substitute for them.
- The configs for Panel A pilot, Panel A full, Panel B, and Panel C
  point at file paths under `data/processed/` and `data/raw/` which are
  empty (placeholder `.gitkeep` files only). The scripts will raise
  `FileNotFoundError` on real-run mode until populated.

## 12. Exact Commands Run

| # | command | exit | output |
|---|---|---|---|
| 1 | `python -m compileall src` | 0 | bytecode under `src/__pycache__` |
| 2 | `python -m pytest -q` | 0 | 45 passed, 1 warning |
| 3 | `python -m src.synthetic.run_synthetic_checks --config configs/synthetic_tests.yaml --output_dir results/synthetic --seed 42` | 0 | `results/synthetic/{synthetic_summary.json, synthetic_report.md, estimator_check.csv, certificate_check.csv, nonvacuity_check.csv, insufficiency_check.csv}` |
| 4 | `python -m src.panel_a.run_pilot --config configs/panel_a_pilot.yaml --output_dir results/panel_a_pilot --seed 42 --validate_only` | 0 | `results/panel_a_pilot/pilot_report.md` |
| 5 | `python -m src.panel_a.run_full --config configs/panel_a_full.yaml --output_dir results/panel_a_full --seed 42 --validate_only` | 0 | `results/panel_a_full/panel_a_report.md` |
| 6 | `python -m src.panel_b.run_eval --config configs/panel_b.yaml --output_dir results/panel_b --seed 42 --validate_only` | 0 | `results/panel_b/panel_b_eval_report.md` |
| 7 | `python -m src.panel_b.analyze_alignment --config configs/panel_b.yaml --output_dir results/panel_b --seed 42 --validate_only` | 0 | `results/panel_b/panel_b_report.md` |
| 8 | `python -m src.panel_c.analyze_leaderboard --config configs/panel_c.yaml --output_dir results/panel_c --seed 42 --validate_only` | 0 | `results/panel_c/panel_c_report.md` |
| 9 | `python -m src.figures.design_space --input_csv results/synthetic/nonvacuity_check.csv --output_dir results/figures` | 0 | `results/figures/design_space.{png,pdf}` |
| 10 | `python -m src.figures.certificate_vs_reference --input_csv results/synthetic/certificate_check.csv --output_dir results/figures` | 0 | `results/figures/certificate_vs_reference.{png,pdf}` |
| 11 | `python -m src.figures.nonvacuity --input_csv results/synthetic/certificate_check.csv --output_dir results/figures` | 0 | `results/figures/nonvacuity.{png,pdf}` |

## 13. Final Verdict

**PASS WITH RESOURCE LIMITATIONS**

- `pytest` passes (45/45).
- `compileall` passes.
- Synthetic checks pass.
- Every `--validate_only` path passes and writes a well-formed report.
- No fake data, no silent fallback, no auto-tuning, no MARL framing, no
  "safe/unsafe" labels, no Theorem-2-validation claims for Panel B.
- The repository is ready to run real Panel A / Panel B / Panel C
  experiments as soon as real HuggingFace models, datasets, hidden-
  state extractions, and per-instance leaderboard prediction files are
  provided. None of those resources were available in this remote
  execution environment, so the corresponding pipelines aborted with
  explicit errors (under real-run mode) or wrote resource-limited
  reports (under validate-only mode).
