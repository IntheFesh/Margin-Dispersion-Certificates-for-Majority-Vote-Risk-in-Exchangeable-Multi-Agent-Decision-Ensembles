# Final Self-Check Report

- datetime_utc: 2026-05-22T17:15:03.701501Z
- git_hash: cee0ed0f7960bed80a4344c4d14f697ef329b38b

## 1) Executive Summary
- status: PASS WITH RESOURCE LIMITATIONS
- Ran compileall, pytest, synthetic checks, panel validate_only commands, and figure generation on synthetic outputs.
- Not run: real Panel A/PB/PC inference because required real model outputs/resources were not provided.

## 2) Repository Structure Check
| Path | Status |
|---|---|
| `configs` | PASS |
| `data` | FAIL |
| `results` | PASS |
| `scripts` | PASS |
| `src` | PASS |
| `tests` | PASS |
| `README.md` | PASS |
| `requirements.txt` | PASS |
| `pyproject.toml` | PASS |
| `src/theory/estimators.py` | PASS |
| `src/theory/majority_risk.py` | PASS |
| `src/theory/certificate.py` | PASS |
| `src/theory/insufficiency.py` | PASS |
| `src/synthetic/simulate_bernoulli_mixture.py` | PASS |
| `src/synthetic/run_synthetic_checks.py` | PASS |
| `src/panel_a/protocols.py` | PASS |
| `src/panel_a/run_pilot.py` | PASS |
| `src/panel_a/run_full.py` | PASS |
| `src/panel_a/analyze_panel_a.py` | PASS |
| `src/panel_b/run_eval.py` | PASS |
| `src/panel_b/pairwise_stats.py` | PASS |
| `src/panel_b/cka.py` | PASS |
| `src/panel_b/permutation_nulls.py` | PASS |
| `src/panel_b/analyze_alignment.py` | PASS |
| `src/panel_c/load_leaderboard_predictions.py` | PASS |
| `src/panel_c/analyze_leaderboard.py` | PASS |
| `src/figures` | PASS |
| `src/io` | PASS |
| `src/config` | PASS |

## 3) Theory Implementation Check
- Estimators: binary checks, K/M guards, unbiased F estimator and clipped field present.
- Majority risk: strict tie-as-failure and mixture integration over alpha values.
- Certificate: eta optimized numerically, refusal logic and outputs include issued/eta_star/R_cert.
- Two-moment insufficiency numbers reproduced by tests.

## 4) Test Results
- compileall: PASS
- pytest -q: PASS (20 passed)
- synthetic checks: PASS with outputs at `results/synthetic/`.

## 5) Panel A Status
- pilot validate_only: PASS
- full validate_only: PASS
- full real pilot run: NOT RUN (missing real model/resource outputs, no fake inference used).
- warning policy retained (no auto-tuning).

## 6) Panel B Status
- run_eval validate_only: PASS
- analyze_alignment validate_only: PASS
- leakage checks in full real run require real probe/eval IDs; not executable without resources.

## 7) Panel C Status
- analyze_leaderboard validate_only: PASS
- per-instance leaderboard prediction files not available in this environment, so full run not executed.

## 8) Fake Data / Fallback Audit
- No fake benchmark/model outputs were generated during this self-check.
- Synthetic outputs are explicitly synthetic and confined to synthetic pipeline outputs.
- Validate-only paths now report resource limitations instead of fabricating metrics.
- Inspected modules: theory, synthetic, panel_a, panel_b, panel_c, adapters, inference parsers.

## 9) Config and Provenance Audit
- Configs found: panel_a_pilot/full, panel_b, panel_c, synthetic_tests.
- `config_used.yaml` copied to each validate/run output directory via runtime config resolver.
- Seed override path verified via CLI commands (`--seed 42`).
- Git hash available and recorded above.

## 10) Known Remaining Limitations
- Real benchmark/model inference not run due to missing external resources and prediction artifacts.
- Panel B full CKA/permutation analysis and Panel C full leaderboard analysis require real input files.

## 11) Exact Commands Run
1. `python -m compileall src tests` -> exit 0
2. `pytest -q` -> exit 0 (20 passed)
3. `python -m src.synthetic.run_synthetic_checks --config configs/synthetic_tests.yaml --output_dir results/synthetic --seed 42` -> exit 0
4. `python -m src.panel_a.run_pilot --config configs/panel_a_pilot.yaml --output_dir results/panel_a_pilot --seed 42 --validate_only` -> exit 0
5. `python -m src.panel_a.run_full --config configs/panel_a_full.yaml --output_dir results/panel_a_full --seed 42 --validate_only` -> exit 0
6. `python -m src.panel_b.run_eval --config configs/panel_b.yaml --output_dir results/panel_b --seed 42 --validate_only` -> exit 0
7. `python -m src.panel_b.analyze_alignment --config configs/panel_b.yaml --output_dir results/panel_b --seed 42 --validate_only` -> exit 0
8. `python -m src.panel_c.analyze_leaderboard --config configs/panel_c.yaml --output_dir results/panel_c --seed 42 --validate_only` -> exit 0
9. `python -m src.figures.design_space --input_csv results/synthetic/estimator_check.csv --output_dir results/figures` -> exit 0
10. `python -m src.figures.certificate_vs_reference --input_csv results/synthetic/certificate_check.csv --output_dir results/figures` -> exit 0

## 12) Final Verdict
**PASS WITH RESOURCE LIMITATIONS**: codebase passes tests and required validate/synthetic checks; full real-resource inference experiments were not run because real external resources were unavailable.
