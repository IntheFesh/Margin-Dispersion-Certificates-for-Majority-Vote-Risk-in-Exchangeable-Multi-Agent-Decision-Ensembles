# Margin-Dispersion Certificates for Majority-Vote Risk

## Research goal
Study when majority voting helps or fails in exchangeable multi-agent decision ensembles.

## Not MARL
This repository studies non-interactive multi-agent decision aggregation and **does not implement MARL**.

## Panel B caveat
Panel B is external grounding for dependence/dispersion signals; it is observational and does not validate theorem statements.

## Structure
- `configs/`: experiment configs
- `src/theory/`: certificate/math core
- `src/synthetic/`: synthetic checks
- `src/panel_a|panel_b|panel_c/`: pipelines
- `src/figures/`: matplotlib figure scripts
- `results/`: all outputs

## Install
`pip install -r requirements.txt`

## Run
- Synthetic: `bash scripts/run_synthetic_tests.sh configs/synthetic_tests.yaml results/synthetic`
- Panel A pilot: `bash scripts/run_panel_a_pilot.sh configs/panel_a_pilot.yaml results/panel_a_pilot`
- Panel A full: `bash scripts/run_panel_a_full.sh configs/panel_a_full.yaml results/panel_a_full`
- Panel B: `bash scripts/run_panel_b_eval.sh ...`, `bash scripts/run_panel_b_cka.sh ...`, `bash scripts/run_panel_b_analysis.sh ...`
- Panel C: `bash scripts/run_panel_c.sh configs/panel_c.yaml results/panel_c`

## Reproducibility
All major scripts accept `--config --output_dir --seed`; CLI overrides YAML for seed/output dir; config copy saved as `config_used.yaml`.
