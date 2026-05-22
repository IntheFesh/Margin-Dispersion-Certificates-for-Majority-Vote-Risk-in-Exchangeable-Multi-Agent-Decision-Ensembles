# Panel A Full Report

- mode: validate_only
- config_path: configs/panel_a_full.yaml
- seed: 42
- datetime_utc: 2026-05-22T22:59:06.444490+00:00
- git_commit: b61706db3c76e7976b345a1107063900d8ca0e86
- num_instances: 0
- num_models_or_samples: 0
- invalid_parse_rate: 0.000000

## Config Used
```yaml
{
  "seed": 42,
  "output_dir": "results/panel_a_full",
  "validate_only": false,
  "M": 500,
  "K_ref": 256,
  "K_est_values": [
    16,
    32,
    64
  ],
  "benchmarks": [
    "ARC-Challenge",
    "GSM8K",
    "MMLU"
  ],
  "N_values": [
    8,
    16,
    32,
    64
  ],
  "global_delta": 0.1,
  "B_boot": 1000,
  "full_correctness_csv": "data/processed/panel_a_full_correctness.csv",
  "strict_majority": true
}
```

## Main Metrics
validate_only run; no inference executed

## Figure Links
- none

## Missing Outputs
- panel_a_cell_metrics.csv
- panel_a_bootstrap_metrics.csv
- panel_a_baseline_comparison.csv
- panel_a_nonvacuity_summary.csv
- panel_a_baseline_summary.csv

## Warnings
- Real Panel A full inference not run; requires real per-(instance, sample) correctness outputs.
