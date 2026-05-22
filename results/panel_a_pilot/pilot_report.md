# Panel A Pilot Report

- mode: validate_only
- config_path: configs/panel_a_pilot.yaml
- seed: 42
- datetime_utc: 2026-05-22T22:34:35.268533+00:00
- git_commit: 32792a2f61cd838630833b2ddd3914bbb4cb1c0c
- num_instances: 0
- num_models_or_samples: 0
- invalid_parse_rate: 0.000000

## Config Used
```yaml
{
  "seed": 42,
  "output_dir": "results/panel_a_pilot",
  "validate_only": false,
  "M": 100,
  "K_ref": 64,
  "K_est": 32,
  "benchmarks": [
    "ARC-Challenge",
    "GSM8K"
  ],
  "protocols": [
    "A1",
    "A2"
  ],
  "N_values": [
    16,
    32,
    64
  ],
  "global_delta": 0.1,
  "pilot_correctness_csv": "data/processed/panel_a_pilot_correctness.csv",
  "strict_majority": true
}
```

## Main Metrics
validate_only run; no inference executed

## Figure Links
- none

## Missing Outputs
- pilot_metrics.csv
- pilot_summary.json

## Warnings
- Real pilot inference not run; pilot requires per-(instance, sample) correctness outputs.
