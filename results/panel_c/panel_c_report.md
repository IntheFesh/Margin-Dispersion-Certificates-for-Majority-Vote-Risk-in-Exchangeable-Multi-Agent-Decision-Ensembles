# Panel C Supplementary Report

- mode: validate_only
- config_path: configs/panel_c.yaml
- seed: 42
- datetime_utc: 2026-05-22T22:34:38.876360+00:00
- git_commit: 32792a2f61cd838630833b2ddd3914bbb4cb1c0c
- num_instances: 0
- num_models_or_samples: 0
- invalid_parse_rate: 0.000000

## Config Used
```yaml
{
  "seed": 42,
  "validate_only": true,
  "output_dir": "results/panel_c",
  "strict_majority": true
}
```

## Main Metrics
validate_only run; no leaderboard predictions consumed

## Figure Links
- none

## Missing Outputs
- panel_c_pairwise_correctness.csv
- panel_c_family_summary.csv

## Warnings
- Panel C is appendix-only.
- No CKA is computed and no representation-alignment claim is made.
- Full Panel C run requires real per-instance leaderboard prediction files.
