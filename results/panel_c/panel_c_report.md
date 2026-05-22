# Panel C Supplementary Report

- mode: validate_only
- config_path: configs/panel_c.yaml
- seed: 42
- datetime_utc: 2026-05-22T22:59:08.212566+00:00
- git_commit: b61706db3c76e7976b345a1107063900d8ca0e86
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
