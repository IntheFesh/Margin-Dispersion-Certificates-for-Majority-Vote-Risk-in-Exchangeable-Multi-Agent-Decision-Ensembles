# Panel B Alignment Report

- mode: validate_only
- config_path: configs/panel_b.yaml
- seed: 42
- datetime_utc: 2026-05-22T22:59:07.792997+00:00
- git_commit: b61706db3c76e7976b345a1107063900d8ca0e86
- num_instances: 0
- num_models_or_samples: 0
- invalid_parse_rate: 0.000000

## Config Used
```yaml
{
  "seed": 42,
  "validate_only": true,
  "output_dir": "results/panel_b",
  "strict_majority": true
}
```

## Main Metrics
validate_only run; no inputs required

## Figure Links
- none

## Missing Outputs
- panel_b_pairwise_stats.csv
- panel_b_cka.csv

## Warnings
- Panel B is external grounding only; observational and non-causal.
- Full alignment analysis requires real pairwise stats and CKA CSVs.
