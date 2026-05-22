# Panel B Evaluation Report

- mode: validate_only
- config_path: configs/panel_b.yaml
- seed: 42
- datetime_utc: 2026-05-22T22:59:06.867643+00:00
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
validate_only run; no inference executed

## Figure Links
- none

## Missing Outputs
- panel_b_evaluation.csv
- panel_b_pairwise_stats.csv

## Warnings
- Panel B is external grounding only; observational and non-causal.
- Full evaluation requires real Hugging Face model resources and probe/eval splits.
