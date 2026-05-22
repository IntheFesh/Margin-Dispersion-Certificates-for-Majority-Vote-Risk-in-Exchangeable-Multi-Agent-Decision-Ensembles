# Synthetic Checks Report

- mode: synthetic
- config_path: configs/synthetic_tests.yaml
- seed: 42
- datetime_utc: 2026-05-22T22:59:04.427507+00:00
- git_commit: b61706db3c76e7976b345a1107063900d8ca0e86
- num_instances: 2000
- num_models_or_samples: 32
- invalid_parse_rate: 0.000000

## Config Used
```yaml
{
  "seed": 42,
  "output_dir": "results/synthetic",
  "global_delta": 0.1,
  "validate_only": false,
  "strict_majority": true,
  "synthetic": {
    "M": 2000,
    "K": 32,
    "N_values": [
      16,
      32,
      64
    ],
    "N": 32,
    "estimator_trials": 100,
    "certificate_trials": 200,
    "alpha_distribution": {
      "name": "beta",
      "a": 12,
      "b": 8
    },
    "nonvacuity_high": {
      "name": "two_point",
      "p": 0.5,
      "a_low": 0.85,
      "a_high": 0.95
    },
    "nonvacuity_low": {
      "name": "two_point",
      "p": 0.5,
      "a_low": 0.48,
      "a_high": 0.52
    }
  }
}
```

## Main Metrics
| metric | value |
|---|---|
| estimator mean error | -0.000006 |
| estimator std error | 0.000441 |
| primary N | 32 |
| primary coverage rate | 1.0000 |
| nominal coverage target | 0.9000 |
| high-setting min empirical R_cert | 0.6993 |
| high-setting min population R_cert | 0.0632 |
| low-setting min empirical R_cert | 1.0000 |

| insufficiency | mu1 | mu2 |
|---|---|---|
| mean | 0.600000 | 0.600000 |
| variance | 0.040000 | 0.040000 |
| R3 | 0.376000 | 0.340000 |

## Figure Links
- none

## Missing Outputs
- none

## Warnings
- none
