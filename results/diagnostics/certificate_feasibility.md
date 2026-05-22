# Certificate Feasibility Diagnostic

Diagnostic-only: evaluates the current Theorem 2 empirical certificate over a
grid of (M, C, N, bar_alpha, F) at delta_global = 0.10, delta_cell = delta_global / C.
No theorem changes; no auto-tuning.

## Grid

- M: [100, 300, 500, 1000, 2000]
- C: [4, 6, 12]
- N: [8, 16, 32, 64]
- bar_alpha: [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
- F: [0.0, 0.01, 0.03, 0.05, 0.1]
- total grid points: 1800

## Overall feasibility

| metric | value |
|---|---|
| fraction m_L <= 0 (certificate refused) | 0.2111 |
| fraction U_F = 1/4 (deterministic clip binding) | 0.3356 |
| fraction R_cert = 1 (vacuous) | 0.9833 |
| fraction R_cert < 0.7 | 0.0000 |
| fraction R_cert < 0.3 | 0.0000 |

## Breakdown by M

| M | mL_le_0 | UF=1/4 | R_cert=1 | R_cert<0.7 | R_cert<0.3 |
|---|---|---|---|---|---|
| 100 | 0.5000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 300 | 0.2222 | 0.5000 | 1.0000 | 0.0000 | 0.0000 |
| 500 | 0.1667 | 0.1778 | 1.0000 | 0.0000 | 0.0000 |
| 1000 | 0.1667 | 0.0000 | 0.9583 | 0.0000 | 0.0000 |
| 2000 | 0.0000 | 0.0000 | 0.9583 | 0.0000 | 0.0000 |

## Breakdown by N

| N | mL_le_0 | UF=1/4 | R_cert=1 | R_cert<0.7 | R_cert<0.3 |
|---|---|---|---|---|---|
| 8 | 0.2111 | 0.3356 | 1.0000 | 0.0000 | 0.0000 |
| 16 | 0.2111 | 0.3356 | 1.0000 | 0.0000 | 0.0000 |
| 32 | 0.2111 | 0.3356 | 1.0000 | 0.0000 | 0.0000 |
| 64 | 0.2111 | 0.3356 | 0.9333 | 0.0000 | 0.0000 |

## Breakdown by C

| C | mL_le_0 | UF=1/4 | R_cert=1 | R_cert<0.7 | R_cert<0.3 |
|---|---|---|---|---|---|
| 4 | 0.2000 | 0.3133 | 0.9833 | 0.0000 | 0.0000 |
| 6 | 0.2000 | 0.3267 | 0.9833 | 0.0000 | 0.0000 |
| 12 | 0.2333 | 0.3667 | 0.9833 | 0.0000 | 0.0000 |

## Where the certificate is informative

- non-vacuous grid points: **30 / 1800 (1.67%)**
- min R_cert observed: **0.9304** at M=2000, C=4, N=64, bar_alpha=0.80, F=0.00
- N values with any non-vacuous cell: [64]
- M values with any non-vacuous cell: [1000, 2000]
- bar_alpha values with any non-vacuous cell: [0.8]

Full non-vacuous slice (sorted by R_cert ascending):

| M | C | N | bar_alpha | F | eps_M | U_F | m_L | R_cert |
|---|---|---|---|---|---|---|---|---|
| 2000 | 4 | 64 | 0.80 | 0.00 | 0.0356 | 0.0913 | 0.2644 | 0.9304 |
| 2000 | 6 | 64 | 0.80 | 0.00 | 0.0370 | 0.0949 | 0.2630 | 0.9367 |
| 2000 | 4 | 64 | 0.80 | 0.01 | 0.0356 | 0.1013 | 0.2644 | 0.9405 |
| 2000 | 6 | 64 | 0.80 | 0.01 | 0.0370 | 0.1049 | 0.2630 | 0.9458 |
| 2000 | 12 | 64 | 0.80 | 0.00 | 0.0393 | 0.1006 | 0.2607 | 0.9460 |
| 2000 | 12 | 64 | 0.80 | 0.01 | 0.0393 | 0.1106 | 0.2607 | 0.9537 |
| 2000 | 4 | 64 | 0.80 | 0.03 | 0.0356 | 0.1213 | 0.2644 | 0.9552 |
| 2000 | 6 | 64 | 0.80 | 0.03 | 0.0370 | 0.1249 | 0.2630 | 0.9592 |
| 2000 | 12 | 64 | 0.80 | 0.03 | 0.0393 | 0.1306 | 0.2607 | 0.9651 |
| 2000 | 4 | 64 | 0.80 | 0.05 | 0.0356 | 0.1413 | 0.2644 | 0.9653 |
| 2000 | 6 | 64 | 0.80 | 0.05 | 0.0370 | 0.1449 | 0.2630 | 0.9684 |
| 2000 | 12 | 64 | 0.80 | 0.05 | 0.0393 | 0.1506 | 0.2607 | 0.9729 |
| 1000 | 4 | 64 | 0.80 | 0.00 | 0.0504 | 0.1284 | 0.2496 | 0.9777 |
| 2000 | 4 | 64 | 0.80 | 0.10 | 0.0356 | 0.1913 | 0.2644 | 0.9797 |
| 1000 | 4 | 64 | 0.80 | 0.01 | 0.0504 | 0.1384 | 0.2496 | 0.9809 |
| 1000 | 6 | 64 | 0.80 | 0.00 | 0.0523 | 0.1334 | 0.2477 | 0.9815 |
| 2000 | 6 | 64 | 0.80 | 0.10 | 0.0370 | 0.1949 | 0.2630 | 0.9816 |
| 1000 | 6 | 64 | 0.80 | 0.01 | 0.0523 | 0.1434 | 0.2477 | 0.9843 |
| 2000 | 12 | 64 | 0.80 | 0.10 | 0.0393 | 0.2006 | 0.2607 | 0.9843 |
| 1000 | 4 | 64 | 0.80 | 0.03 | 0.0504 | 0.1584 | 0.2496 | 0.9859 |
| 1000 | 12 | 64 | 0.80 | 0.00 | 0.0556 | 0.1414 | 0.2444 | 0.9868 |
| 1000 | 6 | 64 | 0.80 | 0.03 | 0.0523 | 0.1634 | 0.2477 | 0.9885 |
| 1000 | 12 | 64 | 0.80 | 0.01 | 0.0556 | 0.1514 | 0.2444 | 0.9890 |
| 1000 | 4 | 64 | 0.80 | 0.05 | 0.0504 | 0.1784 | 0.2496 | 0.9894 |
| 1000 | 6 | 64 | 0.80 | 0.05 | 0.0523 | 0.1834 | 0.2477 | 0.9914 |
| 1000 | 12 | 64 | 0.80 | 0.03 | 0.0556 | 0.1714 | 0.2444 | 0.9921 |
| 1000 | 12 | 64 | 0.80 | 0.05 | 0.0556 | 0.1914 | 0.2444 | 0.9944 |
| 1000 | 4 | 64 | 0.80 | 0.10 | 0.0504 | 0.2284 | 0.2496 | 0.9945 |
| 1000 | 6 | 64 | 0.80 | 0.10 | 0.0523 | 0.2334 | 0.2477 | 0.9958 |
| 1000 | 12 | 64 | 0.80 | 0.10 | 0.0556 | 0.2414 | 0.2444 | 0.9977 |

## Interpretation (diagnostic, not a theorem change)

- $\epsilon_M=\sqrt{\log(4/\delta_{cell})/(2M)}$ dominates U_F at small M. With $\delta_{global}=0.10, C=4, M=100$, $\epsilon_M\approx 0.159$ and U_F clips at 1/4 in every cell. Larger M is required for U_F to escape the deterministic 1/4 clip.
- $L_\alpha = \bar\alpha - \epsilon_M$ falls below 1/2 whenever $\bar\alpha < 0.5 + \epsilon_M$, refusing the certificate. The fraction $\{m_L\le 0\}$ scales with M and not with N.
- Reducing C (e.g. running fewer cells in a single Bonferroni group) loosens $\delta_{cell}$ and shrinks $\epsilon_M$, but the effect is logarithmic in 1/C and small at this grid.
- Within the requested grid, R_cert < 0.7 is unattainable. Stronger informativeness requires $\bar\alpha > 0.8$ or $M \gg 2000$.

No theorem or default config was changed. This is a feasibility map only.

