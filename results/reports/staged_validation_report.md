# Staged Validation Report — Pre-Experiment Audit

- date: 2026-05-22
- branch: claude/exciting-turing-jKd0t
- PR: #5
- mode: pre-experiment staged validation; no full Panel A or Panel B runs
  attempted; no fabricated benchmark / model outputs

## 1. Theory–code consistency audit

| Invariant from the proposal | Code locus | Status |
|---|---|---|
| Strict-majority convention: ties count as failures, $r_N(a)=P(\mathrm{Bin}(N,a)\le \lfloor N/2\rfloor)$ | `theory/majority_risk.binom_strict_failure_prob` (uses `binom.cdf(N//2, N, a)`) | matches |
| K (samples / instance) vs N (target ensemble size) separated | `K_est` / `K_est_values` for estimator; `N_values` for certify; never conflated | matches |
| $\widehat{\mathcal F}$ = within-instance second-order U-statistic minus cross-instance squared-mean U-statistic | `estimators.estimate_F_unbiased = mean(U2_per_instance) - cross_instance_u_statistic(Z)` with $K\ge 2, M\ge 2$ enforced | matches |
| $L_\alpha, U_2, U_F, m_L, R_N^{\mathrm{cert}}$ formulas | `certificate.compute_confidence_bounds` + `empirical_certificate_from_summaries` | matches |
| $\eta$ chosen by 1-D numerical minimisation; no manual fix | `certificate.optimize_eta` uses `scipy.optimize.minimize_scalar(method="bounded")` with a dense 2000-point fallback recorded as `method=dense-grid-fallback` | matches |
| Oracle/reference split disjoint | `panel_a.split_reference_estimation.split_estimation_reference` permutes column indices and asserts `intersect1d(est, ref).size == 0` | matches |
| Panel A randomised exchangeable protocols vs Panel B heterogeneous surrogate diagnostic kept separate | Distinct top-level modules `src/panel_a/` and `src/panel_b/`; distinct configs; distinct result directories; Panel B report includes "external grounding only; observational and non-causal; does not validate Theorem 2" | matches |
| Bonferroni-adjusted $\delta_{\mathrm{cell}}=\delta_{\mathrm{global}}/C$ for Panel A | `run_pilot.py` counts C = unique (protocol, benchmark) cells; `run_full.py` counts C = unique (protocol, benchmark, regime) cells; both compute `delta_cell = delta_global / C` and write it into the metrics CSV and the report | matches |

The audit passes. No code change required in this step.

## 2. Synthetic unit tests on named Bernoulli mixtures

Implemented in `src/tests/test_named_mixtures.py` (8 tests, all PASS).

| Distribution | Theoretical check | Test | Result |
|---|---|---|---|
| $\mu=\delta_{0.7}$ | $\bar\alpha=0.7$, $\mathcal F=0$, $R_N=P(\mathrm{Bin}(N,0.7)\le\lfloor N/2\rfloor)$ closed form | `test_mu_delta_07_F_zero_and_R_N_closed_form` | $\|\widehat{\mathcal F}\|<0.005$ at $M=4000, K=32$; $R_N$ matches `binom.cdf` exactly |
| $\mu=\delta_{0.7}$ | Certificate collapses to Hoeffding $e^{-2Nm^2}$ when $\mathcal F=0$ | `test_mu_delta_07_population_certificate_collapses_to_hoeffding` | matches Hoeffding to within 1e-4 (numerical $\eta$-optimiser cannot hit $\eta=m$ exactly) |
| $\mu=\delta_{0.7}$ | Empirical certificate is a valid upper bound on $R_N$ | `test_mu_delta_07_empirical_certificate_covers_truth` | $R^{\mathrm{cert}}\ge R_{\mathrm{true}}$ at $N\in\{16,32,64\}$, $M=2000, K=32, \delta=0.10$ |
| $\mu=0.5\delta_{0.4}+0.5\delta_{0.8}$ | $\bar\alpha=0.6, \mathcal F=0.04$, $R_N$ closed form | `test_two_point_F_and_R_N_closed_form` | exact to 1e-12 |
| $\mu=0.5\delta_{0.4}+0.5\delta_{0.8}$ | Estimator recovers $\mathcal F=0.04$ on average | `test_two_point_estimator_recovers_F` | $\|\overline{\widehat{\mathcal F}}-0.04\|<0.01$ over 40 trials |
| $\mu=0.5\delta_{0.4}+0.5\delta_{0.8}$ | Population certificate vacuous at low margin $m=0.1$ for all $N\in\{8,16,32,64\}$ — confirms Corollary 1 | `test_two_point_certificate_vacuous_low_margin` | $R^{\mathrm{cert}}\ge 0.99$ for all $N$ |
| Two-moment insufficiency: $\mu_1$ vs $\mu_2$ | $\bar\alpha=0.6, \mathcal F=0.04$ both; $R_3(\mu_1)=0.376, R_3(\mu_2)=0.340$ (Proposition 1) | `test_two_moment_insufficiency_exact_numbers` | reproduced to 1e-9 |
| $\mu_1$ vs $\mu_2$ | Certificate cannot distinguish; this IS the insufficiency | `test_two_moment_insufficiency_certificate_indistinguishable` | $R^{\mathrm{cert}}$ identical for both; true risks differ by 0.036 |

All synthetic checks PASS. Total test count is now **53 passed**.

## 3. Tiny real-data smoke test

Two parts:

### 3a. HF Hub adapters (network-dependent)

ARC-Challenge, GSM8K, MMLU, HellaSwag with `max_instances=3, split="test"`
on the four adapters. **All four raise `FileNotFoundError`** because the
remote sandbox has no outbound network access to `huggingface.co`. This
is the spec'd behaviour:

> "any 缺失文件、schema mismatch、模型加载失败、数据下载失败、… 都必须 raise explicit error。"

No silent fallback occurred. No fake data was generated.

### 3b. In-repo smoke test (network-independent)

`/tmp/smoke_pipeline.py` (kept under `/tmp/`, not committed) exercises
every code path that does not need HF Hub or a GPU:

- 12 answer-extraction cases (`The answer is (B)`, `Option A`, GSM8K
  `#### 42`, `1,234.50`, `-7`, empty, multiple-numbers→last, no-number→
  invalid). All produce the expected `(parsed_answer, invalid_parse)`
  tuples.
- Correctness pipeline: REQUIRED_COLUMNS enforced, `invalid_parse`
  preserved, `raw_output` retained when invalid, `correct=0` when
  invalid; missing-column input raises `ValueError`.
- `split_estimation_reference`: seed-deterministic, disjoint, leakage
  assertion raises.
- Pilot pipeline on a tiny hand-crafted correctness CSV: full schema
  written, Bonferroni $\delta_{\mathrm{cell}}=0.10/4=0.025$ applied
  automatically, 12 metric rows (2 benchmarks × 2 protocols × 3 N) with
  all spec'd columns.

All four sections of the in-repo smoke test PASS.

## 4. Pilot status — NOT RUN on real data

The pilot ($M=100$, $K_{\mathrm{ref}}=64$, ARC-Challenge + GSM8K, A1+A2,
$N\in\{16,32,64\}$, $\delta_{\mathrm{global}}=0.10$) requires:

1. ARC-Challenge and GSM8K datasets (HF Hub) — **unavailable** here.
2. LLM inference for $M\times K_{\mathrm{ref}}\times 2\,\mathrm{benchmarks}\times 2\,\mathrm{protocols}=25\,600$ generations — **unavailable** (no GPU / model resources).

Therefore the real pilot is **not run** in this remote sandbox. The
adapters and pilot pipeline are wired up and would execute end-to-end
the moment those resources are present.

### 4a. Pilot code-readiness dry-run (clearly labelled, NOT a real pilot)

To confirm the pilot pipeline produces the spec'd outputs end-to-end, a
*simulated-correctness* dry-run was performed using Bernoulli-mixture
draws calibrated to plausible per-cell competence distributions. The
numbers below are **not real-LLM pilot results** and must not be
reported as such. They are reproduced here only to demonstrate that
the pilot pipeline works end-to-end and to surface a structural
observation about $M$.

Run config: $M=100, K_{\mathrm{ref}}=64, K_{\mathrm{est}}=32, \delta_{\mathrm{global}}=0.10, C=4,
\delta_{\mathrm{cell}}=0.0250$.

Observed metric rows (per cell, per $N$):

| benchmark | protocol | N | $\widehat{m}$ | $\widehat{\mathcal F}$ | $m_L$ | $U_F$ | $R^{\mathrm{cert}}$ | $R^{\mathrm{MC}}$ | issued |
|---|---|---|---|---|---|---|---|---|---|
| ARC-Challenge | A1 | 16 | 0.3150 | 0.0004 | 0.156 | 0.250 | 1.0000 | 0.028 | True |
| ARC-Challenge | A1 | 32 | 0.3150 | 0.0004 | 0.156 | 0.250 | 1.0000 | 0.009 | True |
| ARC-Challenge | A1 | 64 | 0.3150 | 0.0004 | 0.156 | 0.250 | 1.0000 | 0.003 | True |
| ARC-Challenge | A2 | 16 | 0.2547 | 0.0141 | 0.095 | 0.250 | 1.0000 | 0.114 | True |
| ARC-Challenge | A2 | 32 | 0.2547 | 0.0141 | 0.095 | 0.250 | 1.0000 | 0.089 | True |
| ARC-Challenge | A2 | 64 | 0.2547 | 0.0141 | 0.095 | 0.250 | 1.0000 | 0.076 | True |
| GSM8K | A1 | 16 | 0.1100 | 0.0044 | -0.049 | 0.250 | 1.0000 | 0.284 | False |
| GSM8K | A1 | 32 | 0.1100 | 0.0044 | -0.049 | 0.250 | 1.0000 | 0.220 | False |
| GSM8K | A1 | 64 | 0.1100 | 0.0044 | -0.049 | 0.250 | 1.0000 | 0.180 | False |
| GSM8K | A2 | 16 | 0.0253 | 0.0177 | -0.134 | 0.250 | 1.0000 | 0.485 | False |
| GSM8K | A2 | 32 | 0.0253 | 0.0177 | -0.134 | 0.250 | 1.0000 | 0.439 | False |
| GSM8K | A2 | 64 | 0.0253 | 0.0177 | -0.134 | 0.250 | 1.0000 | 0.409 | False |

Summary: non-vacuity rate 0/12; pointwise coverage vs MC reference
12/12; `warning_near_universal_vacuity = True`.

### 4b. Structural observation about $M$, $\delta_{\mathrm{cell}}$, and $U_F$

The dry-run reveals a non-trivial finite-sample issue that the real
pilot will face for the same reason. With $M=100$ and Bonferroni-
adjusted $\delta_{\mathrm{cell}}=0.0250$,

$$
\epsilon_\delta=\sqrt{\log(4/\delta_{\mathrm{cell}})/(2M)} = \sqrt{\log(160)/200}\approx 0.159.
$$

Therefore $U_F = \min\{1/4,\max(0,\,U_2-L_\alpha^2)\}$ clips at $1/4$ in
every cell (the empirical second-moment inflation exceeds the
deterministic $1/4$ bound on a $[0,1]$-variance). The certificate's
first term becomes $1/4/(1/4+(m_L-\eta)^2)$, which, combined with
$e^{-2N\eta^2}$ on small $m_L$, generally exceeds 1 over $\eta\in(0,m_L)$.

This is the predicted behaviour at small $M$. The pilot is designed to
surface exactly this, and the proposal §4.1 anticipates it:

> "If the pilot shows near-universal vacuity, the full Panel A protocol is adjusted before the expensive run."

Per the no-auto-tuning rule (§0 constraint 5), the pilot does **not**
automatically change M / K_ref / δ / K_est / protocols / N_values. The
warning is the deliverable. The decision to adjust before the full
Panel A is left to the user.

## 5. Recommendation — staged go/no-go

| step | status | go to next? |
|---|---|---|
| Theory–code audit | PASS | yes |
| Synthetic mixture tests (3 distributions) | PASS (8/8) | yes |
| In-repo smoke test (extraction / correctness / split / pilot pipeline) | PASS | yes |
| Real-data adapter smoke test | NOT POSSIBLE (no HF Hub access in sandbox); adapters correctly fail loudly | **blocked on resources** |
| Real LLM pilot | NOT POSSIBLE (no GPU / models in sandbox) | **blocked on resources** |
| Full Panel A | NOT STARTED | **NO — wait for real pilot result** |
| Full Panel B | NOT STARTED | **NO — wait for real pilot result** |

The codebase is theory-consistent and empirically non-degenerate on the
checks we can run here. The remaining gates (real-data adapter smoke
test, real-LLM pilot) require external resources that this sandbox
does not provide. Do not start full Panel A or Panel B until the real
pilot is executed and its `pilot_summary.json` confirms a useful
non-vacuity fraction and the chosen $M, K_{\mathrm{ref}}, K_{\mathrm{est}}$ keep
$\epsilon_\delta$ small enough that $U_F$ does not always clip at $1/4$.
