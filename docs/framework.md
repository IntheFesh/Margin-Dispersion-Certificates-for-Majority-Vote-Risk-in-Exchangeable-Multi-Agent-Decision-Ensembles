# Theoretical framework (v12)

This note restates the certificates implemented in `src/certs/`. We bound the
probability that an **exchangeable ensemble** of `N` decision agents fails to
reach a **strict success-majority** on a random task instance, using only the
first two moments of the per-instance agent success rate. All certificates are
**distribution-free** over the family of instance-success-rate distributions
with the given moments.

## Setup and notation

For a fixed instance, let `a in [0, 1]` be the per-agent success probability
(agents are conditionally i.i.d. given the instance, i.e. exchangeable). Across
the instance population `a` is drawn from a distribution `mu` on `[0, 1]` with

- **mean** `alpha_bar = E_mu[a]` (the mean per-agent success rate), and
- **dispersion** `F = Var_mu(a)` (the instance-to-instance variance).

Define the **mean margin** `m = alpha_bar - 1/2`. The moment-feasible region is
`0 <= F <= alpha_bar (1 - alpha_bar)`; outside it no distribution exists.

With `N` agents, the ensemble fails the strict success-majority on an instance
with conditional probability

```
g_N(a) = P( Bin(N, a) <= floor(N/2) ),
```

i.e. at most `floor(N/2)` of the `N` agents succeed (ties, which occur only for
even `N`, count as failures — hence we use **odd** `N` throughout). The target
risk is the population average

```
R_N(mu) = E_{a ~ mu}[ g_N(a) ].
```

The certificates upper-bound `R_N(mu)` knowing only `(alpha_bar, F)`.

## Theorem 1 — Cantelli–Hoeffding certificate `B_N^CH`

Split the population by a free threshold `eta in (0, m)`: instances with
`a <= 1/2 + eta` are bounded by `1`, the rest by a Hoeffding tail. Optimizing
the Cantelli (one-sided Chebyshev) bound on the low-success mass against the
Hoeffding tail gives

```
B_N^CH(alpha_bar, F, N) = inf_{0 < eta < m} [  F / (F + (m - eta)^2)
                                             + exp(-2 N eta^2)  ].
```

(`src/certs/theorem1.py`.) The first term is the Cantelli bound on
`mu([0, 1/2 + eta])`; the second is the Hoeffding majority-failure tail for
instances with margin at least `eta`. If `m <= 0` the certificate is vacuous
(`= 1`).

## Refinement 1 — boundedness-aware certificate `B_N^CH'`

`B_N^CH` uses the Cantelli bound, which ignores that `a` lives in `[0, 1]`. The
refinement replaces the Cantelli term by the **sharp two-moment bound on
`mu([0, 1/2 + eta])` over distributions supported on `[0, 1]`**, denoted
`C_{[0,1]}(alpha_bar, F, eta)`:

```
B_N^CH'(alpha_bar, F, N) = inf_{0 < eta < m} [ C_{[0,1]}(alpha_bar, F, eta)
                                             + exp(-2 N eta^2) ].
```

`C_{[0,1]}` is **piecewise**, governed by the sign of `p_0`, the lower atom of
the (unconstrained) optimal two-point distribution on `R` that matches the
mean–variance constraints. Writing the boundary `(1 - alpha_bar)(m - eta)`:

- **Case A (`p_0 >= 0`)**, when `F <= (1 - alpha_bar)(m - eta)`: the
  unconstrained Cantelli optimizer already lives in `[0, 1]`, so

  ```
  C_{[0,1]} = F / (F + (m - eta)^2).
  ```

- **Case B (`p_0 = 0`)**, when `F > (1 - alpha_bar)(m - eta)`: the optimizer is
  pushed against the lower boundary `0`, giving

  ```
  C_{[0,1]} = (1 - alpha_bar) + ( alpha_bar (1 - alpha_bar) - F ) / (1/2 - eta).
  ```

(`src/certs/refinement1.py`, functions `C_01` and `B_N_CH_prime`.) The case
split is governed solely by the sign of `p_0`, the position of the lower atom
of the unconstrained optimal two-point distribution: `p_0 >= 0` (Case A) vs
`p_0 = 0` (Case B). Because Case A coincides with the Cantelli term and Case B
can only lower it, `B_N^CH' <= B_N^CH` pointwise.

## Theorem 3 — sharp two-moment envelope `B_N^star`

The sharpest distribution-free bound from two moments is the value of the
**moment problem** itself:

```
B_N^star = sup_{ nu in P([0,1]) }  { integral g_N dnu :
              integral a   dnu = alpha_bar,
              integral a^2 dnu = alpha_bar^2 + F }.
```

This is a linear program in the measure `nu`. We discretize `[0, 1]` on a
uniform grid (default `n_grid = 2001`) and solve the **primal** LP
(`max g^T nu` subject to the `1, a, a^2` moment equalities, `nu >= 0`) via
`scipy.optimize.linprog(method="highs")`. The **dual** is the quadratic-majorant
problem `min b^T y` subject to `y_0 + a y_1 + a^2 y_2 >= g_N(a)` on the grid.

Strong duality holds for the discretized LP, so `|primal - dual|` is a
**numerical-bug detector**, not part of the proof: it is checked in
`src/certs/verify.py` and any violation aborts the pipeline. (On a dense sweep
the gap has a floating-point floor near `1e-7` from grid-node alignment of the
optimal atom; a genuine bug would be orders of magnitude larger.)
(`src/certs/theorem3.py`.)

## The certificate hierarchy

For every `(alpha_bar, F, N)` the three population-level bounds satisfy

```
R_N(mu) <= B_N^star <= B_N^CH' <= B_N^CH.
```

`B_N^star` is the tightest (it IS the two-moment supremum); `B_N^CH'` relaxes it
by the threshold-splitting argument with the sharp bounded-support tail;
`B_N^CH` further relaxes the bounded-support tail to the plain Cantelli bound.
The computable chain `B_N^star <= B_N^CH' <= B_N^CH` is verified per cell by
`check_hierarchy`; the leftmost inequality `R_N(mu) <= B_N^star` is exercised on
known mixtures (Proposition 1) in `tests/test_hierarchy.py`. (`src/certs/verify.py`.)

## Empirical certificates `R_N^cert`, `Q_N^cert`, `R_{N,BA}^cert`

In practice `(alpha_bar, F)` are unknown and estimated from the **estimation
pool only** (the oracle pool never feeds certificate construction). From an
`(M, K_full)` binary success matrix we form the unbiased estimates
`alpha_bar_hat` and `F_hat` and the second-moment estimate `E[a^2]_hat`
(`src/certs/moments.py`), then build one-sided confidence quantities at a
per-cell budget `delta_cell`.

### Four-event Bonferroni budget `delta_cell`

Each cell's bound uses a **four-event** one-sided Hoeffding union bound over the
events `{L_alpha, U_alpha, U_2, reserve}`, each allotted `delta_cell / 4`, so
the per-quantity radius is

```
eps_delta = sqrt( log(4 / delta_cell) / (2 M) ).
```

Across the experiment the global budget is split over the cells:
`delta_cell = delta_global / C`, where `C` is the number of
`(protocol, benchmark, K_est)` cells, computed at run time
(`bonferroni_cell_budget`, `hoeffding_radius` in `src/certs/empirical.py`).

### Confidence-corrected moments and the issued certificate

```
L_alpha = alpha_bar_hat - eps_delta      U_alpha = alpha_bar_hat + eps_delta
U_2     = E[a^2]_hat + eps_delta          U_F     = clip( U_2 - L_alpha^2, 0, 1/4 )
m_L      = L_alpha - 1/2                  m_beta_L = (1 - U_alpha) - 1/2
```

The **success-side certificate** plugs the confidence-corrected
`(L_alpha, U_F)` into the Cantelli–Hoeffding form, when `m_L > 0`:

```
R_N^cert = inf_{0 < eta < m_L} [ U_F / (U_F + (m_L - eta)^2) + exp(-2 N eta^2) ].
```

The **failure-side certificate** `Q_N^cert` is the symmetric quantity using the
failure-indicator margin `m_beta_L` (issued when `m_beta_L > 0`). The
**robust boundedness-aware certificate** `R_{N,BA}^cert` is a secondary,
bounded-support-robust variant on the same margin (`src/certs/robust_ba.py`),
issued only when requested and `m_L > 0`. For each issued certificate we record
the optimizing `eta_star` and whether the **Cantelli** (dispersion) or
**Hoeffding** (budget) term dominates at the optimum. (`src/certs/empirical.py`,
`empirical_certificate`.)

## Assumption 1 — bridge lemma (invalid-parse-is-failure)

The certificates bound the risk of the *observable success indicator*. **Assumption 1**
(the bridge lemma) makes the observable indicator a conservative proxy for true
task success: an answer that cannot be parsed is **kept as a recorded failure**
(`success_indicator = 0`, `invalid_parse = True`) rather than dropped. Because an
invalid parse can only lower the measured success rate, the issued certificate
on the observable indicator remains a valid upper bound on the true
majority-failure risk. No instance or sample is ever silently discarded.

## Refusal taxonomy (four modes)

A cell `(certificate × N)` either issues an informative certificate at a target
`epsilon` or **refuses**. The taxonomy (`src/certs/refusal.py`,
`classify_refusal`) has four modes:

- **`no_direction`** — `m_L <= 0` **and** `m_beta_L <= 0`: neither the success
  nor the failure direction has a usable margin after confidence correction.
- **`success_margin`** — `m_L > 0` but `R_N^cert > epsilon`: a success margin
  exists but the certificate is not informative enough.
- **`failure_margin`** — `m_beta_L > 0` but `Q_N^cert > epsilon`: symmetric on
  the failure side.
- **`bidirectional_cert`** — `min(R_N^cert, Q_N^cert) <= epsilon`: at least one
  issued certificate clears the target (a successful certification).

Within a **margin refusal** we sub-classify by the dominating term at `eta_star`:
**dispersion-dominated** (the Cantelli term dominates — more samples will not
help; the instance population is too dispersed) versus **budget-dominated** (the
Hoeffding term dominates — a larger ensemble `N` would tighten the bound).

Refusal is the maximally conservative outcome and is **never** scored as an
informative result: in the analyses a refused cell counts as vacuous at every
threshold and is excluded (not imputed) from coverage denominators.
