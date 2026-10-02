# Online tail dependence under copula change — step 1: stationary validation

**Direction.** How should tail dependence be estimated online when the copula is subject to
abrupt structural change? Central design: switch copula family at **matched global dependence**,
so the change is in the *shape* (tails), not in rank association.

**Prior work this builds on** (branch `adaptive-memory`): `kalman_changepoint_results.md` §14.13
/ `copula_v1.md` §7.1 ran Gaussian → t(4) at fixed $\rho_S=0.58$. Every $\rho$-monitoring detector
was blind (1–2/20, AUC ≈ 0.53), coefficient monitoring scored 0/20, the `tail_cusum` monitor
($q=0.15$) scored 0/20, and only `gaussian_mle_cusum` had power (17/20, delay ≈ 700) — by
accident, via score-variance inflation. So H1 (global-dependence detectors miss shape change) is
already supported; what is new here is *estimating* tail dependence online, the memory × threshold
trade-off, and more transitions.

Harness: `experiments/tail_validate.cpp` (`make tail_validate && ./tail_validate 0.6 20`).

## Setup

- **Copula space, known uniform margins.** Rank-based margins are step 2.
- **Matched on $\rho_S$, not Kendall's $\tau$.** The DGP (`copula.hpp`) and every estimator in the
  repo target Spearman $\rho_S$; matching on $\tau$ would leave a $\rho_S$ step at the switch.
  Knob $d=0.6$ → target $\rho_S=0.58192$ (the §14.13 level). Families: Gaussian ($\rho=0.6$),
  Clayton ($\theta=1.419$), Gumbel ($\theta=1.712$), $t_4$ ($\rho=0.614$).
- **Samplers** reproduce `copula.hpp`'s formulas with the parameter solved once (the DGP class
  re-solves per draw: 225 µs/draw for $t$). Cross-checked against the DGP classes (1M draws,
  160k for $t$): $\chi_L(.05)$, $\chi_U(.05)$ and $\rho_S$ agree, 12 z-scores all $|z|\le2.1$.
  Achieved $\rho_S$ over 20M draws: 0.5814 / 0.5820 / 0.5822 / 0.5815 — matched to 0.001.
- **Targets.** $\lambda_L,\lambda_U$ in closed form; finite-threshold
  $\chi_L(q)=C(q,q)/q$, $\chi_U(q)=P(U>1-q,V>1-q)/q$ by 20M-draw Monte Carlo, checked against
  closed-form $C(q,q)$ for Clayton/Gumbel (agree to ≤ 0.0012).
- **Estimators** (standard definitions; the Frahm–Junker–Schmidt paper was not available in-session,
  so exact forms should be re-checked against it), exponentially forgotten with ESS $m$, normalised
  by accumulated weight (no startup bias):
  - **SEC** $\hat\lambda_L=\hat C(q,q)/q$;
  - **ratio** $\hat P(V\le q\mid U\le q)$ = joint / marginal corner count — what the rank-based
    empirical copula reduces to, since ranks make the marginal count exactly $qn$;
  - **LOG** $\hat\lambda_L=2-\log(1-2q+\hat C(q,q))/\log(1-q)$.
  Upper tail by $u\mapsto1-u$. $q\in\{.01,.02,.05,.10\}$, $m\in\{250,\dots,50000\}$, 20 seeds,
  300k ticks, scored after 150k.

## Results

### 1. The finite threshold is the dominant error, and memory cannot remove it

$\chi(q)-\lambda$ (the bias of a *perfect* estimator at threshold $q$):

| family, tail | $\lambda$ | $q=.01$ | $q=.02$ | $q=.05$ | $q=.10$ |
| --- | --- | --- | --- | --- | --- |
| Gaussian, both | 0 | **+0.19** | +0.23 | +0.31 | +0.39 |
| Clayton, lower | 0.613 | +0.000 | +0.002 | +0.004 | +0.009 |
| Clayton, upper | 0 | +0.02 | +0.05 | +0.11 | +0.21 |
| Gumbel, upper | 0.501 | +0.005 | +0.009 | +0.018 | +0.038 |
| Gumbel, lower | 0 | +0.10 | +0.14 | +0.23 | +0.32 |
| $t_4$, both | 0.324 | +0.04 | +0.06 | +0.09 | +0.13 |

- **A Gaussian copula looks tail-dependent at every usable threshold** — 0.19 even at $q=0.01$ —
  while its $\lambda$ is 0. Tail-*independent* sides are where the threshold bias is large;
  genuinely tail-dependent sides converge fast.
- So the advertised "$\lambda_L$: 0 → 0.32" of Gaussian → $t$ is, at $q=0.05$, an observable change
  of **0.31 → 0.41**: about a third of the limiting change.
- **Scoring rule adopted:** score each estimator against its own finite-$q$ target and report the
  gap to $\lambda$ separately. Scoring against $\lambda$ would mostly measure the table above.

### 2. Against their own target, all three estimators are unbiased; the ratio form is least noisy

Max $|\text{bias}|$ vs own target at $m\ge1000$: SEC 0.006, ratio 0.006, LOG 0.006. SD relative to
ratio: SEC 0.97–1.66 (median 1.26), LOG 0.99–1.77. The ratio form cancels the marginal-count
fluctuation that SEC and LOG carry with known margins. **Use the ratio (rank) form.**

### 3. Tail noise is exactly binomial thinning at rate $q$

$\mathrm{SD}^2\cdot m\cdot q\,/\,\chi(1-\chi)$ at $m=5000$: **0.89–1.08** across all families and
thresholds. So

$$
\operatorname{Var}\hat\chi(q)\approx\frac{\chi(1-\chi)}{q\,m}:
$$

a tail estimator is a dependence estimator on a stream thinned to rate $q$ — the same structure as
the async study's information-rate law. Prediction for the drift experiments: variance term
$\propto 1/(qm)$, so the optimal memory scales like $(q)^{-1/3}$ in the variance-dominated regime,
while the threshold bias (table 1) is, like async bias, untouched by memory.

### 4. Detectability: SNR of the matched-$\rho_S$ transitions

$|\Delta\chi(q)|/\max(\mathrm{SD}_\text{before},\mathrm{SD}_\text{after})$, ratio estimator:

| transition (tail) | $\Delta\lambda$ | best $q$ | SNR at $m=1000$ | $m$ for SNR ≈ 3 |
| --- | --- | --- | --- | --- |
| Gauss → Clayton (lower) | 0 → 0.61 | 0.10 | 4.7 | ~500 |
| Gauss → Gumbel (upper) | 0 → 0.50 | 0.05–0.10 | 3.0 | ~1000 |
| **Gauss → $t_4$ (lower)** | 0 → 0.32 | 0.05 | **1.5** | **~5000** |
| Clayton → Gumbel (lower) | 0.61 → 0 | 0.10 | 6.3 | ~250 |
| Clayton → Gumbel (upper) | 0 → 0.50 | 0.10 | 6.5 | ~250 |

- **For detection, a moderate threshold beats a small one.** SNR rises from $q=0.01$ to
  $q=0.05$–$0.10$ in every transition: $\Delta\chi$ shrinks slowly with $q$ while variance falls
  like $1/q$. Estimating $\lambda$ wants small $q$; detecting a change wants larger $q$. That is H3,
  measured.
- **Gaussian → $t$ is the hardest transition by a wide margin**, and it is the one §14.13 tested:
  SNR 3 needs memory ≈ 5000 ticks, but that experiment had only 2500 post-switch ticks. This
  explains `tail_cusum`'s 0/20 without any defect in the detector.
- One-shot SNR between two stationary estimates is an upper bound on what a sequential,
  ARL₀-calibrated detector achieves, not a detection rate.

## What this sets up

- Use the ratio estimator; score against $\chi(q)$; report $\chi(q)-\lambda$ separately.
- Step 2 (known changepoint, no detector): sweep $m\times q$ through each transition; the oracle
  restart vs best fixed memory gate. Expect Gaussian → $t$ to need long memory and to gain most from
  restart; Clayton ↔ Gumbel to be easy.
- Step 2 should also switch to rank-based margins (rank-PIT exists on `adaptive-memory`).
- Only one dependence level ($\rho_S=0.58$) and $\nu=4$ so far.
