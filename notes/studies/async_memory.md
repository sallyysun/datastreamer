# Asynchronous arrival and the optimal memory

**Question.** How does asynchronous information arrival change the optimal memory $M^*$ of an
online dependence estimator? Broader framing: how should an online estimator remember the past when
both the dependence *and the rate at which information arrives* can change?

Motivating paper: Chakrabarti & Sen (2022), *Copula Estimation for Nonsynchronous Financial Data*,
Sankhyā. Harness: `experiments/async_memory.cpp`; scaling check: `experiments/async_scaling.py`.
Deliberately scoped: Gaussian copula only, `constant` and linear drifts only, no detectors, no
general-copula (Kendall/regression) correction.

**Status of the laws.** $M^*\propto f^{-1/3}$ is not from the paper. It is the prediction of our
own local-linear-drift bias–variance model (processing-rate study):
$\operatorname{MSE}(M)\approx\sigma^2/(fM)+(dM/2)^2\Rightarrow M^*=(2\sigma^2/d^2)^{1/3}f^{-1/3}$.
The $\bar c^{-2/3}$ extension below is the same heuristic with the corrected variance
$\sigma^2/\bar c^2$. Both are model-derived predictions checked here, not theorems; cube-root
rates from a $1/M$ vs $M^2$ balance are generic, so the novelty is the decomposition, not the root.

**Notation.** $M$ = memory in **clock ticks**; $m\approx fM$ = memory in **processed pairs**
(the estimator's ESS knob; `M_pairs` in the CSV, $M$ = `M_clock`). The same optimum reads
$M^*\propto f^{-1/3}$ or $m^*\propto f^{2/3}$.

## Setup

- **Latent process.** One synchronous Gaussian-copula return stream on a clock of $T=20{,}000$
  steps, dependence $\rho_t$ from the existing `ConstantDependence(0.6)` / `DriftDependence(-0.5, 0.8, T)`.
  Scored after a burn-in of 2,000 clock steps, every 10th step, against the true Spearman $\rho_S(t)$.
  Errors are measured in **clock time** (the estimate is held between pairs), so staleness counts.
- **Arrivals.** Each leg observed independently, Bernoulli($p_x$), Bernoulli($p_y$) per clock step.
  Settings $(p_x,p_y)\in\{(1,1),(0.5,0.5),(0.25,0.25),(0.1,0.1),(0.5,0.1)\}$; $(1,1)$ is the
  fully synchronous baseline. The latent path is shared across settings (common random numbers).
- **Pairing.** Refresh-time pairing on actual timestamps (the paper's algorithm A0): a pair is
  formed once both legs have a new observation; each leg's return runs from its previous paired
  observation to its latest one. Returns are scaled by $\sqrt{\text{interval}}$ so both margins
  stay exactly $N(0,1)$ — the harness's standardisation assumption still holds.
- **Overlap.** For pair $k$, $c_k = I_k/\sqrt{L^x_kL^y_k}\in[0,1]$ ($L$ = each leg's return
  interval, $I_k$ = their common time). The paired scaled returns have correlation $\theta c_k$,
  so pooled naive pairing sees exactly $\theta\,\bar c$ with $\bar c=\frac1n\sum_k c_k$.
  This is **not** the paper's $m(I)/\sqrt{m_1m_2}$ (a ratio of means for *unscaled* returns); it is
  its analogue for $\sqrt{\text{interval}}$-scaled returns, and is what every $\bar c$ below means.
- **Arms** (same refresh times, so identical pair counts):

  | Arm | What | Isolates |
  | --- | --- | --- |
  | `sync` | both legs observed at each refresh time | **information-rate** effect (vs the $(1,1)$ row) |
  | `async` | naive refresh-time pairing | **asynchronicity bias** (vs `sync`) |
  | `corr` | `async` output, Pearson-equivalent divided by running $\bar c$, mapped back to $\rho_S$ | the elliptical correction and its cost |

  `sync` is the matched-rate thinning control: same number of pairs per unit clock time as `async`,
  but no overlap loss.
- **Estimators** — every single-knob memory family: `xiao_ew`, `hermite_ew`, `gaussian_mle_ew`,
  `windowed_rank`, `kalman`. Common grid of effective sample size $M\in\{10,\dots,2560\}$ pairs
  (doubling); EW knobs via $\mathrm{ESS}=(2-\alpha)/\alpha$ (mirror-image λ conventions respected),
  window $=M$, Kalman $\sigma_\omega^2$ from steady-state gain $K=2/(M/20+1)$. $M^*$ is a
  log-parabolic interpolation of the 20-seed mean RMSE curve; 68% CI by seed bootstrap.
  $M=m\times$ (clock steps per pair).

Pairing cost, measured: steps per pair 2.67 / 5.72 / 14.82 / 10.22 and mean overlap
$\bar c$ = 0.847 / 0.733 / 0.664 / 0.861 for (0.5,0.5) / (0.25,0.25) / (0.1,0.1) / (0.5,0.1).
A busy–quiet pair loses *less* overlap than two quiet legs.

## Results (20 seeds, base seed 42)

### `drift` — interior $M^*$ exists

Ratios, range across the five estimators:

| $(p_x,p_y)$ | $M^*_\text{sync}$ / $f^{-1/3}$ prediction | $M^*_\text{async}$ / $M^*_\text{sync}$ | $M^*_\text{corr}$ / $f^{-1/3}\bar c^{-2/3}$ prediction | RMSE$^*$ async / sync | RMSE$^*$ corr / sync |
| --- | --- | --- | --- | --- | --- |
| 0.5, 0.5 | 0.98–1.02 | 0.84–0.90 | 1.02–1.06 | 1.60–1.69 | 1.14–1.19 |
| 0.25, 0.25 | 1.03–1.08 | 0.85–0.96 | 0.98–1.12 | 1.77–1.98 | 1.22–1.30 |
| 0.1, 0.1 | 1.02–1.12 | 0.92–1.05 | 0.91–1.19 | 1.68–1.88 | 1.33–1.41 |
| 0.5, 0.1 | 0.93–1.03 | 0.94–1.00 | 0.92–1.09 | 1.27–1.35 | 1.12–1.15 |

Predictions are anchored on each estimator's own $(1,1)$ optimum ($M^*$ 636–711 ticks), so this
tests the **scaling**, not the constant $(2\sigma^2/d^2)^{1/3}$. Rough absolute check: the $\rho_S$
path rises 1.27 over $T$, $d\approx6.3\times10^{-5}$/tick, and $\sigma^2\approx1$ gives
$M^*\approx790$ — the right size, not a claim.
Example, `xiao_ew` $M^*_\text{clock}$: 664 (sync, 1,1) → 929 sync / 819 async / 1070 corr at (0.5,0.5)
→ 1820 / 1776 / 2546 at (0.1,0.1).

### `constant` — no interior $M^*$

Every `sync` and `corr` cell has its optimum at the grid edge (longest memory), as it must when
nothing changes. `async` sometimes shows an "interior" optimum, but its RMSE curve is flat to
<0.001 across the top of the grid (bias ≫ variance), so that $M^*$ carries no information.
At $m=2560$, `xiao_ew` bias: sync −0.009 / −0.012 / −0.011, **async −0.099 / −0.171 / −0.210**,
corr −0.010 / −0.017 / −0.018 at (0.5,0.5) / (0.25,0.25) / (0.1,0.1), and async −0.090 at
(0.5,0.1) — naive bias 0.09–0.21 across settings. Naive async RMSE is
3.5–8.4× the sync RMSE across the four settings; corrected is 1.3–1.7×.

## Reading

1. **Fewer pairs alone: the $f^{-1/3}$ prediction holds.** The `sync` arm's optimum follows
   $M^*\propto f^{-1/3}$ to 0.93–1.12 across all drifts for the four well-behaved estimators, over
   a 15× range of pair rates (`hermite_ew` excluded — startup bug, see "State-dependent noise").
   Lower liquidity ⇒ longer memory in ticks, *shorter* in pairs ($m^*\propto f^{2/3}$).
2. **Asynchronicity is a bias problem, not a memory problem.** Naive pairing raises the best
   achievable RMSE by 1.3–2× on `drift` and 3.5–8× on `constant`, and no memory choice recovers it.
   Its $M^*$ moves only slightly — and *shorter*, not longer.
3. **Why shorter (and why not general).** The estimate tracks $\bar c\rho_t$, so the error is
   lag $+\,(\bar c-1)\rho_t$. On this drift $\rho_t$ is mostly positive while rising, so the lag and the
   attenuation both point down; longer memory adds lag in the direction the bias already errs,
   which pushes $M^*$ down, outweighing the flatter-slope effect that pushes it up. The sign of
   the cross-term is $\operatorname{sign}(\dot\rho\cdot\bar\rho)$: a drift *toward* zero
   shifts $M^*$ the other way — **confirmed** in both mirror pairs (below).
4. **With the correction, overlap is a second rate.** Correcting removes the bias (residual
   bias ≈ the sync arm's) but divides the signal by $\bar c$, inflating variance by $\bar c^{-2}$.
   The same bias–variance calculation then gives
   $M^*\propto f^{-1/3}\,\bar c^{-2/3}$ to 0.95–1.26 across all drifts (four estimators), and
   to 0.92–1.17 once the measured state-dependent noise is used. Asynchronous
   overlap therefore acts like an *additional* loss of information rate, $f_\text{eff}=f\bar c^{2}$,
   on top of the loss of pairs — the answer to "does $f^{-1/3}$ still describe the shift?": yes,
   with $f$ replaced by $f\bar c^2$, approximately. The price is 1.1–1.4×
   the matched-rate RMSE.

### Directional drifts — testing reading 3

The sign argument predicts naive `async` $M^*$ **shorter** than `sync` when $\rho$ drifts away from
zero (lag and attenuation both point toward zero) and **longer** when it drifts toward zero (they
oppose). Four linear drifts over $T$, same arrivals and seeds; `async`/`sync` $M^*$ ratio, range
across estimators:

| $(p_x,p_y)$ | `away_pos` 0→0.8 | `away_neg` 0→−0.8 | `toward_pos` 0.8→0 | `toward_neg` −0.8→0 |
| --- | --- | --- | --- | --- |
| 0.5, 0.5 | 0.69–0.71 | 0.67–0.71 | 1.06–2.28 | 1.05–2.39 |
| 0.25, 0.25 | 0.72–0.76 | 0.71–0.86 | 3.19–9.64 | 1.01–3.21 |
| 0.1, 0.1 | 0.81–0.88 | 0.75–0.91 | 3.12–7.93 | 1.00–3.66 |
| 0.5, 0.1 | 0.80–0.85 | 0.79–0.89 | 1.06–1.71 | 1.01–1.73 |

**Confirmed, in both mirror pairs.** Every "away" cell is < 1 and every "toward" cell is ≥ 1, for
all five estimators. The low end of each "toward" range is `hermite_ew`, whose long-memory cells
are corrupted by its startup bug (see "State-dependent noise"). For the other four, toward-zero
$M^*$ is 2–3.5× the sync optimum, and the bias *at* that optimum is near zero: the long memory's lag
is being used to cancel the attenuation.

**Consequence: naive pairing can look better than correct pairing, by luck.** In the toward-zero
drifts, naive `async` RMSE$^*$ is sometimes *below* the matched-rate `sync` arm (ratios down to
0.72–0.93, e.g. `windowed_rank`, and most estimators at (0.5, 0.1)), while in the away drifts it is
2.1–3.0× worse. Whether uncorrected asynchronous data helps or hurts depends on the direction of a
dependence change the estimator does not know — so the naive arm cannot be tuned into safety.

The corrected arm has no direction-dependent *bias*. Its scaling fit is reported, together with
the constant- vs state-dependent-noise comparison, in "State-dependent noise" below.

### Random overlaps — is $\bar c$ the right summary?

The derivation replaces the random $c_k$ by $\bar c$. For the **pooled** estimators run here this
is the relevant summary, not just a convenience: pooling unit-variance pairs with correlations
$\theta c_k$ gives attenuation exactly $\bar c$, and the pooled noise is ~$\sigma^2/n$ regardless
of the spread of $c_k$, so the corrected variance is ~$\sigma^2/(n\bar c^2)$. ($\mathbb E[C^{-2}]$
would appear only for per-pair $1/c_k$ correction — a worse estimator.) The *efficient* estimator
weights pair $k$ by its information $\propto c_k^2$ (near $\theta=0$), reaching
$f_\text{eff}=f\,\mathbb E[C^2]\ge f\bar c^2$; the gap $\operatorname{Var}(C)$ is what pooling
discards. Measured (`experiments/async_overlap.py`, 400k ticks):

| $(p_x,p_y)$ | $\bar c$ | $\mathbb E[C^2]$ | $\mathbb E[C^2]/\bar c^2$ |
| --- | --- | --- | --- |
| 0.5, 0.5 | 0.846 | 0.744 | 1.04 |
| 0.25, 0.25 | 0.732 | 0.575 | 1.07 |
| 0.1, 0.1 | 0.663 | 0.486 | 1.11 |
| 0.5, 0.1 | 0.859 | 0.763 | 1.04 |

Heterogeneity is worth ≤ 11% in variance, ≤ $(1.11)^{1/3}\approx3.5\%$ in $M^*$. Under burstier (bimodal-overlap) arrivals it could matter, and
overlap-weighting would then be the principled fix.

### State-dependent noise — diagnostic

Before the Hermite bug was found, all-estimator ranges showed misses up to 2.5× on toward-zero
drifts at low liquidity. Two candidate causes: random-overlap heterogeneity (ruled out above,
≤ 3.5%) and $\rho$-dependent estimator noise. Stated prediction beforehand: *measured
$v_j(\rho)$ should improve toward-zero fits and leave away-from-zero fits mostly unchanged.*

**Measured noise** (`experiments/async_noise.cpp`): constant-$\rho$ synchronous streams,
$v_j(\rho)=m\,\mathrm{Var}(\hat\rho)$ at steady state, 20 seeds. $m=80$ and $m=320$ agree to a
few percent (the $1/m$ assumption holds). For every estimator $v_j$ peaks at $\rho\approx0$ and is
symmetric: ≈1.0–1.2 at 0, falling to 0.18–0.30 at ±0.8 (0.08 for `gaussian_mle_ew`) — a 4–11×
range the constant-$\sigma^2$ model ignores.

**Predictions** (`experiments/async_predict.py`): minimise the path-averaged heuristic MSE with
each estimator's own kernel (EW, or flat for the window; Kalman adds half a block of delay). The
kernel gives both the lag and the local average of $v_j$. For `corr`, noise is $v_j$ at the
attenuated $\rho$, times the $1/\bar c$ and Spearman↔Pearson Jacobian. Constant model: the same
machinery with $v_j$ averaged along the path. Anchored on each estimator's $(1,1)$ optimum;
observed / predicted, range over four estimators × four arrival settings, (mean $|\log|$ error):

| drift | arm | constant $v$ | state-dependent $v_j(\rho)$ |
| --- | --- | --- | --- |
| original | sync | 0.95–1.12 (0.041) | 0.95–1.12 (0.041) |
| original | corr | 0.99–1.20 (0.073) | 0.99–1.10 (0.038) |
| away | sync | 0.94–1.09 (0.027) | 0.94–1.09 (0.028) |
| away | corr | 0.95–1.26 (0.095) | 0.92–1.17 (0.056) |
| toward | sync | 0.93–1.07 (0.030) | 0.93–1.07 (0.031) |
| toward | corr | 0.97–1.24 (0.094) | 0.93–1.15 (0.050) |

**Reading.**
- **The toward-zero "breakdown" was `hermite_ew`, not a property of the law.** Without it, the
  toward-zero fits are as good as the others. The prediction as stated is **falsified**: there
  was no toward-specific gap left to close.
- **State-dependent noise refines the corrected-arm law uniformly** (error roughly halved in every
  direction), because the corrected estimator works at the *attenuated* $\rho$, closer to the noise
  peak, and pays the Jacobian. It leaves `sync` unchanged: within one path, every arrival setting
  shares the same $v_j(\rho)$ profile, so it moves the level of $M^*$, not its $f$-scaling, and
  anchoring cancels it.
- So: $M^*\propto(f\bar c^2)^{-1/3}$ is the constant-noise special case; the refined statement is
  that $M^*$ is set by information rate, overlap loss, and the local estimator noise at the
  (attenuated) dependence level.

**`hermite_ew` startup bug.** With $\lambda>0$, `HermiteEstimator` applies weight $\lambda$ from
the first observation, so its coefficient sums start shrunk toward zero, its running mean sits on
the first observation and its variance on 1, all decaying only over ~$1/\lambda$ observations.
Harmless at the registry's $\lambda=0.01$ over 5,000 ticks; fatal when memory exceeds the data
available (e.g. $m=2560$ with ~1,350 pairs): biases up to +0.76, and an $M^*$ that is not
identified. Unbiased at full data rate (`async_noise`), which isolates it to the transient. Fix:
weight $\max(\lambda,1/n)$ — a core-estimator change, left for a separate decision.

## Caveats

- Gaussian copula and Bernoulli arrivals only. Sums of non-Gaussian copula increments are not
  the one-step copula, and the correction is the elliptical one.
- $\bar c$ is a cumulative running mean — fine here because arrival rates are constant; a
  liquidity *change* needs a forgetting $\bar c$ (phase 2 below).
- The correction is applied to the estimator output via the Gaussian $\rho_S\leftrightarrow r$ map;
  $\rho_S$ of a mixture of correlations is only approximately the map of the mean correlation.
- Only the `drift` slope of the existing harness, stretched to $T=20{,}000$. The $M^*$ values are
  specific to it; the ratios are the result.
- The $f^{-1/3}$ prediction comes from the processing-rate study's local-linear-drift model (not
  on `main`); only its scaling is checked here.

## Next

- Fix the `hermite_ew` startup weighting ($\max(\lambda,1/n)$), then rerun its cells.
- (Optional) overlap-weighted pairing, $f_\text{eff}=f\,\mathbb E[C^2]$.
- **Phase 2 — liquidity change.** Constant $\rho$, arrival rates drop mid-run. Naive estimates
  (and any detector on them) see a spurious dependence change; test whether a forgetting $\bar c$
  fixes the estimate and whether detectors false-alarm.
- Wire timestamp pairing into `src/consumer/` only after the above.
