# Asynchronous arrival and the optimal memory

**Question.** How does asynchronous information arrival change the optimal memory $M^*$ of an
online dependence estimator? Broader framing: how should an online estimator remember the past when
both the dependence *and the rate at which information arrives* can change?

Motivating paper: Chakrabarti & Sen (2022), *Copula Estimation for Nonsynchronous Financial Data*,
Sankhyā. Harness: `experiments/async_memory.cpp`; scaling check: `experiments/async_scaling.py`.
Deliberately scoped: Gaussian copula only, `constant` and `drift` only, no detectors, no
general-copula (Kendall/regression) correction.

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
- **Overlap.** For pair $k$, $c_k = |I^x_k\cap I^y_k|/\sqrt{|I^x_k||I^y_k|}\in[0,1]$. The paired
  scaled returns have correlation $\theta c_k$, so naive pairing sees $\theta\,\bar c$.
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
  $M^*_\text{clock}=M^*\times$ (clock steps per pair).

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

Predictions are anchored on each estimator's own $(1,1)$ optimum ($M^*_\text{clock}$ 636–711).
Example, `xiao_ew` $M^*_\text{clock}$: 664 (sync, 1,1) → 929 sync / 819 async / 1070 corr at (0.5,0.5)
→ 1820 / 1776 / 2546 at (0.1,0.1).

### `constant` — no interior $M^*$

Every `sync` and `corr` cell has its optimum at the grid edge (longest memory), as it must when
nothing changes. `async` sometimes shows an "interior" optimum, but its RMSE curve is flat to
<0.001 across the top of the grid (bias ≫ variance), so that $M^*$ carries no information.
At $M=2560$, `xiao_ew` bias: sync −0.009 / −0.012 / −0.011, **async −0.099 / −0.171 / −0.210**,
corr −0.010 / −0.017 / −0.018 at (0.5,0.5) / (0.25,0.25) / (0.1,0.1). Naive async RMSE is
3.5–8.4× the sync RMSE across the four settings; corrected is 1.3–1.7×.

## Reading

1. **Fewer pairs alone: the $f^{-1/3}$ law holds.** The `sync` arm's clock-time optimum follows
   $M^*_\text{clock}\propto f^{-1/3}$ within ~10% for all five estimators, over a 15× range of
   pair rates. Lower liquidity ⇒ longer memory in clock time, but *shorter* in pairs
   ($\propto f^{2/3}$). Memory must be specified in one unit; clock time is the one the law is stated in.
2. **Asynchronicity is a bias problem, not a memory problem.** Naive pairing raises the best
   achievable RMSE by 1.3–2× on `drift` and 3.5–8× on `constant`, and no memory choice recovers it.
   Its $M^*$ moves only slightly — and *shorter*, not longer.
3. **Why shorter (and why not general).** The estimate tracks $\bar c\rho_t$, so the error is
   lag $+\,(\bar c-1)\rho_t$. On this drift $\rho_t$ is mostly positive while rising, so the lag and the
   attenuation both point down; longer memory adds lag in the direction the bias already errs,
   which pushes $M^*$ down, outweighing the flatter-slope effect that pushes it up. The sign of
   the cross-term is $\operatorname{sign}(\dot\rho\cdot\bar\rho)$: a drift *toward* zero should
   shift $M^*$ the other way. Untested — do not quote the direction as a law.
4. **With the correction, overlap is a second rate.** Correcting removes the bias (residual
   bias ≈ the sync arm's) but divides the signal by $\bar c$, inflating variance by $\bar c^{-2}$.
   The same bias–variance calculation then gives
   $M^*_\text{clock}\propto f^{-1/3}\,\bar c^{-2/3}$, which matches within ~10%. Asynchronous
   overlap therefore acts like an *additional* loss of information rate, $f_\text{eff}=f\bar c^{2}$,
   on top of the loss of pairs — the answer to "does $f^{-1/3}$ still describe the shift?": yes,
   with $f$ replaced by $f\bar c^2$. The price is 1.1–1.4× the matched-rate RMSE.

## Caveats

- Gaussian copula and Bernoulli arrivals only. Sums of non-Gaussian copula increments are not
  the one-step copula, and the correction is the elliptical one.
- $\bar c$ is a cumulative running mean — fine here because arrival rates are constant; a
  liquidity *change* needs a forgetting $\bar c$ (phase 2 below).
- The correction is applied to the estimator output via the Gaussian $\rho_S\leftrightarrow r$ map;
  $\rho_S$ of a mixture of correlations is only approximately the map of the mean correlation.
- Only the `drift` slope of the existing harness, stretched to $T=20{,}000$. The $M^*$ values are
  specific to it; the ratios are the result.
- "The $f^{-1/3}$ processing-rate result" referenced in the handover is not on `main`; this study
  derives and checks it independently for the `sync` arm.

## Next

- **Phase 2 — liquidity change.** Constant $\rho$, arrival rates drop mid-run. Naive estimates
  (and any detector on them) see a spurious dependence change; test whether a forgetting $\bar c$
  fixes the estimate and whether detectors false-alarm.
- Drift toward zero, to test the sign argument in reading 3.
- Wire timestamp pairing into `src/consumer/` only after the above.
