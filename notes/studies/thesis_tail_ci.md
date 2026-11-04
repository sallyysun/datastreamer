# Online confidence intervals for the EW corner counter χ̂(q) (2026-11-04, in progress)

**Goal.** Online intervals for finite-q tail dependence, and their coverage under change. This is a
small extension of `thesis_bank_tails.md`.

**Planned method.** The online bootstrap of Nagler, Brock and Palm (2026), *Online Bootstrap Inference for
the Trend of Nonstationary Time Series*, UAI 2026, PMLR 337:4684–4711; arXiv 2602.23911; code at
github.com/Tobias-Brock/olbootstrapping.

- It covers level estimators that are weighted averages, exponential smoothing included.
- Each bootstrap replicate carries an autoregressive multiplier, so the state per replicate is fixed.
- It gives pointwise intervals and uniform-in-time bands.

**Status.**

- The paper and code could not be read: `arxiv.org`, `proceedings.mlr.press` and `github.com` do not
  resolve from this session.
- The bootstrap is therefore **not implemented yet**.
- What exists is the harness and the analytic baseline it must beat.

| | |
| --- | --- |
| Code | `experiments/probe_tail_ci.cpp` |
| Platform | Linux / g++ `-O2` |
| Seeds | 42–101 (60) |
| Run | `make probe_tail_ci EIGEN=/usr CXX=g++ && for r in gaussian gauss2clayton clayton2gumbel gauss2t gauss_ar t_persist gauss2t_dep; do ./probe_tail_ci $r 42 60; done` (5 s on 4 cores) |

## Analytic baseline

The counter is $\hat\chi=b/a$, with $a=\sum_s w_s1\{\hat U_s\le q\}$, $b=\sum_s w_s1\{\hat U_s\le q,\hat V_s\le q\}$
and EW weights $w_s=\alpha(1-\alpha)^{t-s}$, $\alpha=2/(M+1)$. Both inputs are indicators and the joint
event lies inside the marginal one. So the per-observation variance of the linearised error
$b-\chi a$ is exactly $b(1-\chi)$, and the delta method gives

$$\widehat{\mathrm{Var}}(\hat\chi)=\frac{S_2}{W_1}\cdot\frac{\hat\chi(1-\hat\chi)}{a},\qquad
W_1=\sum_s w_s,\; S_2=\sum_s w_s^2 .$$

- $W_1$ and $S_2$ are tracked recursively, so the variance is exact during warm-up.
- In steady state $S_2/W_1\to\alpha/(2-\alpha)$, which is the $\chi(1-\chi)/(qM)$ law of
  `tail_dependence.md`.
- Two extra doubles per memory; no bootstrap.
- It assumes the pseudo-observations are independent over time.

## Design

**Data.** Copula space, with every estimator on a 500-tick rank PIT. $\rho_S=0.582$; switch at
$\tau=5000$ of $T=10\,000$; scored from tick 1000. Memories M = 399, 1999, 4999; q = 0.05, 0.10; both
tails; nominal 90%.

**Independent regimes.**

- `gaussian` (no switch);
- `gauss2clayton`;
- `clayton2gumbel`;
- `gauss2t`.

**Serially dependent regimes.** Each has the same marginal copula; only the time dependence is added.

- `gauss_ar`: Gaussian copula, with the latent normal pair following AR(1), φ = 0.7.
- `t_persist`: $t_4$ copula whose $\chi^2$ mixing variable is redrawn with probability 0.05 per tick,
  so joint extremes cluster in time.
- `gauss2t_dep`: `gauss_ar` → `t_persist`.

**Two targets.**

- the instantaneous $\chi_t(q)$ of the regime in force;
- the smoothed $\bar\chi_t=\mathrm{EW}_\alpha(\chi_s(q))$, which is what an EW estimator actually
  estimates.

## Results: analytic interval, coverage of a nominal 90% (mean over 60 seeds and both tails)

| regime | target | M = 399 | M = 1999 | M = 4999 |
| --- | --- | --- | --- | --- |
| independent, any window | smoothed | 0.86–0.90 | 0.90–0.94 | 0.88–0.95 |
| `gauss_ar` (φ = 0.7) | smoothed | **0.78–0.80** | 0.84–0.87 | 0.85–0.91 |
| `t_persist` | smoothed | 0.86–0.89 | 0.87–0.93 | 0.89–0.93 |
| switches, first 500 ticks after | instantaneous | 0.59–0.86 | 0.00–0.59 | 0.00–0.28 |
| switches, 2000–5000 ticks after | instantaneous | 0.86–0.91 | 0.89–0.93 | 0.14–0.85 |

- **Under independence the analytic interval is already about right.** It is slightly conservative at
  M ≥ 1999. The rank PIT makes the marginal-corner count exact over its window, which removes some
  variance, the same effect seen in `thesis_bank_tails.md`.
- **Serial dependence breaks it.** On `gauss_ar` it under-covers at 0.78–0.87. This is the gap an
  online bootstrap with autoregressive multipliers is designed to close, and the main test for it.
- **The target matters more than the method after a change.** Coverage of the *smoothed* target stays
  near nominal through every switch. Coverage of the *instantaneous* $\chi_t$ collapses to 0–0.6 for
  hundreds to thousands of ticks, longer as M grows: about 2000 ticks at M = 1999 and 5000 at
  M = 4999. That is the adaptation bias. No variance estimate, bootstrap or analytic, can repair it,
  because the interval is centred on the wrong quantity. The fix is a shorter memory, or a bank such as
  the family bank, which follows a switch in about 200 ticks.

## Next, once the paper and code are reachable

1. Reproduce their exponential-smoothing bootstrap on a scalar series against their Python code.
2. Add it to `probe_tail_ci` as a second interval: pointwise, and a uniform-in-time band over a window.
3. Report coverage on the same grid, with the analytic baseline beside it. Then report width, and time
   and memory per tick against B.
