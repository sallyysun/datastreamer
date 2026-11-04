# Online confidence intervals for the EW corner counter χ̂(q) (2026-11-04)

**Goal.** Online intervals for finite-q tail dependence, and their coverage under change. This is a
small extension of `thesis_bank_tails.md`.

**Planned method.** The online bootstrap of Nagler, Brock and Palm (2026), *Online Bootstrap Inference for
the Trend of Nonstationary Time Series*, UAI 2026, PMLR 337:4684–4711; arXiv 2602.23911; code at
github.com/Tobias-Brock/olbootstrapping.

- It covers level estimators that are weighted averages, exponential smoothing included.
- Each bootstrap replicate carries an autoregressive multiplier, so the state per replicate is fixed.
- It gives pointwise intervals and uniform-in-time bands.

**Source.** Algorithm 1 is implemented from the paper's PDF (§4.2), which was uploaded to the session.
The authors' Python code was not reachable from this session, so the implementation has not been
cross-checked against it.

| | |
| --- | --- |
| Code | `experiments/probe_tail_ci.cpp` |
| Platform | Linux / g++ `-O2` |
| Seeds | 42–101 (60) |
| Run | `make probe_tail_ci EIGEN=/usr CXX=g++ && for r in gaussian gauss2clayton clayton2gumbel gauss2t gauss_ar t_persist gauss2t_dep; do ./probe_tail_ci $r 42 60; done` (about 1 min on 4 cores); figure `python3 plot_tail_ci.py` → `notes/thesis/figures/ch6_tail_ci.png` |

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

## The online bootstrap (Algorithm 1 of Nagler, Brock and Palm 2026), adapted to b/a

**Effective sample size.** For each memory, $\alpha=2/(M+1)$, the effective sample size is
$\nu=(2-\alpha)/\alpha$, and B = 100 replicates (B₁ = 20 for the scale, B₂ = 80 for the sup-quantile).

**Multipliers.**

- $Z^{(b)}_t=\rho Z^{(b)}_{t-1}+\sqrt{1-\rho^2}\,\zeta^{(b)}_t$, with $\rho=1-\nu^{-1/3}$, the
  paper's default persistence exponent $1/3$.
- $V^{(b)}_t=T(Z^{(b)}_t)$, with $T=t^{-1}_{2+\nu^{1/3}}\circ\Phi$ tabulated in $z$.
- The paper's **iid** baseline is the same algorithm with persistence exponent 0, i.e. $\rho=0$.

**Bootstrapped error.** The paper bootstraps $V^{(b)}_t(X_t-\hat\mu(t-1))$ through the same smoother.
$\hat\chi=b/a$ is a ratio, so it is linearised as in the analytic interval: the input is
$r_t=(J_t-b_{t-1})-\hat\chi_{t-1}(K_t-a_{t-1})$, where $J$ and $K$ are the joint and marginal corner
indicators, and $\hat\delta^{*(b)}_t=\mathrm{EW}_\alpha(V^{(b)}r)_t/a_t$.

**Pointwise interval.** $\hat\chi\pm z_{0.95}\hat\sigma^*$, where $\hat\sigma^*$ is the
standard deviation of replicates 1–20.

**Uniform band.** $\hat\chi\pm\hat q^*_k\hat\sigma^*$ over (t₁, T].

- $\hat q^*_k$ is the $(1-\alpha/K)$ quantile, over replicates 21–100, of the running maximum
  $\max_s|\hat\delta^*_s|/\hat\sigma^*_s$.
- It is recalibrated at $t_0+2^{k-1}(t_1-t_0)$: $t_0=600$ (burn-in), $t_1=1000$, K = 5 blocks up to
  T = 10 000.

**Timing.** Intervals are evaluated after tick t is absorbed, which pairs $\hat\chi(t)$ with
$\bar\chi(t)$ as the paper pairs $\hat\mu(t)$ with $\mu_\eta(t)$.

**Cost.** The extra state is B doubles per estimator for the errors, B for the running maxima, and B per
memory for the multipliers; at B = 100 that is about 2.4 kB per (q, M, tail). Time per tick is O(B).

## Results: bootstrap vs analytic, nominal 90%, smoothed target, 60 seeds

**Pointwise coverage, whole run, q = 0.05 / 0.10.**

| data | M | analytic | bootstrap, iid | bootstrap, AR |
| --- | --- | --- | --- | --- |
| Gaussian, independent | 399 | 0.88 / 0.90 | 0.89 / 0.91 | 0.89 / 0.90 |
| | 1999 | 0.92 / 0.92 | 0.91 / 0.91 | 0.91 / 0.91 |
| | 4999 | 0.92 / 0.92 | 0.91 / 0.92 | 0.91 / 0.91 |
| $t_4$, clustered extremes | 399 | 0.87 / 0.88 | 0.88 / 0.90 | 0.87 / 0.90 |
| | 1999 | 0.89 / 0.92 | 0.89 / 0.92 | 0.89 / 0.93 |
| | 4999 | 0.90 / 0.93 | 0.90 / 0.92 | 0.91 / 0.92 |
| **Gaussian, AR(1) φ = 0.7** | 399 | 0.80 / 0.79 | 0.81 / 0.81 | **0.85 / 0.87** |
| | 1999 | 0.85 / 0.85 | 0.85 / 0.85 | **0.91 / 0.92** |
| | 4999 | 0.86 / 0.87 | 0.86 / 0.86 | **0.93 / 0.94** |

**Uniform-in-time coverage over ticks 1000–10 000, both q pooled.**

| data | M | band, iid | band, AR |
| --- | --- | --- | --- |
| Gaussian, independent | 399 / 1999 / 4999 | 0.62 / 0.90 / 0.92 | 0.69 / 0.90 / 0.92 |
| $t_4$, clustered extremes | 399 / 1999 / 4999 | 0.63 / 0.89 / 0.92 | 0.68 / 0.87 / 0.93 |
| Gaussian, AR(1) φ = 0.7 | 399 / 1999 / 4999 | **0.25 / 0.72 / 0.83** | 0.52 / 0.89 / 0.94 |
| the four switches (smoothed target) | 399 / 1999 / 4999 | 0.49 / 0.86 / 0.90 | 0.61 / 0.88 / 0.92 |

**Width.** Mean full width over the whole run, both q.

- **Independent data.** The bootstrap pointwise interval matches the analytic one: 0.32 against 0.29 at
  M = 399, 0.14 at M = 1999, 0.11 at M = 4999.
- **AR process.** It is 25–30% wider (0.18 against 0.14 at M = 1999), which is the dependence
  correction.
- **Uniform band.** About 2.3–2.5× the pointwise width (0.35–0.43 at M = 1999).
- **M = 399.** The band is about 1.0 wide, the whole $[0,1]$ range, so uninformative.

**Findings.**

1. **Serial dependence is where the bootstrap earns its place.** With AR(1) latents, the analytic
   interval and the iid bootstrap both under-cover (0.79–0.87). The AR-multiplier bootstrap brings
   pointwise coverage to 0.91–0.94 at M ≥ 1999, and its band to 0.89–0.94, against 0.72–0.83 for the
   iid band. This reproduces, for a tail statistic, the paper's main finding, that iid multipliers fail
   under dependence.
2. **On independent data the bootstrap adds nothing over the analytic formula** pointwise. It is still
   the only way to get a calibrated *uniform* band, and its bands are nominal at M ≥ 1999.
3. **Small memory breaks everything.** At M = 399 there are only $qM\approx20$ corner events in
   memory. Every method under-covers there, and the bands do so badly (0.25–0.69). This is the
   small-$\nu$ caveat of the paper (§6.2), made worse by the discreteness of an indicator ratio. Tail
   intervals need $M\gtrsim 2000$ at q = 0.05.
4. **Inference cannot fix the adaptation bias.** Against the *instantaneous* $\chi_t$ after a switch,
   AR-bootstrap pointwise coverage is 0.29 (M = 1999) and 0.15 (M = 4999) in the first 500 ticks. The
   bands cover the instantaneous path in only 0.32–0.53 of seeds. Against the smoothed $\bar\chi_t$
   both stay near nominal (0.88–0.94). The interval is calibrated for what an EW estimator estimates;
   the gap to the instantaneous value is the bias side of the memory trade-off, and it needs a shorter
   memory or an adaptive estimator, not a better interval.

## Earlier result: analytic interval alone, coverage of a nominal 90% (mean over 60 seeds and both tails)

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

## Open

- Cross-check the implementation against the authors' code (github.com/Tobias-Brock/olbootstrapping) once
  it is reachable.
- Xiao's ρ_S (a nonlinear function of the counts) and the family bank need their own linearisation.
  Not attempted.
