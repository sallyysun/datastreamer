# Streaming changepoint detection for copula dependence: methodology and results

Companion to [change_detection.md](change_detection.md), which laid out the design space and build
order. This document reports what was actually **built and measured** against that plan: the
local-level Kalman filter, three detectors on top of it (CUSUM, window-limited GLR, BOCPD), the
$\mathrm{ARL}_0$ calibration and multi-seed replication infrastructure needed to make their numbers
mean anything, and the results — including two corrections to claims that did not survive scrutiny.

Branch: `kalman-changepoint` (off `main`). Everything here is reproducible from
`experiments/` — see [§9](#9-reproduction).

---

## 1. Summary of findings

1. **Bridge 1 is exact, not an analogy.** The local-level Kalman filter's steady-state gain matches
   the closed form $K = \tfrac{1}{2}\left(\sqrt{q^2+4q}-q\right)$ to five decimal places across four
   orders of magnitude in $q = \sigma_\omega^2/\sigma_v^2$. Every fixed-$\lambda$ estimator in this
   codebase *is* this filter at steady state, with $\lambda = 1-K$.

2. **BOCPD is the one genuine structural win — it survives fair tuning *and* misspecification.** On
   `jump`, `kalman_bocpd` achieves RMSE $0.0598 \pm 0.0022$ against the *best-tuned* Xiao
   ($0.0798$), Hermite ($0.0815$) and plain Kalman ($0.0840$) — a ~25% improvement that no choice of
   fixed $\lambda$ can replicate, because no fixed forgetting rate can be simultaneously long-memory
   before a break and short-memory after it. [§10](#10-misspecification-the-multi-seed-grid-across-dgp-families)
   extends this to all four DGP families: it is best in column on `jump` for Gaussian, Clayton,
   Gumbel and $t_4$ alike ($0.0598$/$0.0602$/$0.0589$/$0.0633$), which is what a distribution-free
   structural claim should do.

3. **The Kalman filter's apparent `oscillate` win was a tuning artifact — corrected.** Compared at
   default $\lambda$, `kalman_fast` looked like it beat every nonparametric estimator on
   `oscillate`. Tuned fairly, Xiao ($0.1826$) and Hermite ($0.1879$) beat the *best achievable*
   Kalman filter ($0.2256$) by ~19%, and the Kalman curve plateaus — it cannot be tuned past the
   structural cost of block aggregation.

4. **Hard detect-and-restart costs accuracy.** `kalman_cusum`/`kalman_glr` have *worse* RMSE than
   the un-reset filter they wrap, on both `jump` and `oscillate`. By alarm time the filter has
   already partially adapted; the hard reset throws that away and forces a re-climb from a flat
   prior. This is the concrete instance of change_detection.md Part V's claim that VFF and
   detect-and-restart are one dial, not rivals.

5. **Misspecification is real but asymmetric, and the Gaussian copula is a forgiving wrong answer.**
   `gaussian_mle_ew` degrades only mildly off-family (`stochastic`: $0.0558$ on Gaussian →
   $0.0592$–$0.0680$ elsewhere), while `clayton_mle` on Gumbel data is 10× worse than on its own
   ($0.1589$ vs $0.0154$). Online GoF selection (`copula_select`) recovers the generating family
   exactly when run cumulatively, but churns under forgetting — see §10.3.

6. **Four real bugs were found by verification, not by inspection** — see [§7](#7-bugs-found-and-fixed).
   Two were caught only because the replication tier existed: the BOCPD burn-in alarms (multi-seed)
   and a NaN latch in `gaussian_mle` (cross-family grid). Both were invisible at one seed on one
   DGP and unmistakable across twenty.

---

## 2. Models built

| Key | Family | Description |
| --- | --- | --- |
| `kalman`, `kalman_fast` | Kalman filter | Local-level filter on Fisher-$z$'d blockwise Spearman; $\sigma_\omega^2 = 10^{-4}$ / $10^{-2}$ |
| `kalman_cusum` | Hard-reset detector | Two-sided CUSUM on standardized innovations; calls `reset()` on alarm |
| `kalman_glr` | Hard-reset detector | Window-limited GLR (unknown jump size) via matched filter on the step response |
| `kalman_bocpd`, `kalman_bocpd_fast` | Bayesian | Run-length posterior (Adams & MacKay 2007); hazard $1/1000$ / $1/10$; **no reset** |

Pre-existing baselines used for comparison: `xiao`, `xiao_ew`, `xiao_win`, `hermite`, `hermite_ew`,
`windowed_rank`, `gaussian_mle`, `gaussian_mle_ew`, and — from the parametric-copula branch, merged
in for [§10](#10-misspecification-the-multi-seed-grid-across-dgp-families) — `clayton_mle`,
`gumbel_mle`, `t_mle`, `copula_select` and their `_ew` variants. 22 estimators in total; §5 compares
the 13 Gaussian-DGP ones, §10 all 22 across all four DGP families.

**Bayesian vs frequentist.** `kalman`/`kalman_fast` are exact recursive Bayes for a linear-Gaussian
model (they maintain the full posterior $p(z_t \mid y_{1:t}) = \mathcal N(\hat z_t, P_t)$).
`kalman_bocpd`/`kalman_bocpd_fast` are fully Bayesian (posterior over run length, conjugate priors,
model averaging). `kalman_cusum`/`kalman_glr` are **hybrid**: a Bayesian filter with a frequentist
sequential test bolted on. All baselines are frequentist. Note that the $\mathrm{ARL}_0$ calibration
in §3.7 is a frequentist criterion applied to *every* detector, including the Bayesian ones.

---

## 3. Methodology

### 3.1 The observation

All four detectors share one observation channel. Every $m$ ticks (a *block*), compute the empirical
Spearman correlation $\hat\rho^S$ of the block's $m$ pairs and Fisher-transform it:

$$
y_t = \operatorname{atanh}\left(\hat\rho^S_t\right),
\qquad
\sigma_v^2 \;\approx\; \frac{1.06}{m-3}.
$$

The variance is the classical Fieller–Hartley–Pearson approximation for a Fisher-$z$'d Spearman
correlation — inflated ~6% over Pearson's $1/(n-3)$. That $\sigma_v^2$ is **known in closed form** is
the key structural gift: it pins the denominator of the signal-to-noise ratio, leaving a *single*
scalar to choose.

*Why blockwise rather than per-tick:* the per-tick normal-score product $z_xz_y$ has mean $\rho$ and
variance $1+\rho^2$, and the link $\rho = \tanh z$ is nonlinear — an exact per-tick Kalman filter
does not exist without an EKF/UKF. Blockwise, the model is exactly linear-Gaussian. §6.1 quantifies
what this convenience costs.

*Marginal invariance (free).* $\hat\rho^S$ is a rank statistic, so it is invariant to any monotone
transformation of either marginal. This makes the efficient-score refinement of
change_detection.md Part IV.2 unnecessary for this detector line — verified in §4.4.

### 3.2 The local-level Kalman filter

State $z_t = \operatorname{atanh}\rho_t$; the model is a random walk observed in noise:

$$
z_t = z_{t-1} + \omega_t, \qquad \omega_t \sim \mathcal N(0, \sigma_\omega^2)
$$
$$
y_t = z_t + v_t, \qquad\;\; v_t \sim \mathcal N(0, \sigma_v^2)
$$

Per closed block, the recursion is:

$$
\begin{aligned}
\text{predict:} \quad & \hat z_{t|t-1} = \hat z_{t-1}, \qquad P_{t|t-1} = P_{t-1} + \sigma_\omega^2 \\
\text{innovation:} \quad & e_t = y_t - \hat z_{t|t-1}, \qquad F_t = P_{t|t-1} + \sigma_v^2 \\
\text{gain:} \quad & K_t = \frac{P_{t|t-1}}{F_t} = \frac{P_{t|t-1}}{P_{t|t-1} + \sigma_v^2} \\
\text{update:} \quad & \hat z_t = \hat z_{t|t-1} + K_t e_t, \qquad P_t = (1-K_t)P_{t|t-1}
\end{aligned}
$$

and the reported estimate is $\hat\rho_t = \tanh \hat z_t$.

The gain is a precision-weighted average — Gaussian Bayes' rule. When the filter is uncertain
($P_{t|t-1} \gg \sigma_v^2$) it moves to the data; when confident it ignores noisy reads. This
handles cold-start automatically, which is the principled version of the ad-hoc `hermite_ew`
cold-start patch.

### 3.3 Bridge 1: the filter generalizes every fixed-$\lambda$ estimator

At steady state $P_t \to P^\ast$ and $K_t \to K$, and the update collapses to

$$
\hat z_t = (1-K)\hat z_{t-1} + K y_t,
$$

an EWMA with $\lambda = 1-K$, where

$$
K = \frac{\sqrt{q^2+4q}-q}{2},
\qquad
q = \frac{\sigma_\omega^2}{\sigma_v^2}.
$$

Since $K$ depends only on the ratio $q$, and $\sigma_v^2$ is known, choosing $\sigma_\omega^2$ *is*
choosing $\lambda$ — but $\sigma_\omega^2$ is a physical quantity ("how fast does dependence actually
drift?") that data can in principle speak to, whereas $\lambda$ is an arbitrary constant. Estimating
$\sigma_\omega^2$ online is the variable-forgetting-factor problem (still open).

### 3.4 Innovation CUSUM

Under the model the standardized innovations are white:
$\tilde e_t = e_t/\sqrt{F_t} \sim \mathcal N(0,1)$ i.i.d. The filter absorbs gradual drift; what it
*cannot* absorb shows up as a departure from standard normal. Two one-sided CUSUMs:

$$
S^{+}_t = \max\left(0,\; S^{+}_{t-1} + \tilde e_t - k\right),
\qquad
S^{-}_t = \max\left(0,\; S^{-}_{t-1} - \tilde e_t - k\right),
$$

alarm when either exceeds $h$, with reference value $k = \delta/2$. The increment
$\tilde e_t - \delta/2$ is exactly the log-likelihood-ratio for
$\mathcal N(0,1)$ vs $\mathcal N(\delta,1)$ — CUSUM as the max-product (Viterbi) recursion over a
two-state absorbing chain. On alarm, the filter's `reset()` is called.

### 3.5 Window-limited GLR

CUSUM assumes a *known* shift size $\delta$. GLR handles unknown size by matched-filtering the
filter's own **step response**. For this filter a unit jump at $\tau$ contributes

$$
\varrho_s(\tau) = (1-K)^{\,s-\tau}, \qquad s \ge \tau,
$$

derived from the bias recursion $b_s = (1-K)b_{s-1} + K\nu$, giving
$e_s = \nu(1-K)^{s-\tau}$. The statistic over candidate lags $j$:

$$
A(j) = \sum_{i=0}^{j} (1-K)^i \tilde e_{t-i},
\qquad
N(j) = \sum_{i=0}^{j} (1-K)^{2i},
\qquad
g_t = \max_{j} \frac{A(j)^2}{N(j)},
$$

with $\hat\nu = A(j^\ast)/N(j^\ast)$ estimating the jump size. Accumulators update by the
shift-register recurrence $A_t(j) = \tilde e_t + (1-K)A_{t-1}(j-1)$, and $N(j)$ is precomputed once.
Restricting $j \le W$ (Lai 1995) makes this $O(W)$ per block — the unbounded $\sup_\tau$ GLR is
$O(t)$ and **not streaming-admissible**.

### 3.6 BOCPD

Run length $r_t$ = observations since the last changepoint, constant hazard $H$. Each hypothesis
carries a conjugate Normal posterior on its segment mean $\mu$ (known $\sigma_v^2$, prior
$\mathcal N(\mu_0,\tau_0^2)$), so with $n$ observations summing to $S_y$:

$$
\tau_n^2 = \left(\frac{1}{\tau_0^2} + \frac{n}{\sigma_v^2}\right)^{-1},
\qquad
\mu_n = \tau_n^2\left(\frac{\mu_0}{\tau_0^2} + \frac{S_y}{\sigma_v^2}\right),
\qquad
y \mid \text{data} \sim \mathcal N\!\left(\mu_n,\; \tau_n^2 + \sigma_v^2\right).
$$

The recursion, with $L(r)$ the predictive density of $y_t$ under hypothesis $r$:

$$
P(r_t = 0,\, y_{1:t}) = H \sum_r P(r_{t-1}=r,\,y_{1:t-1})\,L(r)
$$
$$
P(r_t = r+1,\, y_{1:t}) = (1-H)\,P(r_{t-1}=r,\,y_{1:t-1})\,L(r)
$$

computed in log-space with log-sum-exp, truncated at $r \le R_{\max}$ for $O(R_{\max})$ per block.
The estimate is the **run-length-weighted posterior mean** (Bayesian model averaging):

$$
\hat\rho_t = \tanh\left(\sum_r P(r_t = r \mid \text{data})\,\mu_{n(r)}\right).
$$

There is no hard reset anywhere — the soft mixing *is* the mechanism.

> ⚠️ **$P(r_t=0)$ is useless as an alarm signal under a constant hazard.** Writing
> $S = \sum_r P(r_{t-1}=r)L(r)$, the reset branch carries mass $H\cdot S$ and the growth branches
> total $(1-H)\cdot S$ — the same $S$. It cancels in normalization, giving
> $$P(r_t = 0 \mid \text{data}) = \frac{HS}{HS + (1-H)S} = H$$
> **identically, on every tick, independent of the data.** Verified empirically (flat at exactly
> $0.0010 = 1/1000$ across a real jump). The detectable signal lives in how $(1-H)$ redistributes
> *within* the growth branch, so the alarm rule instead thresholds
> $P(r_t \le c \mid \text{data})$ with cutoff $c = 2$.

### 3.7 $\mathrm{ARL}_0$ calibration

change_detection.md Part VI: detectors are only comparable at matched in-control average run length,
and because of rank/PIT feedback $\mathrm{ARL}_0$ must be **bootstrapped on the actual
pseudo-observation stream**, not read off a formula. `experiments/calibrate_arl0.cpp` runs each
detector's real `update()` path on stationary Gaussian-copula streams at
$\rho \in \{0.0, 0.3, 0.6, 0.8\}$ (6 seeds $\times$ 5000 blocks per cell), sweeps the threshold, and
selects the smallest one whose **worst case over $\rho$** clears a target of 1000 blocks. Worst-case
selection matters because the true $\rho$ is unknown in live use.

Where the bare-minimum threshold sat exactly at the target with no margin, the next grid point was
taken instead — a deliberate conservatism-for-margin trade, applied uniformly.

### 3.8 Multi-seed replication

`experiments/multiseed.cpp` runs the full estimator set across $n$ seeds (20 used throughout here),
reporting RMSE and — for every `ChangeDetector` — alarm count, false alarms before the jump, and
detection delay as mean $\pm$ standard error, to `results/multiseed_<scenario>[_<family>].csv`.
Scoring matches `./experiment` exactly ($N=5000$, burn-in 500) so numbers are directly comparable.
Both harnesses take `COPULA=gaussian|clayton|gumbel|t`; `run_multiseed_grid.sh` sweeps the full
$5 \times 4$ grid used in §10.

⚠️ **Correction — what multi-seed actually covered.** §8 of this document previously claimed
multi-seed covered "this detector line only", with the Xiao/Hermite/`gaussian_mle` families left
single-seed. **That was wrong when written.** `multiseed.cpp` has always shared its `keys` list with
`main.cpp`, so those families were multi-seeded from the first run; the numbers in §5.1 were never
single-seed. The genuinely uncovered cases were narrower and different in kind:

- `xiao_win` — registered in `registry.hpp` but absent from *both* harnesses' `keys` lists, so it
  had never been run at all, at any number of seeds (§7, bug 5). Now in both, and in §5.1.
- The 8 parametric-copula keys and the three non-Gaussian DGP families — single-seed only, because
  they lived on an unmerged branch. Now covered at 20 seeds in §10.

This correction is recorded rather than silently edited for the same reason as §5.4: a caveat that
overstates a limitation is still a wrong statement about what the harness did, and the next reader
would otherwise re-run work that was already done.

---

## 4. Verification

Each component was checked against an independent reference before being built on.

### 4.1 Bridge 1 identity — exact

Steady-state gain vs the closed form, stationary stream, $m=20$:

| $\sigma_\omega^2$ | $q$ | $K$ (filter) | $K$ (closed form) | difference |
| --- | --- | --- | --- | --- |
| $10^{-5}$ | 0.00044 | 0.02084 | 0.02084 | 0.00000 |
| $10^{-4}$ | 0.00443 | 0.06441 | 0.06441 | 0.00000 |
| $10^{-3}$ | 0.04434 | 0.18956 | 0.18956 | 0.00000 |
| $10^{-2}$ | 0.44340 | 0.48012 | 0.48012 | 0.00000 |
| $10^{-1}$ | 4.43396 | 0.84063 | 0.84063 | 0.00000 |

### 4.2 GLR step response — matches theory through the noise

Monte Carlo, 2000 seeds, real block-Spearman noise, jump $\rho: 0.2 \to 0.85$. Theoretical decay
ratio $(1-K) = 0.672$; empirical ratios between consecutive post-jump blocks: 0.678, 0.652, 0.690,
0.690, 0.657, 0.735, 0.626, 0.683 — consistent at every lag.

### 4.3 BOCPD jump recovery

Point estimate tracks $0.19 \to 0.85$ within two blocks of a synthetic jump with no reset;
$P(r_t \le 2)$ spikes $0.0015 \to 0.53 \to 0.9999$ then returns to baseline. MAP run length collapses
to 1 exactly at the changepoint.

### 4.4 Marginal invariance (distinction #2) — confirmed, no fix needed

Applied a large purely-marginal shift ($x \mapsto 5x+100$, $y \mapsto 0.1y-50$, both monotone) with
the copula held *exactly* constant. Result: **zero alarms** before or after; the CUSUM statistic
never left the noise floor (peak 3.19 vs threshold 3.5); the estimate stayed at 0.468 against a true
$\rho^S$ of 0.483. This is a structural property of the rank-based observation, so it transfers to
every detector in this line without a separate check.

---

## 5. Results

### 5.1 Tracking accuracy (RMSE vs true Spearman $\rho$)

20 seeds, $N=5000$, burn-in 500, mean $\pm$ SE. **Bold** = best in column.

⚠️ Every row here is at its **configured** setting, so "best in column" means best among these
configurations — *not* best achievable. §5.4 shows the distinction matters: a well-tuned Xiao
reaches 0.1826 on `oscillate`, far below the 0.2695 bolded here. Read this table as "what the
shipped configs do", and §5.4 as "what each family can do". Conflating the two is exactly the
mistake §5.4 corrects.

| Model | `constant` | `drift` | `jump` | `stochastic` | `oscillate` |
| --- | --- | --- | --- | --- | --- |
| `xiao` | 0.0128±0.0015 | 0.3893±0.0027 | 0.3199±0.0020 | 0.1539±0.0140 | 0.4107±0.0003 |
| `xiao_ew` | 0.0470±0.0011 | 0.0669±0.0014 | 0.0798±0.0014 | 0.0594±0.0044 | 0.3484±0.0015 |
| `xiao_win` | 0.0275±0.0013 | 0.0784±0.0020 | 0.1183±0.0018 | 0.0835±0.0067 | 0.3636±0.0005 |
| `hermite` | **0.0121±0.0012** | 0.3902±0.0029 | 0.3179±0.0021 | 0.1524±0.0141 | 0.4107±0.0003 |
| `hermite_ew` | 0.0520±0.0013 | 0.0700±0.0016 | 0.0815±0.0014 | 0.0630±0.0037 | 0.3490±0.0015 |
| `windowed_rank` | 0.0271±0.0013 | 0.0778±0.0020 | 0.1170±0.0018 | 0.0832±0.0069 | 0.3632±0.0005 |
| `gaussian_mle` | 0.0129±0.0014 | 0.4044±0.0041 | 0.3157±0.0019 | 0.1535±0.0148 | 0.4112±0.0003 |
| `gaussian_mle_ew` | 0.0375±0.0011 | **0.0624±0.0015** | 0.0745±0.0011 | **0.0558±0.0047** | 0.3508±0.0018 |
| `kalman` | 0.0208±0.0010 | 0.1274±0.0021 | 0.1288±0.0015 | 0.0956±0.0083 | 0.4153±0.0005 |
| `kalman_fast` | 0.0712±0.0012 | 0.0921±0.0014 | 0.0880±0.0014 | 0.0735±0.0050 | 0.3166±0.0023 |
| `kalman_cusum` | 0.0721±0.0014 | 0.0930±0.0017 | 0.1146±0.0027 | 0.0792±0.0044 | 0.3477±0.0025 |
| `kalman_glr` | 0.0721±0.0014 | 0.0928±0.0016 | 0.1110±0.0032 | 0.0776±0.0047 | 0.3556±0.0029 |
| `kalman_bocpd` | **0.0189±0.0021** | 0.1214±0.0024 | **0.0598±0.0022** | 0.0856±0.0059 | 0.3683±0.0027 |
| `kalman_bocpd_fast` | 0.0613±0.0027 | 0.0961±0.0021 | 0.0760±0.0023 | 0.0800±0.0052 | **0.2695±0.0020** |

### 5.2 Detection metrics (`jump`, changepoint at $t=2500$)

| Detector | alarms/run | false alarms before $\tau$ | detection delay (ticks) | seeds detecting |
| --- | --- | --- | --- | --- |
| `kalman_cusum` | 1.10±0.10 | 0.05±0.05 | 46.4±4.9 | 19/20 |
| `kalman_glr` | 0.90±0.10 | 0.00±0.00 | 49.6±6.9 | 17/20 |
| `kalman_bocpd` | 1.45±0.20 | 0.10±0.07 | 31.9±3.4 | 17/20 |
| `kalman_bocpd_fast` | 0.85±0.22 | 0.10±0.07 | **25.7±3.3** | 9/20 |

Note the trade `kalman_bocpd_fast` makes: fastest mean delay *when* it fires, but it detects in fewer
than half the seeds. Detection **speed** and detection **reliability** are separate axes.

*Why the fast hazard detects less reliably.* A high hazard raises the stationary baseline of the
alarm statistic, because "the run is short" is already a priori common. Measured on a stationary
stream:

| hazard | baseline $P(r_t \le 2)$ | at the jump | ratio |
| --- | --- | --- | --- |
| $1/1000$ | 0.0014 – 0.0027 (flat) | $\to 0.53 \to 0.9999$ | ~330× |
| $1/10$ | 0.14 – 0.26 (noisy) | $\to 0.99 \to 0.9999$ | ~6× |

Calibration must then push the threshold far higher (0.95 vs 0.15) to hold the same
$\mathrm{ARL}_0$, which leaves a genuine jump much less headroom to clear it. Same evidence, thinner
margin — hence the missed seeds.

### 5.2b Alarm rates across all scenarios

Alarms per 5000-tick run (250 blocks), 20 seeds:

| Detector | `constant` | `drift` | `jump` | `stochastic` | `oscillate` |
| --- | --- | --- | --- | --- | --- |
| `kalman_cusum` | 0.05±0.05 | 0.15±0.08 | 1.10±0.10 | 0.20±0.09 | 24.75±0.18 |
| `kalman_glr` | 0.05±0.05 | 0.05±0.05 | 0.90±0.10 | 0.15±0.08 | 17.50±0.52 |
| `kalman_bocpd` | 0.35±0.13 | 0.30±0.18 | 1.45±0.20 | 0.40±0.13 | 17.00±0.98 |
| `kalman_bocpd_fast` | 0.10±0.10 | 0.20±0.09 | 0.85±0.22 | 0.20±0.12 | 4.40±0.56 |

All four are appropriately quiet on `constant`/`drift`/`stochastic` (consistent with the calibrated
$\mathrm{ARL}_0 \approx 2000$ blocks against a 250-block run), fire about once on `jump`, and fire
repeatedly on `oscillate` — correctly, since each half-period genuinely *is* a large change. That
`drift` produces almost no alarms is the intended division of labour: the filter absorbs gradual
drift, so there is nothing left for the detector to flag.

### 5.3 Calibrated thresholds

Target $\mathrm{ARL}_0 = 1000$ blocks, worst case over $\rho$.

| Detector | swept parameter | bare minimum | **chosen** | worst-case $\mathrm{ARL}_0$ | margin |
| --- | --- | --- | --- | --- | --- |
| `kalman_cusum` | $h$ | 3.5 | **3.5** | 2143 | 2.1× |
| `kalman_glr` | $h$ | 10.0 | **10.0** | 2727 | 2.7× |
| `kalman_bocpd` | $P(r\le2)$ threshold | 0.10 | **0.15** | 1875 | 1.9× |
| `kalman_bocpd_fast` | $P(r\le2)$ threshold | 0.90 | **0.95** | 2143 | 2.1× |

$\mathrm{ARL}_0$ rises roughly exponentially in the threshold, as CUSUM theory predicts (e.g.
`kalman_cusum`: $h=1 \to 7$ blocks; $h=2.5 \to 150$; $h=3.5 \to 2143$; $h=5 \to \ge 30{,}000$).

### 5.4 Fair per-family tuning — the correction

Each family's own forgetting parameter swept on the same scenario, 20 seeds:

**`oscillate`** (period 400):

| Xiao ($\lambda$) | RMSE | Hermite ($\lambda$) | RMSE | Kalman ($\sigma_\omega^2$) | RMSE |
| --- | --- | --- | --- | --- | --- |
| 0.99 | 0.3484 | 0.01 | 0.3490 | $10^{-4}$ | 0.4153 |
| 0.98 | 0.2642 | 0.02 | 0.2656 | $10^{-3}$ | 0.4058 |
| **0.95** | **0.1826** | **0.05** | **0.1879** | $10^{-2}$ | 0.3166 |
| 0.92 | 0.1939 | 0.10 | 0.2210 | $3\times10^{-2}$ | 0.2592 |
| 0.90 | 0.2110 | 0.15 | 0.2685 | $10^{-1}$ | 0.2279 |
| 0.85 | 0.2581 | 0.20 | 0.3094 | **$3\times10^{-1}$** | **0.2256** |

**`jump`**:

| Xiao ($\lambda$) | RMSE | Hermite ($\lambda$) | RMSE | Kalman ($\sigma_\omega^2$) | RMSE |
| --- | --- | --- | --- | --- | --- |
| **0.99** | **0.0798** | **0.01** | **0.0815** | $10^{-4}$ | 0.1288 |
| 0.98 | 0.0834 | 0.02 | 0.0887 | **$10^{-3}$** | **0.0840** |
| 0.95 | 0.1187 | 0.05 | 0.1291 | $10^{-2}$ | 0.0880 |
| 0.90 | 0.1714 | 0.10 | 0.1849 | $10^{-1}$ | 0.1298 |

Against these best-tuned baselines, `kalman_bocpd`'s **0.0598** on `jump` still wins by ~25%.

### 5.5 BOCPD hazard rate as a tuning knob

Hazard is BOCPD's analog of $\sigma_\omega^2$ — and no single value is optimal across regimes:

| hazard $H$ | `constant` | `drift` | `jump` | `oscillate` |
| --- | --- | --- | --- | --- |
| $1/1000$ | **0.0189** | 0.1214 | **0.0598** | 0.3683 |
| $1/20$ | 0.0440 | **0.0918** | 0.0642 | 0.2889 |
| $1/10$ | 0.0613 | 0.0961 | 0.0760 | 0.2695 |
| $1/5$ | 0.0884 | 0.1148 | 0.0984 | 0.2487 |
| $1/2$ | 0.1387 | 0.1661 | 0.1449 | 0.2285 |
| $1/1.4$ | 0.1572 | 0.1859 | 0.1622 | **0.2278** |

`kalman_bocpd_fast` uses $H = 1/10$, chosen to match `oscillate`'s true half-period (10 blocks)
rather than the numerically optimal $\approx 1/1.4$ — a principled setting tied to a real DGP
timescale, not one fitted to the test scenario.

---

## 6. Analysis

### 6.1 What block aggregation costs

The linear-Gaussian tractability of §3.1 is bought with two structural penalties, neither removable
by tuning:

- **Lag.** A statistic computed from the last $m$ ticks describes their *average*, centred $\approx
  m/2$ ticks in the past. At $m=20$ that is a ~10-tick delay baked into the observation before the
  filter runs.
- **A noise floor.** Each observation carries $\sigma_v^2 \approx 1.06/(m-3) \approx 0.062$, and the
  filter's innovation variance obeys $F_t = P_{t|t-1} + \sigma_v^2 \ge \sigma_v^2$ always. Even a
  fully-converged filter cannot read more precisely than one 20-point correlation allows.

These pull opposite ways on $m$ (larger $m$: less noise, more lag), so $m=20$ is a compromise, not a
free choice. Measured directly on `jump`: `kalman_fast` scores 0.0921 over all ticks but 0.0828 when
read only at block boundaries — so roughly half its deficit versus `xiao_ew` (0.0798) is the
staircase hold, and the remainder is the noise floor. **The `oscillate` plateau in §5.4 is this
ceiling made visible**: as $\sigma_\omega^2 \to \infty$, $K \to 1$ and the filter simply echoes its
observation — which still carries both penalties.

Per-tick estimators (`xiao_ew`, `hermite_ew`) never batch, so they pay neither.

The block-size sweep that set $m=20$ shows both penalties trading off directly (single seed, RMSE):

| $m$ | $\sigma_\omega^2$ | `constant` | `drift` | `jump` | `stochastic` | `oscillate` |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | $10^{-2}$ | 0.0890 | 0.1130 | 0.1034 | 0.0879 | **0.2681** |
| 20 | $10^{-2}$ | 0.0713 | 0.0904 | 0.0943 | 0.0755 | 0.3247 |
| 50 | $10^{-2}$ | **0.0493** | **0.0657** | **0.0885** | **0.0654** | 0.4447 |

Large $m$ wins everywhere the signal is slow (less observation noise) and loses badly on `oscillate`
(more lag, and the within-block-constant assumption breaks outright when $\rho$ moves substantially
inside one block). $m=20$ is the compromise; at $m=50$ the filter was *worse than non-adaptive
estimators* on `oscillate`, which is how that default was caught.

### 6.2 Why BOCPD's win is structural and the Kalman filter's was not

A fixed-$\lambda$ estimator must commit to one memory length: long memory gives low variance while a
regime is stable but slow adaptation after a break; short memory does the reverse. Any single
$\lambda$ is one point on that trade-off, which is why sweeping $\lambda$ per scenario (§5.4) closed
the `oscillate` gap entirely — `kalman_fast` was simply at a better point on the *same* curve than
`xiao_ew`'s default, not on a better curve.

BOCPD is not on that curve at all. Its run-length posterior *switches* effective memory length based
on evidence: long while the segment persists, collapsing to short the moment the data stop fitting.
No fixed $\lambda$ can do that at any tuning, which is exactly why this gap survives fair tuning and
the other did not.

### 6.3 The tuning sweeps are a direct measurement of Part 0's prediction

[change_detection.md](change_detection.md) Part 0 derives the trade-off theoretically. Writing
$\mathrm{ESS} \approx (2-\alpha)/\alpha$ for the effective sample size of a geometric-forgetting
estimator:

$$
\mathrm{MSE}(\mathrm{ESS}) \;\approx\;
\underbrace{\frac{\sigma^2}{\mathrm{ESS}}}_{\text{variance}}
\;+\;
\underbrace{\left(\dot\theta \, \frac{\mathrm{ESS}}{2}\right)^{\!2}}_{\text{lag}},
$$

whose minimiser depends on $\dot\theta$ — how fast dependence is currently moving. The prediction is
that the optimal memory length is a **property of the regime, not of the estimator**, and that it
should shorten as $\dot\theta$ grows.

The §5.4 sweeps measure exactly this. Converting each family's optimum to $\mathrm{ESS}$ (Xiao:
$\alpha = 1-\lambda$; Hermite: $\alpha = \lambda$ — the mirror-image convention Part 0 tabulates):

| Scenario | Xiao optimum | Hermite optimum | implied $\mathrm{ESS}$ | regime $\dot\theta$ |
| --- | --- | --- | --- | --- |
| `constant` | cumulative | cumulative | $\to \infty$ | zero |
| `jump` | $\lambda = 0.99$ | $\lambda = 0.01$ | $\approx 199$ ticks | zero except at one instant |
| `oscillate` | $\lambda = 0.95$ | $\lambda = 0.05$ | $\approx 39$ ticks | large, continuous |

**Two independent estimator families land on the same $\mathrm{ESS}$ in each regime** — 199 ticks on
`jump`, 39 ticks on `oscillate` — despite having different internals, different bin/basis
representations, and reciprocal parameter conventions. That is the theory's central claim confirmed
quantitatively: the optimum is set by the data-generating process, not the estimator, and it shortens
by a factor of ~5 as $\dot\theta$ goes from "rare jump" to "continuously oscillating".

It also sharpens why §6.2's BOCPD result matters. Part 0's boxed conclusion — *no fixed $\lambda$
dominates, so the estimator must set its own memory* — is a statement about a curve every
fixed-$\lambda$ estimator is stuck on. The `oscillate` correction in §5.4 shows `kalman_fast` was
merely at a **better point on that same curve** than `xiao_ew`'s default, which is why sweeping
$\lambda$ erased its advantage entirely. BOCPD is the one estimator here that is not on the curve at
all, and correspondingly the one whose advantage survived.

### 6.4 Why hard reset costs RMSE

By the time an alarm fires (~46 ticks on `jump`), the un-reset filter has already partly adapted
toward the new level. `reset()` discards that and restarts from a flat prior, so the estimator
re-climbs ground it had already covered. Hence `kalman_cusum` (0.1146) and `kalman_glr` (0.1110) both
score *worse* than the `kalman_fast` (0.0880) they wrap. BOCPD, which never resets, avoids this
entirely — direct support for treating detect-and-restart and adaptive forgetting as one dial at
different decision hardness (change_detection.md Part V), with soft mixing the better setting here.

### 6.4 Practical guidance

| Regime | Best choice | RMSE |
| --- | --- | --- |
| Stationary (`constant`) | `hermite` / `xiao` (cumulative) | ~0.012 |
| Gradual or noisy (`drift`, `stochastic`) | `gaussian_mle_ew` | 0.056–0.062 |
| Rare discrete jumps (`jump`) | `kalman_bocpd` | 0.060 |
| Fast repeated switching (`oscillate`) | well-tuned `xiao`/`hermite`, else `kalman_bocpd_fast` | 0.18 / 0.27 |

Absent knowledge of the regime, `gaussian_mle_ew` is the cheap, safe default. Averaged (unweighted)
across all five scenarios:

| Model | mean RMSE |
| --- | --- |
| `gaussian_mle_ew` | **0.1162** |
| `kalman_bocpd_fast` | 0.1166 |
| `xiao_ew` | 0.1203 |
| `hermite_ew` | 0.1231 |
| `kalman_fast` | 0.1283 |
| `kalman_bocpd` | 0.1308 |
| `kalman_cusum` | 0.1413 |
| `kalman_glr` | 0.1418 |
| `kalman` | 0.1576 |

The sophisticated machinery does not beat a plain EWMA *on average*; it **redistributes** accuracy
toward change-heavy regimes. Its justified use is where discrete regime change is the thing you
actually care about detecting — not as a general-purpose upgrade. (The averaging is unweighted over
five scenarios chosen for this suite, not a real-world change-frequency distribution, so treat it as
a summary rather than a ranking with authority.)

---

## 7. Bugs found and fixed

**1. $P(r_t=0)$ as an alarm signal (BOCPD).** Derived that it is identically $H$ under a constant
hazard (§3.6) and confirmed empirically before shipping — the alarm as first written could never
have fired. Replaced with $P(r_t \le 2)$.

**2. Burn-in false alarms (BOCPD) — caught only by multi-seed replication.** The single-seed run
showed 2 false alarms before the `jump`, plausible as variance; 20 seeds showed the *same* ~2.1 on
nearly every seed — systematic, not noise. Cause: with fewer than $c+2$ blocks of history, every
active hypothesis trivially satisfies $r \le c$, so $P(\text{short run}) \approx 1$ regardless of
data. Verified at 100% alarm rate on blocks 0–1 across 200 stationary seeds, 0% after the fix.
False alarms before $\tau$ fell from $2.10\pm0.07$ to $0.10\pm0.07$. **This is the concrete case for
multi-seed replication as a correctness tool, not just an error-bar formality.**

**3. Overclaim on `oscillate` (documentation).** Corrected in `notes/todo.md` — see §5.4.

**4. `GaussianCopulaMLE` latches to NaN on heavy-tailed data — caught by the cross-family grid.**
`gaussian_mle` and `gaussian_mle_ew` returned NaN on the Gumbel DGP; one seed in 20 (seed 57,
`constant`) poisoned the whole mean. `GaussianCopulaObjective::score` carries $1/(1-\rho^2)^2$ and
$\partial\rho/\partial\theta = 1-\rho^2$ cancels only one power, so
$s_\theta \sim (1-\rho^2)^{-1}$. A joint extreme — which Gumbel's upper-tail dependence supplies
and a Gaussian DGP essentially never does — drives $\theta$ past the saturation of $\tanh$; $\rho$
is then exactly 1 in floating point, the next step is infinite, and two opposite-signed infinities
latch $\theta$ at NaN for the remainder of the stream. Fixed by clamping $\theta$ to
$\operatorname{atanh}(0.999)$ and skipping non-finite scores — the same two guards, and the same
bound, that `ParametricCopulaMLE` already applied; `GaussianCopulaMLE` simply predated the
convention. Verified that all 13 previously published Gaussian `jump` numbers are unchanged, so the
clamp does not bind on Gaussian data.

**5. Silent coverage gap (harness).** `xiao_win` was registered in `registry.hpp` but absent from
both `main.cpp` and `multiseed.cpp` `keys` lists, so it had never been run — no error, just no row.
The two lists are now noted as an invariant in `CLAUDE.md`.

**6. Unknown `COPULA` fell back to Gaussian silently.** `dgp::make_copula` returns a Gaussian copula
for any unrecognised name. Harmless interactively, but in a scripted grid a typo would have produced
Gaussian results labelled as another family. `multiseed.cpp` now rejects unknown names up front.

Also fixed along the way: a Gumbel-style grid artifact in the block-size default (`block=50` broke
the within-block-constant assumption on `oscillate`, making the filter worse than non-adaptive
estimators; swept and corrected to 20), and a `%.1f` print-precision bug in the calibration sweep
that made distinct swept thresholds print identically.

---

## 8. Limitations

- **Single DGP family.** Everything in *this* section is Gaussian-copula. `gaussian_mle_ew`'s strong
  showing is partly the reward of correct specification; the nonparametric estimators' apparent
  inefficiency is the price they pay for robustness this table never tests. **Superseded** — §10
  extends the grid to the Clayton/Gumbel/$t$ DGPs, and finds the effect real but mild.
- **Detection-delay metrics are partial.** $\mathrm{ARL}_0$ calibration controls the false-alarm side
  only. Delay is reported but not compared against the Lorden/KL optimal-delay bound
  (change_detection.md Part IV.4).
- **$\sigma_\omega^2$, hazard, and $m$ are all fixed hyper-parameters.** Estimating them online (VFF)
  is the natural next step and is what §3.3 argues the Bridge-1 parameterization makes possible.
- **No live-path integration.** All results are offline; nothing here runs in `src/consumer/` yet.

---

## 9. Reproduction

```bash
cd experiments
mkdir -p results

make                       # ./experiment  -- single-seed run + per-tick CSV
make multiseed             # ./multiseed   -- multi-seed replication with error bars
make calibrate_arl0        # ./calibrate_arl0 -- ARL0 threshold calibration

for s in constant drift jump stochastic oscillate; do ./multiseed $s 20; done
./calibrate_arl0 1000

./run_multiseed_grid.sh 20 42   # all 5 scenarios x all 4 DGP families (~50 min, measured)
```

Both binaries take `COPULA=gaussian|clayton|gumbel|t` to choose the DGP family (default
`gaussian`); non-Gaussian families get a filename suffix so runs do not clobber each other.

Outputs land in `results/` (gitignored): `experiment_<scenario>[_<family>].csv` (per-tick estimates
plus per-detector alarm columns) and `multiseed_<scenario>[_<family>].csv` (mean/SE summaries).

Implementation: `src/estimators/dependence/kalman_local_level.hpp`, `kalman_cusum.hpp`,
`kalman_glr.hpp`, `kalman_bocpd.hpp`, `change_detector.hpp`; the §10 parametric estimators in
`dependence/parametric_copula.hpp` and `dependence/copula_selector.hpp` with the family math in
`estimators/copula_family.hpp`; the non-Gaussian DGP in
`src/streamgenerator/datasource/copula.hpp`. Configs and their calibration rationale in
`src/estimators/registry.hpp`.

---

## 10. Misspecification: the multi-seed grid across DGP families

Everything above is Gaussian-copula. This section runs the same 20-seed protocol over all four DGP
families — Gaussian, Clayton, Gumbel, Student-$t$ ($\nu=4$) — and all 22 registered estimators
(`./run_multiseed_grid.sh 20 42`). It closes the two real coverage gaps: `xiao_win`, which was
registered in `registry.hpp` but present in neither harness's `keys` list and so had **never been
run**, and the 8 parametric-copula keys, which were single-seed and Gaussian-DGP only.

### 10.1 ⚠️ Cross-family RMSE is only comparable on three of the five scenarios

Clayton and Gumbel are **positive-dependence-only** families. `dgp::make_copula` maps the dependence
knob $d$ to a target $\rho_S$, but the Archimedean parameter is floored at independence, so the
negative half of any path that crosses zero is clipped. The realized ground-truth path is therefore
*not the same signal* across families:

| scenario | Gaussian / $t$: min, max, sd | Clayton / Gumbel: min, max, sd | comparable? |
|---|---|---|---|
| `constant`   | $+0.582, +0.582$, sd $0.000$ | $+0.582, +0.582$, sd $0.000$ | ✅ identical |
| `jump`       | $+0.191, +0.786$, sd $0.297$ | $+0.191, +0.786$, sd $0.297$ | ✅ identical |
| `stochastic` | $+0.352, +0.826$, sd $0.107$ | $+0.352, +0.826$, sd $0.107$ | ✅ identical |
| `drift`      | $-0.483, +0.786$, sd $0.363$ | $\mathbf{+0.001}, +0.786$, sd $0.259$ | ❌ clipped |
| `oscillate`  | $-0.582, +0.582$, sd $0.410$ | $\mathbf{+0.001}, +0.582$, sd $0.225$ | ❌ clipped |

So the Clayton/Gumbel columns on `drift` and `oscillate` look dramatically better (e.g. `xiao_ew`
$0.2010$ on Clayton `oscillate` vs $0.3484$ on Gaussian) **purely because the target barely moves**:
sd $0.225$ against $0.410$. That is a measurement artifact, not robustness. Read those two columns
*down* (which estimator wins on this family) and never *across* (which family is easier).

### 10.2 RMSE by DGP family (20 seeds, $N=5000$, burn-in 500)

Best in column in **bold**. As in §5.1, "best" means best among these shipped configurations, not
best achievable — §5.4's fair-tuning caveat applies here unchanged.

| estimator | \|  Gau `const` | `jump` | `stoch` | \| Clay `const` | `jump` | `stoch` | \| Gum `const` | `jump` | `stoch` | \| $t$ `const` | `jump` | `stoch` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xiao              | 0.0128 | 0.3199 | 0.1539 | 0.0149 | 0.3166 | 0.1371 | **0.0139** | 0.3145 | 0.1420 | 0.0155 | 0.3200 | 0.1556 |
| xiao_ew           | 0.0470 | 0.0798 | 0.0594 | 0.0543 | 0.0859 | 0.0637 | 0.0525 | 0.0834 | 0.0618 | 0.0567 | 0.0871 | 0.0680 |
| xiao_win          | 0.0275 | 0.1183 | 0.0835 | 0.0337 | 0.1223 | 0.0796 | 0.0326 | 0.1189 | 0.0787 | 0.0354 | 0.1234 | 0.0866 |
| hermite           | **0.0121** | 0.3179 | 0.1524 | 0.0155 | 0.3126 | 0.1344 | 0.0170 | 0.3111 | 0.1394 | 0.0150 | 0.3163 | 0.1540 |
| hermite_ew        | 0.0520 | 0.0815 | 0.0630 | 0.0604 | 0.0897 | 0.0696 | 0.0566 | 0.0855 | 0.0659 | 0.0604 | 0.0902 | 0.0708 |
| windowed_rank     | 0.0271 | 0.1170 | 0.0832 | 0.0333 | 0.1213 | 0.0791 | 0.0324 | 0.1183 | 0.0781 | 0.0341 | 0.1225 | 0.0859 |
| gaussian_mle      | 0.0129 | 0.3157 | 0.1535 | **0.0142** | 0.3128 | 0.1430 | 0.0174 | 0.3072 | 0.1391 | 0.0142 | 0.3138 | 0.1519 |
| gaussian_mle_ew   | 0.0375 | 0.0745 | **0.0558** | 0.0482 | 0.0866 | 0.0671 | 0.0463 | 0.0804 | 0.0592 | 0.0575 | 0.0869 | 0.0680 |
| kalman            | 0.0208 | 0.1288 | 0.0956 | 0.0243 | 0.1288 | 0.0878 | 0.0250 | 0.1272 | 0.0907 | 0.0250 | 0.1293 | 0.0973 |
| kalman_fast       | 0.0712 | 0.0880 | 0.0735 | 0.0790 | 0.0952 | 0.0793 | 0.0784 | 0.0934 | 0.0777 | 0.0801 | 0.0966 | 0.0823 |
| kalman_cusum      | 0.0721 | 0.1146 | 0.0792 | 0.0797 | 0.1237 | 0.1023 | 0.0844 | 0.1219 | 0.0985 | 0.0845 | 0.1234 | 0.1031 |
| kalman_glr        | 0.0721 | 0.1110 | 0.0776 | 0.0797 | 0.1136 | 0.0866 | 0.0797 | 0.1145 | 0.0881 | 0.0816 | 0.1161 | 0.0928 |
| **kalman_bocpd**  | 0.0189 | **0.0598** | 0.0856 | 0.0196 | **0.0602** | 0.0818 | 0.0219 | **0.0589** | 0.0825 | 0.0209 | **0.0633** | 0.0894 |
| kalman_bocpd_fast | 0.0613 | 0.0760 | 0.0800 | 0.0780 | 0.0885 | 0.0886 | 0.0736 | 0.0821 | 0.0836 | 0.0734 | 0.0893 | 0.0894 |
| clayton_mle       | 0.1257 | 0.2985 | 0.2105 | 0.0154 | 0.2465 | 0.1379 | 0.1589 | 0.3148 | 0.2306 | 0.0820 | 0.3055 | 0.1966 |
| clayton_mle_ew    | 0.1330 | 0.1255 | 0.1375 | 0.0339 | 0.0607 | **0.0551** | 0.1646 | 0.1621 | 0.1546 | 0.0905 | 0.1022 | 0.1121 |
| gumbel_mle        | 0.0491 | 0.2532 | 0.1639 | 0.0790 | 0.2966 | 0.1804 | 0.0168 | 0.2428 | 0.1357 | 0.0183 | 0.2659 | 0.1651 |
| gumbel_mle_ew     | 0.0644 | 0.0761 | 0.0838 | 0.0876 | 0.1033 | 0.0990 | 0.0409 | 0.0643 | **0.0564** | 0.0415 | 0.0708 | 0.0784 |
| t_mle             | 0.0601 | 0.3040 | 0.1533 | 0.0294 | 0.3004 | 0.1367 | 0.0258 | 0.3013 | 0.1370 | **0.0131** | 0.2997 | 0.1490 |
| t_mle_ew          | 0.0741 | 0.0835 | 0.0727 | 0.0531 | 0.0824 | 0.0648 | 0.0526 | 0.0782 | 0.0617 | 0.0460 | 0.0781 | **0.0610** |
| copula_select     | 0.0277 | 0.2592 | 0.1487 | 0.0154 | 0.2465 | 0.1302 | 0.0168 | 0.2433 | 0.1287 | **0.0131** | 0.2983 | 0.1490 |
| copula_select_ew  | 0.0402 | 0.0724 | 0.0617 | 0.0341 | 0.0653 | 0.0636 | 0.0417 | 0.0683 | 0.0681 | 0.0465 | 0.0778 | 0.0618 |

Full five-scenario tables per family are in `results/multiseed_<scenario>[_<family>].csv`.

### 10.3 What the grid shows

**1. BOCPD's `jump` win is family-robust — the strongest result in this document.** `kalman_bocpd`
is best in column on `jump` for *all four* families: $0.0598$ / $0.0602$ / $0.0589$ / $0.0633$. The
margin over the best fixed-$\lambda$ nonparametric is 25–30% everywhere. §6.2 argued this is
structural — no fixed $\lambda$ can be long-memory before a break and short-memory after — and the
argument never used Gaussianity. The grid is the evidence: the win survives tail dependence
(Clayton, lower), tail dependence (Gumbel, upper) and heavy joint tails ($t_4$), which is what a
distribution-free structural claim should do. Note the ranking is otherwise remarkably stable across
families — the whole column order barely moves, which is itself the marginal-invariance property of
§4.4 showing up at the level of the leaderboard.

**2. Correct specification pays, but misspecification is wildly asymmetric.** §8 predicted
`gaussian_mle_ew` would "be expected to lose" off-Gaussian. Directionally right, but the effect is
mild: it wins `stochastic` on Gaussian ($0.0558$), and off-family it degrades only to $0.0671$
(Clayton) / $0.0592$ (Gumbel) / $0.0680$ ($t$) — never catastrophic, still mid-pack. The Gaussian
copula is a forgiving wrong answer *in $\rho_S$ space*. `clayton_mle` is not: on Gumbel data it
posts $0.1589$ `constant` (10× its own on-family $0.0154$) and on Gaussian $0.1257$. Gumbel-on-
Clayton is bad but half as bad ($0.0790$). The practical reading: **fitting Clayton to
non-Clayton data is the one genuinely dangerous choice here**, and it is dangerous because Clayton's
lower-tail structure is actively wrong for upper-tail data, not merely uninformative.

**3. `copula_select` recovers the generating family, cumulatively.** Compare the selector against
the *correct* family's MLE, on each family's own data: Clayton $0.0154$ vs $0.0154$; Gumbel $0.0168$
vs $0.0168$; $t$ $0.0131$ vs $0.0131$ — exact matches on `constant`, and near-exact on `jump`
($0.2465$/$0.2465$, $0.2433$/$0.2428$). Online GoF selection works and costs essentially nothing
when it is right. But `copula_select_ew` is *worse* than the right family's `_ew` on Clayton `drift`
($0.1039$ vs $0.0717$): under forgetting, the selector is choosing on a short window and churns.
Selection and forgetting interact badly, which is a concrete open problem, not a tuning detail.

**4. `xiao_win` — measured for the first time — adds nothing over `windowed_rank`.** They agree to
~0.5% on every family and scenario ($0.0275$ vs $0.0271$, $0.1183$ vs $0.1170$, …). That is the
expected result (both are exact sliding windows of the same length; Xiao's binning is the only
difference) and it is mild evidence the binning is not costing accuracy. It does cost less storage —
5824 B vs 8000 B — so it is the better of the two, marginally. Not a thesis result on its own; worth
recording so the next person does not re-run it.

**5. The grid found a real bug** (§7, `gaussian_mle` NaN latch). That is now the second bug this
document's replication tier has caught that single-seed runs missed, after the BOCPD burn-in alarms.
Both were invisible at one seed and obvious at twenty.

### 10.4 What this does *not* settle

- Only `constant`/`jump`/`stochastic` support cross-family reading (§10.1). Extending
  `drift`/`oscillate` needs a rotated or mixture Archimedean construction that can represent
  negative dependence.
- $\nu=4$ is fixed for both the $t$ DGP and `t_mle`, so the $t$ column is a matched-$\nu$ best case
  and overstates how well `t_mle` would do against unknown tail weight.
- The detectors are still calibrated on Gaussian innovations only (§3.7). The $\mathrm{ARL}_0$
  thresholds were never re-derived per family; that the alarm counts stay sane off-family is
  encouraging but is not calibration.

---

## 11. References

- Adams, R. P. & MacKay, D. J. C. (2007). *Bayesian Online Changepoint Detection.*
- Page, E. S. (1954). *Continuous inspection schemes.* — CUSUM.
- Willsky, A. S. & Jones, H. L. (1976). *A generalized likelihood ratio approach to the detection and
  estimation of jumps in linear systems.*
- Lai, T. L. (1995). *Sequential changepoint detection in quality control and dynamical systems.* —
  window-limited GLR.
- Basseville, M. & Nikiforov, I. (1993). *Detection of Abrupt Changes.* — innovation-based detection.
- Fieller, E. C., Hartley, H. O. & Pearson, E. S. (1957). — the $1.06/(n-3)$ variance approximation.

Citations other than Adams & MacKay are from memory and should be checked against a bibliographic
database before use in the thesis — the same caveat that applies to the reference list in
[change_detection.md](change_detection.md).
