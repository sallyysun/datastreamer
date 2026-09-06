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

2. **BOCPD is the one genuine structural win, and it survives fair tuning.** On `jump`,
   `kalman_bocpd` achieves RMSE $0.0598 \pm 0.0022$ against the *best-tuned* Xiao ($0.0798$),
   Hermite ($0.0815$) and plain Kalman ($0.0840$) — a ~25% improvement that no choice of fixed
   $\lambda$ can replicate, because no fixed forgetting rate can be simultaneously long-memory
   before a break and short-memory after it.

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

5. **Two real bugs were found by verification, not by inspection** — see [§7](#7-bugs-found-and-fixed).
   One was caught only because multi-seed replication existed.

---

## 2. Models built

| Key | Family | Description |
| --- | --- | --- |
| `kalman`, `kalman_fast` | Kalman filter | Local-level filter on Fisher-$z$'d blockwise Spearman; $\sigma_\omega^2 = 10^{-4}$ / $10^{-2}$ |
| `kalman_cusum` | Hard-reset detector | Two-sided CUSUM on standardized innovations; calls `reset()` on alarm |
| `kalman_glr` | Hard-reset detector | Window-limited GLR (unknown jump size) via matched filter on the step response |
| `kalman_bocpd`, `kalman_bocpd_fast` | Bayesian | Run-length posterior (Adams & MacKay 2007); hazard $1/1000$ / $1/10$; **no reset** |

Pre-existing baselines used for comparison: `xiao`, `xiao_ew`, `hermite`, `hermite_ew`,
`windowed_rank`, `gaussian_mle`, `gaussian_mle_ew`.

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
detection delay as mean $\pm$ standard error, to `results/multiseed_<scenario>.csv`. Scoring matches
`./experiment` exactly ($N=5000$, burn-in 500) so numbers are directly comparable.

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

| Model | `constant` | `drift` | `jump` | `stochastic` | `oscillate` |
| --- | --- | --- | --- | --- | --- |
| `xiao` | 0.0128±0.0015 | 0.3893±0.0027 | 0.3199±0.0020 | 0.1539±0.0140 | 0.4107±0.0003 |
| `xiao_ew` | 0.0470±0.0011 | 0.0669±0.0014 | 0.0798±0.0014 | 0.0594±0.0044 | 0.3484±0.0015 |
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

### 6.3 Why hard reset costs RMSE

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

Also fixed along the way: a Gumbel-style grid artifact in the block-size default (`block=50` broke
the within-block-constant assumption on `oscillate`, making the filter worse than non-adaptive
estimators; swept and corrected to 20), and a `%.1f` print-precision bug in the calibration sweep
that made distinct swept thresholds print identically.

---

## 8. Limitations

- **Single DGP family.** Everything here is Gaussian-copula. `gaussian_mle_ew`'s strong showing is
  partly the reward of correct specification; on the Clayton/Gumbel/$t$ DGPs built on the
  parametric-copula branch it would be expected to lose. The nonparametric estimators' apparent
  inefficiency is the price they pay for robustness this suite never tests.
- **Detection-delay metrics are partial.** $\mathrm{ARL}_0$ calibration controls the false-alarm side
  only. Delay is reported but not compared against the Lorden/KL optimal-delay bound
  (change_detection.md Part IV.4).
- **Multi-seed covers this detector line only** — the Xiao/Hermite/`gaussian_mle` families and all
  parametric-copula-branch work remain single-seed.
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
```

Outputs land in `results/` (gitignored): `experiment_<scenario>.csv` (per-tick estimates plus
per-detector alarm columns) and `multiseed_<scenario>.csv` (mean/SE summaries).

Implementation: `src/estimators/dependence/kalman_local_level.hpp`, `kalman_cusum.hpp`,
`kalman_glr.hpp`, `kalman_bocpd.hpp`, `change_detector.hpp`; configs and their calibration rationale
in `src/estimators/registry.hpp`.

---

## 10. References

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
