# Streaming changepoint detection for copula dependence: methodology and results

Companion to [change_detection.md](change_detection.md), which laid out the design space and build
order. This document reports what was actually **built and measured** against that plan: the
local-level Kalman filter, three detectors on top of it (CUSUM, window-limited GLR, and BOCPD —
Bayesian Online Changepoint Detection), the
$\mathrm{ARL}_0$ calibration and multi-seed replication infrastructure needed to make their numbers
mean anything, and the results — including two corrections to claims that did not survive scrutiny.

Branch: `kalman-changepoint` (off `main`). Everything here is reproducible from
`experiments/` — see [§11](#16-reproduction).

[§0](#0-orientation) is a from-scratch primer on the machinery. It assumes expectation, variance, the
Gaussian, Bayes' rule and likelihood, and builds everything else — copulas, Spearman's $\rho$, the
Fisher transform, state-space models, the Kalman filter, CUSUM, GLR, BOCPD and $\mathrm{ARL}_0$ —
from those. Readers who already have this can start at §1.

---

> **Citation convention.** Inline references are highlighted as **[Author Year]** ⚠️ and collected
> in [§12](#17-references). ⚠️ Except where noted, they are **from memory** — years, volumes and
> pages must be checked against a bibliographic database before entering `bibliography.bib`. None
> of them are currently in the Zotero library except **[Roberts 1959]** ✓ and **[Gregory 2019]** ✓.

## 0. Orientation

### 0.1 The three layers

The methods only look like rivals because they get listed flat. They sit on **three
layers**, and you pick one thing from each.

```text
LAYER 1:  WHAT DO I ASSUME?          LAYER 3:  HOW DO I HANDLE CHANGE?
          │                                    │
          ├── Parametric                       ├── Continuous adaptation
          │     Gaussian / Clayton             │     ├── Fixed forgetting
          │     Gumbel / t                     │     └── VFF (rate itself moves)
          │                                    │
          └── Nonparametric                    └── Explicit detection
                Xiao / Hermite                       ├── CUSUM
                                                     ├── GLR
LAYER 2:  HOW DO I ESTIMATE?                         ├── EWMA chart
          │                                          ├── Shiryaev-Roberts
          ├── Parametric                             └── BOCPD
          │     MLE / Fisher scoring                       │
          │                                                └── optional restart
          └── Nonparametric
                Xiao / Hermite
                State-space: Kalman / EKF / UKF / particle
```

⚠️ **"EWMA" means two different things in this document.**

| | Layer | What it is |
| --- | --- | --- |
| **EWMA smoothing** | 2/3 | the `_ew` variants — *fixed forgetting*, an **estimator** |
| **EWMA control chart** | 3 | `kalman_ewma_chart` — watches innovations, a **detector** |

Same recursion, opposite jobs: one tracks ρ, the other alarms on it.

⚠️ **State-space is not "nonparametric" as such.** It sits on a different axis. The filter
assumes *no copula family* — it consumes Spearman's ρ, which is why it lands under layer 1
nonparametric. But it is parametric about **how ρ moves**: random walk, Gaussian noise, two
variances. Nonparametric in the copula, parametric in the dynamics.

**Estimator choice and adaptation strategy are separate axes.** Pick a layer-2 estimator,
then pick a layer-3 strategy; any combination is legal. §0.3 shows which ones actually exist.

The Kalman filter is a **layer 2** choice, not a detector — a detector sits on top of it and
reads its innovations.

### 0.2 Where this document sits

Mostly **layer 3**, on one layer 2 choice, tested against several layer 1 choices.

| Layer | Question | Sections |
| --- | --- | --- |
| **1 — assume** | Which family generated the data? | [§8](#8-misspecification-the-multi-seed-grid-across-dgp-families) |
| **2 — estimate** | How do we track ρ as it moves? | §0.10, §3.1–3.3, §3.9, §4.1, §5.1, §6.1 |
| **3 — handle change** | Forget, or detect and restart? | §0.11, §3.4–3.7, §4.2–4.4, §5.2–5.5, §6.2, §6.4, [§9](#14-the-reset-is-the-cost-why-detection-loses-to-plain-forgetting-on-jump) |
| *infrastructure* | Do the numbers mean anything? | §2, §3.8, [§7](#7-bugs-found-and-fixed), §10, [§11](#16-reproduction) |

**The headline is a layer-3 result:** on a jump, continuous adaptation *beat* explicit
detection — the restart discarded a nearly-converged estimate.

### 0.3 What we have tried, and what we have not

24 estimators are registered. Laid out on the three layers, the coverage is lopsided.

```text
LAYER 1  WHAT DO I ASSUME?
  ✓ Nonparametric    xiao, xiao_ew, xiao_win, hermite, hermite_ew, windowed_rank
  ✓ Parametric       gaussian_mle, clayton_mle, gumbel_mle, t_mle, copula_select
  ✓ DGP families     gaussian, clayton, gumbel, t          (all four, §8)
  ~ Marginals        rank-PIT built for Hermite only (hermite_pit, section 9);
                     Xiao / MLE / Kalman still standardize

LAYER 2  HOW DO I ESTIMATE?
  ✓ Sliding window   windowed_rank
  ✓ Fixed forgetting every _ew variant (EWMA smoothing — same thing)
  ✓ Linear-Gaussian  kalman, kalman_fast
  ✓ VFF              Kalman, Hermite, MLE (section 10)
  ✓ EKF / UKF        ekf / ukf (section 12) — layer 2 complete
  ✓ Particle filter  particle / particle_fast (section 11)
  ⬜ Coefficient      only scalar rho is ever monitored; the Hermite a_jk
    monitoring       vector is computed, then discarded

LAYER 3  HOW DO I HANDLE CHANGE?
  ✓ Detectors        CUSUM, GLR, EWMA chart, Shiryaev-Roberts, BOCPD
  ✓ Hard reset       the only alarm response tested
  ⬜ Covariance bump  proposed in §10.4, never built
  ⬜ No-restart       detector as a signal without restarting — never tried
```

**The biggest gap is a hole in the middle of the matrix.** Every detector wraps
`KalmanLocalLevel`. None has ever been put on `xiao`, `hermite`, or an MLE.

|              | Fixed forgetting (EWMA smoothing) | VFF | CUSUM | GLR | EWMA chart | SR | BOCPD |
| ------------ | ---------------- | --- | ----- | --- | ---- | -- | ----- |
| **Xiao**     | ✓ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **Hermite**  | ✓ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **MLEs**     | ✓ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **Kalman**   | ✓ | ⬜ | ✓ | ✓ | ✓ | ✓ | ✓ |

CUSUM, GLR, the EWMA chart, SR and BOCPD have all been **run and measured** — every one of
them on `KalmanLocalLevel`, and only there.

**The empty rows are not a wiring gap.** All five detectors watch
`standardized_innovation()` — a surprise with a known scale. Only the Kalman filter emits one.

| Estimator | Emits an innovation? | |
| --- | --- | --- |
| Kalman | ✓ | $e_t/\sqrt{F_t}$, standard normal under the null |
| MLEs | ✓ | the **score**, mean zero, variance already tracked in `info_` |
| Hermite | ✓ | the coefficient surprise $\lVert h_xh_y^\top - A\rVert$ |
| Xiao | ⬜ | $\hat\rho$ only — the residual has to be invented |

So every detection result here is **conditional on one layer-2 choice**. It says detection does
not help *the Kalman filter* on a jump ([§9](#14-the-reset-is-the-cost-why-detection-loses-to-plain-forgetting-on-jump)),
not that it fails in general.

**Where the particle filter goes.** Layer 2, replacing the Kalman filter — not competing with
the detectors. Kalman assumes the state moves linearly with Gaussian noise; ρ is bounded and its
block estimate is not Gaussian near the edges. Costs O(particles) per tick instead of O(1).
Affordable under the relaxed budget, untested.

**Three gaps that matter most**

1. **Detectors on non-Kalman estimators** — closes the empty matrix above.
2. **VFF** — it is the direct rival to detection in layer 3, and §9 says the rival is
   currently winning. Not having built it means the comparison is incomplete.
3. **Alarm response other than hard reset** — §9 shows the reset *is* the cost, so the
   fix belongs here.

### 0.4 The copula, and what ρ throws away

Two stocks, two questions: *how does each behave alone* (marginals) and *how do they move
together* (dependence).

**The separation trick.** Don't record "+2.3%". Record **"AAPL's 87th-percentile day"**.
Every observation becomes a pair like $(0.87, 0.91)$. Both coordinates are now uniform on
$[0,1]$ by construction, so all information about *how big* the moves were is deleted on
purpose. What survives is only how the percentiles move together — and that joint
distribution is the copula $C(u,v)$.

**[Sklar 1959]** ⚠️ (textbook: **[Nelsen 2006]** ✓) says the split is always possible and
unique for continuous margins:

$$F(x,y) = C\big(F_X(x),\, F_Y(y)\big) \qquad\text{joint} = \text{marginals} + \text{copula}$$

Spearman's ρ is **one number computed off that surface**, $\rho_s = 12\iint C - 3$.

> **The copula is a photograph. Spearman's ρ is its average brightness.**

A Gaussian copula and a Clayton copula can both sit at ρ = 0.5 while one crashes together
in the lower tail and the other doesn't. **Every detector here watches the scalar**, so a
change of shape at fixed ρ is invisible to all of them (distinction 1). Monitoring the
Hermite coefficient vector instead — an O(k) object the estimator already computes and
then discards — is the cheapest available fix.

### 0.5 Ranks, and why the current code is exposed

**Rank-PIT**: replace a value by its rank over what you've seen. 87th of 100 → 0.87. No
assumption about the distribution's shape.

> **Marginal invariance.** If AAPL's volatility doubles, the *ranks don't change* — the
> biggest day is still the biggest day. A copula estimated from ranks doesn't move when
> the marginals move.

That is precisely what a dependence detector needs. **This codebase doesn't do it**: it
standardizes (Welford running mean/sd), which is exact only for Gaussian data.
**[Fantazzini 2009]** ✓ measured the cost — with skewed data and symmetric marginals
assumed, elliptical copula correlations were biased downward by **up to 70% of the true
value**. Rank-PIT is open in [todo.md](../todo.md).
[§4.4](#44-marginal-invariance-distinction-2--confirmed-no-fix-needed) shows the current
approach survives the *simulated* DGPs; that is not a defence against real returns.

### 0.6 The five distinctions

Why a copula changepoint is harder than the changepoints most detectors target. Cited by
number elsewhere; full statements in [roadmap.md](roadmap.md).

| # | Short name | The problem |
| --- | --- | --- |
| 1 | Joint, not scalar | The thing changing is a surface; ρ is one number off it |
| 2 | Marginal confound | Ignore volatility changes, detect dependence changes |
| 3 | Rank feedback | Ranks shift as data arrives → the stream is not i.i.d. |
| 4 | Weak signal | Dependence converges slowly → delay is fundamental |
| 5 | Drift, not jumps | Real dependence mostly drifts; jumps are rare |

**Distinction 3 undermines the theory**: every optimality result in
[§12](#17-references) assumes i.i.d. pre-change data. Rank streams are not.

### 0.7 Three terms used later

**Functional** — a function that eats a whole object and returns a number. ρ is a
functional of the copula, and a lossy one. *You choose which functional the detector
watches.*

**Multiplier bootstrap** — how to set a threshold with no usable formula. Distinction 3
rules out the textbook null distribution, so simulate: multiply the existing terms by
random mean-zero weights and recompute. Theory for empirical copula processes is
**[Segers 2012]** ✓ and **[Bücher & Volgushev 2013]** ✓.

**Anytime-valid / e-values** — a p-value is only trustworthy if you look *once*, at a
pre-planned time; peek repeatedly and chance alone eventually gives a false alarm.
Anytime-valid means look as often as you like. An **e-value** is a betting score — how
much you'd have won betting against "nothing changed" — and it holds without i.i.d.,
which is exactly what distinction 3 destroys.
[change_detection.md](change_detection.md) Part IV.1 calls this the strongest novelty
candidate in the project; none of that literature is in the library yet.
### 0.8 Forgetting, and why an optimal memory exists

The standard streaming device is the **EWMA**:

$$\hat\theta_t = (1-\alpha)\,\hat\theta_{t-1} + \alpha\, y_t$$

Weights decay geometrically, so it forgets at a fixed rate in $O(1)$ state.

**Effective sample size** — how many equally-weighted points carry the same variance:
$\text{ESS} = (2-\alpha)/\alpha \approx 2/\alpha$. So $\alpha = 0.01$ means about 200
points of memory.

If observations have variance $\sigma^2$ and the truth drifts at $\dot\theta$ per step:

$$
\mathrm{MSE}(\text{ESS}) \approx \underbrace{\frac{\sigma^2}{\text{ESS}}}_{\text{variance}}
+ \underbrace{\left(\frac{\dot\theta \cdot \text{ESS}}{2}\right)^{\!2}}_{\text{lag bias}}
$$

A U-shape with an interior minimum. **This one expression explains most of §5**: a cumulative
estimator ($\text{ESS} \to \infty$) wins `constant` and is catastrophic on `drift`; a
short-memory one is the reverse.

⚠️ **λ means opposite things across estimators.** Xiao and the MLEs: λ is *retention*, bigger
= longer memory ($\alpha = 1-\lambda$). Hermite: λ is the *update weight*, smaller = longer
memory ($\alpha = \lambda$). Getting it backwards silently inverts an experiment. See the
header of `registry.hpp`.

### 0.9 Blocks, and the Fisher transform

Two obstacles stop us filtering $\hat\rho_S$ tick by tick.

**ρ is bounded.** It lives in $[-1,1]$, so its noise must shrink as $|\rho| \to 1$ — a
Gaussian noise model is false near the ends. The **Fisher transform** fixes it:

$$z = \operatorname{atanh}\rho = \tfrac12\log\frac{1+\rho}{1-\rho} \in \mathbb R$$

$z$ is unbounded and its variance is roughly constant in ρ, which is what a linear-Gaussian
model needs.

⚠️ **Approximately, not exactly.** Fisher-$z$ is a variance-stabilising *normalising* transform,
normal only asymptotically, and $m=20$ is not asymptotic. Block Spearman is a rank statistic on
discrete support. The approximation is weakest near $|\rho| \to 1$ and at small $m$. Two things
limit the damage: the Kalman filter remains the minimum-MSE **linear** estimator without
normality (it just stops being *the* optimal filter), and ARL₀ is bootstrapped on the real
stream rather than read off a Gaussian formula (§3.7). A particle filter is the honest version
— see §0.3, untested.

**One tick isn't enough.** So aggregate: every $m$ ticks (a **block**), compute the block's
$\hat\rho_S$ and transform it. The variance is known in closed form:

$$\sigma_v^2 \approx \frac{1.06}{m-3}$$

**[Fieller, Hartley & Pearson 1957]** ⚠️, for Fisher-$z$'d *Spearman* — about 6% wider than
Pearson's $1/(m-3)$. **$\sigma_v^2$ being known rather than estimated is the structural
gift**: it pins the noise scale, leaving one parameter to choose instead of two.

Blocking costs lag and a noise floor (§6.1), but buys a model close enough to linear-Gaussian
that the filter is closed-form — no EKF or UKF required to *run* it, though one may be required
to do it justice.

### 0.10 State-space models and the Kalman filter

A hidden state moving by a linear rule, seen through noise (**[Kalman 1960]** ⚠️;
**[Harvey 1989]** ⚠️, **[Durbin & Koopman 2001]** ⚠️):

$$z_t = z_{t-1} + \omega_t, \quad \omega_t \sim \mathcal N(0, \sigma_\omega^2)$$
$$y_t = z_t + v_t, \quad\;\; v_t \sim \mathcal N(0, \sigma_v^2)$$

This one — a random walk seen in noise — is the **local-level model**. Here
$z_t = \operatorname{atanh}\rho_t$ and $y_t$ is the block measurement of §0.9. The two
variances say everything: $\sigma_\omega^2$ is *how fast we think the world changes*,
$\sigma_v^2$ is *how noisy the measurement is*.

**The idea.** Carry a belief, not a number: a mean $\hat z$ and a variance $P$. Waiting makes
you less sure ($P$ grows by $\sigma_\omega^2$); measuring makes you more sure. Combining the
two is Bayes for two Gaussians — each weighted by precision, the more confident one wins.

$$
\begin{aligned}
\textbf{predict:} \quad & \hat z_{t|t-1} = \hat z_{t-1}, && P_{t|t-1} = P_{t-1} + \sigma_\omega^2 \\
\textbf{innovation:} \quad & e_t = y_t - \hat z_{t|t-1}, && F_t = P_{t|t-1} + \sigma_v^2 \\
\textbf{gain:} \quad & K_t = P_{t|t-1} / F_t && \\
\textbf{update:} \quad & \hat z_t = \hat z_{t|t-1} + K_t e_t, && P_t = (1-K_t) P_{t|t-1}
\end{aligned}
$$

Read the update as **new estimate = old estimate + gain × surprise**.

- **$e_t$, the innovation**, is the part the model didn't predict. If the model is right,
  $e_t/\sqrt{F_t}$ is i.i.d. standard normal. **That is what every detector monitors**: a real
  change makes the filter wrong in a systematic direction, so surprises stop looking like noise.
- **$K_t \in (0,1)$, the gain**, is how much of the surprise you act on. Uncertain filter
  ⇒ $K \to 1$, follow the data. Confident filter ⇒ $K \to 0$, ignore a noisy read. Cold start
  handles itself.

**Steady state.** $P_t$ converges. With $q = \sigma_\omega^2/\sigma_v^2$:

$$K = \frac{\sqrt{q^2 + 4q} - q}{2}$$

and the update collapses to $\hat z_t = (1-K)\hat z_{t-1} + K y_t$ — **exactly the EWMA of
§0.8 with $\boxed{\lambda = 1 - K}$**. Every fixed-λ estimator in this codebase *is* this
filter at steady state. Not analogous — the same recursion. That is Bridge 1 (§3.3, verified
in §4.1).

The filter's extra content is the **transient**: it starts uncertain and tightens, whereas a
fixed λ is born at its final memory and can never change it.

### 0.11 Detection: from likelihood ratios to five detectors

**Definition (log-likelihood ratio).** Two competing explanations of the data, $H_0$ (no change) and
$H_1$ (change), compared by

$$
\Lambda_t \;=\; \log \frac{p(y_{1:t} \mid H_1)}{p(y_{1:t} \mid H_0)} .
$$

Large $\Lambda_t$ means $H_1$ explains the data better. Every detector below is a way of computing
or approximating this online — they differ only in what they do with the unknown change time $\nu$:
**maximise** over it (CUSUM **[Page 1954]** ⚠️, GLR **[Willsky & Jones 1976]** ⚠️), **sum** over it
(Shiryaev–Roberts **[Shiryaev 1963]** ⚠️), **smooth** it (EWMA chart **[Roberts 1959]** ✓), or carry a
**posterior** over it (BOCPD **[Adams & MacKay 2007]** ⚠️). All of them run on the standardized innovations $e_t/\sqrt{F_t}$,
which are i.i.d. $\mathcal N(0,1)$ under $H_0$ — so "has something changed?" becomes "have these
stopped looking like standard normal noise?"

**CUSUM** (Page 1954). Detect a mean shift of known size $\delta$. Under $H_1$ the innovations have
mean $\delta$ instead of $0$, and the per-observation log-LR is $\delta(e_t/\sqrt{F_t} - \delta/2)$.
Accumulate it, flooring at zero so evidence for "no change" cannot bank credit:

$$
S_t^{\pm} = \max\Bigl(0,\; S_{t-1}^{\pm} \pm \frac{e_t}{\sqrt{F_t}} - k\Bigr),
\qquad k = \frac{\delta}{2},
\qquad \text{alarm when } \max(S^+_t, S^-_t) > h .
$$

Two copies catch increases and decreases. The floor is what makes it a *detector* rather than a
random walk: it forgets accumulated non-evidence.

**GLR** (Willsky & Jones 1976; Lai 1995). We rarely know $\delta$. The generalized likelihood ratio
maximises it out, and also over the unknown changepoint time $\tau$:

$$
g_t \;=\; \max_{t-N \le \tau \le t-d} \; \max_{\delta} \; \log\frac{p(y \mid \tau, \delta)}{p(y \mid H_0)} .
$$

The inner maximum is available in closed form, leaving a matched filter over candidate lags. The
window $N$ is what makes it $O(N)$ per step rather than $O(t)$ — "window-limited" GLR.

**BOCPD** — **B**ayesian **O**nline **C**hangepoint **D**etection (Adams & MacKay 2007). Fully
Bayesian, and structurally different: instead of testing a hypothesis, carry a posterior over the
**run length** $r_t$ = number of blocks since the last change.

$$
p(r_t \mid y_{1:t}) \;\propto\; \sum_{r_{t-1}} \underbrace{p(r_t \mid r_{t-1})}_{\text{hazard}} \;
\underbrace{p(y_t \mid r_{t-1}, y^{(r)})}_{\text{predictive}} \; p(r_{t-1} \mid y_{1:t-1}),
$$

with a constant hazard $H$ (change probability per block, so run lengths are geometric with mean
$1/H$). Each run length carries its own conjugate posterior over $z$, and the reported estimate
averages over them. **This is why it wins `jump`** (§6.2): after a break the short-run hypotheses
gain mass, so the estimator effectively uses only post-break data — long memory before, short memory
after, which no fixed $\lambda$ can do.

⚠️ Under a constant hazard, $P(r_t = 0) = H$ *identically* — the marginal likelihood cancels — so it
carries no information and cannot be an alarm signal. That bug is §7.1; the fix is to threshold
$P(r_t \le c)$ for $c > 0$.

**Definition (average run length, $\mathrm{ARL}_0$).** The expected number of blocks to a false alarm
when nothing is changing:

$$
\mathrm{ARL}_0 \;=\; \mathbb E[\,T_{\text{alarm}} \mid \text{no change}\,].
$$

A threshold with no $\mathrm{ARL}_0$ attached is meaningless — any detector looks fast if allowed to
cry wolf. Comparing detectors requires first tuning each threshold to a *common* $\mathrm{ARL}_0$
(here 1000 blocks, §3.7), and only then comparing detection delay. This is the frequentist criterion
applied even to the Bayesian detectors, so all four sit on one axis.

### 0.12 Reading the results

**RMSE** is against the *known* true $\rho_S$ path, which the simulator supplies:

$$
\mathrm{RMSE} \;=\; \sqrt{\frac{1}{T - t_0}\sum_{t=t_0}^{T} \bigl(\hat\rho_{S,t} - \rho_{S,t}\bigr)^2},
$$

with $t_0 = 500$ discarded as burn-in. Lower is better; it mixes variance and lag-bias exactly as
§0.4 describes.

Every headline number is a mean over 20 random seeds with its standard error
$\mathrm{SE} = s/\sqrt{n}$. Two numbers differing by less than a couple of SEs are not distinguishable.
Single-seed numbers are for development only — §7 records two real bugs that a single seed hid.

⚠️ **"Best in column" means best among the shipped configurations, not best achievable.** A model can
look bad because its $\lambda$ was set for a different scenario. §5.4 tunes each family fairly and
reverses one of the apparent results. Always check there before quoting a ranking.

### 0.13 Symbols

| Symbol | Meaning |
| --- | --- |
| $\rho$, $\rho_S$ | copula (linear) correlation; Spearman's rank correlation — all estimators report $\rho_S$ |
| $m$ | block size in ticks |
| $z_t = \operatorname{atanh}\rho_t$ | latent state: Fisher-transformed dependence |
| $y_t$ | observation: Fisher-$z$'d block Spearman estimate |
| $\sigma_\omega^2$ | state (process) variance — how fast we believe dependence moves |
| $\sigma_v^2 \approx 1.06/(m-3)$ | observation variance — known in closed form |
| $q = \sigma_\omega^2/\sigma_v^2$ | signal-to-noise ratio |
| $\hat z_t$, $P_t$ | filtered mean and variance (the belief) |
| $e_t$, $F_t$ | innovation (surprise) and its variance |
| $K_t$, $K$ | Kalman gain; steady-state gain $K = (\sqrt{q^2+4q}-q)/2$ |
| $\lambda$, $\alpha$ | forgetting parameter and rate; $\lambda = 1-K$ at steady state |
| $\text{ESS} = (2-\alpha)/\alpha$ | effective sample size — memory length in points |
| $h$, $k$, $\delta$ | detector threshold; CUSUM slack; assumed jump size |
| $N$, $d$ | GLR window length and minimum lag |
| $H$, $r_t$ | BOCPD hazard rate and run length |
| $\mathrm{ARL}_0$ | expected blocks to a false alarm under no change |
| $\tau$ | true changepoint time |

---

## 1. Summary of findings

**0. One trade, measured three independent ways.** Each pair below is a simpler method
that assumes more against a general one that assumes less:

| Assuming method | General method | The general one wins when |
| --- | --- | --- |
| standardization | rank-PIT ([§9](#9-rank-pit-the-marginals-finally-handled-properly)) | marginals are not Gaussian — 3.7× better under $x^3$, and *exactly* invariant |
| fixed λ | VFF ([§10](#10-variable-forgetting-adaptation-without-a-detector)) | the fixed λ was badly chosen — 15.6% on Kalman, ~1% on Hermite and the MLE |
| Kalman | particle filter ([§11](#11-the-particle-filter-generality-that-mostly-costs)) | ρ moves fast — wins `oscillate`, loses everywhere else |

**In every case the assuming method wins where its assumption holds and loses unboundedly
where it does not.** Three independent mechanisms giving the same answer is a stronger
claim than any one of them, and it is the efficiency↔adaptation question of
[roadmap.md](roadmap.md) stated in measurements rather than in theory.

**1. Every fixed-λ estimator here *is* a Kalman filter at steady state.** Not an analogy.
The gain matches $K = \tfrac{1}{2}(\sqrt{q^2+4q}-q)$ to five decimal places across four
orders of magnitude in $q$, with $\lambda = 1-K$.

**2. BOCPD is the one real structural win.** On `jump` it gets RMSE $0.0598 \pm 0.0022$
against the *best-tuned* Xiao (0.0798), Hermite (0.0815) and plain Kalman (0.0840). About
25% better, and no fixed λ can match it — no single forgetting rate is both long-memory
before a break and short-memory after it. It stays best across all four DGP families
(0.0598 / 0.0602 / 0.0589 / 0.0633), which is what a distribution-free claim should do.

**3. The Kalman filter's `oscillate` win was a tuning artifact.** ⚠️ Corrected. At default
λ it looked like it beat everything. Tuned fairly, Xiao (0.1826) and Hermite (0.1879) beat
the *best achievable* Kalman (0.2256) by ~19%, and the Kalman curve plateaus — block
aggregation is a floor it cannot be tuned past.

**4. Hard detect-and-restart costs accuracy.** `kalman_cusum` and `kalman_glr` are *worse*
than the filter they wrap, on both `jump` and `oscillate`. By alarm time the filter has
already adapted; the reset throws that away.
[§9](#14-the-reset-is-the-cost-why-detection-loses-to-plain-forgetting-on-jump) traces it
tick by tick.

**5. The Gaussian copula is a forgiving wrong answer.** `gaussian_mle_ew` degrades mildly
off-family (`stochastic`: 0.0558 → 0.0592–0.0680). `clayton_mle` on Gumbel data is 10×
worse than on its own (0.1589 vs 0.0154). `copula_select` recovers the true family when run
cumulatively, but churns under forgetting (§8.3).

**6. Four bugs were found by verification, not by reading the code** ([§7](#7-bugs-found-and-fixed)).
Two showed up only because the replication tier existed — BOCPD burn-in alarms and a NaN
latch in `gaussian_mle`. Both invisible at one seed, unmistakable across twenty.

---

## 2. Models built

| Key | Family | What it is |
| --- | --- | --- |
| `kalman`, `kalman_fast` | Layer 2 | Local-level filter on Fisher-$z$'d block Spearman; $\sigma_\omega^2 = 10^{-4}$ / $10^{-2}$ |
| `kalman_cusum` | Layer 3, hard reset | Two-sided CUSUM on standardized innovations |
| `kalman_glr` | Layer 3, hard reset | Window-limited GLR, unknown jump size |
| `kalman_ewma_chart` | Layer 3, hard reset | EWMA control chart on the same innovations |
| `kalman_sr` | Layer 3, hard reset | Shiryaev–Roberts; sums over change time instead of maximising |
| `kalman_bocpd`, `_fast` | Layer 3, no reset | Run-length posterior; hazard $1/1000$ / $1/10$ |

Baselines compared against: `xiao`, `xiao_ew`, `xiao_win`, `hermite`, `hermite_ew`,
`windowed_rank`, `gaussian_mle`, `gaussian_mle_ew`, plus `clayton_mle`, `gumbel_mle`,
`t_mle`, `copula_select` and their `_ew` variants. §5 compares the 13 Gaussian-DGP ones;
[§8](#8-misspecification-the-multi-seed-grid-across-dgp-families) runs all of them
across all four families.

**One thing worth flagging.** `kalman` and the BOCPD variants are Bayesian; CUSUM, GLR,
EWMA and SR are frequentist tests bolted onto a Bayesian filter. The ARL₀ calibration in
§3.7 is a frequentist criterion, and it gets applied to all of them, Bayesian included.

---

## 3. Methodology

### 3.1 The observation

Every $m$ ticks (a **block**), take the block's empirical Spearman correlation and
Fisher-transform it:

$$y_t = \operatorname{atanh}\big(\hat\rho^S_t\big), \qquad \sigma_v^2 \approx \frac{1.06}{m-3}$$

All five detectors share this one channel. §0.9 covers why.

**Why blockwise and not per-tick.** The per-tick normal-score product $z_xz_y$ has mean ρ and
variance $1+\rho^2$, and $\rho = \tanh z$ is nonlinear — so a per-tick Kalman filter needs an
EKF or UKF. Blockwise, the model is *approximately* linear-Gaussian, which is enough for a
closed-form filter. §6.1 prices that convenience; §0.9 states what the approximation costs.

**Marginal invariance comes free.** $\hat\rho^S$ is a rank statistic, so it is invariant to
any monotone transform of either marginal. That makes the efficient-score refinement of
change_detection.md Part IV.2 unnecessary here — verified in §4.4.

### 3.2 The filter

State $z_t = \operatorname{atanh}\rho_t$, local-level model, recursion exactly as §0.10.
Reported estimate is $\hat\rho_t = \tanh\hat z_t$. Two configurations:

| Key | $\sigma_\omega^2$ | Steady-state gain | Equivalent λ |
| --- | --- | --- | --- |
| `kalman` | $10^{-4}$ | slow | long memory |
| `kalman_fast` | $10^{-2}$ | $K \approx 0.36$ | $\lambda \approx 0.64$ |

The precision-weighted gain handles cold start automatically — the principled version of the
ad-hoc `hermite_ew` cold-start patch.

### 3.3 Bridge 1

$K$ depends only on $q = \sigma_\omega^2/\sigma_v^2$, and $\sigma_v^2$ is known. **So choosing
$\sigma_\omega^2$ *is* choosing λ.** The difference is what the parameter means:
$\sigma_\omega^2$ is physical — *how fast does dependence actually drift* — and data can speak
to it. λ is an arbitrary constant and cannot be estimated. Estimating $\sigma_\omega^2$ online
is the VFF problem, still open. Derivation in §0.10, verified numerically in §4.1.

### 3.4 Innovation CUSUM

**[Page 1954]** ⚠️, innovation form from **[Basseville & Nikiforov 1993]** ⚠️. Exactly
minimax-optimal for Lorden's worst-case delay — **[Lorden 1971]** ⚠️, **[Moustakides 1986]** ⚠️.

The filter absorbs gradual drift; what it *can't* absorb shows up as innovations that stop
looking standard normal. Two one-sided sums catch increases and decreases:

$$
S^{\pm}_t = \max\big(0,\; S^{\pm}_{t-1} \pm \tilde e_t - k\big),
\qquad k = \delta/2,
\qquad \text{alarm when } \max(S^+, S^-) > h
$$

The increment is exactly the log-LR for $\mathcal N(0,1)$ vs $\mathcal N(\delta,1)$. **The floor
at zero is what makes it a detector** rather than a random walk — it refuses to bank credit for
evidence of *no* change. On alarm it calls the filter's `reset()`.

### 3.5 Window-limited GLR

CUSUM needs a known shift size δ. GLR maximises δ out, matched-filtering the filter's own step
response (**[Willsky & Jones 1976]** ⚠️; window-limited form **[Lai 1995]** ⚠️). A unit jump at
τ contributes $\varrho_s(\tau) = (1-K)^{s-\tau}$, so over candidate lags $j$:

$$
A(j) = \sum_{i=0}^{j} (1-K)^i \tilde e_{t-i},
\qquad
N(j) = \sum_{i=0}^{j} (1-K)^{2i},
\qquad
g_t = \max_{j} \frac{A(j)^2}{N(j)}
$$

$A$ updates by a shift register, $N$ is precomputed. **Capping $j \le W$ is what makes it
streaming-admissible** — the unbounded $\sup_\tau$ GLR is $O(t)$ per step.

### 3.6 BOCPD

**[Adams & MacKay 2007]** ⚠️. Structurally different from the others: instead of testing a
hypothesis, it carries a posterior over the **run length** $r_t$ — blocks since the last change.
CUSUM asks *is there enough evidence to declare a change?*; BOCPD asks *how likely is each
possible time-since-change?* and never declares anything.

Each hypothesis keeps a conjugate Normal posterior on its segment mean. With constant hazard $H$
and $L(r)$ the predictive density of $y_t$ under hypothesis $r$:

$$P(r_t = 0,\, y_{1:t}) = H \sum_r P(r_{t-1}=r,\,y_{1:t-1})\,L(r)$$
$$P(r_t = r{+}1,\, y_{1:t}) = (1-H)\,P(r_{t-1}=r,\,y_{1:t-1})\,L(r)$$

In log-space with log-sum-exp, truncated at $R_{\max}$ for $O(R_{\max})$ per block. The estimate
is the run-length-weighted posterior mean — Bayesian model averaging.

**There is no reset anywhere. The soft mixing *is* the mechanism**, which is why §9 finds it
escapes the reset cost entirely.

> ⚠️ **$P(r_t=0)$ is useless as an alarm under a constant hazard.** Write
> $S = \sum_r P(r_{t-1}=r)L(r)$. The reset branch carries $H\cdot S$, the growth branches
> $(1-H)\cdot S$ — the same $S$, which cancels in normalization:
> $$P(r_t = 0 \mid \text{data}) = \frac{HS}{HS + (1-H)S} = H$$
> **identically, every tick, whatever the data.** Confirmed empirically: flat at exactly
> $0.0010 = 1/1000$ straight through a real jump. The signal lives in how $(1-H)$ redistributes
> *within* the growth branch, so the alarm thresholds $P(r_t \le 2)$ instead.

### 3.7 $\mathrm{ARL}_0$ calibration

**Detectors are only comparable at matched in-control run length.** And because of rank/PIT
feedback (distinction 3), ARL₀ has to be *bootstrapped on the actual stream*, not read off a
formula.

`calibrate_arl0.cpp` runs each detector's real `update()` path on stationary Gaussian-copula
streams at $\rho \in \{0, 0.3, 0.6, 0.8\}$ (6 seeds × 5000 blocks per cell), sweeps the
threshold, and takes the smallest one whose **worst case over ρ** clears 1000 blocks.
Worst-case because the true ρ is unknown in live use. Where the minimum threshold sat exactly
on target with no margin, the next grid point was taken instead.

### 3.8 Multi-seed replication

`multiseed.cpp` runs the full estimator set across 20 seeds, reporting RMSE and — for every
`ChangeDetector` — alarm count, false alarms before the jump, and detection delay as mean ± SE.
Scoring matches `./experiment` exactly ($N=5000$, burn-in 500).

**Does it actually change anything?** Measured on `jump`, seed 42 alone against the 20-seed
mean:

| estimator | seed 42 | 20 seeds | off by |
| --- | --- | --- | --- |
| `xiao_ew` | 0.0782 | 0.0798 | −1.2 SE |
| `hermite_ew` | 0.0798 | 0.0815 | −1.3 SE |
| `gaussian_mle_ew` | 0.0760 | 0.0745 | +1.3 SE |
| `kalman_fast` | 0.0921 | 0.0880 | +2.9 SE |
| `kalman_cusum` | 0.1177 | 0.1146 | +1.2 SE |
| `kalman_sr` | 0.1177 | 0.1139 | +1.6 SE |
| **`kalman_ewma_chart`** | **0.0921** | **0.1042** | **−3.5 SE** |
| `kalman_bocpd` | 0.0640 | 0.0598 | +1.9 SE |

Most land within 1–2 SE and the ranking survives. **`kalman_ewma_chart` does not, and it flips
the conclusion.** On seed 42 it scores 0.0921 — identical to the bare `kalman_fast` it wraps —
so a single seed says the EWMA detector is free. Across 20 seeds it costs 18%. The cause is in
[§9](#14-the-reset-is-the-cost-why-detection-loses-to-plain-forgetting-on-jump): it misses 8 of
20 jumps, so its per-seed spread is wide and seed 42 is a lucky one.

**This is the same failure mode as the BOCPD burn-in bug** (§7): one seed looking plausible
while the effect is systematic.

⚠️ **Correction.** §10 once claimed multi-seed covered "this detector line only", leaving
Xiao/Hermite/`gaussian_mle` single-seed. **That was wrong when written** — `multiseed.cpp` has
always shared its `keys` list with `main.cpp`, so those were multi-seeded from the first run.
The genuinely uncovered cases were different: `xiao_win`, which was in neither harness and had
never run at all (§7), and the parametric-copula keys plus non-Gaussian families, which lived
on an unmerged branch. Both now covered.

Recorded rather than silently edited: a caveat that *overstates* a limitation is still a wrong
statement about what the harness did, and the next reader would re-run work already done.

### 3.9 Slow vs `_fast`

Each `_fast` variant differs by **one number** — a prior about how often the world changes.
Nothing to do with compute speed.

**Kalman.** The knob is $\sigma_\omega^2$. At $m=20$, so $\sigma_v^2 = 0.0623$:

| key | $\sigma_\omega^2$ | $q$ | $K$ | $\lambda$ | ESS |
| --- | --- | --- | --- | --- | --- |
| `kalman` | $10^{-4}$ | 0.0016 | 0.0393 | 0.9607 | 50 blocks ≈ **999 ticks** |
| `kalman_fast` | $10^{-2}$ | 0.1604 | 0.3282 | 0.6718 | 5.1 blocks ≈ **102 ticks** |

100× more $\sigma_\omega^2$ says the truth wanders 100× more per block, so the filter trusts
its prior less. Note the compression: 100× in $\sigma_\omega^2$ buys only ~10× in memory,
because $K \approx \sqrt{q}$ for small $q$.

**BOCPD.** The knob is the hazard $H$ — a geometric prior on run length with mean $1/H$:

| key | $H$ | prior mean run length |
| --- | --- | --- |
| `kalman_bocpd` | $1/1000$ | 1000 blocks ≈ **20 000 ticks** |
| `kalman_bocpd_fast` | $1/10$ | 10 blocks ≈ **200 ticks** |

**These are not the same knob.** $\sigma_\omega^2$ fixes a memory; $H$ sets a prior the
posterior can override, so BOCPD's effective memory moves with the evidence. That is the §6.2
distinction, and why the thresholds differ too (0.15 vs 0.95) — a model expecting frequent
change finds short runs likelier, so it needs a higher bar for the same ARL₀.

**Priors** are all deliberately vague: $\hat z_0 = 0$ (independence) with $P_0 = 1$;
BOCPD's $\mu_0 = 0$, $\tau_0^2 = 1$; `CopulaFit` starts with one pseudo-observation of
information so the first step can't divide by zero. $P_0 = 1$ against $\sigma_v^2 \approx 0.062$
makes the prior ~16× wider than one observation, so the first block is taken almost at face
value ($K_1 = 0.94$ for both variants). **The fast/slow split is about steady state, not the
start** — both begin equally credulous.

---

## 4. Verification

Each component checked against an independent reference before anything was built on it.

**4.1 Bridge 1 is exact.** Steady-state gain vs closed form, $m=20$:

| $\sigma_\omega^2$ | $q$ | $K$ (filter) | $K$ (closed form) | diff |
| --- | --- | --- | --- | --- |
| $10^{-5}$ | 0.00044 | 0.02084 | 0.02084 | 0.00000 |
| $10^{-4}$ | 0.00443 | 0.06441 | 0.06441 | 0.00000 |
| $10^{-3}$ | 0.04434 | 0.18956 | 0.18956 | 0.00000 |
| $10^{-2}$ | 0.44340 | 0.48012 | 0.48012 | 0.00000 |
| $10^{-1}$ | 4.43396 | 0.84063 | 0.84063 | 0.00000 |

**4.2 GLR step response matches theory.** 2000 seeds, real block-Spearman noise, jump
$0.2 \to 0.85$. Theory says consecutive post-jump blocks decay by $(1-K) = 0.672$. Measured:
0.678, 0.652, 0.690, 0.690, 0.657, 0.735, 0.626, 0.683.

**4.3 BOCPD recovers a jump.** Tracks $0.19 \to 0.85$ within two blocks, no reset.
$P(r_t \le 2)$ spikes $0.0015 \to 0.53 \to 0.9999$ then returns to baseline; MAP run length
collapses to 1 exactly at the changepoint.

**4.4 Marginal invariance holds (distinction 2).** Applied a large purely-marginal shift
($x \mapsto 5x+100$, $y \mapsto 0.1y-50$, both monotone) with the copula held *exactly*
constant. **Zero alarms.** The CUSUM statistic never left the noise floor (peak 3.19 vs
threshold 3.5); the estimate stayed at 0.468 against a true $\rho^S$ of 0.483.

This is structural to the rank-based observation, so it transfers to every detector in this
line without a separate check. ⚠️ It does *not* clear the marginals for real data — see §0.5.

---

## 5. Results

### 5.1 Tracking accuracy (RMSE vs true Spearman $\rho$)

20 seeds, $N=5000$, burn-in 500, mean $\pm$ SE. **Bold** = best in column.

⚠️ **"Best in column" is bolded from point estimates and several are ties.** `multiseed`
now prints distance from the best row in pooled SEs, with `~` for anything under 2σ — on
`constant`, `xiao` (0.0128) and `gaussian_mle` (0.0129) are **0.4σ** from `hermite`'s
0.0121, so that bold is not a ranking. Re-read every bold in this section that way until
the table is regenerated.

⚠️ Every row is at its **shipped** setting, so "best in column" means best *among these
configs*, not best achievable. A tuned Xiao reaches 0.1826 on `oscillate` against the 0.2695
bolded here. This table = what the shipped configs do. §5.4 = what each family can do.
Conflating them is the mistake §5.4 corrects.

`kalman_ewma_chart` and `kalman_sr` are not in this table — they were added later; their `jump`
numbers are in [§9](#14-the-reset-is-the-cost-why-detection-loses-to-plain-forgetting-on-jump).

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

**Speed and reliability are separate axes.** `kalman_bocpd_fast` has the fastest mean delay
*when* it fires — and fires in fewer than half the seeds.

Why: a high hazard raises the alarm statistic's stationary baseline, because "the run is short"
is already common a priori.

| hazard | baseline $P(r_t \le 2)$ | at the jump | ratio |
| --- | --- | --- | --- |
| $1/1000$ | 0.0014 – 0.0027 (flat) | $\to 0.53 \to 0.9999$ | ~330× |
| $1/10$ | 0.14 – 0.26 (noisy) | $\to 0.99 \to 0.9999$ | ~6× |

Calibration then has to push the threshold far higher (0.95 vs 0.15) to hold the same ARL₀,
leaving a real jump much less headroom. **Same evidence, thinner margin** — hence the misses.

### 5.2b Alarm rates across all scenarios

Alarms per 5000-tick run (250 blocks), 20 seeds:

| Detector | `constant` | `drift` | `jump` | `stochastic` | `oscillate` |
| --- | --- | --- | --- | --- | --- |
| `kalman_cusum` | 0.05±0.05 | 0.15±0.08 | 1.10±0.10 | 0.20±0.09 | 24.75±0.18 |
| `kalman_glr` | 0.05±0.05 | 0.05±0.05 | 0.90±0.10 | 0.15±0.08 | 17.50±0.52 |
| `kalman_bocpd` | 0.35±0.13 | 0.30±0.18 | 1.45±0.20 | 0.40±0.13 | 17.00±0.98 |
| `kalman_bocpd_fast` | 0.10±0.10 | 0.20±0.09 | 0.85±0.22 | 0.20±0.12 | 4.40±0.56 |

Quiet on `constant`/`drift`/`stochastic`, about one alarm on `jump`, many on `oscillate` —
correctly, since each half-period genuinely *is* a large change.

**`drift` producing almost no alarms is the intended division of labour**: the filter absorbs
gradual drift, so nothing is left for the detector to flag.

### 5.3 Calibrated thresholds

Target $\mathrm{ARL}_0 = 1000$ blocks, worst case over $\rho$.

| Detector | swept parameter | bare minimum | **chosen** | worst-case $\mathrm{ARL}_0$ | margin |
| --- | --- | --- | --- | --- | --- |
| `kalman_cusum` | $h$ | 3.5 | **3.5** | 2143 | 2.1× |
| `kalman_glr` | $h$ | 10.0 | **10.0** | 2727 | 2.7× |
| `kalman_bocpd` | $P(r\le2)$ threshold | 0.10 | **0.15** | 1875 | 1.9× |
| `kalman_bocpd_fast` | $P(r\le2)$ threshold | 0.90 | **0.95** | 2143 | 2.1× |

ARL₀ rises roughly exponentially in the threshold, as CUSUM theory predicts —
`kalman_cusum`: $h=1 \to 7$ blocks, $2.5 \to 150$, $3.5 \to 2143$, $5 \to \ge 30{,}000$.

⚠️ `kalman_ewma_chart` ($L$) and `kalman_sr` ($h$) have **not** been calibrated. Their
thresholds are textbook defaults, so §9's comparison is not yet at matched ARL₀.

### 5.4 Fair per-family tuning — the correction

Each family's own forgetting parameter swept on the same scenario, 20 seeds. This is what
§5.1 cannot show.

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

`kalman_bocpd_fast` uses $H = 1/10$ — `oscillate`'s true half-period — rather than the
numerically optimal ≈ 1/1.4. **Tied to a real DGP timescale, not fitted to the test scenario.**

---

## 6. Analysis

### 6.1 What block aggregation costs

Two penalties, neither fixable by tuning:

- **Lag.** A statistic over the last $m$ ticks describes their *average*, centred about
  $m/2$ ticks in the past. At $m=20$ that is ~10 ticks of delay before the filter even runs.
- **A noise floor.** Each observation carries $\sigma_v^2 \approx 1.06/(m-3) \approx 0.062$,
  and $F_t = P_{t|t-1} + \sigma_v^2 \ge \sigma_v^2$ always. A perfectly converged filter
  still cannot read more precisely than one 20-point correlation allows.

They pull opposite ways on $m$, so $m=20$ is a compromise. Measured on `jump`:
`kalman_fast` scores 0.0921 over all ticks but 0.0828 read only at block boundaries — so
about half its gap to `xiao_ew` (0.0798) is the staircase hold, the rest is the floor.
**The `oscillate` plateau in §5.4 is this ceiling made visible.**

Per-tick estimators (`xiao_ew`, `hermite_ew`) never batch, so they pay neither.

Block-size sweep, single seed, RMSE:

| $m$ | $\sigma_\omega^2$ | `constant` | `drift` | `jump` | `stochastic` | `oscillate` |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | $10^{-2}$ | 0.0890 | 0.1130 | 0.1034 | 0.0879 | **0.2681** |
| 20 | $10^{-2}$ | 0.0713 | 0.0904 | 0.0943 | 0.0755 | 0.3247 |
| 50 | $10^{-2}$ | **0.0493** | **0.0657** | **0.0885** | **0.0654** | 0.4447 |

Big $m$ wins wherever the signal is slow and loses badly on `oscillate`, where ρ moves
substantially *inside* one block and the within-block-constant assumption breaks. At
$m=50$ the filter was worse than non-adaptive estimators — which is how the default got
caught.

### 6.2 Why BOCPD's win is structural and the Kalman filter's was not

A fixed λ commits to one memory length — one point on the variance/adaptation trade-off.
Sweeping λ slides along that curve. That is why §5.4 erased `kalman_fast`'s `oscillate`
advantage: it was at a better *point*, not on a better curve.

**BOCPD is not on the curve.** Its run-length posterior *switches* memory on evidence —
long while a segment holds, short the moment the data stop fitting. No fixed λ does that
at any tuning. That is why this gap survived fair tuning and the other did not.

### 6.3 The sweeps measure the theory directly

[change_detection.md](change_detection.md) Part 0 predicts the optimal memory is a property
of the *regime*, not the estimator, and should shorten as ρ moves faster. Writing
$\mathrm{ESS} \approx (2-\alpha)/\alpha$:

$$
\mathrm{MSE}(\mathrm{ESS}) \;\approx\;
\underbrace{\frac{\sigma^2}{\mathrm{ESS}}}_{\text{variance}}
\;+\;
\underbrace{\left(\dot\theta \, \frac{\mathrm{ESS}}{2}\right)^{\!2}}_{\text{lag}}
$$

| Scenario | Xiao optimum | Hermite optimum | implied ESS | regime $\dot\theta$ |
| --- | --- | --- | --- | --- |
| `constant` | cumulative | cumulative | $\to \infty$ | zero |
| `jump` | $\lambda = 0.99$ | $\lambda = 0.01$ | ≈ 199 ticks | zero except at one instant |
| `oscillate` | $\lambda = 0.95$ | $\lambda = 0.05$ | ≈ 39 ticks | large, continuous |

**Two unrelated estimator families land on the same ESS in each regime** — 199 ticks on
`jump`, 39 on `oscillate` — despite different internals and reciprocal λ conventions. The
optimum is set by the data, not the estimator, and shortens ~5× as the signal speeds up.

### 6.4 Why hard reset costs RMSE

By alarm time (~46 ticks on `jump`) the filter has already partly adapted. `reset()`
discards that and restarts from a flat prior, so the estimator re-climbs ground it had
covered. `kalman_cusum` (0.1146) and `kalman_glr` (0.1110) both score worse than the
`kalman_fast` (0.0880) they wrap. BOCPD never resets and avoids it entirely.
[§9](#14-the-reset-is-the-cost-why-detection-loses-to-plain-forgetting-on-jump) traces
this tick by tick.

### 6.5 Practical guidance

| Regime | Best choice | RMSE |
| --- | --- | --- |
| Stationary (`constant`) | `hermite` / `xiao`, cumulative | ~0.012 |
| Gradual or noisy (`drift`, `stochastic`) | `gaussian_mle_ew` | 0.056–0.062 |
| Rare discrete jumps (`jump`) | `kalman_bocpd` | 0.060 |
| Fast switching (`oscillate`) | tuned `xiao`/`hermite`, else `kalman_bocpd_fast` | 0.18 / 0.27 |

Not knowing the regime, `gaussian_mle_ew` is the cheap safe default. Mean RMSE over all
five scenarios:

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

**The sophisticated machinery does not beat a plain EWMA on average.** It *redistributes*
accuracy toward change-heavy regimes. Use it when discrete regime change is the thing you
care about, not as a general upgrade.

⚠️ The average is unweighted over five scenarios chosen for this suite, not a real-world
change-frequency distribution. Treat it as a summary, not a ranking.

---

## 7. Bugs found and fixed

**1. The BOCPD alarm could never fire.** $P(r_t=0)$ is identically $H$ under a constant
hazard (§3.6), so it carried no information. Replaced with $P(r_t \le 2)$.

**2. BOCPD burn-in false alarms — found only by multi-seed.** One seed showed 2 false
alarms before the jump, easily dismissed as variance. Twenty seeds showed the *same* ~2.1
on nearly every seed: systematic. Cause: with fewer than $c+2$ blocks of history every
hypothesis trivially satisfies $r \le c$, so $P(\text{short run}) \approx 1$ whatever the
data says. 100% alarm rate on blocks 0–1 across 200 stationary seeds before the fix, 0%
after. False alarms fell from $2.10\pm0.07$ to $0.10\pm0.07$.
**This is the case for multi-seed as a correctness tool, not an error-bar formality.**

**3. `gaussian_mle` latched to NaN on heavy tails — found by the cross-family grid.** One
seed in 20 (seed 57, `constant`, Gumbel DGP) poisoned the whole mean. The score carries
$1/(1-\rho^2)^2$ while $\partial\rho/\partial\theta = 1-\rho^2$ cancels only one power, so
$s_\theta \sim (1-\rho^2)^{-1}$. A joint extreme — which Gumbel's upper tail supplies and a
Gaussian DGP essentially never does — pushes $\theta$ past $\tanh$ saturation, $\rho$ becomes
exactly 1 in floating point, and two opposite infinities latch $\theta$ at NaN for the rest
of the stream. Fixed by clamping $\theta$ to $\operatorname{atanh}(0.999)$ and skipping
non-finite scores — guards `ParametricCopulaMLE` already had. All 13 published Gaussian
`jump` numbers are unchanged, so the clamp doesn't bind on Gaussian data.

**4. `xiao_win` had never run.** Registered in `registry.hpp` but missing from the `keys`
lists in `main.cpp` and `multiseed.cpp`. No error — just no row. The two lists are now an
invariant in `CLAUDE.md`.

**5. An unknown `COPULA` silently fell back to Gaussian.** Harmless interactively; in a
scripted grid a typo would have produced Gaussian results labelled as another family.
`multiseed.cpp` now rejects unknown names up front.

**6. Overclaim on `oscillate`** — documentation, corrected in §5.4.

**7. Two lists that had to agree, three times over.** `registry.hpp` keeps
`make_estimator` branches and `all_estimator_keys()` separately, and every combination of
disagreement has now bitten: `xiao_win` had a branch and no entry (ran never, reported
nothing); the three `_vff` keys had entries and no branch (`make_estimator` returned
`nullptr`, the harness dereferenced it, segfault with no diagnostic); `particle` had a
branch and no entry again.

Both directions are now checked. `check_registry()` runs at the top of `main()` in both
harnesses and refuses to start if a listed key does not construct; `make check-registry`
compares the two lists in the source and fails on either kind of mismatch.

Also fixed: `block=50` broke the within-block-constant assumption on `oscillate` (swept,
corrected to 20), and a `%.1f` print bug made distinct calibration thresholds print
identically.

---

## 8. Misspecification: the multi-seed grid across DGP families

Everything above is Gaussian-copula. This section runs the same 20-seed protocol over all four DGP
families — Gaussian, Clayton, Gumbel, Student-$t$ ($\nu=4$) — and all 22 registered estimators
(`./run_multiseed_grid.sh 20 42`). It closes the two real coverage gaps: `xiao_win`, which was
registered in `registry.hpp` but present in neither harness's `keys` list and so had **never been
run**, and the 8 parametric-copula keys, which were single-seed and Gaussian-DGP only.

### 8.1 ⚠️ Cross-family RMSE is only comparable on three of the five scenarios

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

### 8.2 RMSE by DGP family (20 seeds, $N=5000$, burn-in 500)

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

All 22 × 5 × 4 cells, colour-coded, are in `results/multiseed_heatmap.png`
(`python3 plot_multiseed_heatmap.py`); the raw numbers are in
`results/multiseed_<scenario>[_<family>].csv`. Colour there is normalised *within* each column,
since absolute RMSE spans 0.012 (`constant`) to 0.44 (`oscillate`) and a global scale shows nothing;
the clipped `drift`/`oscillate` columns are outlined for the §8.1 reason.

### 8.3 What the grid shows

**1. BOCPD's `jump` win survives every family — the strongest result here.** Best in column
on `jump` for all four: 0.0598 / 0.0602 / 0.0589 / 0.0633, beating the best fixed-λ
nonparametric by 25–30% everywhere. §6.2 argued this is structural, and that argument never
used Gaussianity. The grid is the evidence: it holds under lower-tail dependence (Clayton),
upper-tail (Gumbel) and heavy joint tails ($t_4$).

The whole column order barely moves across families — that is §4.4's marginal invariance
showing up at the level of the leaderboard.

**2. Misspecification is wildly asymmetric.** `gaussian_mle_ew` degrades only mildly
off-family (0.0558 → 0.0671 / 0.0592 / 0.0680) and stays mid-pack. **The Gaussian copula is a
forgiving wrong answer in ρ-space.** `clayton_mle` is not: 0.1589 on Gumbel `constant`, 10× its
own on-family 0.0154. Gumbel-on-Clayton is half as bad (0.0790).

**Fitting Clayton to non-Clayton data is the one genuinely dangerous choice** — its lower-tail
structure is actively wrong for upper-tail data, not merely uninformative.

**3. `copula_select` finds the right family — cumulatively.** It matches the correct family's
MLE exactly on that family's own data (Clayton 0.0154, Gumbel 0.0168, $t$ 0.0131 on `constant`).
Costs nothing when right.

But `copula_select_ew` loses to the right family's `_ew` on Clayton `drift` (0.1039 vs 0.0717).
Under forgetting it chooses on a short window and churns. **Selection and forgetting interact
badly — an open problem, not a tuning detail.**

**4. `xiao_win`, measured for the first time, adds nothing over `windowed_rank`.** They agree
to ~0.5% (0.0275 vs 0.0271; 0.1183 vs 0.1170), as two exact sliding windows differing only in
Xiao's binning should. It uses less storage (5824 B vs 8000 B), so it is marginally better.
Recorded so nobody re-runs it.

**5. The grid found a real bug** — the `gaussian_mle` NaN latch (§7). Second one the
replication tier caught that single seeds missed.

### 8.4 What this does not settle

- **Only `constant`/`jump`/`stochastic` read across families** (§8.1). `drift`/`oscillate`
  need a rotated or mixture Archimedean construction that can represent negative dependence.
- **$\nu=4$ is fixed** for both the $t$ DGP and `t_mle`, so that column is a matched-ν best
  case and overstates `t_mle` against unknown tail weight.
- **Detectors are calibrated on Gaussian innovations only** (§3.7). Thresholds were never
  re-derived per family. Alarm counts staying sane off-family is encouraging, not calibration.

---

## 9. Rank-PIT: the marginals, finally handled properly

Added 2026-09-08. Layer 1. Everything above this section was estimated from
**standardized** marginals — Welford mean and variance — which is exact only if the
data is Gaussian. `hermite_pit` replaces that with rank-PIT plus a probit.

### 9.1 The mechanism

```cpp
sx = RankPit::probit(pit_x_.transform(x));   // rank -> u in (0,1) -> normal score
```

`transform` ranks the value against a bounded sorted window (500 by default), returning
$(r + \tfrac12)/(n+1)$; the probit puts it back on the real line where the Hermite basis
expects it. Exact rank-PIT needs every past observation, which is not streaming-admissible;
the window is what buys $O(\log W)$ ranking and bounded memory.

### 9.2 The decisive test

A strictly monotone transform of a marginal leaves the copula **exactly** unchanged. So any
movement in $\hat\rho$ is pure marginal contamination. Gaussian copula, $\rho=0.6$
(true $\rho_S = 0.5819$), 10 seeds, $N=20{,}000$, mean $|\hat\rho - \rho_S|$:

| marginal $x \mapsto$ | `hermite_ew` | `hermite_pit` | `gaussian_mle_param` | `gaussian_mle_pit` | `kalman_mid` |
| --- | --- | --- | --- | --- | --- |
| identity | **0.0187** | 0.0274 | 0.0273 | 0.0249 | 0.0198 |
| $e^x$ (skewed) | 0.0386 | 0.0274 | 0.0424 | 0.0249 | **0.0198** |
| $x^3$ (heavy) | 0.0695 | 0.0274 | 0.0365 | 0.0249 | **0.0198** |

**Three estimators are exactly invariant**: `hermite_pit`, `gaussian_mle_pit`, and
`kalman_mid` — the last with no work at all, because `block_spearman()` ranks within each
block, so the Kalman observation channel never had the marginal problem (§3.1 said so; this
is the measurement). `kalman_mid` is also the most accurate of the invariant three.

**Rank-PIT is free for the MLE and costs efficiency for Hermite.** `gaussian_mle_pit`
(0.0249) beats the standardized `gaussian_mle_param` (0.0273) even on clean Gaussian
marginals, whereas `hermite_pit` (0.0274) loses to `hermite_ew` (0.0187) there. The MLE's
`pseudo()` divides by a *running* Welford variance that is noisy in its own right; a
500-window rank is better conditioned. **So the efficiency↔robustness trade is not
universal — it depends on what the "efficient" method is actually doing.**

**`hermite_pit` is exactly constant.** Not approximately — ranks are literally unchanged by
a monotone map, so the estimate cannot move. `hermite_ew` degrades **3.7×** under the cube
transform.

### 9.3 What it means

**This is the efficiency↔robustness trade of the thesis question, in one table.** On clean
Gaussian marginals standardization is *better* (0.0187 vs 0.0274) — it is the efficient
choice when its assumption holds. The 0.0274 is the price of ranks, the same kind of
constant as **[Fieller, Hartley & Pearson 1957]**'s 1.06 inflation.

The moment the marginals are not Gaussian, that price buys unbounded protection.

**It also explains why §8's family sweep saw only 2–8%.** Those DGPs have Gaussian
marginals, so standardization is nearly correct and there is almost nothing for ranks to
recover:

| DGP family | `hermite_ew` | `hermite_pit` | |
| --- | --- | --- | --- |
| gaussian | 0.0520 | 0.0486 | −6.5% |
| clayton | 0.0604 | 0.0558 | −7.6% |
| gumbel | 0.0566 | 0.0553 | −2.3% |
| t | 0.0604 | 0.0575 | −4.8% |

**Changing the copula family does not test the marginal assumption.** Only changing the
marginals does. §4.4's invariance check used an *affine* shift — precisely what
standardization already removes — so it could never have caught this.

### 9.4 ⚠️ What this does not establish

- **Xiao is still not converted.** Hermite and the copula MLEs now have rank-PIT variants;
  the Kalman channel never needed one. Xiao bins marginals into equiprobable cells with
  estimated cutpoints, which is rank-*like* but not rank-exact — untested.
- **The `_pit` variants are additions, not replacements.** Every number in §5–§8 is still
  on the old marginals.
- **The window is a new hyper-parameter** (500). Too short and the rank scale is noisy;
  too long and it cannot follow a genuinely changing marginal.
- **Real returns are not tested.** `exp` and `x^3` are severe but synthetic.
- **Distinction 3 is now live.** Ranks shift as data arrives, so the pseudo-observations are
  not i.i.d. — which is exactly the assumption every ARL₀ result in
  [§13](#17-references) relies on. Rank-PIT fixes the marginal problem and sharpens the
  calibration one.

---

## 10. Variable forgetting: adaptation without a detector

Added 2026-09-07. VFF is the layer-3 rival to explicit detection — adapt the memory
continuously instead of alarming and restarting. Built for all three estimator families,
not just the filter.

### 10.1 The mechanism, three lines each

**Kalman** — standardized innovations have variance 1 under a correct model, so a
sustained excess means the filter is under-modelling change:

```cpp
const double e2 = (innovation * innovation) / F;   // averages 1 when correct
C_ += vff_rate * (e2 - C_);
sigma_omega2_ *= std::pow(C_, vff_rate);           // scale memory by the excess
```

**MLE** — the score has mean zero when θ is right, so a systematically one-signed score
means θ is lagging:

```cpp
score_mean_ += vff_rate_ * (score_phi - score_mean_);
score_var_  += vff_rate_ * (score_phi * score_phi - score_var_);
const double drift = score_mean_ * score_mean_ / score_var_;
lambda_ = lambda0_ * std::pow(1.0 + drift, -10.0 * vff_rate_);   // retention: down = shorter
```

**Hermite** — ratio of recent to long-run coefficient surprise:

```cpp
fast_ += 0.05  * (surprise - fast_);
slow_ += 0.002 * (surprise - slow_);
lambda_ = config_.lambda * std::pow(fast_ / slow_, config_.vff_rate);  // weight: up = shorter
```

⚠️ Note the mirror image: for the MLE λ is *retention* so shortening memory **lowers** it;
for Hermite λ is the *update weight* so shortening **raises** it. Same trap as `registry.hpp`.

### 10.2 Results (20 seeds)

| scenario | `hermite_ew` → `_vff` | `gaussian_mle_ew` → `_vff` | `kalman_fast` → `kalman_vff` |
| --- | --- | --- | --- |
| `constant` | 0.0520 → **0.0495** | 0.0375 → 0.0514 | 0.0712 → **0.0341** |
| `drift` | 0.0700 → **0.0689** | 0.0624 → 0.0786 | 0.0921 → **0.0773** |
| `jump` | 0.0815 → **0.0806** | 0.0745 → **0.0708** | 0.0880 → 0.0993 |
| `oscillate` | 0.3490 → **0.3469** | 0.3508 → **0.3175** | 0.3166 → **0.2690** |
| **mean** | 0.1381 → 0.1365 (−1.2%) | 0.1313 → 0.1296 (−1.3%) | 0.1420 → **0.1199 (−15.6%)** |

### 10.3 What it means

**VFF is insurance against bad tuning, not a free upgrade.** It recovers 15.6% on the
Kalman filter, whose two shipped configs sit an order of magnitude apart in
$\sigma_\omega^2$ with neither right for most scenarios, and ~1% on Hermite and the MLE,
which were already near a good operating point.

**It also beats detection on detection's own ground.** On `jump`, `kalman_vff` scores
0.0993 against `kalman_cusum`'s 0.1146 — continuous adaptation over detect-and-restart,
which is what [§10](#14-the-reset-is-the-cost-why-detection-loses-to-plain-forgetting-on-jump)
predicted. No reset, so nothing to throw away.

### 10.4 ⚠️ What this does not establish

- **`vff_rate` is itself a hand-picked constant** (0.05 / 0.05 / 0.5). The knob moved up a
  level rather than disappearing. Whether VFF is genuinely tuning-free needs a sweep.
- **No sweep against best-tuned fixed λ.** §5.4 shows a tuned `xiao` reaches 0.1826 on
  `oscillate`; `kalman_vff`'s 0.2690 is well short of that. VFF beats *these* fixed configs,
  not the best achievable ones.
- **The MLE variant is worse on `constant` and `drift`.** The score-drift signal appears to
  over-react on slow regimes. Not diagnosed.

### 10.5 A harness trap worth recording

`hermite_vff`, `gaussian_mle_vff` and `kalman_vff` were added to `all_estimator_keys()`
while their `make_estimator` branches were missing. `make_estimator` returns `nullptr` for
an unknown key and the harness dereferences it — **segfault, no diagnostic**.

This is the mirror of the `xiao_win` bug (§7): that key was *constructible but unlisted*, so
it silently never ran; these were *listed but unconstructible*, so they crashed. Both come
from the same two-lists-must-agree invariant. A null check in the harness would turn the
crash into a message naming the key.

---

## 11. The particle filter: generality that mostly costs

Added 2026-09-08. Layer 2. The Kalman filter is optimal only if the state moves by
Gaussian increments — one $\sigma_\omega^2$ serving both quiet stretches and breaks. A
particle filter carries the posterior as samples, so the increment distribution is free.
Here it is a jump-diffusion mixture: mostly small steps, a rare large one. Long memory
while a regime holds, an instant move when it does not, **with no detector and no reset**.

Observation likelihood is Student-$t$ ($\nu=5$) rather than Gaussian, so one odd block
cannot collapse the weights. Systematic resampling when ESS drops below half. $O(P)$ per
block with $P = 400$.

### 11.1 Results (20 seeds)

| scenario | `kalman_fast` | `kalman_vff` | `particle` | `particle_fast` |
| --- | --- | --- | --- | --- |
| `constant` | 0.0712 | **0.0341** | 0.0872 | 0.1165 |
| `drift` | 0.0921 | **0.0773** | 0.1199 | 0.1472 |
| `jump` | **0.0880** | 0.0993 | 0.1099 | 0.1288 |
| `oscillate` | 0.3166 | 0.2690 | 0.3250 | **0.2610** |

**It wins only on `oscillate`**, where ρ moves fastest and the Gaussian-increment
assumption is most wrong. Everywhere else the strictly more general model loses to the
simpler one.

### 11.2 Why generality costs here

400 particles **approximate** the posterior the Kalman filter computes **exactly**. When
the linear-Gaussian model is roughly right, that approximation is pure added variance. The
jump component makes it worse: 1% of particles take a large step every block whether or not
anything has happened, so the filter pays a standing variance premium for readiness it
rarely uses.

### 11.3 The same trade, three times over

| Simpler, more assuming | More general | The general one wins when |
| --- | --- | --- |
| standardization | rank-PIT (§9) | marginals are not Gaussian |
| fixed λ | VFF (§10) | the fixed λ was badly chosen |
| Kalman | particle filter (§11) | ρ moves fast (`oscillate`) |

Three independent mechanisms, one shape: **the assuming method wins where its assumption
holds and loses unboundedly where it does not.** That is the efficiency↔adaptation question
of [roadmap.md](roadmap.md), measured three separate ways.

### 11.4 ⚠️ What this does not establish

- **Nothing is tuned.** `particles`, `jump_prob`, `jump_scale` and `nu` are all first
  guesses. A sweep might change the ordering; §5.4 is the warning about exactly this.
- **The comparison is not like-for-like on cost.** $O(400)$ per block against $O(1)$.
- **EKF and UKF are still untried** — they sit between these two and would separate "the
  posterior is non-Gaussian" from "the transition is non-linear".

---

## 12. EKF and UKF: does the Fisher transform earn its place?

Added 2026-09-08. Layer 2, completing it.

An EKF on `KalmanLocalLevel`'s model would be pointless — after the Fisher transform the
observation is *already* linear in the state, so it would reduce to the Kalman filter. The
only meaningful nonlinear version is to **drop the transform**: keep the state as
$z = \operatorname{atanh}\rho$ but observe $\hat\rho$ raw through $h(z) = \tanh z$, with the
variance the raw statistic actually has:

$$\operatorname{Var}(\hat\rho) \approx \frac{1.06\,(1-\rho^2)^2}{m-1}$$

That heteroscedasticity is precisely what §0.9's transform exists to remove. So the
comparison asks one question: **is variance stabilization worth more than handling the
nonlinearity properly?**

### 12.1 Results, at matched process noise

$\sigma_\omega^2 = 10^{-3}$ throughout — `kalman_mid` exists only so the comparison is
like-for-like, the sole difference being the observation model. 20 seeds.

| scenario | `kalman_mid` (Fisher) | `ekf` | `ukf` |
| --- | --- | --- | --- |
| `constant` | **0.0387** | 0.0477 | 0.0460 |
| `drift` | **0.0689** | 0.0722 | 0.0716 |
| `jump` | **0.0840** | 0.0977 | 0.0968 |
| `oscillate` | 0.4058 | **0.3992** | 0.3997 |

**The Fisher transform wins on three of four**, and `oscillate` is close enough to be a
wash. Transforming the nonlinearity away beats handling it directly.

### 12.2 Two things this shows

**Variance stabilization is doing more work than linearization.** The transform buys both,
and the EKF/UKF recover only the second. They still have to model the heteroscedastic
noise, and the $(1-\rho^2)^2$ form is itself an approximation — so the nonlinear filters
trade one approximation for another while giving up the constant-variance gift that pins
$\sigma_v^2$ (§3.1).

**UKF ≈ EKF everywhere.** Differences are inside one SE. $\tanh$ is only mildly nonlinear
over the range ρ actually occupies, so the UKF's second-order correction has nothing to
correct. That is a useful negative: it locates the problem in the *noise model*, not in the
curvature.

### 12.3 ⚠️ A near-miss worth recording

The first run had `ekf`/`ukf` at $\sigma_\omega^2 = 10^{-3}$ against `kalman`
($10^{-4}$) and `kalman_fast` ($10^{-2}$), and the EKF appeared to *win* on `drift`
(0.0722 vs `kalman_vff`'s 0.0773). At matched process noise `kalman_mid` scores 0.0689 and
the win evaporates.

**That is §5.4's mistake, live.** Any comparison across filter families has to fix the
forgetting parameter first, or it measures the tuning rather than the method.

### 12.4 Confirmed across all four DGP families

20 seeds, full grid. `constant` / `jump`, RMSE:

| estimator | gaussian | clayton | gumbel | $t$ |
| --- | --- | --- | --- | --- |
| `kalman_mid` | **0.0387** / **0.0840** | **0.0444** / **0.0898** | **0.0452** / **0.0877** | **0.0455** / 0.0901 |
| `ekf` | 0.0477 / 0.0977 | 0.0530 / 0.1028 | 0.0531 / 0.1013 | 0.0548 / 0.1026 |
| `ukf` | 0.0460 / 0.0968 | 0.0514 / 0.1021 | 0.0515 / 0.1005 | 0.0531 / 0.1019 |
| `kalman_vff` | 0.0341 / 0.0993 | 0.0482 / 0.0993 | 0.0463 / 0.0978 | 0.0508 / **0.0948** |
| `hermite_ew` | 0.0520 / 0.0815 | 0.0604 / 0.0897 | 0.0566 / 0.0855 | 0.0604 / 0.0902 |
| `hermite_pit` | 0.0486 / 0.0804 | 0.0558 / 0.0869 | 0.0553 / 0.0843 | 0.0575 / 0.0878 |

**The Fisher transform beats EKF/UKF in every cell.** Not a Gaussian artifact.

**`hermite_pit` beats `hermite_ew` in every cell too** — consistently, though only by 2–8%,
because these DGPs have Gaussian marginals (§9.3).

⚠️ **`kalman_vff`'s advantage is partly Gaussian-specific.** On `constant` it leads at
0.0341 on Gaussian but degrades to 0.0482–0.0508 off-family, while `kalman_mid` moves only
0.0387→0.0455. The innovation-variance signal VFF drives on assumes a correctly specified
noise model, and off-family that assumption weakens. Same pattern as everything else in §1.

---

## 13. Fisher scoring for the copula MLEs: which of three ideas survive

Added 2026-09-08. The MLE's optimiser, not a detector. All three candidates are $O(1)$ and
change no interface.

### 13.1 The ablation (20 seeds, Gaussian DGP)

`gaussian_mle_param` is the control — same class, none of the changes.

| scenario | control | expected info | Polyak–Ruppert | damped | einfo+damped | all three |
| --- | --- | --- | --- | --- | --- | --- |
| `constant` | 0.0427 | 0.0375 | 0.0376 | 0.0363 | **0.0362** | 0.0289* |
| `drift` | 0.0719 | 0.0624 | 0.0730 | 0.0630 | **0.0619** | 0.0617* |
| `jump` | 0.0847 | 0.0745 | 0.0988 | **0.0722** | 0.0740 | 0.0875 |
| `oscillate` | 0.3829 | **0.3508** | 0.4619 | 0.3842 | 0.3578 | 0.4619 |

\* the all-three column inherits Polyak–Ruppert's damage on `jump` and `oscillate`.

**Verdict**

- **Expected information — adopt**, but for *stability*: the mean gain is significant only on `oscillate` (§13.2), while the across-seed SE drops 5–9× everywhere.
- **Damped step — optional.** Helps three of four; replaces the `kMaxStep` clamp with
  something that has a reason behind it and needs no per-family tuning.
- **Polyak–Ruppert — reject.** It recovers asymptotic efficiency *for an estimator
  converging to a fixed point*. A forgetting estimator is deliberately never converging; it
  is tracking. Averaging the iterates only adds lag, which is why it helps `constant` and
  costs 21% on `oscillate`. Textbook-correct, wrong problem.

### 13.2 Expected information buys stability, not accuracy

⚠️ **Correction to 13.1.** The einfo column was read as "wins all four scenarios". With
error bars it wins **one**. Difference of means against the pooled SE:

| scenario | control | expected info | difference | significance |
| --- | --- | --- | --- | --- |
| `constant` | 0.0427 ± 0.0061 | 0.0375 ± 0.0011 | 0.0053 | 0.9σ — noise |
| `drift` | 0.0719 ± 0.0073 | 0.0624 ± 0.0015 | 0.0094 | 1.3σ — noise |
| `jump` | 0.0847 ± 0.0099 | 0.0745 ± 0.0011 | 0.0101 | 1.0σ — noise |
| `oscillate` | 0.3829 ± 0.0023 | 0.3508 ± 0.0018 | 0.0321 | **11.0σ** |

**The consistent effect is in the SE, not the mean.** Observed information has **5–9× the
across-seed spread** in every scenario — 0.0061 vs 0.0011, 0.0073 vs 0.0015, 0.0099 vs
0.0011. Same expected accuracy, far less reliable.

That resolves what looked like a contradiction. A standalone probe at fixed $\rho$ compared
*means* and found the two indistinguishable at every $\rho \in \{0.3, 0.6, 0.85, 0.95\}$ —
correctly, because the means *are* the same. The harness means merely looked separated
because the observed-information mean is dragged around by its own variance.

It also explains why `oscillate` is the one genuine win: ρ sweeps its entire range there, so
the observed score's $1/(1-\rho^2)$ factor — the same factor behind §7's NaN bug — is
excited repeatedly, and each excitation inflates `info_` for the ~100 blocks the $\lambda=0.99$
decay retains it.

**Revised verdict: adopt expected information for variance reduction**, not for accuracy.
For a thesis that reports mean ± SE over seeds, halving the SE is worth having on its own.

⚠️ Gaussian only. No closed form was derived for Clayton, Gumbel or $t$, so those families
still use the observed information.

⚠️ **Method note.** §13.1's table was written from point estimates alone. The significance
test should have come first — it is the same discipline §5.4 and §12.3 record, applied to
error bars rather than to tuning.

---

## 14. The reset is the cost: why detection loses to plain forgetting on `jump`

Added 2026-09-06, after registering `kalman_ewma_chart` and `kalman_sr`.

### 14.1 The comparison is controlled

The two detectors added here are the EWMA control chart (**[Roberts 1959]** ✓) and Shiryaev–Roberts
(**[Shiryaev 1963]** ⚠️; asymptotically optimal for Pollak's conditional-average delay criterion,
**[Pollak 1985]** ⚠️). All four detectors wrap `KalmanLocalLevel` at `sigma_omega2 = 1e-2` — **the same
configuration as `kalman_fast`**. So `kalman_fast` vs any detector is the same base
filter with and without an alarm mechanism, not fast-vs-slow.

`jump`, Gaussian DGP, 20 seeds:

| estimator | RMSE | alarms/run | delay (ticks) | misses |
| --- | --- | --- | --- | --- |
| `kalman_fast` | **0.0880** | 0.00 | — | — |
| `kalman_ewma_chart` | 0.1042 | 0.60 | 74.0 | 8 |
| `kalman_glr` | 0.1110 | 0.90 | 49.6 | 3 |
| `kalman_sr` | 0.1139 | 1.00 | 51.6 | 1 |
| `kalman_cusum` | 0.1146 | 1.10 | 46.4 | 1 |
| `kalman_bocpd` | 0.0598 | 1.45 | 31.9 | 3 |

Adding a detector costs 18–30% RMSE, **monotonically in alarm count** — order the
hard-reset rows by alarms and the RMSE ordering follows exactly.

### 14.2 Per-tick trace: the reset destroys a nearly-converged estimate

Jump at $t=2500$, $\rho_s: 0.191 \to 0.786$. Gap % is the fraction of the jump
still unclosed.

| ticks after | blocks | `kalman_fast` | gap | `kalman_cusum` | gap |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0 | 0.167 | 104% | 0.167 | 104% |
| 20 | 1.0 | 0.410 | 63% | 0.410 | 63% |
| 40 | 2.0 | 0.517 | 45% | 0.517 | 45% |
| 60 | 3.0 | 0.679 | **18%** | **0.000** | **132%** |
| 80 | 4.0 | 0.641 | 24% | 0.000 | 132% |
| 100 | 5.0 | 0.699 | 15% | 0.688 | 16% |
| 140 | 7.0 | 0.747 | 6% | 0.751 | 6% |

The steady-state gain is $K \approx 0.36$ per block, so the unaided filter closes
$\approx 37\%$ of the gap per block and the gap decays as $0.64^n$. By block 3 it has
covered 82% of the jump on its own.

That is exactly when the detector fires — and `reset()` sends the estimate to
**0.000**, a 132% gap, *worse than before the jump*. It sits there two blocks, then
returns to 0.688, roughly where the unaided filter already was when the alarm rang.
Forty ticks at $\hat\rho=0$ against a truth of 0.786 is $\approx 0.62$ squared error
per tick, which accounts for the whole RMSE gap.

**The detector is not wrong about *when*.** Three blocks after a real change is
respectable. The problem is *what the alarm does*: `reset()` to an empty buffer
teleports the estimator to ignorance, not to the new truth, and it must then
re-accumulate at the same rate the filter was already converging.

### 14.3 At matched ARL₀ the four hard-reset detectors are indistinguishable

Re-run after calibration (§5.3), `jump`, 20 seeds:

| detector | RMSE | alarms/run | delay (ticks) | detected |
| --- | --- | --- | --- | --- |
| `kalman_glr` | 0.1110 ± 0.0032 | 0.90 | 49.6 ± 6.9 | 17/20 |
| `kalman_ewma_chart` | 0.1124 ± 0.0029 | 0.95 | 47.9 ± 4.9 | 18/20 |
| `kalman_sr` | 0.1139 ± 0.0023 | 1.00 | 51.6 ± 4.4 | 19/20 |
| `kalman_cusum` | 0.1146 ± 0.0027 | 1.10 | 46.4 ± 4.9 | 19/20 |
| `kalman_bocpd` | **0.0598 ± 0.0022** | 1.45 | **31.9 ± 3.4** | 17/20 |

**The four hard-reset detectors are within ~1σ of each other on RMSE (spread 0.0036
against SEs of ~0.003) and well inside 1σ on delay (46–52 against SEs of 4–7).**

That is the empirical answer to the Lorden-vs-Pollak question of §3.4: at matched
false-alarm rate, maximising over the change time (CUSUM, GLR), summing over it
(Shiryaev–Roberts) and smoothing it (EWMA chart) land in the same place on this problem.
The optimality criteria differ; the measured behaviour does not.

**Calibration also fixed the EWMA chart.** At the textbook $L=3.0$ it detected 12/20 with a
74-tick delay; at the calibrated $L=2.5$ it detects 18/20 at 47.9. The eight misses were the
threshold, not the method — which is what §14.5 warned about and is now closed.

### 14.4 The monotone alarm cost survives calibration

| alarms/run | RMSE |
| --- | --- |
| 0.90 (`kalman_glr`) | 0.1110 |
| 0.95 (`kalman_ewma_chart`) | 0.1124 |
| 1.00 (`kalman_sr`) | 0.1139 |
| 1.10 (`kalman_cusum`) | 0.1146 |

Still perfectly monotone, and now with the alarm rates set by a common ARL₀ rather than by
arbitrary thresholds. **Every reset costs accuracy, and the cost scales with how often you
reset.** `kalman_bocpd` alarms most (1.45) and scores best (0.0598) because it never
resets — the soft-decision control.

### 14.5 BOCPD is the control

`kalman_bocpd` (**[Adams & MacKay 2007]** ⚠️) alarms **most** (1.45/run) and has the **best** RMSE
(0.0598), breaking the monotone pattern outright. It never hard-resets — it reweights a
distribution over run lengths. The cost is not detecting; it is *committing*.

This is the hard-decision/soft-decision axis of
[change_detection.md](change_detection.md) appearing directly in RMSE.

### 14.4 Open — the obvious next test

Candidate fix: make `reset()` inflate the state covariance (the standard adaptive
filtering "covariance bump") or re-seed from the current estimate, rather than
clearing state. That keeps the alarm's information — *distrust the past* — without
discarding the estimate. **Untested.**

A cheap mechanism check: run a detector with `reset()` made a no-op. If RMSE returns
to `kalman_fast`'s 0.0880, the reset cost is confirmed as the entire story.

### 14.5 ⚠️ What this does not establish

- **Thresholds are uncalibrated.** `L = 3.0`, `h = 5.0` are textbook defaults. The
  spread of alarm rates is what makes the monotone relationship visible — it reads as
  a natural experiment, not a designed one. Not a thesis result until
  `calibrate_arl0` puts all four at matched $\mathrm{ARL}_0$.
- `jump` + Gaussian only; the other 19 grid files are unexamined.
- `kalman_fast`'s $\lambda$ was not tuned against these scenarios either.

---

## 15. Limitations

- **Only one layer-2 choice.** Every detector wraps the Kalman filter. Nothing here says
  whether detection helps estimators *in general* — see §0.3.
- **Detection delay is only half-measured.** ARL₀ calibration controls false alarms.
  Delay is reported but never compared against the Lorden/KL optimal-delay bound.
- **$\sigma_\omega^2$, hazard and $m$ are hand-picked.** Estimating them online (VFF) is
  the obvious next step and is what §3.3's parameterization makes possible.
- **Nothing runs live.** All results are offline; none of this is in `src/consumer/`.
- ~~Single DGP family~~ — superseded by
  [§8](#8-misspecification-the-multi-seed-grid-across-dgp-families).

---

## 16. Reproduction

```bash
cd experiments
mkdir -p results

make                    # ./experiment      single seed + per-tick CSV
make multiseed          # ./multiseed       error bars
make calibrate_arl0     # ./calibrate_arl0  threshold calibration

for s in constant drift jump stochastic oscillate; do ./multiseed $s 20; done
./calibrate_arl0 1000

./run_multiseed_grid.sh 20 42            # 5 scenarios x 4 families, ~12 min
./run_multiseed_grid_parallel.sh 20 42   # same, across all cores, ~3 min
```

`COPULA=gaussian|clayton|gumbel|t` picks the DGP family (default `gaussian`). Non-Gaussian
families get a filename suffix so runs don't clobber each other. Outputs land in `results/`
(gitignored).

Code: `src/estimators/dependence/` — `kalman_local_level.hpp`, `kalman_cusum.hpp`,
`kalman_glr.hpp`, `kalman_ewma_chart.hpp`, `kalman_shiryaev_roberts.hpp`, `kalman_bocpd.hpp`,
`change_detector.hpp`. Parametric estimators in `parametric_copula.hpp` and
`copula_selector.hpp`, family math in `estimators/copula_family.hpp`, the DGP in
`src/streamgenerator/datasource/copula.hpp`. Configs in `registry.hpp`.

---

## 17. References

**✓ = in the Zotero library. ⚠️ = from memory, verify before it enters `bibliography.bib`.**
The ⚠️ set is tracked as an acquisition item in [todo.md](../todo.md).

**Layer 1 — what we assume**

- ✓ **[Sklar 1959]** — the decomposition: joint = marginals + copula.
- ✓ **[Nelsen 2006]** — *An Introduction to Copulas.* Standard reference.
- ✓ **[Fantazzini 2009]** — misspecified marginals bias elliptical copula correlations by
  up to 70%. The argument for rank-PIT (§0.5).
- ✓ **[Kojadinovic & Yan 2010]** — rank-based goodness-of-fit; the batch counterpart to
  `copula_select`.
- ✓ **[Patton 2006]**, ✓ **[Opschoor et al. 2021]** ✓ — dynamic / score-driven copulas, the
  literature this state-space framing sits beside.

**Layer 2 — how we estimate**

- ⚠️ **[Kalman 1960]** — the filter. ⚠️ **[Harvey 1989]**, ⚠️ **[Durbin & Koopman 2001]** —
  local-level model.
- ✓ **[Xiao 2019]** — online empirical-copula Spearman estimator.
- ✓ **[Stephanou & Varughese 2021]**, ✓ **[hermiter 2024]** ✓ — sequential Spearman via
  Hermite series; `external/hermiter`.
- ✓ **[Segers 2012]**, ✓ **[Bücher & Volgushev 2013]** ✓ — weak convergence of empirical and
  *sequential* empirical copula processes under serial dependence, with multiplier
  bootstrap. The theory §3.7's bootstrapped ARL₀ currently substitutes for empirically.
- ✓ **[Gregory 2019]** — streaming empirical copula with error bounds. Cumulative: no
  forgetting.
- ✓ **[Cormode 2011]** — sketches; the basis for defending `store_bytes`.
- ⚠️ **[Fieller, Hartley & Pearson 1957]** — the $1.06/(m-3)$ variance for Fisher-$z$'d
  Spearman (§0.8, §3.1). This pins $\sigma_v^2$, so it fixes the scale every detector
  threshold is expressed in.
- ⚠️ **[Welford 1962]** — the running mean/variance the standardization uses.
- ✓ **[Pereira, Costa & Gonçalves 2026]** — bootstrap-enhanced Fisher scoring for SSM
  parameters. Batch, not online, but the principled way to choose $\sigma_\omega^2$
  instead of hand-picking it.

**Layer 3 — the detectors**

- ⚠️ **[Page 1954]** — CUSUM.
- ✓ **[Roberts 1959]** — EWMA control chart; `kalman_ewma_chart`. In Zotero as the 2000
  Technometrics reprint.
- ⚠️ **[Shiryaev 1963]** — the Shiryaev–Roberts statistic; `kalman_sr`.
- ⚠️ **[Willsky & Jones 1976]** — GLR. ⚠️ **[Lai 1995]** ⚠️ — the window-limited form that is
  streaming-admissible.
- ⚠️ **[Adams & MacKay 2007]** — BOCPD. Taken from the paper itself, not memory.

**Layer 3 — optimality theory**

- ⚠️ **[Lorden 1971]** — the minimax criterion CUSUM optimises: worst case over change
  time *and* pre-change history.
- ⚠️ **[Pollak 1985]** — replaces Lorden's inner ess-sup with a conditional average;
  Shiryaev–Roberts is asymptotically optimal for it.
  > These two are why `kalman_cusum` vs `kalman_sr` compares **optimality criteria**, not
  > algorithms: one likelihood-ratio recursion, maximised over $\nu$ versus summed over it.
- ⚠️ **[Moustakides 1986]** — CUSUM is *exactly*, not merely asymptotically, Lorden-optimal.
- ⚠️ **[Basseville & Nikiforov 1993]** — innovation-based detection.
- ⚠️ **[Tartakovsky, Nikiforov & Basseville 2014]** — the modern survey.

**Not yet acquired** — nothing from these lines is in the library, and both are flagged as
high value: variable-forgetting-factor RLS (Haykin; Paleologu et al.), and conformal /
e-value anytime-valid testing (Vovk; Ramdas), which
[change_detection.md](change_detection.md) Part IV.1 calls the strongest novelty candidate.
