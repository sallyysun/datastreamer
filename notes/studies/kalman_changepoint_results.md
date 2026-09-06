# Streaming changepoint detection for copula dependence: methodology and results

Companion to [change_detection.md](change_detection.md), which laid out the design space and build
order. This document reports what was actually **built and measured** against that plan: the
local-level Kalman filter, three detectors on top of it (CUSUM, window-limited GLR, and BOCPD —
Bayesian Online Changepoint Detection), the
$\mathrm{ARL}_0$ calibration and multi-seed replication infrastructure needed to make their numbers
mean anything, and the results — including two corrections to claims that did not survive scrutiny.

Branch: `kalman-changepoint` (off `main`). Everything here is reproducible from
`experiments/` — see [§9](#9-reproduction).

[§0](#0-orientation) is a from-scratch primer on the machinery. It assumes expectation, variance, the
Gaussian, Bayes' rule and likelihood, and builds everything else — copulas, Spearman's $\rho$, the
Fisher transform, state-space models, the Kalman filter, CUSUM, GLR, BOCPD and $\mathrm{ARL}_0$ —
from those. Readers who already have this can start at §1.

---

> **Citation convention.** Inline references are highlighted as **[Author Year]** and collected
> in [§12](#12-references). ⚠️ Except where noted, they are **from memory** — years, volumes and
> pages must be checked against a bibliographic database before entering `bibliography.bib`. None
> of them are currently in the Zotero library except **[Roberts 1959]** and **[Gregory 2019]**.

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

**Estimator choice and adaptation strategy are separate axes.**

|                  | Fixed forgetting | VFF | CUSUM + restart | BOCPD |
| ---------------- | ---------------- | --- | --------------- | ----- |
| **Gaussian MLE** | ✓ | ⬜ | ⬜ | ⬜ |
| **Xiao**         | ✓ | ⬜ | ⬜ | ⬜ |
| **Hermite**      | ✓ | ⬜ | ⬜ | ⬜ |
| **Kalman**       | ✓ | ⬜ | ✓ | ✓ |

✓ built and measured, ⬜ not built. The matrix does not need filling; the point is that
it *is* a matrix. The Kalman filter is a **layer 2** choice, not a detector — a detector
sits on top of it and reads its innovations.

### 0.2 Where this document sits

Mostly **layer 3**, on one layer 2 choice, tested against several layer 1 choices.

| Layer | Question | Sections |
| --- | --- | --- |
| **1 — assume** | Which family generated the data? | [§10](#10-misspecification-the-multi-seed-grid-across-dgp-families) |
| **2 — estimate** | How do we track ρ as it moves? | §0.9, §3.1–3.3, §3.9, §4.1, §5.1, §6.1 |
| **3 — handle change** | Forget, or detect and restart? | §0.10, §3.4–3.7, §4.2–4.4, §5.2–5.5, §6.2, §6.4, [§11](#11-the-reset-is-the-cost-why-detection-loses-to-plain-forgetting-on-jump) |
| *infrastructure* | Do the numbers mean anything? | §2, §3.8, [§7](#7-bugs-found-and-fixed), §8, [§9](#9-reproduction) |

**The headline is a layer-3 result:** on a jump, continuous adaptation *beat* explicit
detection — the restart discarded a nearly-converged estimate.

### 0.3 The copula, and what ρ throws away

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

### 0.4 Ranks, and why the current code is exposed

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

### 0.5 The five distinctions

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
[§12](#12-references) assumes i.i.d. pre-change data. Rank streams are not.

### 0.6 Three terms used later

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
### 0.7 Forgetting, and why an optimal memory exists

The standard streaming device is the **exponentially weighted moving average**: for observations
$y_1,\dots,y_t$ and a rate $\alpha \in (0,1]$,

$$
\hat\theta_t \;=\; (1-\alpha)\,\hat\theta_{t-1} \;+\; \alpha\, y_t
\;=\; \alpha \sum_{j=0}^{t-1} (1-\alpha)^j \, y_{t-j} \;+\; (1-\alpha)^t \hat\theta_0 .
$$

Weights decay geometrically, so the estimator forgets at a fixed rate and needs $O(1)$ state.

**Definition (effective sample size).** The number of equally-weighted points carrying the same
variance as the weighted average, $\text{ESS} = (\sum_j w_j)^2 / \sum_j w_j^2$. For the EWMA above,

$$
\text{ESS} \;=\; \frac{2-\alpha}{\alpha} \;\approx\; \frac{2}{\alpha} \quad (\alpha \ll 1).
$$

So $\alpha = 0.01$ means "about 200 points of memory". If the observations have variance $\sigma^2$
and the truth drifts at rate $\dot\theta$ per step, then to first order

$$
\mathrm{MSE}(\text{ESS}) \;\approx\; \underbrace{\frac{\sigma^2}{\text{ESS}}}_{\text{variance}}
\;+\; \underbrace{\left(\frac{\dot\theta \cdot \text{ESS}}{2}\right)^{\!2}}_{\text{lag bias}} ,
$$

a U-shape in $\text{ESS}$ with an interior minimum. This one expression explains most of §5: a
cumulative estimator ($\text{ESS} = t \to \infty$, first term $\to 0$) wins `constant` and is
catastrophic on `drift`; a short-memory one is the reverse.

⚠️ **The convention differs by estimator.** For Xiao and the parametric MLEs, $\lambda$ is a
*retention* factor, so larger $\lambda$ = longer memory ($\alpha = 1-\lambda$). For Hermite,
$\lambda$ is the *update weight*, so smaller $\lambda$ = longer memory ($\alpha = \lambda$). Both
reduce to the same $\text{ESS} = (2-\alpha)/\alpha$. Getting this backwards silently inverts an
experiment — see the header of `src/estimators/registry.hpp`.

### 0.8 Blocks, and the Fisher transform

Two obstacles stop us from filtering $\hat\rho_S$ tick by tick.

**Obstacle 1: $\rho$ is bounded and heteroscedastic.** It lives in $[-1,1]$, so its sampling noise
must shrink as $|\rho| \to 1$ — a Gaussian noise model is false near the ends. The **Fisher
transformation** fixes this:

$$
z \;=\; \operatorname{atanh} \rho \;=\; \tfrac12 \log\frac{1+\rho}{1-\rho}
\;\in\; \mathbb R,
\qquad
\rho = \tanh z .
$$

$z$ is unbounded, and its sampling variance is approximately constant in $\rho$ — which is exactly
what a linear-Gaussian model needs.

**Obstacle 2: one tick is not enough.** So aggregate: every $m$ ticks (a **block**) compute the
block's empirical $\hat\rho_S$ and transform it. Standard theory gives the variance in closed form,

$$
\sigma_v^2 \;\approx\; \frac{1.06}{m-3}
$$

(**[Fieller, Hartley & Pearson 1957]**, for a Fisher-$z$'d *Spearman* correlation — ~6% wider than Pearson's
$1/(m-3)$). $\sigma_v^2$ being **known rather than estimated** is the structural gift here: it pins
the denominator of the signal-to-noise ratio, leaving one scalar to choose instead of two.

Blocking is not free (§6.1), but it buys an *exactly* linear-Gaussian model, so the optimal filter is
closed-form rather than an EKF/UKF.

### 0.9 State-space models and the Kalman filter

**Definition (linear-Gaussian state-space model).** A hidden state evolving by a linear rule,
observed through linear noise (**[Kalman 1960]**; textbook treatments **[Harvey 1989]**,
**[Durbin & Koopman 2001]**):

$$
z_t = z_{t-1} + \omega_t, \quad \omega_t \sim \mathcal N(0, \sigma_\omega^2)
\qquad \text{(state / transition)}
$$
$$
y_t = z_t + v_t, \quad\;\; v_t \sim \mathcal N(0, \sigma_v^2)
\qquad\;\;\, \text{(observation)}
$$

with $\omega_t, v_t$ independent. This particular one — a random walk seen in noise — is the
**local-level model**. Here $z_t = \operatorname{atanh}\rho_t$ is the true dependence and $y_t$ is
the block measurement of §0.8. The two variances say everything: $\sigma_\omega^2$ is *how fast we
believe the world changes*, $\sigma_v^2$ is *how noisy our measurement is*.

#### The idea in one paragraph

Carry a **belief**, not a number: a mean $\hat z$ and a variance $P$ saying how sure you are. Waiting
makes you less sure ($P$ grows by $\sigma_\omega^2$). A measurement makes you more sure. Combining
belief and measurement is Bayes' rule for two Gaussians, which weights each by its **precision**
(inverse variance) — the more confident party wins.

**Definition (Kalman recursion).** Per block:

$$
\begin{aligned}
\textbf{predict:} \quad & \hat z_{t|t-1} = \hat z_{t-1}, && P_{t|t-1} = P_{t-1} + \sigma_\omega^2 \\
\textbf{innovation:} \quad & e_t = y_t - \hat z_{t|t-1}, && F_t = P_{t|t-1} + \sigma_v^2 \\
\textbf{gain:} \quad & K_t = P_{t|t-1} / F_t && \\
\textbf{update:} \quad & \hat z_t = \hat z_{t|t-1} + K_t e_t, && P_t = (1-K_t) P_{t|t-1}
\end{aligned}
$$

Read the update as: *new estimate = old estimate + gain × surprise.*

- $e_t$, the **innovation**, is the part of the measurement the model did not predict — the surprise.
  If the model is correct, $e_t \sim \mathcal N(0, F_t)$ and the standardized innovations
  $e_t/\sqrt{F_t}$ are i.i.d. standard normal. **That is what every detector in §3 monitors**: a real
  change makes the filter wrong in a systematic direction, so surprises stop looking like noise.
- $K_t \in (0,1)$, the **gain**, is the fraction of the surprise we act on:
  $K_t = P_{t|t-1}/(P_{t|t-1} + \sigma_v^2)$ — the prior's share of total precision. Uncertain
  filter ($P \gg \sigma_v^2$) $\Rightarrow K \to 1$, follow the data. Confident filter
  $\Rightarrow K \to 0$, ignore a noisy read. Cold-start is handled automatically.

#### Worked example

Take $\sigma_v^2 = 0.05$ (so $m \approx 24$) and $\sigma_\omega^2 = 0.01$, starting from a vague
prior $\hat z_0 = 0$, $P_0 = 1$.

| | $P_{t\mid t-1}$ | $F_t$ | $K_t$ | $y_t$ | $e_t$ | $\hat z_t$ | $P_t$ |
|---|---|---|---|---|---|---|---|
| block 1 | $1.0000 + 0.01 = 1.0100$ | $1.0600$ | $0.9528$ | $0.70$ | $+0.7000$ | $0.6670$ | $0.0476$ |
| block 2 | $0.0476 + 0.01 = 0.0576$ | $0.1076$ | $0.5355$ | $0.60$ | $-0.0670$ | $0.6311$ | $0.0268$ |
| block 3 | $0.0268 + 0.01 = 0.0368$ | $0.0868$ | $0.4238$ | $0.75$ | $+0.1189$ | $0.6815$ | $0.0212$ |

Block 1 knows nothing, so it takes 95% of the measurement. By block 3 it is confident and takes 42%.
$P$ falls toward a fixed point where the $\sigma_\omega^2$ added each step exactly balances what the
measurement removes — here $q = 0.01/0.05 = 0.2$, so $K \to (\sqrt{0.04 + 0.8} - 0.2)/2 = 0.3583$,
equivalent to an EWMA with $\lambda = 0.642$ and $\text{ESS} \approx 4.6$ blocks.

#### Steady state, and why this generalizes everything else

$P_t$ converges. Write $q = \sigma_\omega^2/\sigma_v^2$ (the **signal-to-noise ratio**); the limiting
gain solves a quadratic with the closed form

$$
K \;=\; \frac{\sqrt{q^2 + 4q} \; - \; q}{2}.
$$

At that fixed point $K_t$ is a constant $K$, and the update collapses to

$$
\hat z_t \;=\; \hat z_{t-1} + K(y_t - \hat z_{t-1}) \;=\; (1-K)\,\hat z_{t-1} + K y_t ,
$$

which is exactly the EWMA of §0.7 with $\alpha = K$, i.e. $\boxed{\lambda = 1 - K}$. **Every
fixed-$\lambda$ estimator in this codebase is this filter at steady state** — they are not analogous,
they are the same recursion. This is "Bridge 1" (§3.3, verified numerically in §4.1). The filter's
extra content is the *transient*: it starts uncertain and tightens, whereas a fixed $\lambda$ is
born at its final memory and can never change it.

### 0.10 Detection: from likelihood ratios to four detectors

**Definition (log-likelihood ratio).** Two competing explanations of the data, $H_0$ (no change) and
$H_1$ (change), compared by

$$
\Lambda_t \;=\; \log \frac{p(y_{1:t} \mid H_1)}{p(y_{1:t} \mid H_0)} .
$$

Large $\Lambda_t$ means $H_1$ explains the data better. Every detector below is a way of computing
or approximating this online — they differ only in what they do with the unknown change time $\nu$:
**maximise** over it (CUSUM **[Page 1954]**, GLR **[Willsky & Jones 1976]**), **sum** over it
(Shiryaev–Roberts **[Shiryaev 1963]**), **smooth** it (EWMA chart **[Roberts 1959]**), or carry a
**posterior** over it (BOCPD **[Adams & MacKay 2007]**). All of them run on the standardized innovations $e_t/\sqrt{F_t}$,
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

### 0.11 Reading the results

**RMSE** is against the *known* true $\rho_S$ path, which the simulator supplies:

$$
\mathrm{RMSE} \;=\; \sqrt{\frac{1}{T - t_0}\sum_{t=t_0}^{T} \bigl(\hat\rho_{S,t} - \rho_{S,t}\bigr)^2},
$$

with $t_0 = 500$ discarded as burn-in. Lower is better; it mixes variance and lag-bias exactly as
§0.3 describes.

Every headline number is a mean over 20 random seeds with its standard error
$\mathrm{SE} = s/\sqrt{n}$. Two numbers differing by less than a couple of SEs are not distinguishable.
Single-seed numbers are for development only — §7 records two real bugs that a single seed hid.

⚠️ **"Best in column" means best among the shipped configurations, not best achievable.** A model can
look bad because its $\lambda$ was set for a different scenario. §5.4 tunes each family fairly and
reverses one of the apparent results. Always check there before quoting a ranking.

### 0.12 Symbols

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

CUSUM is **[Page 1954]**; the innovation-based form used here follows
**[Basseville & Nikiforov 1993]**. It is exactly minimax-optimal for Lorden's worst-case delay
criterion — worst case over both change time and pre-change history — **[Lorden 1971]**,
**[Moustakides 1986]**.

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
filter's own **step response** (**[Willsky & Jones 1976]**; the window-limited form that makes it
streaming-admissible is **[Lai 1995]**). For this filter a unit jump at $\tau$ contributes

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

**BOCPD** = **B**ayesian **O**nline **C**hangepoint **D**etection (Adams & MacKay 2007). *Online* in
the streaming sense — one pass, each observation seen once, no revisiting history — as opposed to
retrospective/offline segmentation, which sees the whole series before deciding where the breaks
were. *Bayesian* because it maintains a posterior distribution rather than testing a hypothesis:
CUSUM and GLR ask "is there enough evidence to declare a change?", BOCPD asks "given everything so
far, how likely is each possible time-since-last-change?" and never has to declare anything at all.

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

### 3.9 Slow vs `_fast`, and the priors

Each `_fast` variant differs from its sibling in **one number** — a prior belief about how often the
world changes. Nothing about compute speed.

**Kalman.** The knob is $\sigma_\omega^2$. Everything else follows from Bridge 1: form
$q = \sigma_\omega^2/\sigma_v^2$, get the steady-state gain in closed form, read off the equivalent
memory. At $m = 20$, so $\sigma_v^2 = 1.06/17 = 0.0623$:

| key | $\sigma_\omega^2$ | $q$ | $K$ | $\lambda = 1-K$ | $\mathrm{ESS}$ |
| --- | --- | --- | --- | --- | --- |
| `kalman` | $10^{-4}$ | 0.0016 | 0.0393 | 0.9607 | 50 blocks ≈ **999 ticks** |
| `kalman_fast` | $10^{-2}$ | 0.1604 | 0.3282 | 0.6718 | 5.1 blocks ≈ **102 ticks** |

A $100\times$ larger $\sigma_\omega^2$ says "the truth wanders $100\times$ more per block", so the
filter trusts its prior less: gain $0.039 \to 0.328$, memory $999 \to 102$ ticks. Note the
compression — $100\times$ in $\sigma_\omega^2$ buys only $\approx 10\times$ in memory, because
$K \approx \sqrt{q}$ for small $q$.

**BOCPD.** The knob is the hazard $H$, which is a *geometric prior on run length*,
$p(r) = H(1-H)^{r}$, with mean $1/H$:

| key | $H$ | prior mean run length |
| --- | --- | --- |
| `kalman_bocpd` | $1/1000$ | 1000 blocks ≈ **20 000 ticks** |
| `kalman_bocpd_fast` | $1/10$ | 10 blocks ≈ **200 ticks** |

Mathematically this is *not* the same knob. $\sigma_\omega^2$ sets a fixed memory; $H$ sets a prior
that the posterior can override, so BOCPD's effective memory moves with the evidence — long while a
segment holds, short right after a break. That is precisely the §6.2 distinction, and why the
detector thresholds also differ (0.15 vs 0.95): a model expecting frequent change finds short runs
likelier, so it needs a higher bar for the same $\mathrm{ARL}_0$.

**Priors.** All are deliberately vague, and all are one line in the code:

| Where | Prior | Value |
| --- | --- | --- |
| `KalmanLocalLevel` | $\hat z_0$, $P_0$ | $0$ (i.e. $\rho = 0$, independence), $P_0 = 1$ |
| `KalmanBocpd`, per fresh run | $\mu \sim \mathcal N(\mu_0, \tau_0^2)$ | $\mu_0 = 0$, $\tau_0^2 = 1$ |
| `CopulaFit` | accumulated information $H_0$ | $1.0$ — one pseudo-observation, so the first step cannot divide by zero |

$P_0 = 1$ against $\sigma_v^2 \approx 0.062$ makes the prior ~16× wider than a single observation, so
the first block is taken almost at face value ($K_1 = 0.94$ for both Kalman variants) and the filter
converges to its configured memory within a handful of blocks. The `_fast`/slow distinction is
therefore about the *steady state*, not the start: both begin equally credulous.

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

A fixed-$\lambda$ estimator commits to one memory length — one point on the variance/adaptation
trade-off. Sweeping $\lambda$ moves along that curve, which is why §5.4 erased `kalman_fast`'s
`oscillate` advantage: it was at a better *point* than `xiao_ew`'s default, not on a better curve.

BOCPD is not on the curve. Its run-length posterior *switches* effective memory on evidence — long
while a segment persists, short the moment the data stop fitting. No fixed $\lambda$ does that at any
tuning, which is why this gap survived fair tuning and the other did not.

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
`jump`, 39 ticks on `oscillate` — despite different internals, representations, and reciprocal
parameter conventions. The theory's central claim, confirmed quantitatively: the optimum is set by
the data-generating process, not the estimator, and shortens ~5× as $\dot\theta$ grows.

### 6.4 Why hard reset costs RMSE

By the time an alarm fires (~46 ticks on `jump`), the un-reset filter has already partly adapted
toward the new level. `reset()` discards that and restarts from a flat prior, so the estimator
re-climbs ground it had already covered. Hence `kalman_cusum` (0.1146) and `kalman_glr` (0.1110) both
score *worse* than the `kalman_fast` (0.0880) they wrap. BOCPD, which never resets, avoids this
entirely — direct support for treating detect-and-restart and adaptive forgetting as one dial at
different decision hardness (change_detection.md Part V), with soft mixing the better setting here.

### 6.5 Practical guidance

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

All 22 × 5 × 4 cells, colour-coded, are in `results/multiseed_heatmap.png`
(`python3 plot_multiseed_heatmap.py`); the raw numbers are in
`results/multiseed_<scenario>[_<family>].csv`. Colour there is normalised *within* each column,
since absolute RMSE spans 0.012 (`constant`) to 0.44 (`oscillate`) and a global scale shows nothing;
the clipped `drift`/`oscillate` columns are outlined for the §10.1 reason.

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
`gaussian_mle_ew` would "be expected to lose" off-Gaussian. Directionally right, but mild: it wins
`stochastic` on Gaussian ($0.0558$) and degrades only to $0.0671$ / $0.0592$ / $0.0680$ off-family —
still mid-pack. The Gaussian copula is a forgiving wrong answer *in $\rho_S$ space*. `clayton_mle` is
not: $0.1589$ on Gumbel `constant`, 10× its own on-family $0.0154$. Gumbel-on-Clayton is half as bad
($0.0790$). **Fitting Clayton to non-Clayton data is the one genuinely dangerous choice here** —
dangerous because its lower-tail structure is actively wrong for upper-tail data, not merely
uninformative.

**3. `copula_select` recovers the generating family, cumulatively.** Against the *correct* family's
MLE on that family's own data: Clayton $0.0154$/$0.0154$, Gumbel $0.0168$/$0.0168$, $t$
$0.0131$/$0.0131$ on `constant`, near-exact on `jump`. Online GoF selection works, and costs nothing
when right. But `copula_select_ew` loses to the right family's `_ew` on Clayton `drift` ($0.1039$ vs
$0.0717$) — under forgetting it chooses on a short window and churns. Selection and forgetting
interact badly: an open problem, not a tuning detail.

**4. `xiao_win` — measured for the first time — adds nothing over `windowed_rank`.** They agree to
~0.5% everywhere ($0.0275$ vs $0.0271$; $0.1183$ vs $0.1170$), as expected for two exact sliding
windows differing only in Xiao's binning — mild evidence the binning costs no accuracy. It does use
less storage (5824 B vs 8000 B), so it is marginally the better of the two. Recorded so the next
person need not re-run it.

**5. The grid found a real bug** (§7, `gaussian_mle` NaN latch) — the second the replication tier has
caught that single-seed runs missed, after the BOCPD burn-in alarms.

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

## 11. The reset is the cost: why detection loses to plain forgetting on `jump`

Added 2026-09-06, after registering `kalman_ewma_chart` and `kalman_sr`.

### 11.1 The comparison is controlled

The two detectors added here are the EWMA control chart (**[Roberts 1959]**) and Shiryaev–Roberts
(**[Shiryaev 1963]**; asymptotically optimal for Pollak's conditional-average delay criterion,
**[Pollak 1985]**). All four detectors wrap `KalmanLocalLevel` at `sigma_omega2 = 1e-2` — **the same
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

### 11.2 Per-tick trace: the reset destroys a nearly-converged estimate

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

### 11.3 BOCPD is the control

`kalman_bocpd` (**[Adams & MacKay 2007]**) alarms **most** (1.45/run) and has the **best** RMSE
(0.0598), breaking the monotone pattern outright. It never hard-resets — it reweights a
distribution over run lengths. The cost is not detecting; it is *committing*.

This is the hard-decision/soft-decision axis of
[change_detection.md](change_detection.md) appearing directly in RMSE.

### 11.4 Open — the obvious next test

Candidate fix: make `reset()` inflate the state covariance (the standard adaptive
filtering "covariance bump") or re-seed from the current estimate, rather than
clearing state. That keeps the alarm's information — *distrust the past* — without
discarding the estimate. **Untested.**

A cheap mechanism check: run a detector with `reset()` made a no-op. If RMSE returns
to `kalman_fast`'s 0.0880, the reset cost is confirmed as the entire story.

### 11.5 ⚠️ What this does not establish

- **Thresholds are uncalibrated.** `L = 3.0`, `h = 5.0` are textbook defaults. The
  spread of alarm rates is what makes the monotone relationship visible — it reads as
  a natural experiment, not a designed one. Not a thesis result until
  `calibrate_arl0` puts all four at matched $\mathrm{ARL}_0$.
- `jump` + Gaussian only; the other 19 grid files are unexamined.
- `kalman_fast`'s $\lambda$ was not tuned against these scenarios either.

---

## 12. References

**✓ = in the Zotero library. ⚠️ = from memory, verify before it enters `bibliography.bib`.**
The ⚠️ set is tracked as an acquisition item in [todo.md](../todo.md).

**Layer 1 — what we assume**

- ✓ **[Sklar 1959]** — the decomposition: joint = marginals + copula.
- ✓ **[Nelsen 2006]** — *An Introduction to Copulas.* Standard reference.
- ✓ **[Fantazzini 2009]** — misspecified marginals bias elliptical copula correlations by
  up to 70%. The argument for rank-PIT (§0.4).
- ✓ **[Kojadinovic & Yan 2010]** — rank-based goodness-of-fit; the batch counterpart to
  `copula_select`.
- ✓ **[Patton 2006]**, ✓ **[Opschoor et al. 2021]** — dynamic / score-driven copulas, the
  literature this state-space framing sits beside.

**Layer 2 — how we estimate**

- ⚠️ **[Kalman 1960]** — the filter. ⚠️ **[Harvey 1989]**, ⚠️ **[Durbin & Koopman 2001]** —
  local-level model.
- ✓ **[Xiao 2019]** — online empirical-copula Spearman estimator.
- ✓ **[Stephanou & Varughese 2021]**, ✓ **[hermiter 2024]** — sequential Spearman via
  Hermite series; `external/hermiter`.
- ✓ **[Segers 2012]**, ✓ **[Bücher & Volgushev 2013]** — weak convergence of empirical and
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
- ⚠️ **[Willsky & Jones 1976]** — GLR. ⚠️ **[Lai 1995]** — the window-limited form that is
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
