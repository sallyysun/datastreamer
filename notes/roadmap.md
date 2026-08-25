# Tying it together
## The central question
> **How should we estimate a time-varying copula dependence parameter when observations arrive sequentially?**

Then build around this:
$$
\boxed{\text{stationary efficiency} \quad \leftrightarrow \quad \text{adaptation to change}}
$$

## Progress

**1. Baseline: online MLE / Fisher scoring**

Start with simplest sequential estimator:

$$
\theta_t
=
\theta_{t-1}
+
I(\theta_{t-1})^{-1}s_t(\theta_{t-1})
$$

This gives the cleanest connection to **Fisher information** and lets us derive/measure estimation efficiency.

Then introduce **exponential forgetting**:

$$
\hat\theta_t
=
\text{weighted contribution of recent observations}.
$$

Study how $\lambda$ affects:

- variance in a stationary environment
- adaptation speed after a change
- bias
- recovery time.

This should probably be the **core**.


**2. Explicit change detection + restart**

Then compare forgetting with:

$$
\boxed{\text{estimate continuously} \quad vs \quad \text{detect change + restart}}
$$

For example:

$$
\text{online estimator}
\rightarrow
\text{CUSUM}
\rightarrow
\text{change detected}
\rightarrow
\text{restart estimator}.
$$

This is particularly interesting because it directly addresses the original idea:

> Is it better to continuously forget old information, or explicitly detect structural changes and reset?

That's a very clean thesis contribution.


**3. Dynamic latent-process model**

A model for the *true data-generating process* in simulations.
$$
z_t = z_{t-1}+\omega_t,
\qquad
\rho_t=\tanh(z_t).
$$

For example, generate data under:

* constant $\rho$
* smoothly changing $\rho_t$
* abrupt change in $\rho$
* stochastic latent $\rho_t$.

Then we know the true $\rho_t$, allowing us to properly evaluate estimators.

**4. Bayesian sequential estimation — optional extension**

Then ask:
> What if instead of maintaining a point estimate, we maintain a posterior?
$$
p(\theta_t\mid x_{1:t})
$$

- Implement simple **sequential Bayesian update**.
- Probably not start with full MCMC over the entire history. That's computationally heavier and starts pulling thesis toward "Bayesian dynamic copula modelling"

**5. Particle filter / SMC — only if time permits**

This is the most natural Bayesian method if we treat
$$
\theta_t
$$
as a latent stochastic process.

Particles represent possible current states:
$$
{\theta_t^{(1)},\ldots,\theta_t^{(N)}}.
$$

Then:
$$
\text{predict}
\rightarrow
\text{observe}
\rightarrow
\text{weight}
\rightarrow
\text{resample}.
$$

## Research hierarchy
$$
\boxed{
\begin{array}{c}
\textbf{Dynamic copula DGP} \\
\downarrow \\
\textbf{Online MLE / Fisher scoring} \\
\downarrow \\
\textbf{Exponential forgetting} \\
\downarrow \\
\textbf{Change detection + restart} \\
\downarrow \\
\textbf{Bayesian sequential estimation} \\
\downarrow \\
\textbf{Particle filter (optional)}
\end{array}
}
$$
And evaluate everything on the **same simulated changing-dependence environments**.

## Conclusion
**Question:** how should an estimator adapt?

→ **Frequentist baseline:** online Fisher scoring
→ **Adaptation mechanism:** forgetting
→ **Alternative adaptation:** change detection/restart
→ **Alternative paradigm:** Bayesian filtering
→ **Advanced version:** particle filtering.

And `dcvar`'s dynamic copula tells me *how dependence can evolve*, while my work asks *how an estimator should track that evolution online*.

# Non-parametric
We have **two levels**:

**1. Estimate the copula itself**
$$
C_t(u,v)
$$

**2. Or estimate a parameter describing the copula**
$$
\theta_t \quad\text{or}\quad \rho_t
$$

Xiao and Hermite are primarily relevant to **(1)**.

## Xiao: online empirical copula
The Xiao/streaming approach gives you a way to estimate the **empirical copula directly as observations arrive**.

Instead of assuming
$$
C_t = C_{\theta_t}
$$

you essentially estimate
$$
\widehat C_t(u,v)
$$
directly from the observations.

Using pseudo-observations $U_t,V_t$, followed by an empirical copula
$$
\widehat C_n(u,v)
=
\frac1n\sum_{i=1}^n
1{U_i\leq u,V_i\leq v}.
$$

Then your main contribution is to make this **streaming + exponentially weighted**:
$$
\widehat C_t^{(\lambda)}
=
\lambda\widehat C_{t-1}^{(\lambda)}
+
(1-\lambda)
1{U_t\leq u,V_t\leq v}.
$$

So:

> **Xiao = computational/online way of maintaining the empirical copula.**

## Hermite: another way to represent/estimate the copula
Hermite is useful when you don't want to store the entire empirical copula surface.

Instead of representing $C(u,v)$ on a grid, you represent it through a **basis expansion**:
$$
C(u,v)
\approx
\sum_{j,k} a_{jk} H_j(u)H_k(v)
$$
(or the corresponding Hermite representation).

Then you only need to maintain/update the coefficients $a_{jk}$.

So conceptually:

- **Xiao → online empirical/checkerboard representation of (C_t)**
- **Hermite → compact basis representation of (C_t)**

Both are ways of estimating the **full dependence structure without imposing a Gaussian/t/Clayton/etc. copula family**.

## Exponential forgetting

This is actually the really interesting part.

You can apply exponential forgetting to **either representation**.

For Xiao:
$$
\widehat C_t
=
\lambda\widehat C_{t-1}
+
(1-\lambda)C_t^{\text{new}}.
$$

For Hermite coefficients:
$$
\hat a_{jk,t}
=
\lambda\hat a_{jk,t-1}
+
(1-\lambda),h_{jk}(U_t,V_t).
$$

So the **core research question** becomes:

> How should a nonparametric online copula estimator trade off statistical efficiency against adaptation to changing dependence?

That's exactly aligned with your existing motivation around exponential forgetting and effective sample size. 

## And this is different from the dynamic parametric copula paper

This distinction is important for your thesis.

|                    | Your Xiao/Hermite approach     | Dynamic copula paper                |
| ------------------ | ------------------------------ | ----------------------------------- |
| Copula             | **Nonparametric**              | Parametric                          |
| Estimate           | $C_t(u,v)$                     | $\theta_t$                          |
| Dependence family  | Not assumed                    | Gaussian/t/etc.                     |
| Data processing    | **Online/sequential**          | Uses full sample likelihood         |
| Forgetting         | **Yes**                        | Generally no                        |
| Main question      | How should estimator adapt?    | How does dependence evolve?         |
| Representation     | Empirical/basis                | Parametric copula                   |
| Fisher information | Can study estimator efficiency | Used for likelihood-based inference |
| Change detection   | Central to your project        | Not the main objective              |

So **Hermite/Xiao aren't competing with the dynamic copula model**. They're providing your **nonparametric estimator**.

And that gives you a very clean thesis structure:
$$
\boxed{
\text{Dynamic DGP}
\rightarrow
\begin{cases}
    \text{Xiao: online empirical copula}\\
    \text{Hermite: online basis copula}
\end{cases}
\rightarrow
\text{forgetting}
\rightarrow
\text{change detection}
}
$$

Then, if you want, you can add **parametric Fisher-scoring/MLE** as a benchmark:
$$
\text{Dynamic DGP}
\rightarrow
\text{Parametric online MLE/Fisher scoring}
\rightarrow
\text{forgetting}
$$

That comparison is actually quite strong: **nonparametric online estimation vs parametric online estimation**, both under stationary, smoothly drifting, and abruptly changing dependence. Your current draft already frames the experiments around exactly those three regimes. 

And **Hermite/Xiao are therefore not extra random methods to bolt on**—they are the natural estimators for the nonparametric branch of what you are already doing.


# Plan
1. Go through code and sort through all the code
2. Non-parametric: try faster events, tie in real life data
3. parametric: mle, fisher
    - depends on whether the model fits the data
    - develop the "ultimate" change-point detection "streaming" algorithm for copulas 
4. Application ~ pairs-trading: how would you look at that through a copula? 
    - since its just more info on the correlation
    - pairs-trading strategy based on copula (an application)
    - or time-shifted dependence (in either direction)

# The "ultimate" streaming estimator

## Terminology
- **streaming** = live, single-pass, O(1) per update, bounded memory, real-time over TCP
  (the hard constraint). Our claim: not just online but *streaming-admissible*.
- **online** = sequential/incremental but allowed to be heavier (may revisit, larger state).

Making the distinction explicit turns the streaming constraints (bounded state, O(1), latency)
into first-class, measurable requirements — which is what separates this from batch copula
modelling.

## Core idea: make the memory itself adaptive
Every experiment so far shows **no fixed $\lambda$ is ever right**: cumulative wins when
stationary, forgetting wins when changing, and the crossover is set by
(rate of change) vs (memory window / ESS). So the ultimate estimator should not *pick* a memory
length — it should **tune its own forgetting rate online** from the data. That single principle
dissolves the efficiency$\leftrightarrow$adaptation tradeoff: behave cumulative-like when quiet,
forget fast when things move.

Four ways to realize it (increasing sophistication):

1. **Variable forgetting factor (VFF)** — one estimator, but $\lambda_t$ is driven by the
   innovation / prediction error. Small residuals -> grow the window (efficiency); residual spike
   -> shrink it (adapt). Standard in adaptive RLS filtering; cheapest win, still O(1), one extra
   scalar. **Prototype this first — it is the novel contribution.**
2. **Detect-and-restart** — CUSUM / Page-Hinkley on the estimate stream; on trigger, `reset()`.
   The bang-bang version of VFF (infinite memory until change, then dump). The `reset()` hook is
   already in the estimator interface. "Continuous forgetting vs detect+restart" = VFF vs CUSUM.
3. **Multi-timescale ensemble** — a bank of estimators at several $\lambda$ in parallel, combined
   by recent predictive performance (exponential weights / Bayesian model averaging over
   $\lambda$). Leans on the long-memory expert when stationary, the short one when changing.
   Robust, cheap, sidesteps picking $\lambda$ at all.
4. **Bayesian online changepoint (BOCPD) / particle filter** — treat
   $z_t = \operatorname{atanh}\rho_t$ as a latent random walk and filter online. BOCPD keeps a
   distribution over run-length-since-last-change, unifying gradual drift and abrupt jumps in one
   framework. The principled gold-standard reference.

**Recommendation:** build **#1 (VFF)** as the contribution, benchmark against **#2** (hard-decision
version) and **#4** (Bayesian version). Thesis story: *fixed-$\lambda$ estimators are dominated;
here is an estimator that learns its own memory — in a frequentist (VFF), a hard-decision (CUSUM),
and a Bayesian (BOCPD) flavour.*

## Differentiator we already have
Most copula-dependence papers are batch. We have a genuinely streaming TCP system with a live probe
measuring latency, throughput, and post-change recovery. So evaluate on **streaming-specific**
metrics, not just RMSE: O(1) update cost, bounded memory, and **recovery time / detection delay**
after a change (the `lag` metric the probe already computes). Framing: *how should an estimator
adapt under streaming constraints?*

## Detail on the four plan points
1. **Sort the code** — fold in the debts found in experiment 1: the `hermite_ew` cold-start fix
   (decouple cumulative Welford variance from EWMA coefficients); NaN-before-calibration so plots
   don't draw a fake 0 line; and **multi-seed replication for error bars** (everything is
   single-seed — do this before any number is treated as a result).
2. **Faster events + real data** — `FileSource` exists but is not wired into the server; wiring is
   small. The real challenge: real data has **no ground-truth $\rho$**, so RMSE is unavailable.
   Switch the metric to **one-step predictive log-likelihood** (or agreement with a batch
   rolling-window estimate as pseudo-truth). That metric also serves the parametric branch.
3. **Parametric MLE / Fisher** — "depends on whether the model fits" is a **goodness-of-fit**
   question, and it is a feature: run an online copula GoF test and let the data choose parametric
   vs nonparametric — an estimator that decides how much structure to assume.
4. **Pairs-trading via copula (strongest application)** — the copula gives the *conditional*
   distribution $P(Y\le y \mid X=x) = h(y\mid x) = \partial C/\partial u$. The trading signal is a
   tail event: when the observed pair lands in the tail of that conditional (one leg cheap/rich
   given the other), trade. Richer than a linear-correlation z-score because it captures
   **tail / asymmetric** co-movement a Pearson spread misses. **Time-shifted dependence** is the
   novel angle and ties straight to the `lag` metric: estimate the cross-copula at lag $k$ and take
   $\arg\max_k$ dependence to reveal **lead-lag structure** — a streaming lead-lag copula for pairs.

# What makes a copula changepoint different

Motivation for "the ultimate streaming changepoint algorithm *for copulas*": a copula changepoint
is a genuinely different (and harder) object than the changepoints most detectors target — a change
in the **dependence structure** while the marginals may be doing something else entirely. Five
distinctions, almost none of which standard changepoint methods face at once:

1. **A joint structure, not a scalar/mean.** Classic detectors watch a 1-D signal (mean, variance,
   rate). A copula changepoint is a change in a 2-D+ object $C_t(u,v)$ — how streams *co-move*. It
   is **bounded and nonlinear** ($\rho\in(-1,1)$), so $0.9\to0.99$ matters far more than
   $0.0\to0.09$ despite equal $+0.09$; work in $z=\operatorname{atanh}\rho$ so equal changes are
   equally detectable. Nonparametrically a change can be in **shape** (tail dependence appears,
   asymmetry flips) at the *same* Spearman $\rho$ — a scalar detector on $\rho$ misses it entirely.

2. **The marginal-vs-dependence confound (the defining difficulty).** The copula lives on
   pseudo-observations (PIT/ranks), so the marginals must be removed first — but they change online
   too. Two superimposed change types: a **marginal** change (one series' vol regime) to *ignore*,
   and a **dependence** change to *detect*. A naive detector fires on marginal changes → false
   copula alarms. So the detector must be **invariant to marginal change** — a constraint ordinary
   changepoint methods never face (they *want* the marginal shift). Same decoupling lesson as the
   `hermite_ew` cold-start: filter marginals separately, detect on decoupled scores/ranks.

3. **Rank/PIT feedback breaks the i.i.d. assumption.** Changepoint theory assumes pre-change data
   are i.i.d. from a fixed law. But streaming pseudo-observations aren't fixed — each new point
   re-ranks history, so past $U_t$ drift as data arrives. Even under *constant* dependence the
   transformed stream is mildly non-stationary and serially dependent, so off-the-shelf
   false-alarm-rate guarantees don't transfer. Need a stable streaming-PIT or statistics robust to
   this drift.

4. **Weaker signal → intrinsically harder detection.** Dependence is a second-moment / joint-tail
   property, estimated less efficiently than a first moment (cf. the oscillate results: even good
   estimators lag/attenuate). For the same statistical "size," a copula changepoint needs **more
   observations to detect** → longer detection delay is fundamental, not a tuning failure. In $d>2$
   the object is a whole correlation matrix / high-dim copula and a change can hide in one pair —
   a multiple-testing / high-dimensional detection problem.

5. **Drift is the common case, not the exception.** Classic models assume piecewise-constant signal
   with rare abrupt breaks. Financial dependence mostly **drifts** and only occasionally **jumps**
   (crises). A pure jump detector either chatters on drift or desensitizes; the method must
   distinguish structural breaks from ongoing drift — why the Kalman-within-run + BOCPD-across-runs
   hybrid matters more here than in a typical changepoint setting.

**Together:** a copula changepoint is a change in a bounded, nonlinear, higher-dimensional joint
object, detected invariantly to simultaneously-changing marginals, from feedback-contaminated rank
streams, with an inherently weaker signal, against a drift-plus-jump background. Standard
changepoint literature rarely faces all five — that gap is what makes the "ultimate streaming
changepoint algorithm for copulas" a contribution rather than an application of an existing detector.

# Change-detection methods to survey

> Developed further in [change_detection.md](change_detection.md): the state-space view
> ($\lambda$ *is* a Kalman gain; CUSUM *is* a 2-state filter), the three-axis detector design
> space, the $\mathrm{ARL}_0$ calibration trap, and a build order.

Classics (CUSUM, Page-Hinkley) are the start; the broader landscape, grouped by fit to a
streaming, O(1), copula setting:

**Sequential / control-chart (cheap, classic)**
- **CUSUM** — cumulative log-likelihood-ratio; minimax-optimal (Lorden) for a *known* post-change
  shift. Fast on small persistent shifts.
- **Page-Hinkley** — CUSUM's mean-shift special case; the streaming-ML default.
- **EWMA / Shiryaev-Roberts** — SR is the Bayesian-optimal sibling of CUSUM, often faster for
  unknown change times.
- **GLR (generalized likelihood ratio)** — maximizes over the *unknown* post-change parameter.
  The honest version for copulas, since we don't know the new $\rho$.

**Bayesian (unifies detection + estimation)**
- **BOCPD** (Adams & MacKay 2007) — run-length posterior, online, O(1)-ish with pruning; detection
  *and* current estimate in one object. See the dedicated section below.
- **Particle filters / SMC with jump priors** — $\rho_t$ can drift *or* jump; heavier gold standard.

**Nonparametric / distribution-free (matches the nonparametric branch)**
- **Kernel change detection (MMD / KCUSUM)** — recent vs reference window in an RKHS; no
  distributional assumption. Natural pairing with the nonparametric estimators.
- **Rank / sign-based CUSUM** — we already work in Spearman/rank space, so this is nearly free and
  robust to marginals.
- **ADWIN** (Bifet & Gavalda) — variable-length window auto-dropped on change = *automatic adaptive
  forgetting*, with guarantees. The bridge between change detection and the VFF idea.

**Streaming-ML drift detectors (easy baselines)**
- **DDM / EDDM, KSWIN, HDDDM** — cheap online drift detectors; good live baselines.

**Copula/dependence-specific (closest to the actual quantity)**
- Tests for **constancy of the copula / Kendall's $\tau$** over time (CUSUM-of-$\tau$;
  Bucher-Kojadinovic-type sequential copula tests). Monitor *dependence* rather than marginals —
  the right target, with a citable literature.

**Steer:** three do the most work — (1) **GLR or rank-CUSUM** (honest sequential detector for
unknown post-change $\rho$, in rank space); (2) **ADWIN** (bridge to adaptive forgetting, has
theory); (3) **BOCPD** (Bayesian unification + gold standard). All run on the estimate/innovation
stream the live probe already produces, and are scored with the recovery-time / detection-delay
metrics we already have.

# Bayesian vs frequentist for this problem

**Untie one knot first:** a prior does **not** make streaming updates computationally faster —
that conflates two meanings of "faster".
- **Computational speed (per update):** set by the model, not the prior. A **conjugate** Bayesian
  update is O(1), same shape as a frequentist recursive update. Non-conjugate (particle, BOCPD) is
  *heavier*. The prior buys nothing on latency.
- **Statistical efficiency (samples to a good estimate):** here the prior helps — but only at the
  **cold start** / small samples (exactly the `hermite_ew` sag a scale prior would have killed).
  Once a few hundred points arrive, the likelihood dominates and the prior washes out.

**You don't "update the prior" on a schedule.** Recursive filtering is
$\text{posterior}_{t-1} \xrightarrow{\text{evolve}} \text{prior}_t \xrightarrow{\text{observe } x_t}
\text{posterior}_t$: the prior is set **once**, then the **posterior updates every tick** and
becomes the next step's prior automatically. Periodically re-injecting a fixed prior would discard
learned information (a bug, not a knob).

**The real knob is the evolve step** — the process noise $\sigma_\omega^2$ (or a power-prior
discount $\alpha$), and *it is the Bayesian $\lambda$*:
- small $\sigma_\omega^2$ → belief barely widens → long memory → efficient, slow to adapt ($\approx$
  cumulative);
- large $\sigma_\omega^2$ → belief widens fast → short memory → adapts, noisier ($\approx$
  aggressive forgetting).

So the efficiency$\leftrightarrow$adaptation tradeoff collapses to one variance parameter, and the
adaptive-forgetting move becomes **infer $\sigma_\omega^2$ online** (or let BOCPD's run-length do
it).

**Recommendation — use both as a ladder:**
- **Frequentist (VFF / recursive MLE + CUSUM):** cheapest, genuinely streaming (O(1), tiny state),
  easiest to defend as streaming-admissible. The workhorse + live-demo estimator.
- **Bayesian (Gaussian state-space filter / BOCPD):** principled reference — posterior (uncertainty
  bands, not just a point), unifies drift + jumps, adaptive forgetting via process-noise/run-length.
  The gold standard to compare against, not necessarily the thing under the TCP latency budget.

The contrast is itself a contribution: *"a cheap frequentist adaptive-forgetting estimator matches
the Bayesian gold standard's tracking at a fraction of the per-update cost"* — measurable directly
with the live probe (RMSE/recovery **and** per-update time).

**Concrete entry point:** conjugate Gaussian state-space on $z_t=\operatorname{atanh}\rho_t$: latent
random walk $z_t=z_{t-1}+\omega_t,\ \omega_t\sim\mathcal N(0,\sigma_\omega^2)$, Kalman-style update.
O(1), gives uncertainty, and $\sigma_\omega^2$ is the single forgetting knob — the cleanest Bayesian
analog of what we already have.

# BOCPD for copulas

BOCPD is a *framework* with two plug-in pieces; applying it to copulas means choosing them so the
thing that can change is the **dependence**. It maintains a posterior over the **run length** $r_t$
(steps since last changepoint):

$$P(r_t \mid x_{1:t}) \propto \sum_{r_{t-1}} P(r_t \mid r_{t-1})\,P(x_t \mid r_{t-1}, x^{(r)})\,
P(r_{t-1}\mid x_{1:t-1})$$

Supply: a **hazard** $H(r)$ (constant $1/\lambda_h$ ⇒ geometric run lengths; $\lambda_h$ = expected
time between changes) and an **underlying predictive model (UPM)** $P(x_t\mid\text{data since last
changepoint})$, ideally conjugate for O(1)-per-run sufficient statistics. Applying to copulas = make
the UPM a copula model whose dependence resets at each changepoint. Two conjugate versions:

**Parametric-copula BOCPD (recommended first).** Work in normal scores
$z_x=\Phi^{-1}(u),\ z_y=\Phi^{-1}(v)$; a Gaussian copula's dependence is just the correlation of a
bivariate normal.
- UPM = bivariate normal with a **Normal-inverse-Wishart** prior on the $2\times2$ covariance.
- Each run keeps NIW sufficient stats (count, sum, sum of outer products) — O(1)/step.
- Predictive is a closed-form **bivariate Student-$t$**.
- Reported estimate $\hat\rho_t = \sum_r P(r_t\mid x_{1:t})\,\rho^{(r)}$ (run-length-weighted average
  of per-run correlations).

**Nonparametric-copula BOCPD (ties to Xiao).** UPM = a checkerboard/histogram copula = **Xiao's bin
grid with a Dirichlet prior**; predictive of the new bin $(i,j)$ is the **Dirichlet-multinomial**
posterior predictive, O(bins)/run. A changepoint resets the grid — the existing estimator becomes
the UPM.

**Copula-specific gotcha.** The copula lives on pseudo-observations (ranks/PIT), but online the
ranks move as marginals are estimated. So (a) use a **streaming PIT** (rolling empirical CDF / the
running ranks we already keep), and (b) a **marginal** change (vol regime shift) is *not* a
**dependence** change — a contaminated PIT leaks marginal jumps into false dependence changepoints.
**Fix: decouple** — standardize each series' marginal separately, run the dependence-BOCPD on the
decoupled normal scores (same lesson as the `hermite_ew` cold-start).

**Drift vs jump — the hybrid.** BOCPD is a *jump* model (piecewise-constant $\rho$ between
changepoints); pure drift is faked with many tiny changepoints. Make the UPM itself a **within-run
random walk** on $z_t=\operatorname{atanh}\rho_t$ (Kalman inside each run) → drift absorbed by the
within-run state, jumps by the changepoints. Kalman-within-run + BOCPD-across-runs **is** the
"ultimate" estimator: gradual adaptation *and* clean resets in one object.

**What you get:** a posterior over $\rho_t$ (uncertainty bands), an explicit $P(\text{changepoint
now})$ stream, and automatic adaptive forgetting. **Prune** low-probability run lengths (cap max run
length) to bound per-step cost. **Steer:** prototype the NIW-on-normal-scores version first
(conjugate, O(1), closed-form); the Dirichlet-on-Xiao-grid version reuses existing code.

# Application track 2 — hedging & the covariance matrix

Moved to [application_2.md](application_2.md) — the covariance/hedging/portfolio thread
(factor models, sparse/vine residual dependence, shrinkage/composition, the dynamic
streaming vine, and the analysis/diagnostics layer).
