# Change detection for streaming copulas

Companion to [roadmap.md](roadmap.md). The roadmap surveys *which* detectors exist
("Change-detection methods to survey") and sketches BOCPD. This doc does two things that
survey does not:

1. shows that CUSUM, Shiryaev-Roberts, GLR, BOCPD and HMM regime-switching are **one recursion
   with three knobs**, not ten separate methods — so the thesis chooses a *cell of a design
   space* rather than benchmarking a grab-bag;
2. develops the **state-space (SSM) view**, which is the missing link between the exponential
   forgetting already implemented and the detectors we want to add.

---

# Part 0 — The setup

Stated plainly, because the rest of this doc assumes it.

## What forgetting is

Every estimator in `src/estimators/dependence/` maintains a running summary and updates it once
per tick. Adding forgetting means the summary is a **weighted** average in which recent
observations count more:

$$
\hat\theta_t=(1-\alpha)\,\hat\theta_{t-1}+\alpha\,(\text{new observation}),
\qquad \alpha\in(0,1)
$$

Unrolled, observation $t-k$ carries weight $\alpha(1-\alpha)^k$ — geometric decay. Nothing is
ever deleted; old data just fades. This is what makes the estimators *streaming*: the whole past
is compressed into fixed-size state, so cost per tick is O(1) and memory is bounded no matter how
long the stream runs.

The useful summary of "how much data am I effectively using" is the **effective sample size**:

$$
\mathrm{ESS}\approx\frac{2-\alpha}{\alpha}\ \approx\ \frac{2}{\alpha}\ \ \text{for small }\alpha .
$$

⚠️ In this codebase $\alpha$ is spelled differently per estimator, which is the mirror-image
convention `registry.hpp` warns about:

| Estimator | Config $\lambda$ | Relation to $\alpha$ | Configured value | ESS |
| --- | --- | --- | --- | --- |
| Xiao | retention | $\alpha = 1-\lambda$ | $\lambda=0.99$ | $\approx 200$ |
| Hermite | update weight | $\alpha = \lambda$ | $\lambda=0.01$ | $\approx 100$ |

Same mechanism, reciprocal spelling. $\lambda=0$ means *cumulative* — no forgetting, ESS grows
with $t$.

## Why no single $\lambda$ works

The tension is a bias-variance trade-off, and both halves scale with ESS in opposite directions.

**Variance.** Averaging $\mathrm{ESS}$ effective observations of noise $\sigma^2$ gives

$$
\operatorname{Var}(\hat\theta_t)\ \approx\ \frac{\sigma^2}{\mathrm{ESS}} .
$$

Longer memory ⇒ smaller variance. In a stationary world, more memory is always better, and
cumulative ($\mathrm{ESS}\to t$) wins outright.

**Lag.** But the estimator is an average over the last $\sim\mathrm{ESS}$ ticks, so it reports
roughly where $\theta$ *was*, about $\mathrm{ESS}/2$ ticks ago. If $\theta$ is moving at rate
$\dot\theta$, that is a systematic error

$$
\text{bias}\ \approx\ \dot\theta\cdot\frac{\mathrm{ESS}}{2},
$$

and after a step change of size $\Delta$ the estimator needs $\sim\mathrm{ESS}$ ticks to catch up.
Longer memory ⇒ larger bias. Putting the two together,

$$
\mathrm{MSE}(\mathrm{ESS})\ \approx\ \underbrace{\frac{\sigma^2}{\mathrm{ESS}}}_{\text{variance}}
\;+\;\underbrace{\Big(\dot\theta\,\frac{\mathrm{ESS}}{2}\Big)^{2}}_{\text{lag}} .
$$

**This is the whole thesis in one line.** The minimising $\mathrm{ESS}$ depends on $\dot\theta$ —
how fast dependence is currently moving. Fast-moving ⇒ short memory; quiet ⇒ long memory.

## Why that is a problem rather than a tuning exercise

If $\dot\theta$ were a constant you would simply solve for the optimal ESS once and hard-code it.
It is not. Real dependence is quiet for long stretches and then moves abruptly, so the optimal
memory length is **itself time-varying** — and any fixed $\lambda$ is the right answer only for
whichever regime you happened to tune it on.

The offline harness shows exactly this: cumulative variants win on `constant`, forgetting variants
win on `jump`, and no configured $\lambda$ wins both. The crossover is set by rate of change
versus memory length, exactly as the MSE expression predicts.

$$
\boxed{\text{No fixed }\lambda\text{ dominates}\ \Longrightarrow\ \text{the estimator must set its own memory.}}
$$

## The two ways to do that

- **Gradually** — let $\lambda_t$ move with the data. Small prediction errors ⇒ lengthen memory
  (buy efficiency); a spike ⇒ shorten it (buy adaptation). This is the *variable forgetting factor*
  (VFF).
- **Abruptly** — watch for a change and, on detection, discard history and restart. The `reset()`
  hook in `estimator.hpp:16` exists for exactly this.

The roadmap frames these as competing options. Part V shows they are the same architecture at
different decision hardness — but the setup above is what both are trying to solve, and it is why
everything that follows is about **memory**, not about correlation formulas.

---

# Part I — Two bridges

Both of these are identities, not analogies, and both shorten the work.

## Bridge 1: our $\lambda$ *is* a Kalman gain

Take the local level model on $z_t=\operatorname{atanh}\rho_t$:

$$
z_t = z_{t-1} + \omega_t,\quad \omega_t\sim\mathcal N(0,\sigma_\omega^2)
\qquad\text{(state)}
$$
$$
y_t = z_t + v_t,\quad\ \ v_t\sim\mathcal N(0,\sigma_v^2)
\qquad\text{(observation)}
$$

The Kalman recursion is

$$
P_{t|t-1}=P_{t-1}+\sigma_\omega^2,\qquad
F_t=P_{t|t-1}+\sigma_v^2,\qquad
K_t=\frac{P_{t|t-1}}{F_t},
$$
$$
e_t = y_t-\hat z_{t-1},\qquad
\hat z_t=\hat z_{t-1}+K_te_t .
$$

At steady state $K_t\to K$ and this collapses to

$$
\boxed{\;\hat z_t=(1-K)\,\hat z_{t-1}+K\,y_t\;}
\qquad\text{i.e. an EWMA with }\lambda=1-K,
$$
$$
K=\frac{\sqrt{q^2+4q}-q}{2},
\qquad
q=\frac{\sigma_\omega^2}{\sigma_v^2}\ \ \text{(signal-to-noise ratio)} .
$$

**Consequences.**

- Every fixed-$\lambda$ estimator in `src/estimators/dependence/` is the **steady-state limit of
  the simplest possible SSM**. The SSM is a strict generalization, not a competing branch — it
  slots in beside the existing estimators.
- "Choose $\lambda$" becomes "choose $q$", a *signal-to-noise ratio*, which the data can speak to.
  $\lambda$ is a tuning constant; $q$ is a quantity with a likelihood.
- **VFF $\equiv$ estimating $q$ online.** This is the same statement as the roadmap's
  "the real knob is the process noise $\sigma_\omega^2$ — it is the Bayesian $\lambda$", but now
  with the exact map between the two parameterizations.
- The transient $K_t>K$ handles **cold start** automatically: the filter is wide-open early and
  tightens as $P_t$ shrinks. That is the principled version of the `hermite_ew` cold-start patch.

## Bridge 2: CUSUM is already a state-space filter

Model the change as a **2-state absorbing Markov chain** (pre-change $\to$ post-change, no return).
Then:

- the **forward / sum-product** recursion over that chain gives the **Shiryaev-Roberts** statistic
  $R_t=(1+R_{t-1})\Lambda_t$;
- the **max-product (Viterbi)** recursion over the *same chain* gives **CUSUM**
  $S_t=\max(0,S_{t-1}+\log\Lambda_t)$;
- refine the latent state from a binary flag to a **run-length counter** and the same forward
  recursion is **BOCPD**;
- make the chain **recurrent over $K$ states** instead of absorbing and it is a **regime-switching
  HMM**.

Same graphical model, different semiring and different latent alphabet. This makes
CUSUM $\to$ SR $\to$ BOCPD $\to$ HMM one ladder rather than four unrelated citations.

---

# Part II — The detector design space

Every detector runs the same loop: *at each tick, ask how much better the data looks if a change
had happened; accumulate that; alarm when it gets big.* What separates the named methods is how
they answer two questions — both of which are about things you do not know.

## Question 1 — when did it change?

- **Take the best guess.** Try every possible change time, keep whichever fits best.
  → **CUSUM**, **GLR**. (Formally: $\max$ over $\tau$ — max-product / Viterbi.)
- **Average over all of them.** Give each possible change time a probability and add them up.
  → **Shiryaev-Roberts**, **BOCPD**. (Formally: $\sum$ over $\tau$ — forward filtering.)

## Question 2 — what did it change *to*?

- **Assume you already know** — e.g. "I am watching for $\rho$ jumping to 0.8." → **CUSUM**.
- **Fit it** — whatever value fits best. → **GLR**.
- **Average over the possibilities** with a prior. → **BOCPD**.

## The methods are just the combinations

| | Guess the change time | Average over change times |
| --- | --- | --- |
| **Know the new $\rho$** | CUSUM | Shiryaev-Roberts |
| **Fit the new $\rho$** | GLR | — |
| **Average over new $\rho$** | — | BOCPD |

So it is one accumulator loop with two flags: `max`/`sum` for the time, and
`known`/`fit`/`average` for the value. Six named methods, one implementation.

**Question 1 is also the frequentist/Bayesian split**, exactly: treating $\tau$ as a fixed
unknown you maximise over is frequentist (CUSUM is minimax-optimal under Lorden's criterion);
treating it as a random variable with a hazard prior you average over is Bayesian. Nothing
philosophical is at stake — it is $\max$ versus $\sum$ in the same recursion.

## A third knob: what you feed the loop

Raw estimates / prediction errors (innovations) / likelihood scores / kernel embeddings.
This one matters more for results than either question above — it is the difference between a
detector that chatters under drift and one that does not (Part III, Level 1) — and it is where
Page-Hinkley (raw observations), the sequential score test, and MMD differ from each other.

**Why this matters for the thesis.** The claim is not "we benchmarked ten detectors" (which answers
to *it depends*), it is "the design space has three axes; here is which cell wins for copulas and
why." Page-Hinkley and the EWMA chart fall out as **degenerate cells** — Page-Hinkley is CUSUM with a
mean-shift plug-in, the EWMA chart is the steady-state Kalman of Bridge 1 with a threshold — so they
need no separate experiments. That is real work saved.

## Regime-switching vs change-point is a hazard choice, not a paradigm

It is tempting to file HMMs under a different question ("which regime am I in?" vs "when did it
change?"). They are the same object; what differs is **what happens to $\theta$ at a transition**:

| Model | Latent chain | $\theta$ after a transition |
| --- | --- | --- |
| CUSUM | 2-state, absorbing | fixed known $\theta_1$ |
| BOCPD | run-length $r_t$ | drawn **fresh from the prior** |
| HMM | $K$-state, recurrent | **reused** from a finite remembered set |

**The interesting consequence: BOCPD is amnesiac.** Financial dependence regimes *recur* —
calm $\to$ crisis $\to$ calm — and BOCPD discards everything it learned about the crisis regime the
moment it leaves. An HMM re-enters a known regime instantly, with **zero relearning delay**. The
synthesis (Fearnhead & Liu 2007; sticky HDP-HMM, Fox et al.) lets a new run either **reuse an old
regime or spawn a new one**.

$$
\boxed{\text{Does regime memory reduce detection delay on \emph{recurring} dependence shifts?}}
$$

Sharp, novel, answerable, and it needs only the `oscillate` / regime-switching DGP we already
generate. Strong candidate for a thesis chapter.

---

# Part III — The SSM ladder

Four levels, increasing sophistication, each reusing the previous.

## Level 1 — SSM whitens, CUSUM detects (Basseville & Nikiforov)

Run the Kalman filter of Bridge 1, then run CUSUM / Page-Hinkley on the **standardized
innovations** $e_t/\sqrt{F_t}$, which are iid $\mathcal N(0,1)$ *under the model*.

**Why this is the highest-value thing to build.** It fixes distinction #5 in the roadmap
("drift is the common case"). CUSUM on raw $\hat\rho_t$ chatters under drift because the pre-change
stream is not stationary, so Lorden-type optimality does not apply. The SSM **absorbs the drift**,
the innovations become white, and the classical theory transfers. Division of labour:

$$
\text{SSM handles gradual drift}\quad\Big|\quad\text{CUSUM handles what the SSM cannot absorb.}
$$

**Practical observation equation.** Do *not* feed one tick at a time — the per-tick normal-score
product $z_{x,t}z_{y,t}$ has mean $\rho$ and variance $1+\rho^2$, and the link $\rho=\tanh z$ is
nonlinear, so you need an EKF/UKF. Instead emit a Fisher-transformed windowed Spearman every $m$
ticks:

$$
y_t=\operatorname{atanh}\hat\rho^{\,S}_{t},
\qquad
\sigma_v^2\approx\frac{1.06}{m-3}\ \ \text{(known!)}
$$

$\sigma_v^2$ being **known** is a gift specific to this problem: it pins down the denominator of $q$,
leaving a *single* scalar $\sigma_\omega^2$ to fit. O(1) amortized, linear-Gaussian, exact filter.

## Level 2 — GLR on the state (Willsky & Jones 1976)

The honest version when the post-change $\rho$ is unknown. A jump of unknown size $\nu$ at unknown
time $\tau$ leaves a *deterministic signature* on the innovation sequence — the filter's step
response $\varrho_s(\tau)$ — so the GLR is a matched filter maximized over $\tau$:

$$
g_t=\max_{\tau}\ \frac{\big(\sum_s \varrho_s(\tau)e_s/F_s\big)^2}{\sum_s \varrho_s(\tau)^2/F_s}\ >\ h .
$$

For the local level model $\varrho_s(\tau)$ is closed-form. This is exactly "CUSUM, but the SSM tells
you what a change *looks like*".

> ⚠️ **Streaming caveat.** A GLR with $\sup$ over all $\tau\le t$ is $O(t)$ per step and is
> **not streaming-admissible** — it breaks the central claim of the project. Use the
> **window-limited GLR** (Lai 1995): restrict $\tau\in[t-W,\,t-W_0]$, giving $O(W)$ per step with
> near-optimal delay. Non-negotiable if GLR goes in.

## Level 3 — switching SSMs (soft detection, no hard reset)

- **Harrison & Stevens multiprocess DLM (1976)** — each step is a mixture over
  {normal, outlier, level shift}; collapse via GPB1 to stay O(1). Detect-and-restart *without* a
  bang-bang decision.
- **IMM filter** (target-tracking literature) — a bank of Kalman filters at different
  $\sigma_\omega^2$, Markov-switched and soft-mixed. This is the roadmap's "multi-timescale
  ensemble" idea, but with a transition model and decades of theory attached.
- **West's cumulative Bayes factor monitoring (1986)** —
  $H_t=B_t\cdot\min(1,H_{t-1})$, which is *literally CUSUM in log space* with log-Bayes-factor
  increments. The tightest link between our CUSUM and the DLM world.

## Level 4 — no detector at all

Give the process noise a heavy tail (Student-$t$) or a spike-and-slab / Laplace prior. A jump is
then just a large draw from the tail and the filter adapts on its own. The MAP under Laplace process
noise is **fused-lasso / $\ell_1$ trend filtering**; the streaming version is an online proximal
step. Cheap, and it sidesteps the false-alarm-threshold question entirely.

## Aside — when the filter has to go nonlinear (Copula Particle Filter)

Everything above stays linear-Gaussian because the observation is a blockwise Fisher-$z$'d Spearman
(Level 1). If you ever feed **per-tick** normal-score products instead, the link $\rho=\tanh z$ is
nonlinear and the exact Kalman filter is gone — you need SMC. **Rodríguez & Walker (2021), *Copula
Particle Filters*** is the cheapest route I have seen:

$$
p(x_{t+1}\mid y^{t})=p(x_{t+1}\mid y^{t-1})\;
c\big(P(x_{t+1}\mid y^{t-1}),\,P(y_t\mid y^{t-1})\big),
\qquad
P(x_{t+1}\mid y^{t})=\tfrac{\partial C}{\partial v}(\cdot,\cdot)
$$

so particle propagation is *draw $u\sim U(0,1)$, solve $\partial C/\partial v=u$* — closed form for
the Gaussian copula, $P(x\mid x_d)=\Phi(z\sqrt{1-\rho^2}+\rho\,\Phi^{-1}(s))$, $z\sim\mathcal N(0,1)$.
No importance sampling in the propagation step.

**Why it matters here:** the algorithm accumulates the **marginal likelihood as a by-product**
($\text{log-lik} \mathrel{+}= \log k - \log M$ each step). That is exactly the machinery needed to
estimate $\sigma_\omega^2$ — equivalently $q$ — online, i.e. the VFF problem from Part I. A filter
that hands you the likelihood for free is a route to *"infer your own forgetting rate."*

**Where it sits:** O($M$) per step, $M\ge 100$ recommended when the link is nonlinear, plus
root-finding for any copula without a closed-form conditional. Not the live TCP path — a
**gold-standard reference to benchmark against**, the slot this doc already reserves for the
Bayesian branch. R code and both real datasets ship as supplementary material, so validating an
implementation against theirs is realistic.

> ⚠️ **Do not conflate the two $\rho$'s.** In this thesis $\rho$ is the *estimand*. In that paper
> $\rho$ is a **tuning knob of the sampler** — they recommend trying $0.1, 0.5, 0.8$, and report it
> interacting badly with Metropolis–Hastings acceptance (one example accepted almost nothing at
> $\rho=0.9$; another needed $\rho>0.9$ to converge). Same symbol, opposite role. Say so explicitly
> if you cite it.

**What it is not:** a change-detection paper. No changepoints, regimes, or CUSUM anywhere in it, and
its examples (Gaussian SSM, stochastic volatility, VGT log-returns, unemployment) are all
*univariate* state problems. It is filtering machinery, not a competitor to anything in this doc.

---

# Part IV — What the standard survey misses

Five gaps, roughly in order of value to this thesis.

**1. Test martingales / e-values (Vovk; Ramdas — safe anytime-valid inference).**
The biggest gap. Every classical detector needs a threshold calibrated against a **known**
pre-change law — but roadmap distinction #3 says rank/PIT feedback makes the pre-change stream
non-iid, so those ARL guarantees *do not transfer*. Conformal test martingales give anytime-valid
false-alarm control **without trusting the null**. Recent (2019-2024) literature, essentially
untouched in the copula change-point space, and it solves a problem we have already identified as
fundamental. **Strongest novelty candidate in this document.**

**2. The *efficient* (orthogonalized) score, not the raw score.**
Partition $\theta=(\theta_{\text{marg}},\theta_{\text{dep}})$. The **efficient score** for the
dependence block — the score minus its projection onto the nuisance tangent space,
$\tilde S_{\text{dep}} = S_{\text{dep}} - I_{\text{dm}}I_{\text{mm}}^{-1}S_{\text{marg}}$ — yields a
CUSUM that is **invariant to marginal changes by construction**. This is the principled fix for
distinction #2 (the marginal-vs-dependence confound), rather than hoping the ranks are clean.
Fisher scoring already computes every ingredient. Score-CUSUM is also the *locally most powerful*
test, hence optimal for **small** changes — exactly the copula regime (distinction #4, weak signal).
**Rank this above GLR.**

**3. Multivariate CUSUM (Crosier MCUSUM / MEWMA) on Hermite coefficients.**
Every detector in the standard list is scalar and therefore misses distinction #1 by design: a copula
can change *shape* (tail dependence appears, asymmetry flips) at **constant** Spearman $\rho$. Take
the state to be the Hermite coefficient vector $a_t\in\mathbb R^K$ (random walk), observations
$h_{jk}(U_t,V_t)$ — already computed in
[hermiter.hpp](../../src/estimators/dependence/hermiter.hpp) — and detect on the innovation *vector*:

$$
S_t=\max\!\big(0,\ \|S_{t-1}+e_t\|_{\Sigma^{-1}}-k\big).
$$

No published streaming-copula equivalent that I am aware of.

**4. The optimal-delay benchmark.**
Lorden / Lai: $\mathbb E[\text{delay}]\approx \log(\mathrm{ARL}_0)/\mathrm{KL}(\theta_1\|\theta_0)$.
For Gaussian copulas the KL between $\rho_0$ and $\rho_1$ is closed form (KL between two bivariate
normals with unit variances). So we can **compute the theoretically minimal detection delay** for any
$\rho$ jump and plot every detector against it. This upgrades the evaluation chapter from
"we compared some things" to **"here is the efficiency frontier and how close each detector gets."**

**5. Window-limited GLR** — see the caveat in Level 2.

---

# Part V — Architecture: the loop, not the pipeline

The obvious architecture is one-way:

```text
estimator ──► detector ──► { keep updating | restart }
```

That topology is wrong, or at least uninteresting. The detector's output should **modulate the
estimator's memory**:

```text
   innovations ──► detector ──► P(change | data) ──► λ_t ──┐
        ▲                                                   │
        └────────────────── estimator ◄─────────────────────┘
```

Hard reset is simply $\lambda_t\to 0$ when the posterior crosses a threshold. Which means:

$$
\boxed{\text{VFF and detect-and-restart are not rivals — they are one dial at different decision hardness.}}
$$

with Harrison-Stevens / IMM soft-mixing sitting in between. The roadmap currently frames these as
competing options (#1 vs #2 of "the ultimate streaming estimator"); they are the same architecture,
which is both a cleaner story and less to build.

---

# Part VI — Evaluation

## The trap: detectors are only comparable at matched $\mathrm{ARL}_0$

Every detector has a threshold $h$ that trades false alarms against delay. Comparing two detectors at
arbitrary thresholds is **meaningless**. Calibrate all of them to the same in-control average run
length $\mathrm{ARL}_0$ under stationarity *first*, then compare delay.

And because of rank feedback (distinction #3), $\mathrm{ARL}_0$ **must be calibrated by
simulation / bootstrap on the actual pseudo-observation stream**, not read off a formula. The
harness does not measure $\mathrm{ARL}_0$ at all yet — this is the first thing to add, before any
detector number is a result.

## Metrics

$$
\boxed{
\begin{array}{ll}
\mathrm{ARL}_0 & \text{in-control average run length (false alarms)}\\
\mathbb E[\text{delay}] & \text{detection delay, at matched }\mathrm{ARL}_0\\
\mathbb E[\text{delay}]/\text{bound} & \text{efficiency vs the Lorden/KL bound (Part IV.4)}\\
\text{recovery time} & \text{ticks until }|\hat\rho_t-\rho_t|\text{ back within tolerance}\\
\text{post-change RMSE} & \\
\text{stationary variance} & \text{the cost paid when nothing happens}\\
\text{per-update cost} & \text{ns/tick and bytes of state (streaming-admissibility)}
\end{array}}
$$

Multi-seed with error bars — see the open item in [todo.md](../todo.md).

## The sharpened research question

Not "how does the choice of detector affect the trade-off" (answers to *it depends*), but:

> **How closely can a streaming, $O(1)$, marginal-invariant copula change-point detector approach the
> information-theoretic detection-delay bound — and does adaptive memory dominate the entire
> fixed-$\lambda$ family on the $\mathrm{ARL}_0$ / delay / stationary-variance frontier?**

---

# Build order

1. **$\mathrm{ARL}_0$ calibration in the harness.** Bootstrap under stationarity. Nothing else is
   interpretable without it.
2. **Local-level Kalman** on Fisher-$z$'d blockwise Spearman (`kalman_local_level.hpp`). ~150 lines,
   drops in beside the existing estimators, and Bridge 1 means it is a strict generalization of the
   fixed-$\lambda$ estimators rather than a new branch.
3. **Innovation CUSUM** on $e_t/\sqrt{F_t}$, wired to the existing `reset()` hook
   (`estimator.hpp:16`).
4. **Efficient-score CUSUM** (Part IV.2) — marginal-invariant, reuses the Fisher-scoring machinery.
5. **Window-limited GLR** (Part III, Level 2) as the unknown-$\rho_1$ comparator.
6. **BOCPD** (NIW-on-normal-scores, per roadmap) as the Bayesian gold standard.
7. *Then* the novelty bets: **test martingales** (IV.1), **MCUSUM on Hermite coefficients** (IV.3),
   **regime memory** (Part II).

Steps 2-3 alone give the thesis its cleanest single result: *the fixed-$\lambda$ estimators are one
point on a curve the Kalman filter parameterizes, and the drift the filter absorbs is exactly what
was making CUSUM chatter.*

---

# References to chase

Everything below except the first entry was written from memory and has **not** been checked against
a bibliographic database. Treat them as leads, not citations.

**Read in full:**

- **Rodríguez, C. E. & Walker, S. G. (2021), "Copula Particle Filters", *Computational Statistics
  and Data Analysis* 161, 107230.** doi:10.1016/j.csda.2021.107230. Copula as a device to eliminate
  the propagation integral in an SSM; likelihood as a by-product; R code + datasets in the
  supplement. See the Level-4 aside in Part III — **and the $\rho$-conflation warning there.**

**From memory, unverified:**

- Basseville & Nikiforov (1993), *Detection of Abrupt Changes* — innovation-based detection; the
  nuisance-rejection / parity-space material behind Part IV.2.
- Willsky & Jones (1976) — GLR for jumps in linear systems.
- Lai (1995, 1998) — window-limited GLR, near-optimal delay.
- Harrison & Stevens (1976); West & Harrison, *Bayesian Forecasting and Dynamic Linear Models* —
  multiprocess DLM, discounting, cumulative Bayes factor monitoring.
- Adams & MacKay (2007) — BOCPD. Fearnhead & Liu (2007) — online multiple change-points with
  parameter reuse.
- Creal, Koopman & Lucas (2013) *Generalized Autoregressive Score models*; Harvey (2013) — **our
  online Fisher scoring with forgetting is a GAS/DCS model with $B=\lambda$**, and they have a
  time-varying copula example. Free citation for the parametric branch.
- Vovk (conformal test martingales); Ramdas et al. (safe anytime-valid inference, e-values).
- Crosier (1988) MCUSUM; Lowry et al. (1992) MEWMA.
- Bücher & Kojadinovic — sequential tests for constancy of the copula.
