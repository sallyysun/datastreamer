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

Every method in the roadmap's survey list is the same sequential recursion. Only three things vary:

| Knob | Options | Who is who |
| --- | --- | --- |
| **Unknown change time $\tau$** | $\max$ over $\tau$ / $\sum$ over $\tau$ | CUSUM, GLR $=\max$ (Viterbi). SR, BOCPD $=\sum$ (forward filter). |
| **Unknown post-change $\theta_1$** | known plug-in / maximize / integrate against prior | CUSUM $=$ known. GLR $=$ maximize. SR-$r$, BOCPD $=$ integrate. |
| **Input statistic** | raw observation / score / log-LR / kernel embedding | Page-Hinkley, sequential score test, CUSUM, MMD. |

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
