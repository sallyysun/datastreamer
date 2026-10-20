> **Status.** This is `COPULA V1` — the consolidated rewrite of
> `notes/studies/kalman_changepoint_results.md` (2868 lines, organised chronologically).
> Sections fill in one at a time; anything marked *stub* is not yet written and its content
> still lives in the old document. Nothing measured has been discarded.

## 1. Introduction

### 1.1 The four layers

Each estimator is one choice from each of three modelling layers, under a fourth that is a budget.

```text
LAYER 1   WHAT DO I ASSUME?             LAYER 3   HOW DO I HANDLE CHANGE?
          ├── Dependence                          │  (one question: how long should
          │     ├── Parametric                    │   my memory be, right now?)
          │     │     Gaussian / Clayton          │
          │     │     Gumbel / t                  ├── fixed                lambda
          │     │       the SAME families         ├── continuously         VFF
          │     │       generate the data,        │   adapted
          │     │       + Frechet, mixgauss       ├── hard window          drop past W
          │     │       as DGP only               ├── averaged over        banks, BOCPD
          │     └── Nonparametric                 └── jump-changed         detect, then
          │           no family assumed               at detected              reset / inflate
          │                                           instants                 / deflate / nothing
          └── Marginals
                ├── standardization (Welford)
                └── rank-PIT

LAYER 2   HOW DO I ESTIMATE?
          │  (which algorithm, and what it assumes about DYNAMICS)
          ├── Binned empirical copula      Xiao
          ├── Orthogonal basis             Hermite
          ├── Likelihood                   MLE / Fisher scoring
          └── State space                  Kalman / EKF / UKF / particle
                                           no copula assumed, but a
                                           parametric random walk

────────────────────────────────────────────────────────────────────────────────
LAYER 4   CAN I AFFORD IT?          applies to whatever was picked above
          ├── memory      store_bytes
          ├── time        ticks/sec
          └── degradation when the arrival rate exceeds the processing rate
```

- Layers 1–3 are independent. **Layer 4 is a constraint, not an alternative — it can flip the
  ranking.** The best estimator by RMSE is 11× slower than the default it beats.
- **State space is the one off-diagonal**: no copula family (L1), parametric random walk (L2).
- **Every layer-3 response is a memory change.** Reset → zero; inflate $P$ → higher gain; deflate
  `info_` → bigger step. The detector only decides *when*.

⚠️ **"EWMA" means two different things below.**

| | Layer | |
| --- | --- | --- |
| **EWMA smoothing** | 2 | the `_ew` variants — an **estimator** |
| **EWMA control chart** | 3 | `kalman_ewma_chart` — a **detector** |

⚠️ **λ is mirror-image.** Xiao and the MLEs: *retention*, larger = longer memory. Hermite: *update
weight*, smaller = longer. Both give $\mathrm{ESS}=(2-\alpha)/\alpha$, so λ = 0.99 and λ = 0.01 are
the same memory. Getting it backwards inverts an experiment.

### 1.2 What has been tried

```text
LAYER 1  WHAT DO I ASSUME?
  Dependence
    ✓ Parametric        Gaussian / Clayton / Gumbel / t — one family list, used three ways
        as estimator      gaussian_mle, clayton_mle, gumbel_mle, t_mle, copula_select
        as DGP            all four, + frechet and mixgauss (DGP only)      (6 families)
        as a change       gauss2t — Gaussian -> t at matched rho_S
    ✓ Nonparametric     no family assumed — the Xiao, Hermite and state-space branches
  Marginals
    ✓ standardization   Welford; the default everywhere
    ✓ rank-PIT          Hermite, the copula MLEs, and Xiao (14.19)

LAYER 2  HOW DO I ESTIMATE?
  ✓ Binned empirical   xiao            8 / 20 / 40 bins measured
  ✓ Orthogonal basis   hermite         order 10 / 20 / 40 measured
  ✓ Likelihood         the four MLEs, copula_select, + a Fisher-scoring ablation
  ✓ State space        kalman, ekf, ukf, particle
  ✗ Coefficient        built and measured; no detection power, twice
    monitoring

LAYER 3  HOW DO I HANDLE CHANGE?      how long should my memory be, right now?
  ✓ fixed              every _ew variant; a 7-point sigma_omega2 sweep and a
                       5-point lambda sweep
  ✓ hard window        windowed_rank, xiao_win
  ✓ continuously       VFF for Kalman, Hermite and the MLE
    adapted
  ✓ averaged over      kalman_bank, xiao_bank, hermite_bank, mle_bank; BOCPD
  ✓ jump-changed       CUSUM, GLR, EWMA chart, Shiryaev-Roberts
      responses          hard reset, no-op, covariance inflation, information
                         deflation, weight boost (unstable)
      calibration        every threshold at matched ARL0 = 1000 blocks
  ✗ shape detection    tail monitor built and calibrated; 0/20

LAYER 4  CAN I AFFORD IT?
  ✓ Memory             store_bytes, every estimator
  ✓ Time               ticks/sec, 8 estimators — a 74x spread
  ⬜ Rate budget        what happens when arrival outruns processing
```

**Detector × estimator.** Every detector used to wrap `KalmanLocalLevel` — it was the only
estimator emitting a standardized innovation. No longer true.

| | Fixed λ | VFF | Bank | CUSUM | GLR | EWMA chart | SR | BOCPD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Xiao** | ✓ | ⬜ | ✓ | ✓ | ⬜ | ⬜ | ⬜ | ⬜ |
| **Hermite** | ✓ | ✓ | ✓ | ✓ | ⬜ | ⬜ | ⬜ | ⬜ |
| **MLEs** | ✓ | ✓ | ✓ | ✓ | ⬜ | ⬜ | ⬜ | ⬜ |
| **Kalman** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

What closed the CUSUM column was a **statistic**, not wiring — each detector needs a surprise with
a known scale:

| | Innovation? | |
| --- | --- | --- |
| Kalman | ✓ | $e_t/\sqrt{F_t}$, standard normal under the null |
| MLEs | ✓ | the **score** — mean zero, variance in `info_` |
| Hermite | ✓ | coefficient surprise $\lVert h_xh_y^\top - A\rVert$ — measured, no power |
| Xiao | ⬜ → ✓ | none natively; a per-tick rank residual supplies one |

⚠️ The remaining ⬜ are not one kind of gap. GLR, the EWMA chart and Shiryaev–Roberts were each
found indistinguishable from CUSUM *on* the Kalman filter at matched ARL₀, so porting them would
likely reproduce the CUSUM row. Whether they read as "not done" or "not worth doing" is undecided.

### 1.3 How this document is arranged

§3 states two findings. There are about fifteen results, so the findings quote perhaps four of them
and the rest appear only in the layer chapters.

| | holds |
| --- | --- |
| §4 | the protocol. What was run, on what, and how each piece was checked |
| §5 | **Layer 1.** The misspecification grid, rank-PIT, and the Fréchet copula where the nonparametrics finally win |
| §6 | **Layer 2.** The process-noise sweep and the bank, the particle filter, EKF/UKF, the Fisher-scoring ablation |
| §7 | **Layer 3.** The detectors and their calibration, the reset finding, the score-CUSUM, the shape change, the tail monitor |
| §8 | **Layer 4.** `store_bytes` and ticks/sec across the whole set, a 74× spread |
| A–C | the DGP families; constants, verification and harness bugs; retracted claims |

A finding is a claim and the layer chapters are its evidence. §3 asserts 8.2σ; §6 is where that
number was measured and against what. The negative results live only in the layer chapters, and a
reader needs them to avoid repeating work that has already failed.

## 2. Background

Statements and formulas only. §2.4, §2.6 and §2.7 compress
`03_estimation/change_point_detection.qmd` rather than replace it.

### 2.1 Definitions

**The five distinctions** — why a copula changepoint is harder than most. Cited by number
throughout; full statements in `notes/studies/roadmap.md`.

| # | | |
| --- | --- | --- |
| 1 | Joint, not scalar | the thing changing is a surface; ρ is one number off it |
| 2 | Marginal confound | ignore volatility changes, detect dependence changes |
| 3 | Rank feedback | ranks shift as data arrives → the stream is not i.i.d. |
| 4 | Weak signal | dependence converges slowly → delay is fundamental |
| 5 | Drift, not jumps | real dependence mostly drifts; jumps are rare |

⚠️ **Distinction 3 undermines the theory** — every optimality result in §2.7 assumes i.i.d.
pre-change data. Rank streams are not.

**Two terms.**

- **Functional** — a map from a whole object to a number. $\rho_S$ is a lossy functional of the
  copula; §2.2 shows exactly what it loses. *Which functional the detector watches* is the design
  choice §7 finds matters most.
- **Multiplier bootstrap** — a null distribution by simulation, for when distinction 3 rules the
  textbook one out: multiply existing terms by random mean-zero weights, recompute.
  **[Segers 2012]** ✓, **[Bücher & Volgushev 2013]** ✓.

⚠️ Neither term resolves distinction 3 — they only make it tractable. What *would* is an
anytime-valid threshold; parked in §9 rather than developed here.

### 2.2 Copulas

**[Sklar 1959]** ⚠️ (textbook **[Nelsen 2006]** ✓) — for continuous margins the split into
marginals and dependence is unique:

$$F(x,y) = C\big(F_X(x),\, F_Y(y)\big)$$

$C$ is a **surface** on the unit square. Every estimator here reports one number off it:

$$\rho_S = 12\iint_{[0,1]^2} C(u,v)\,du\,dv - 3$$

**Read that integral as an average.** $\rho_S$ is an affine rescaling of the mean height of $C$
over $[0,1]^2$; the 12 and the 3 only stretch it onto $[-1,1]$:

Checked on the three extreme copulas — the **Fréchet–Hoeffding bounds** and independence, between
which every copula on the unit square is trapped:

$$W(u,v) = \max(u+v-1,\,0) \;\le\; C(u,v) \;\le\; \min(u,v) = M(u,v)$$

| $C$ | | $\iint C$ | $\rho_S$ |
| --- | --- | --- | --- |
| $M(u,v)=\min(u,v)$ | comonotone, $V=U$ | $1/3$ | $+1$ |
| $\Pi(u,v)=uv$ | independence | $1/4$ | $0$ |
| $W(u,v)=\max(u+v-1,0)$ | countermonotone, $V=1-U$ | $1/6$ | $-1$ |

Mixing these three is the **Fréchet family** — the adversarial DGP of Appendix A.5, whose density
is a singular ridge on the diagonal plus a flat background.

So $\rho_S$ is a **functional** — a map from a whole function to a number — and it is **lossy**,
because an average is not injective:

```text
C(u,v)   a surface on [0,1]^2        infinite-dimensional
   |
   |   ∫∫ · du dv    <- keeps the mean height, discards the shape
   v
rho_S    one number in [-1,1]        1-dimensional
```

Infinitely many copulas share one $\rho_S$. **Which ones differ is the whole problem**, because
what they differ in is how much mass sits in the corners:

$$\lambda_L = \lim_{q\to 0^+}\frac{C(q,q)}{q}, \qquad
\lambda_U = \lim_{q\to 1^-}\frac{1-2q+C(q,q)}{1-q}$$

```text
   v=1  +---------------+
        | .           X |   <- lambda_U : a LIMIT at the corner
        |   .       .   |
        |     .   .     |      rho_S : an AVERAGE over the whole box
        |   .       .   |
        | X           . |   <- lambda_L : a LIMIT at the corner
   v=0  +---------------+
       u=0             u=1
```

Away from the extremes, fixing the average barely constrains the corners — mass can be moved into
them with the mean height held flat. That is **distinction 1**, and it is not hypothetical: the
`gauss2t` DGP (Appendix A.7) switches Gaussian → t(4) with

```text
rho_S      flat to five decimals     <- the functional sees nothing
lambda_U   0  ->  0.3                <- the copula changed a great deal
```

⚠️ **A detector watching $\rho_S$ is structurally blind to any change that leaves the mean height
alone.** Not a tuning failure and not a power shortfall — the change lies in the kernel of the
functional being monitored. §7 measures it: every detector lands at 1–2/20, its false-alarm rate.

**One thing the averaging buys back.** $\rho_S$ is **linear in $C$**, so a mixture
$C=\sum_i w_i C_i$ has $\rho_S=\sum_i w_i\,\rho_S(C_i)$ exactly. That is what gives the Fréchet and
mixture-Gaussian families in Appendix A closed-form ground truth rather than a calibrated one.

Family definitions, Kendall/Spearman relations and samplers: **Appendix A**.

### 2.3 Ranks

- Replace a value by its rank among what has been seen. No assumption about the marginal's shape.
- $\hat\rho_S$ is a rank statistic ⇒ **invariant to any monotone transform of either marginal**.
  Volatility doubles, ranks do not move. That is distinction 2.
- **The default here is not ranks** — running Welford standardization, exact only for Gaussian
  margins. **[Fantazzini 2009]** ✓: with skewed data and symmetric marginals assumed, elliptical
  copula correlations biased down by **up to 70%**.
- **Rank-PIT.** The probability integral transform says that if $X\sim F_X$ then
  $F_X(X)\sim U(0,1)$ — apply it to both legs and what is left is the copula alone. Using it needs
  $F_X$, and there are only three places to get one:

```text
the TRUE F        unavailable
an ASSUMED F      Welford standardization -- assumes Gaussian.  <- the default here
the EMPIRICAL F   ranks -- assumes nothing at all.              <- rank-PIT
```

  So "by ranks" is *as opposed to by an assumed parametric marginal*. Replace each observation by
  its normalised rank among the $n$ seen so far:

$$\hat u_i = \frac{\operatorname{rank}(x_i)}{n+1}, \qquad
\hat v_i = \frac{\operatorname{rank}(y_i)}{n+1}, \qquad \hat u,\hat v \in (0,1)$$

  Since $\operatorname{rank}(x_i)/n \to F_X(x_i)$, the pair $(\hat u,\hat v)$ converges to a draw
  from the copula itself with the marginals stripped **exactly** — no shape assumed anywhere. The
  $n+1$ denominator keeps the values off the closed endpoints, where a copula density can diverge.
  ⚠️ An exact rank needs the whole history, so this is a layer-1 choice with a layer-4 bill; §5
  covers the streaming form and what it costs. Built for Hermite, the MLEs, and Xiao.

### 2.4 Exponential forgetting

The project's central device — the first two words of the thesis title.

$$\hat\theta_t = (1-\alpha)\,\hat\theta_{t-1} + \alpha\, y_t
\qquad\Longleftrightarrow\qquad
\hat\theta_t = \alpha\sum_{k\ge 0}(1-\alpha)^k\, y_{t-k}$$

- **Weights decay geometrically** — $y_{t-k}$ carries $\alpha(1-\alpha)^k$. That is what makes it
  *exponential*, and what makes it $O(1)$ state: the whole past is one number.
- Written as retention, $\lambda = 1-\alpha$, so $\hat\theta_t = \lambda\hat\theta_{t-1} +
  (1-\lambda)y_t$. ⚠️ Hermite spells λ the other way — §1.1.

$$\mathrm{ESS} = \frac{2-\alpha}{\alpha} \approx \frac{2}{\alpha}$$

how many equally-weighted points carry the same variance. With observation variance $\sigma^2$ and
truth drifting at $\dot\theta$ per step:

$$\mathrm{MSE}(\mathrm{ESS}) \;\approx\;
\underbrace{\frac{\sigma^2}{\mathrm{ESS}}}_{\text{variance}} \;+\;
\underbrace{\Big(\frac{\dot\theta \cdot \mathrm{ESS}}{2}\Big)^{\!2}}_{\text{lag bias}}$$

- A U-shape with an interior minimum. **This drives most of §6**: cumulative
  ($\mathrm{ESS}\to\infty$) wins `constant` and is catastrophic on `drift`; short memory is the
  reverse.
- It is also why the alternative to *choosing* $\alpha$ is carrying a posterior over it (§6).

**This is one of five memory strategies** (§1.1's layer-3 ladder). The other four — hard window,
VFF, posterior over memory, and jump-changed at a detected instant — are defined where they are
measured, in §6 and §7.

**Variable forgetting — the `_vff` suffix.** Rather than fixing the memory, adapt it from how
surprising the data has been. On the Kalman line $\tilde e_t^{\,2}$ averages 1 under a correct
model, so a sustained excess means the filter is under-modelling change:

$$C_t = (1-r)\,C_{t-1} + r\,\tilde e_t^{\,2},
\qquad
\sigma^2_{\omega} \leftarrow \sigma^2_{\omega}\cdot C_t^{\,r}$$

- $C_t > 1$ — more surprise than the model allows ⇒ inflate $\sigma_\omega^2$ ⇒ **shorter memory**.
- $C_t < 1$ — less ⇒ deflate ⇒ **longer memory**.
- $r$ is `vff_rate`, the same kind of rate knob as $p$ below and $c$ in §2.7.
- It is a **heuristic, not a posterior** — one tracked scalar, not a belief over memory length.
  §6 measures what that costs against the bank.
- ⚠️ The MLE and Hermite variants apply the identical idea to their own surprise, but λ is
  mirror-image between them (§1.1): shortening memory *lowers* the MLE's λ and *raises* Hermite's.

**Sticky weights — the `_sticky` suffix.** The same forgetting idea applied to a *set of models*
rather than to a value. Given weights $w_k$ over $K$ candidate filters, mix a little uniform mass
back in every step:

$$w_k \;\leftarrow\; (1-p)\,w_k \;+\; \frac{p}{K}, \qquad p = \texttt{switch\_prob}$$

- $p$ is a **hazard rate over which model is right** — exactly what $H$ is for BOCPD's run length.
- $p \to 0$ — weights persist, but once they converge they can never move again. That is the
  cumulative-estimator trap in a new place.
- $p$ large — weights are dragged back to uniform and learn nothing.
- `_sticky` marks $p = 0.002$: small, but deliberately nonzero. §6 measures the ordering
  $0.002 > 0.02 > 0.15 > \text{frozen}$.

### 2.5 Blocks and the Fisher transform

$$z = \operatorname{atanh}\rho = \tfrac12\log\frac{1+\rho}{1-\rho} \in \mathbb{R}
\qquad
\sigma_v^2 \approx \frac{1.06}{m-3}$$

- ρ is bounded, so its noise shrinks as $|\rho|\to1$ — a Gaussian noise model is false near the
  ends. Fisher-$z$ removes that.
- One tick cannot estimate a correlation, so $\hat\rho_S$ is computed per **block** of $m$ ticks.
- **[Fieller, Hartley & Pearson 1957]** ⚠️, for Fisher-$z$'d *Spearman* — ~6% wider than Pearson's
  $1/(m-3)$.
- **$\sigma_v^2$ being known rather than estimated is the structural gift** — it pins the noise
  scale, leaving one free parameter instead of two.

⚠️ Approximately, not exactly: Fisher-$z$ is normal only asymptotically, $m=20$ is not asymptotic,
and block Spearman is a rank statistic on discrete support. Two things limit the damage — the
Kalman filter stays the minimum-MSE *linear* estimator without normality, and ARL₀ is bootstrapped
on the real stream (§4).

### 2.6 State space and the Kalman filter

Local-level model, $z_t = \operatorname{atanh}\rho_t$, $y_t$ the block measurement of §2.5:

$$z_t = z_{t-1} + \omega_t,\quad \omega_t \sim \mathcal N(0,\sigma_\omega^2)
\qquad
y_t = z_t + v_t,\quad v_t \sim \mathcal N(0,\sigma_v^2)$$

$$
\begin{aligned}
\textbf{predict:} \quad & \hat z_{t|t-1} = \hat z_{t-1}, && P_{t|t-1} = P_{t-1} + \sigma_\omega^2 \\
\textbf{innovation:} \quad & e_t = y_t - \hat z_{t|t-1}, && F_t = P_{t|t-1} + \sigma_v^2 \\
\textbf{gain:} \quad & K_t = P_{t|t-1} / F_t && \\
\textbf{update:} \quad & \hat z_t = \hat z_{t|t-1} + K_t e_t, && P_t = (1-K_t)P_{t|t-1}
\end{aligned}
$$

- Under the model $\tilde e_t = e_t/\sqrt{F_t} \sim \mathcal N(0,1)$, i.i.d. **Every detector in §7
  monitors this.**
- **$K \in (0,1)$ is how much of the surprise you act on.** Uncertain filter ⇒ $K \to 1$, follow
  the data; confident filter ⇒ $K \to 0$, ignore a noisy read. So *raising $P$ raises $K$ raises
  the gain* — which is why inflating the covariance and shortening the memory are the same
  operation (§1.1), and why a larger $\sigma_\omega^2$ beat a detector-with-inflation (§7).
**Bridge 1 — a settled Kalman filter *is* an EWMA.** Run the recursion above on a stationary
stream and $P_t$ stops moving. A constant $P$ means a constant $K$, and a constant $K$ means the
update is a fixed-weight blend of "what I thought" and "what I just saw":

```text
warming up    P_t still shrinking   ->   K_t still moving   ->   the gain ADAPTS
settled       P_t constant          ->   K constant         ->   a FIXED-weight update
```

With $q = \sigma_\omega^2/\sigma_v^2$ the settled gain has a closed form, and the update collapses:

$$K = \frac{\sqrt{q^2+4q}-q}{2}
\qquad\Longrightarrow\qquad
\hat z_t = (1-K)\,\hat z_{t-1} + K\,y_t$$

That right-hand side is §2.4's EWMA, with $\boxed{\lambda = 1-K}$ — read it as *keep $\lambda$ of
what you had, take $K$ of what arrived*.

- **Every fixed-λ estimator here *is* this filter at steady state** — the same recursion, not an
  analogy. Verified to 5 dp in §4.
- Since $\sigma_v^2$ is known, choosing $\sigma_\omega^2$ *is* choosing λ — but $\sigma_\omega^2$
  is physical, so data can speak to it.
- **[Kalman 1960]** ⚠️, **[Harvey 1989]** ⚠️, **[Durbin & Koopman 2001]** ⚠️.

**So what does the filter buy?** Bridge 1 makes the two the same recursion, which makes this the
obvious question.

- ⚠️ **Not accuracy.** Best achievable Kalman on `oscillate` is 0.2256; tuned Xiao 0.1826,
  `gaussian_mle_l95` 0.1710. Block aggregation is a floor it cannot tune past (§6.2). If all you
  want is $\hat\rho$, a fixed λ on a per-tick estimator is better.
- **$P_t$ — the uncertainty.** Every detector thresholds $e_t/\sqrt{F_t}$, and $F_t = P_t +
  \sigma_v^2$. **No $P$, no standardized innovation, no calibrated detector.** That is exactly why
  §1.2's matrix was Kalman-only for months — the others had no uncertainty to divide by.
- **$\sigma_\omega^2$ is physical; λ is not.** λ parameterises nothing in the world, so it cannot
  be estimated. $\sigma_\omega^2$ is *how fast dependence moves*, so data can speak to it. VFF and
  the posterior-over-memory both exist only because of this.
- **It generalises.** EKF, UKF, particle filter, BOCPD, time-varying $\sigma_v^2$, missing data —
  all reachable from a state-space model, none from a fixed λ. Every one was built here.
- **The transient.** Starts uncertain and tightens, so cold start handles itself; a fixed λ is born
  at its final memory. The principled version of the ad-hoc `hermite_ew` cold-start patch.

> **A fixed λ is the filter's *answer*. The filter is the *reasoning*.** Bridge 1 does not say the
> filter is redundant — it says every `_ew` estimator was already running one without knowing, and
> therefore silently discarding $P$.

⚠️ §6.4 builds the same posterior-over-memory off a *non*-Kalman estimator, using the per-tick rank
residual as a stand-in predictive density — and `mle_bank` then beats `kalman_bank` *(retracted:
`mle_bank` was never measured; see `copula_v2.md` §E and `thesis_audit.md` §1.4 item 3)*. The filter is
where these ideas came from, not the only place they can live.

### 2.7 Detectors

All five run on $\tilde e_t$, i.i.d. $\mathcal N(0,1)$ under $H_0$. They differ only in what they
do with the unknown change time — **maximise** over it, **sum** over it, **smooth** it, or carry a
**posterior** over it.

**CUSUM** **[Page 1954]** ⚠️ — **minimax-optimal** for Lorden's delay **[Lorden 1971]** ⚠️,
**[Moustakides 1986]** ⚠️. *Minimax* = best worst case: at a matched false-alarm rate, no detector
has a shorter delay against the **least favourable** change time and pre-change history. It says
nothing about the average case — that is Shiryaev–Roberts' criterion below.

$$S^{\pm}_t = \max\big(0,\; S^{\pm}_{t-1} \pm \tilde e_t - k\big), \quad k = \delta/2,
\qquad \text{alarm if } \max(S^+,S^-) > h$$

**CUSUM** is *cumulative sum*, and both halves of the recursion are derivable rather than chosen:

- **The increment is exactly a log-likelihood ratio.** For $\mathcal N(0,1)$ against
  $\mathcal N(\delta,1)$,

$$\log\frac{\phi(\tilde e_t-\delta)}{\phi(\tilde e_t)}
= \delta\,\tilde e_t - \frac{\delta^2}{2}
= \delta\Big(\tilde e_t - \frac{\delta}{2}\Big)$$

  so up to the positive factor $\delta$, the increment *is* $\tilde e_t - k$ with $k=\delta/2$.
  Summing it accumulates the log-LR for "a change occurred at $\tau$" against "no change".
- **The floor at zero is a maximisation over $\tau$.** Unrolling the recursion gives
  $S_t = \max_{0\le\tau\le t}\sum_{i=\tau+1}^{t}(\tilde e_i - k)$ — resetting to zero discards any
  candidate change time that has accumulated negative evidence, in favour of starting fresh. That
  is what makes it a detector rather than a random walk, and it is why CUSUM needs no memory of
  $\tau$ at all.

**Window-limited GLR** — *generalised likelihood ratio*. **[Willsky & Jones 1976]** ⚠️,
**[Lai 1995]** ⚠️.

- CUSUM has to be **told** the jump size $\delta$ in advance; it is baked into $k=\delta/2$.
- A likelihood ratio is called **generalised** when an unknown parameter is replaced by the value
  that maximises the likelihood, and the test then run as usual. Here *both* unknowns — the jump
  size $\delta$ and the change time $\tau$ — are maximised out.
- Maximising over $\tau$ is what turns it into a **matched filter** on the filter's own step
  response: a unit jump at $\tau$ contributes $(1-K)^{s-\tau}$ to the standardised innovation, so
  each candidate lag $j$ carries its own accumulator.

$$A(j) = \sum_{i=0}^{j}(1-K)^i\,\tilde e_{t-i}, \quad
N(j) = \sum_{i=0}^{j}(1-K)^{2i}, \quad
g_t = \max_{j}\frac{A(j)^2}{N(j)}$$

- $g_t$ is the best-matching lag's squared, variance-normalised response. $A$ updates by a shift
  register and $N$ is precomputed, so a block costs $O(W)$.
- **Capping $j\le W$ is what makes it streaming-admissible.** The unbounded $\sup_\tau$ version is
  $O(t)$ per step — its cost grows with the length of the stream, which is disqualifying here.

**Shiryaev–Roberts** **[Shiryaev 1963]** ⚠️ — sums over the change time; optimal for Pollak's
*conditional-average* delay **[Pollak 1985]** ⚠️:

$$R_t = (1 + R_{t-1})\,\Lambda_t, \qquad \text{alarm if } R_t > h$$

where $\Lambda_t = p_1(\tilde e_t)/p_0(\tilde e_t)$ is the likelihood ratio of the post- against
the pre-change density — so $\mathbb E_0[\Lambda_t] = 1$, and $R_t$ is a running **sum of products**
of these, one product per candidate change time.

**EWMA chart** **[Roberts 1959]** ✓ — smooths instead:

$$g_t = (1-r)g_{t-1} + r\,\tilde e_t, \qquad
\text{alarm if } |g_t| > L\sqrt{\tfrac{r}{2-r}}$$

**BOCPD** — *Bayesian online changepoint detection*. **[Adams & MacKay 2007]** ⚠️. Instead of
testing a hypothesis it carries a posterior over the **run length** $r_t$ — blocks elapsed since
the last change. With hazard $H$ and $L(r)$ the predictive density of $y_t$ under hypothesis $r$:

$$P(r_t = 0,\,y_{1:t}) = H\sum_r P(r_{t-1}=r,\,y_{1:t-1})L(r)$$
$$P(r_t = r{+}1,\,y_{1:t}) = (1-H)\,P(r_{t-1}=r,\,y_{1:t-1})L(r)$$

**No reset anywhere — the soft mixing is the mechanism**, which is why §7 finds it escapes the
reset cost entirely.

> ⚠️ $P(r_t=0)$ is **useless as an alarm** under constant hazard. With
> $S=\sum_r P(r_{t-1}=r)L(r)$: reset branch $HS$, growth branches $(1-H)S$, so
> $P(r_t=0\mid\text{data}) = HS/(HS+(1-H)S) = H$ identically, every tick, whatever the data. The
> alarm thresholds $P(r_t \le c)$, $c>0$. §4 records it as a bug found by verification.

**$\mathrm{ARL}_0$** $= \mathbb E[\,T_{\text{alarm}} \mid \text{no change}\,]$, in blocks.

- **A threshold with no ARL₀ attached is meaningless** — any detector looks fast if allowed to cry
  wolf. Every threshold here is calibrated to 1000 blocks (§4); only then are delays compared.
- Distinction 3 means it must be bootstrapped on the real stream, not read off a formula.

**Standardising the statistic — the `_rob` suffix.** Every detector above divides by an assumed
scale. On the Kalman line that scale is $\sqrt{F_t}$, and it is genuinely known. For the MLE score
CUSUM it is the Fisher information, $\tilde e_t = s_t/\sqrt{I}$ — which silently assumes the
**information matrix equality** $\operatorname{Var}[s]=I$, true only when the family is right
(**[White 1982]** ⚠️). Off-family the two come apart. The `_rob` variants stop trusting the model's
scale and measure one instead:

$$\hat J_t = (1-c)\,\hat J_{t-1} + c\,s_t^2,
\qquad
\tilde e^{\,\text{rob}}_t = \frac{s_t}{\sqrt{\hat J_t}}$$

- That is a **robust (sandwich) standard error** in the ordinary statistical sense: don't trust the
  model's variance formula, estimate the variance from the data. "Robust" here always means
  *robust to misspecification*.
- ⚠️ $c$ must be far slower than the change being detected — $c = 0.001$ against a $\lambda$ of
  0.99. A fast scale estimate **chases the signal and divides it away**.
- Note the form: this is the sticky step of §2.4, pointed at a variance instead of a weight
  vector. §3 shows the two land on opposite sides of one rule.

### 2.8 One dial, many names

Nine symbols in this document describe how long an estimator remembers, or how fast something about
that changes. They sit in three tiers, and conflating the tiers is the main way this gets confusing.

**Tier 1 — the memory itself.** One quantity in five costumes, linked by Bridge 1:

$$\sigma_\omega^2 \;\longrightarrow\; q \;\longrightarrow\; K \;\longrightarrow\; \lambda
\;\longrightarrow\; \mathrm{ESS}$$

| estimator | its knob | value | **memory, in ticks** |
| --- | --- | --- | --- |
| `kalman` | $\sigma_\omega^2$ | $10^{-4}$ | **≈ 1000** |
| `kalman_fast` | $\sigma_\omega^2$ | $10^{-2}$ | **≈ 100** |
| `xiao_ew` | $\lambda$ (retention) | $0.99$ | **≈ 200** |
| `hermite_ew` | $\lambda$ (update weight) | $0.01$ | **≈ 200** |

Those last two rows are the mirror-image trap of §1.1. Written as ticks it disappears: $0.99$ and
$0.01$ are the same memory. **Wherever a value appears, the memory in ticks appears with it.**

**Tier 2 — hazards.** $H$ for BOCPD, over run length. $p$ for the bank, over which memory is right.
These are *not* memory. A hazard is a prior the data can override; a memory is fixed until something
changes it. That distinction is exactly why BOCPD can beat a fixed $\lambda$ (§6).

**Tier 3 — adaptation rates.** $r$ for VFF, $c$ for `_rob`. How fast a meta-parameter moves. Both
fail at either extreme: too slow and the parameter freezes, too fast and it chases the signal (§3).

| tier | members | always called |
| --- | --- | --- |
| 1 | $\sigma_\omega^2$, $q$, $K$, $\lambda$, $\mathrm{ESS}$ | **memory**, quoted in ticks |
| 2 | $H$, $p$ | a **hazard** |
| 3 | $r$, $c$ | an **adaptation rate** |

### 2.9 Symbols

| Symbol | Meaning |
| --- | --- |
| $\rho$, $\rho_S$ | copula (linear) correlation; Spearman's — **all estimators report $\rho_S$** |
| $\lambda_L$, $\lambda_U$ | lower / upper tail dependence |
| $m$ | block size in ticks |
| $z_t = \operatorname{atanh}\rho_t$ | latent state |
| $y_t$ | observation: Fisher-$z$'d block Spearman |
| $\sigma_\omega^2$ | process variance — how fast dependence is believed to move |
| $\sigma_v^2 \approx 1.06/(m-3)$ | observation variance — known in closed form |
| $q = \sigma_\omega^2/\sigma_v^2$ | signal-to-noise ratio |
| $\hat z_t$, $P_t$ | filtered mean and variance |
| $e_t$, $F_t$ | innovation and its variance; $\tilde e_t = e_t/\sqrt{F_t}$ |
| $K_t$, $K$ | Kalman gain; steady state $K=(\sqrt{q^2+4q}-q)/2$ |
| $\lambda$, $\alpha$ | forgetting parameter and rate; $\lambda = 1-K$ at steady state |
| $\mathrm{ESS}=(2-\alpha)/\alpha$ | memory length in points |
| $h$, $k$, $\delta$, $L$ | detector threshold; CUSUM slack; assumed jump size; chart width |
| $N$, $W$ | GLR window length |
| $H$, $r_t$ | BOCPD hazard rate and run length |
| $\mathrm{ARL}_0$ | expected blocks to a false alarm under no change |
| $\tau$, $\nu$ | true changepoint time |

## 3. Finding

One finding, reached twice by separate routes. Everything else is a result and lives in §5–8 with
its evidence.

**The algorithm is not where the performance is.**

This project built two zoos. In both, the members turned out to be close to interchangeable, and
what decided the outcome sat outside the algorithm.

**The estimators — four unrelated derivations.**

- Xiao bins the copula. Hermite expands it in orthogonal polynomials. The MLEs fit a parametric
  family. The Kalman filter models a random walk.
- On smooth families they track within a few percent of one another.
- What separates them is **memory**. Sweeping it moves RMSE about four times more than the choice
  between these four does.

**The detectors — five different optimality proofs.**

- CUSUM is a running sum. GLR is a matched filter. Shiryaev–Roberts sums over the change time.
  BOCPD carries a Bayesian posterior over run length.
- On a copula shape change all five score 1/20 to 5/20, together.
- A plain CUSUM pointed at a different number scores 17/20. What separates them is **which number
  they watched**.

**The one apparent counterexample fits too.**

- BOCPD does beat every fixed-memory estimator on `jump`, by 25–30% across all four DGP families.
- But §6 attributes that to its run-length posterior *switching memory mid-stream*, which no fixed
  value can do.
- So it is a memory result, not an algorithm result.

In the language of §1.1:

| layer | | how much it decided |
| --- | --- | --- |
| 1 | what do I assume — **what you watch** | a lot |
| 2 | how do I estimate — **the algorithm** | least |
| 3 | how do I handle change — **how long you remember** | a lot |

**Layer 2 is where most of the literature sits, and it is the layer that mattered least here.**

The two routes to that conclusion are §3.1 and §3.2.

### 3.1 Learned memory length

Every estimator here is an average over the past. Its **memory** is how far back that average
reaches.

| memory | good at | bad at |
| --- | --- | --- |
| long | averaging noise away | noticing a change |
| short | noticing a change | averaging noise away |

**The memory you want changes while the stream runs.**

- In a calm stretch, long memory.
- In the ticks after a change, short memory, because the old data describes a regime that no longer
  exists.
- Once the new regime settles, long again.
- Those are opposite requirements. **No fixed setting is right, and the reason is not that nobody
  found the right one.**

**This is the knob that matters most, and it had never been swept.**

- Changing it moves RMSE about four times more than the choice of detector does.
- Every detector in this document nevertheless inherited a memory of ~100 ticks
  ($\sigma_\omega^2 = 10^{-2}$).

**The fix is to stop choosing it.**

- Run seven filters at seven memory lengths. Weight each by how well it predicted the block that
  just arrived.
- It costs 736 bytes.
- Through a jump it cuts its own memory roughly 300-fold within 60 ticks, then relaxes back over the
  next thousand. No detector, no threshold, no reset anywhere in that.
- It beats `known_best` by 8.2σ on `drift` and 9.3σ on `jump`, where `known_best` is handed the
  winning fixed value for each scenario in advance. → §6

### 3.2 The monitored input decides what can be detected

**Five detectors share one input**, the blockwise Spearman reading.

- They are very different algorithms. CUSUM is a running sum, GLR a matched filter, BOCPD a
  Bayesian posterior over run length.
- On a copula shape change at constant $\rho_S$ all five fail together, scoring 1/20 to 5/20.
- None of them is badly built. **The number they watch did not move**, because $\rho_S$ is an
  average over the unit square and the change moved mass into the corners (§2.2).

**One detector sees it, and it is the simplest one there.**

- `gaussian_mle_cusum` scores 17/20 on the same data.
- It is a plain CUSUM, not a better detector.
- It watches a different number, the MLE's score, which responds to the whole fit rather than to a
  single summary of it.

Algorithm sophistication bought nothing here. Choice of input bought everything. → §7

### 3.3 A rule for what you are allowed to adapt

- §3.1 lets the data set the memory. It wins by 8–9σ.
- §7.2 lets the data set a *different* parameter. Detection collapses.
- One rule separates the two cases.

| what gets adapted | what that quantity is | outcome |
| --- | --- | --- |
| the memory (§3.1) | *how fast is the truth moving?* Pure nuisance. Never the thing being detected. | free win, 8–9σ |
| the yardstick (§7.2) | *how badly does my model fit?* Nuisance and signal at once. | deletes the signal, 17/20 → 10/20 |

**This is not an analogy.** VFF (§2.4) and `_rob` (§2.7) run the identical recursion on the
identical quantity, an exponentially weighted mean of a squared standardised residual:

$$C_t = (1-r)\,C_{t-1} + r\,\tilde e_t^{\,2}
\qquad\text{alongside}\qquad
\hat J_t = (1-c)\,\hat J_{t-1} + c\,s_t^{\,2}$$

They then put that number to opposite use:

| | what it does with the surprise | reading |
| --- | --- | --- |
| VFF | $\sigma_\omega^2 \leftarrow \sigma_\omega^2\,C_t^{\,r}$ — shortens memory | **acts on it**: *I am surprised, so adapt faster* |
| `_rob` | $\tilde e_t \leftarrow s_t/\sqrt{\hat J_t}$ — divides it out | **explains it away**: *I am surprised, so that is normal now* |

Each adaptation also carries a **rate**, and fails at both extremes:

| rate | failure | seen in |
| --- | --- | --- |
| too slow | the parameter freezes and buys nothing | frozen bank weights; $c\to0$ |
| too fast | it chases the signal and divides it away | `hermite_cusum`; $c$ large |

**Adapt freely to a nuisance. Adapt slowly, or not at all, to anything that doubles as your
signal.**

**Appendix C** tabulates ten claims this project made and then disproved itself.

- Each row carries the control that killed it.
- Only six failure modes recur: a missing control, an effect too small, an unfair comparison, a
  test with no power, a confounded comparison, and a claim inferred rather than measured.
- Knowing which one a claim is exposed to is how the next control gets designed.

## 4. Method

The protocol only: what was run, on what, with which constants, and how each was checked. The
estimators themselves are defined in the layer chapter that measures them — §5 for what they
assume, §6 for how they estimate, §7 for how they handle change.

### 4.1 What was run

| | |
| --- | --- |
| scenarios | `constant`, `drift`, `jump`, `stochastic`, `oscillate` |
| DGP families | Gaussian, Clayton, Gumbel, Student-$t$ (+ Fréchet, mixture-Gaussian and `gauss2t` as generator-only) |
| estimators | 22, every key in `all_estimator_keys()` |
| seeds | 20 |
| ticks per run | $N = 5000$ |
| burn-in | 500 ticks, discarded before scoring |
| score | RMSE against the true $\rho_S$, reported as mean $\pm$ SE across seeds |
| detector metrics | alarms/run, false alarms before $\tau$, hit rate, detection delay — same convention |

Two harnesses share one key list:

- `./multiseed` — 20 seeds with error bars. **Every number in this document comes from it.**
- `./experiment` — one seed plus a per-tick CSV, for inspecting a trace. Diagnostic only.
- Both read `all_estimator_keys()` from `registry.hpp`. A key registered but absent from that list
  runs never and reports nothing — see §4.6, bug 4.

### 4.2 The observation channel

Every $m$ ticks, take the block's empirical Spearman correlation and Fisher-transform it:

$$y_t = \operatorname{atanh}\big(\hat\rho^{\,S}_t\big),
\qquad
\sigma_v^2 \approx \frac{1.06}{m-3},
\qquad
m = 20 \;\Rightarrow\; \sigma_v^2 \approx 0.0624$$

All five detectors share this one channel, so a change to it moves all of them together.

**Tick and block.** A **tick** is one arriving pair $(x,y)$. A **block** is $m=20$ consecutive
ticks.

The filter needs an observation to update on. A single pair cannot give it one, because a
correlation cannot be computed from one point. So there are two ways to proceed.

| | the observation | how it relates to the state $z=\operatorname{atanh}\rho$ |
| --- | --- | --- |
| **per tick** | the product of the two normal scores, $z_xz_y$, which has mean $\rho$ | **nonlinear.** It gives you $\rho$, and $\rho=\tanh z$ |
| **per block** | $\operatorname{atanh}$ of Spearman's $\rho$ over the 20 ticks | **linear.** It gives you $z$ directly, plus noise |

The per-block route is the one that keeps the filter closed-form. A nonlinear observation forces an
EKF or UKF; a linear one does not. It also hands you $\sigma_v^2$ in closed form, which leaves the
filter with one free parameter instead of two.

That convenience is not free:

- You only get an observation every 20 ticks, so the reading trails the truth by about $m/2 = 10$
  ticks.
- $\rho$ is assumed constant *within* a block. At $m=50$ that assumption broke on `oscillate` and
  made the filter worse than non-adaptive estimators (B.3, bug 8).

§6 measures what the lag costs.

**Marginal invariance is free.** $\hat\rho^S$ is a rank statistic, hence invariant to any monotone
transform of either marginal — so the efficient-score refinement is unnecessary on this channel.
⚠️ Verified in §4.5, but only on the channel that already passes; see R7.

### 4.3 $\mathrm{ARL}_0$ calibration

**A detector's threshold is one dial between fast and accurate.**

| threshold | detection is | false alarms are |
| --- | --- | --- |
| low | **fast** | many |
| high | slow | **few** |

- You cannot have both, so **a delay figure means nothing on its own**. Any detector can be made
  faster by accepting more false alarms.
- Comparing two detectors therefore requires holding one side fixed and measuring the other.
- $\mathrm{ARL}_0$ is the side held fixed: expected blocks to a false alarm when nothing is
  changing. Every threshold here is set so $\mathrm{ARL}_0 = 1000$ blocks. Only then are delays
  compared.

**It has to be measured, not computed.**

- The textbook formulas assume i.i.d. pre-change data, and rank feedback breaks that (distinction 3).
- `calibrate_arl0.cpp` runs each detector's real `update()` path on stationary streams at
  $\rho\in\{0,0.3,0.6,0.8\}$ and sweeps the threshold.
- It takes the smallest value whose **worst case over $\rho$** clears the target, because the true
  $\rho$ is unknown in live use.
- ⚠️ Changing $m$, the memory, or a hazard invalidates every threshold. Re-run it.

Verification and the eight bugs found along the way are in **Appendix B**. They are consulted, not
read.

### 4.4 Reproduction

```bash
cd experiments && mkdir -p results

make                    # ./experiment       single seed + per-tick CSV
make multiseed          # ./multiseed        error bars -- the published numbers
make calibrate_arl0     # ./calibrate_arl0   threshold calibration

for s in constant drift jump stochastic oscillate; do ./multiseed $s 20; done
./calibrate_arl0 1000

./run_multiseed_grid.sh 20 42            # 5 scenarios x 4 families, ~12 min
./run_multiseed_grid_parallel.sh 20 42   # same, all cores, ~3 min
```

`COPULA=gaussian|clayton|gumbel|t` selects the generator family; non-Gaussian families take a
filename suffix so runs cannot clobber each other. Output lands in `results/`, which is gitignored.

Code: estimators and detectors in `src/estimators/dependence/`, family mathematics in
`src/estimators/copula_family.hpp`, the generator in
`src/streamgenerator/datasource/copula.hpp`, and every constant in §4.3 in
`src/estimators/registry.hpp`.

## 5. Layer 1 — What do I assume

**Worry about your marginals, not your copula family.**

Layer 1 is two separate guesses, and they are not equally important.

| | the guess | getting it wrong costs | the fix |
| --- | --- | --- | --- |
| **the marginals** | what each series looks like on its own. Standardization assumes Gaussian. | **≈ 270%** | **rank-PIT.** Exact rather than approximate, and free (§5.1). |
| **the copula family** | how the two are linked. Gaussian, Clayton, Gumbel, $t$. | **≈ 20%** | `copula_select`, or accept the Gaussian as a forgiving default (§5.2). |

- Almost all of the copula literature argues about the second one.
- It is the cheaper one to get wrong by roughly an order of magnitude.

The analogy: the marginals are your **ruler**, the copula is your **theory**. A slightly wrong theory
fitted to good measurements still lands near the right answer. A stretchy ruler corrupts every
number before the theory ever sees it.

### 5.1 The marginals — the expensive assumption

Everything in §5.2–5.3 was estimated from **standardized** marginals (Welford mean and variance),
which is exact only if the data is Gaussian. The `_pit` variants replace that with rank-PIT (§2.3),
ranking against a bounded window of 500 so cost stays $O(\log W)$.

**The decisive test.** A strictly monotone transform of a marginal leaves the copula *exactly*
unchanged, so any movement in $\hat\rho$ is pure marginal contamination. Gaussian copula,
$\rho_S = 0.5819$, 10 seeds, $N=20{,}000$, mean $|\hat\rho - \rho_S|$:

| $x \mapsto$ | `hermite_ew` | `hermite_pit` | `gaussian_mle_param` | `gaussian_mle_pit` | `kalman_mid` |
| --- | --- | --- | --- | --- | --- |
| identity | 0.0187 | 0.0274 | 0.0273 | 0.0249 | 0.0198 |
| $e^x$ (skewed) | 0.0386 | 0.0274 | 0.0424 | 0.0249 | **0.0198** |
| $x^3$ (heavy) | 0.0695 | 0.0274 | 0.0365 | 0.0249 | **0.0198** |

- **Standardization degrades 3.7× under the cube.** Read down the `hermite_ew` column: 0.0187 →
  0.0695. That is the same estimator, same copula, same true $\rho_S$. Only the ruler changed.
- **Three estimators are exactly invariant** — `hermite_pit`, `gaussian_mle_pit`, `kalman_mid`. Not
  approximately. Ranks are literally unchanged by a monotone map, so the estimate cannot move.
- **`kalman_mid` gets it for free**, and is not a rank-PIT variant at all. It is a plain local-level
  filter. `block_spearman()` already ranks within each block, so that channel never had the problem.

**Ranks are free, and the earlier claim that they cost efficiency was confounded.**

- ⚠️ §9.2 of the source read the identity row as *the price of ranks*: `hermite_ew` 0.0187 against
  `hermite_pit` 0.0274.
- Those two differ in **two** things. `hermite_pit` is order 20 with burn-in 100; `hermite_ew` is
  order 10 with burn-in 50. The comparison never isolated the marginals.
- Against a matched control (`hermite_ew20`: order 20, burn-in 100, standardized) the ratio is
  **1.00 in all 20 grid cells**, with no cell significant at 2 SE. The gap was Hermite *order*.
- `xiao_pit` against `xiao_ew` likewise: ratio 0.98–1.01, nothing significant.
- `gaussian_mle_pit` **wins outright** on 11 of 20 cells (gumbel `jump` 0.1129 → 0.0840; $t$ `drift`
  0.1252 → 0.0914) and cuts the SE 6× on gumbel `jump`. Its `pseudo()` divides by a running Welford
  variance that is noisy in its own right; a 500-window rank is better conditioned.
- **So there is no efficiency-versus-robustness trade to agonise over here.** Ranks cost nothing on
  this grid and gain for the MLE. This is R8 in Appendix C.

⚠️ What this leaves open:

- The `_pit` variants are still **additions, not replacements**. Every number in §5.2–5.3 is on the
  old marginals.
- The robustness case now rests **entirely on the static-transform table above**, not on the grid.
  The grid cannot show it, for the reason in §5.2.
- The window (500) is a new hyper-parameter. Too short and the rank scale is noisy; too long and it
  cannot follow a genuinely changing marginal.
- $e^x$ and $x^3$ are severe but synthetic. Real returns are untested, and they are the case that
  motivates the whole section.
- **Distinction 3 goes live.** Ranks shift as data arrives, so the pseudo-observations are not
  i.i.d. — the assumption every $\mathrm{ARL}_0$ result depends on. Rank-PIT fixes the marginal
  problem and sharpens the calibration one.

### 5.2 The copula family — the cheap assumption

**Guessing Gaussian and being wrong is cheap. Guessing Clayton and being wrong is not.**

- `gaussian_mle_ew` off-family: $0.0558 \to 0.0671 / 0.0592 / 0.0680$. Worst case about **+20%**, and
  it stays mid-pack throughout. That number is the ≈20% quoted at the top of §5.
- `clayton_mle` on Gumbel `constant`: **0.1589**, against its own on-family 0.0154. Ten times worse.
- Clayton's lower-tail structure is *actively wrong* for upper-tail data, not merely uninformative.
  Gaussian's is merely uninformative, which is why it degrades gracefully.
- **Of the four families, only one is dangerous to guess.**

That is measured over 20 seeds, four generator families (Gaussian, Clayton, Gumbel, Student-$t$ with
$\nu=4$) and all 22 registered estimators.

⚠️ **Two of the five scenarios cannot be read across families.** Clayton and Gumbel are
positive-dependence-only, so the generator floors the parameter at independence and clips the
negative half of any path that crosses zero:

| scenario | Gaussian / $t$ — min, max, sd | Clayton / Gumbel — min, max, sd | comparable? |
| --- | --- | --- | --- |
| `constant` | $+0.582,\ +0.582$, sd $0.000$ | $+0.582,\ +0.582$, sd $0.000$ | ✅ |
| `jump` | $+0.191,\ +0.786$, sd $0.297$ | $+0.191,\ +0.786$, sd $0.297$ | ✅ |
| `stochastic` | $+0.352,\ +0.826$, sd $0.107$ | $+0.352,\ +0.826$, sd $0.107$ | ✅ |
| `drift` | $-0.483,\ +0.786$, sd $0.363$ | $\mathbf{+0.001},\ +0.786$, sd $0.259$ | ❌ clipped |
| `oscillate` | $-0.582,\ +0.582$, sd $0.410$ | $\mathbf{+0.001},\ +0.582$, sd $0.225$ | ❌ clipped |

- Clayton's `oscillate` column looks far better — `xiao_ew` 0.2010 against 0.3484 on Gaussian —
  **purely because the target barely moves** (sd 0.225 against 0.410).
- That is a measurement artifact, not robustness. Read those two columns *down*, never *across*.

**Can you avoid choosing a family at all?** Partly. `copula_select` scores all four by prequential
likelihood and reports the winner's $\rho_S$.

- **When cumulative it works**, matching the correct family's own MLE exactly — Clayton 0.0154,
  Gumbel 0.0168, $t$ 0.0131 on `constant`. It costs nothing when it is right.
- ⚠️ **Under forgetting it breaks.** `copula_select_ew` loses to the right family's `_ew` on Clayton
  `drift`, 0.1039 against 0.0717. It chooses on a short window and churns. *(Superseded: mostly a
  `NonparamCopula` bug, fixed 2026-10-20 — `kalman_changepoint_results.md` §8.5. The selector's
  score is also only partly prequential — `thesis_audit.md` §1.6.)*
- **Selection and forgetting interact badly.** That is an open problem, not a tuning detail, and it
  is the interesting form of the model-selection question for a streaming thesis (§9).

**Two results from other sections survive the family sweep.**

- **BOCPD's `jump` win holds on all four families** — best in column at 0.0598 / 0.0602 / 0.0589 /
  0.0633, beating the best fixed-memory nonparametric by 25–30%. §6's structural argument never used
  Gaussianity, and this is the evidence for that.
- **The column order barely moves across families**, which is §4.2's marginal invariance appearing at
  the level of the leaderboard rather than in a single estimate.

**One piece of housekeeping.**

- **`xiao_win`, run here for the first time, adds nothing** over `windowed_rank`. They agree to
  ~0.5%, as two exact sliding windows differing only in binning should. It uses less storage
  (5824 B against 8000 B), so it is marginally preferable. Recorded so nobody re-runs it.

⚠️ $\nu=4$ is fixed for both the $t$ generator and `t_mle`, so that column is a matched-$\nu$ best
case and overstates `t_mle` against unknown tail weight. Detector thresholds were calibrated on
Gaussian innovations only, so alarm counts staying sane off-family is encouraging, not calibration.

**And every generator here has Gaussian marginals.** So this entire grid ran with a correct ruler
and was blind to the assumption that matters more. Changing the copula family does not test the
marginal assumption — only changing the marginals does. That is also why B.2's invariance check,
run with an *affine* shift, could never have caught it (R5, R7).

### 5.3 The exception — when being nonparametric pays

Across Clayton, Gumbel and $t$ the Gaussian MLE wins once and ties twice, so **being nonparametric
had never paid for itself.** Two families outside the Gaussian's neighbourhood settle whether that
is a fact about the Gaussian copula or about the family set: the **Fréchet mixture** (§2.2,
Appendix A.5), whose density is a singular ridge on the diagonal, and **mixture-Gaussian**, the same
construction made smooth.

`xiao_ew` against `gaussian_mle_ew` on `jump`, 20 seeds:

| DGP | `xiao_ew` | `gaussian_mle_ew` | |
| --- | --- | --- | --- |
| Gaussian | 0.0798 | **0.0745** | MLE, 2.9σ |
| Clayton | 0.0859 | 0.0866 | tie, 0.2σ |
| Gumbel | 0.0834 | 0.0804 | tie, 1.3σ |
| $t$ | 0.0871 | 0.0869 | tie, 0.1σ |
| **Fréchet** | **0.0841** | 0.0914 | **Xiao, 2.7σ** |
| **mixgauss** | 0.0888 | 0.0932 | Xiao, 1.4σ |

- **The first significant win for a nonparametric estimator in the project.** It holds on `drift`
  too (0.0728 vs 0.0834, 4.3σ).
- ⚠️ **The margin is 8–37%, not an order of magnitude.** Even against a copula with a singular
  component the Gaussian MLE only loses moderately, because the 40% independent part of the mixture
  punishes $\rho\to1$ hard enough to keep the fit sensible.
- So the forgiveness result survives in a **stronger** form: the Gaussian copula is forgiving even
  of copulas that are not remotely Gaussian. The other parametrics are not — `t_mle` is 11× worse,
  `copula_select` 4.6× — making this a property of the Gaussian family specifically, not of
  parametric estimation.

**It also separates Xiao from Hermite,** which had tracked within 2–4% on every smooth family and
read as two routes to one answer. On `constant`, cumulative:

| | Gaussian DGP | Fréchet DGP | |
| --- | --- | --- | --- |
| `xiao` — $20\times20$ bin counts | 0.0128 | **0.0152** | indifferent |
| `hermite` — order-10 basis | 0.0121 | 0.0288 | **4.7σ worse** |
| `gaussian_mle` — wrong family | 0.0129 | 0.0241 | degrades gracefully |

- **A smooth orthogonal-polynomial expansion cannot represent a singular density.** Hermite's basis
  *is* the assumption that fails, and it fails hard enough to put Hermite behind the misspecified
  Gaussian MLE.
- A histogram does not care whether the density it counts is smooth. On `mixgauss`, where the
  singularity is gone, Hermite recovers (0.0208 vs Xiao's 0.0168, 1.7σ). So it is the
  **singularity**, not the mixture.
- **This is the first result giving Xiao a reason to exist that Hermite does not share**, and it
  costs 3824 bytes against Hermite's 1192. A real trade, now with something on both sides.

⚠️ The Fréchet mixture is deliberately adversarial. It is a real family and finance does use mixture
copulas, but nothing here says real returns look like it.

## 6. Layer 2 — How do I estimate

*Stub for the particle filter (old §11), EKF/UKF (old §12) and the Fisher-scoring ablation (old
§13). The forgetting-rate sections below are written; they hold the evidence for finding 1.*

### 6.1 Trying seven memory lengths

**The best memory is different in every scenario, and no fixed value is close to good everywhere.**

Every detector in this document wraps a filter with a memory of about 100 ticks
($\sigma_\omega^2 = 10^{-2}$). That value was never swept. It should have been, because memory moves
RMSE about four times more than the choice of detector does.

| memory (ticks) | $\sigma_\omega^2$ | `constant` | `drift` | `jump` | `oscillate` |
| --- | --- | --- | --- | --- | --- |
| ≈ 300 | $10^{-3}$ | **0.0208** | 0.1274 | 0.1288 | 0.4153 |
| ≈ 100 | $10^{-2}$ | 0.0712 | **0.0921** | **0.0880** | 0.3166 |
| ≈ 30 | $3\times10^{-1}$ | 0.1419 | 0.1757 | 0.1507 | **0.2256** |

Read down each column. The winning row is different every time, and the full sweep spans two and a
half orders of magnitude in $\sigma_\omega^2$.

- Long memory wins when nothing changes. Short memory wins when everything does.
- This is the efficiency against adaptation trade, with no detector anywhere in it.
- No single value is good everywhere. That is the shape of the problem, not a tuning failure (§2.4).

### 6.2 Three ways to set the memory

In increasing order of letting the data decide.

| | how it is set | `drift` | `jump` | `oscillate` |
| --- | --- | --- | --- | --- |
| `known_best` | one fixed value, chosen per scenario *after* seeing the answer | 0.0921 | 0.0880 | **0.2256** |
| `kalman_vff` | a heuristic nudges it each block (§2.4) | 0.0773 | 0.0993 | 0.2690 |
| **`kalman_bank`** | a posterior over 7 values | **0.0734** | **0.0675** | 0.2352 |

- `known_best` cheats. It is told the scenario and handed the winning value. The other two rows use
  one setting for all five scenarios.
- VFF beats `known_best` on `drift` without being told the scenario. Adaptivity pays even from a
  heuristic.
- The bank beats `known_best` on `drift` by 8.2σ and on `jump` by 9.3σ. It also beats VFF on all
  five.
- The bank loses narrowly on `constant` and `oscillate`, by 3.0σ and 3.2σ. In those two the truth's
  behaviour never changes, so one fixed value genuinely suffices and the bank pays a small premium
  for flexibility it does not need.
- Beating a cheating baseline is possible because the bank changes its memory mid-stream. No fixed
  value can.

### 6.3 The device

Seven filters at $q = 10^{-4}\ldots10^{0}$. Each is scored by the predictive likelihood it gave the
block before seeing it, reweighted by Bayes, then passed through the sticky step of §2.4.

- It costs **736 bytes**, against `kalman_fast`'s 512 and BOCPD's 2968. The block Spearman is
  computed once and shared.
- Freezing the weights uniform makes it worse than everything on all five scenarios. The win comes
  from learning the weights, not from carrying several filters.

### 6.4 The mechanism, watched directly

Posterior-mean $\sigma_\omega^2$ through the jump at $t=2500$.

| $t$ | posterior-mean $\sigma_\omega^2$ | |
| --- | --- | --- |
| 2400 | 0.00102 | long memory, quiet stretch |
| — | — | *jump at $t = 2500$* |
| 2560 | 0.30869 | **300× shorter memory**, caught up |
| 3000 | 0.02018 | relaxing back |
| 3800 | 0.00817 | long again |

Long memory, then short, then long again, automatically. There is no detector, no threshold, no
reset and nothing calibrated. §2.4's U-shaped MSE says no fixed λ can produce that.

⚠️ The grid $\{10^{-4}\ldots10^{0}\}$ was picked from the sweep in §6.1, so it is mildly informed by
the answer. A grid too narrow to contain the truth would fail, and that has not been tested.

## 7. Layer 3 — How do I handle change

*Stub for the detectors and their calibration (old §5, §6), the reset finding (old §14.1–14.7), the
score-CUSUM (old §14.10), the rank score-CUSUM (old §14.14) and the tail monitor. The shape-change
sections below are written; they hold the evidence for finding 2.*

### 7.1 A shape change at constant ρ

**Five detectors with five different optimality proofs fail together, and the simplest detector in
the set beats all of them by watching a different number.**

The test: at $t=2500$ the generator switches from a Gaussian copula to $t(4)$, holding the target
$\rho_S$ at 0.58192. Tail dependence rises from 0 to 0.31. The reported truth for $\rho_S$ is
identical on both sides.

⚠️ The *achieved* $\rho_S$ is not quite identical. Empirical Spearman drops from 0.5818 to 0.5734,
because `spearman_from_param` for the $t$ family is quadrature rather than exact. In Fisher-$z$ that
is 0.012 against a block innovation SD of 0.25, so 0.05σ per block. It is far below anything a
detector could use, but it is not zero, and every RMSE in this section therefore carries a small
step that belongs to the generator rather than to the estimator.

**Results, 20 seeds.** For scale, these same detectors manage 20/20 at delay 47 on a $\rho$ jump.

| detector | the number it watches | hits |
| --- | --- | --- |
| `kalman_glr`, `kalman_ewma_chart`, `kalman_sr` | block Spearman | 1/20 |
| `kalman_cusum` | block Spearman | 2/20 |
| `kalman_bocpd` | block Spearman | 5/20 |
| `hermite_cusum2` | coefficient surprise | 0/20 |
| `gaussian_mle_cusum` | the MLE's score | **17/20** |

The first three rows sit at their own false-alarm rate. The last row detects the change.

⚠️ **`kalman_bocpd`'s 5/20 is not partial detection.** On a matched stationary run it alarms 14/40
against 12/40 "hits" on the change. Its apparent edge *is* its false-alarm rate.

**Why the Spearman input cannot work.** It carries one number per block, and that number is an
average over the unit square (§2.2). The change moved mass into the corners and left the average
where it was. This is distinction 1.

⚠️ **They do not see literally nothing, and the residual is not dependence.** AUC is threshold-free
— the probability a post-change block scores above a pre-change one — so it measures the statistic
rather than the cutoff. Measured over 40 seeds against a matched stationary control:

| detector | hits | AUC raw | AUC on the null | **corrected** |
| --- | --- | --- | --- | --- |
| `kalman_cusum` | 3/40 | 0.551 | 0.500 | **0.551 ± 0.012** |
| `kalman_sr` | 1/40 | 0.542 | 0.495 | **0.548 ± 0.010** |
| `kalman_glr` | 3/40 | 0.538 | 0.504 | **0.534 ± 0.009** |
| `kalman_ewma_chart` | 1/40 | 0.522 | 0.493 | **0.529 ± 0.007** |
| `kalman_bocpd` | 12/40 | 0.593 | 0.571 | **0.522 ± 0.006** |
| `gaussian_mle_cusum` | 36/40 | 0.646 | 0.504 | **0.641 ± 0.013** |

- **0.50 is nothing. 0.64 is the detector that works. These sit at 0.53.** Three to five SEs above
  chance, so not literally blind.
- **But the reaction is not the dependence.** Across the switch the block reading's mean barely
  moves ($\rho_S$ 0.5801 → 0.5857) while its **SD moves 0.2521 → 0.2771**, about 10%. They are
  twitching at readings that got noisier, not at dependence that changed.
- That is the same misfit `gaussian_mle_cusum` reads, an order of magnitude weaker. So the two
  halves of this section are one mechanism at two strengths, not a contrast.
- ⚠️ The correction matters: `kalman_bocpd` reads 0.593 raw and `tail_cusum` 0.810 raw, almost
  entirely **accumulator drift**. CUSUM, GLR and SR are sums, so their statistics climb under the
  null. An uncorrected before-versus-after AUC measures that climb.
- ⚠️ Part of the residual is the generator's own imperfection, not shape. The Gaussian and $t$ legs
  do not match $\rho_S$ exactly, which accounts for roughly AUC 0.509 on its own.

**Why the score input does work.** The score is the gradient of the log-likelihood, so it responds
to the whole fit rather than to one summary of it. A $t$ copula fits a Gaussian model badly, which
inflates $\operatorname{Var}[s]/I$ from 0.93 to 1.87. The CUSUM assumes that variance is 1, so an
inflated one trips it. Nobody designed it to do this.

**The coefficient input is the control.** `hermite_cusum2` watches a third number, the surprise
across 441 basis coefficients. It scores 0/20, on the change it was built for. A richer input is
therefore not automatically a better one. A norm over 441 numbers is too diffuse to isolate a change
confined to the tails.

§2.1 claims that which functional a detector watches is the design choice that matters most. This
is the measurement behind that claim.

⚠️ 17/20 at delay 702 ticks is not a working detector. It is roughly 15× slower than the same
machinery on a $\rho$ jump, and no threshold here was calibrated for a shape change. Only one family
pair was tested, at one $\rho$, at one switch time.

### 7.2 Robustness and detection are the same statistic

The score's sensitivity to fit is also a liability. §5.1 shows that a threshold calibrated on
Gaussian data does not hold off-family, because the same inflation appears whenever the family is
merely *wrong* rather than *changing*.

The detector only ever sees one number, and these two situations produce the identical value:

| | ask the model what is normal | check recent history for what is normal |
| --- | --- | --- |
| data was **always** $t$ | alarms forever. Wrong. | quiet. Right. |
| data **became** $t$ | alarms. Right. | quiet. Missed it. Wrong. |

Each option is correct in one row and wrong in the other. Measured, at matched $\mathrm{ARL}_0$
($h=4.0$, both at 0.30 false alarms before the switch, so the whole difference is post-switch):

| | what it divides by | hits |
| --- | --- | --- |
| `gaussian_mle_cusum` | the model's Fisher information | **17/20** |
| `gaussian_mle_cusum_rob` | the empirical score variance (§2.7) | 10/20 |

So a detector must be built either to survive misspecification or to report it. It cannot do both
from one accumulator, and nothing in this project had forced that choice before.

⚠️ It is a dial rather than a switch. The rate $c$ controls how fast the empirical variance adapts,
and $c=0.001$ is already 1000× slower than $\lambda$. The 17/20 → 10/20 loss is what that costs even
at the slow setting.

## 8. Layer 4 — Can I afford it

*stub — two subsections measured (`store_bytes` and ticks/sec for the whole set; where the cost
actually goes), two still open (blocks counted in ticks rather than time; does the ranking
survive a rate budget).*

## 9. Limitations

*stub*

⬜ **RMSE is the only accuracy metric, and it has two known blind spots.**

- It lives in $\rho_S$, which §2.2 shows is a lossy average. A copula with completely wrong tails
  but the correct $\rho_S$ scores **perfectly**. That is exactly the territory of §7.1.
- It sums lag bias and variance into one number, and those two are the entire subject of the thesis.
  §2.4 theorises the trade as a U-curve, then every table reports their sum. **Decomposing RMSE into
  bias and variance would measure the trade directly** rather than its total.
- It says nothing about whether the filter's $P_t$ is honest. Accurate-but-overconfident scores the
  same as accurate-and-calibrated.

⬜ **AUC would test a claim §7.1 currently asserts.**

- Detection is reported as hit rate at a *calibrated threshold*, so "1/20 means blind" is an
  inference from the rate matching the null rate, not a measurement.
- AUC is **threshold-free**: the probability that a post-change block scores above a pre-change one.
  If those five detectors are genuinely blind their statistics give AUC ≈ 0.5; if the signal is
  there and the threshold is discarding it, AUC > 0.5 and "blind" is too strong.
- It would also separate the two effects in §5.2's off-family result. If *"the detection survives,
  the calibration does not"* is right, AUC should hold up off-family while the fixed-threshold hit
  rate collapses.
- ⚠️ AUC ignores timing entirely, so it complements delay-at-matched-$\mathrm{ARL}_0$ rather than
  replacing it. A detector can have excellent AUC and useless delay.

➖ **AIC and BIC are deliberately not used.** They need a fixed $n$, a comparable likelihood across
parametric and nonparametric estimators, and a well-defined parameter count — none of which survive
a stream with exponential forgetting. The streaming replacement is prequential log-likelihood, which
`copula_selector.hpp` and the bank already use. Using it as an *evaluation* metric rather than only
a selection one is open, and would score the whole density instead of one functional of it.

⬜ **Anytime-valid thresholds — parked, not developed.** §14.12's off-family result is a
*calibration* failure: a bootstrapped ARL₀ does not transfer across DGP families.

An **e-value** is a bet against "nothing changed" — a statistic $E_t$ with $\mathbb E_0[E_t]\le1$,
read as the wealth of someone who staked \$1 on a change. Ville's inequality then gives
$P(E_t \text{ ever} \ge 1/\alpha) \le \alpha$: a threshold of $1/\alpha$ stays valid while you look
**continuously**, because you cannot get rich betting against something true. Crucially it needs a
fair *bet*, not i.i.d. *data* — the assumption distinction 3 destroys and every §2.7 optimality
result relies on.

Shiryaev–Roberts is already a running sum of likelihood-ratio products, and $\mathbb E_0[\Lambda_t]=1$
makes each such product an e-value — so this is a reframing, not a build. **One experiment, not a
chapter:** run SR off-family against its bootstrapped threshold and see which holds ARL₀.
⚠️ Nothing in Zotero — Ville, Wald's SPRT, the modern safe-testing line.

## A. Appendix — the DGP families, constructed

*stub — Gaussian, Clayton, Gumbel, Student t, Fréchet mixture, mixture Gaussian, the switching
family, and the two identities the detectors use. Already written; moves across unchanged.*

## B. Appendix — constants, verification, and harness bugs

### B.1 Every tuned constant

**"Inherited" marks a value that was never swept before the result that swept it.** Two of the
seven retractions in Appendix C trace directly to a row of that kind.

| constant | value | memory (ticks) | provenance |
| --- | --- | --- | --- |
| block $m$ | 20 | — | swept; $m=50$ broke the within-block-constant assumption on `oscillate` (B.3, bug 8) |
| $\sigma_\omega^2$ — `kalman` | $10^{-4}$ | **999** | the long-memory setting |
| $\sigma_\omega^2$ — `kalman_fast`, **and every detector** | $10^{-2}$ | **102** | **inherited**, unswept until §6. This is the finding. |
| $\lambda$ — `xiao_ew`, `gaussian_mle_ew` | $0.99$ | **199** | retention |
| $\lambda$ — `hermite_ew` | $0.01$ | **199** | update weight. The *same* memory, mirror convention (§1.1, §2.8). |
| order — `hermite_ew` | 10 | — | burn-in 50 |
| order — `hermite_pit` | 20 | — | burn-in 100. **Differs from `hermite_ew` in order as well as marginals**, which is R8. |
| order — `hermite_ew20` | 20 | — | burn-in 100, standardized. Added as the matched control `hermite_pit` never had. |
| $\lambda$ — `xiao`, `hermite`, `gaussian_mle` | $0$ | ∞ | cumulative, never forgets |
| window — `xiao_win` | 500 | 500 | a hard window rather than a decay |
| bins — Xiao | 20 | 8 / 20 / 40 measured (§6); 40 retracted as R6 |
| order — Hermite | 10 | 10 / 20 / 40 measured (§6) |
| $\delta$, $h$ — `kalman_cusum` | $1.0$, $3.5$ | $h$ calibrated, §4.3 |
| $h$ — `kalman_glr` | $10$ | calibrated, §4.3 |
| hazard $H$ — `kalman_bocpd` | $1/1000$ | set to match the $\mathrm{ARL}_0$ target directly |
| hazard $H$ — `kalman_bocpd_fast` | $1/10$ | `oscillate`'s true half-period — a DGP timescale, deliberately *not* the RMSE-minimising value |
| alarm threshold — bocpd / bocpd_fast | $0.15$ / $0.95$ | calibrated, §4.3 |
| reset inflation | $0,2,5,10,20,50$ | swept (§7); $0$ is the no-op control behind R1 |
| $p$ — `_sticky` (§2.4) | $0.002$ | measured ordering $0.002 > 0.02 > 0.15 > \text{frozen}$ |
| $c$ — `_rob` (§2.7) | $0.001$ | deliberately $\sim\!1000\times$ slower than $\lambda=0.99$, so the scale estimate cannot chase the signal |

⚠️ Transcribed from `registry.hpp` and the sections that set each value. **Regenerate it from the
header before submission** — a constants table that has drifted from the code is worse than none.

### B.2 Verification

Each component was checked against an independent reference before anything was built on it.

**Bridge 1 is exact.** Steady-state gain against the closed form $K=(\sqrt{q^2+4q}-q)/2$:

| $\sigma_\omega^2$ | $q$ | $K$ (filter) | $K$ (closed form) | difference |
| --- | --- | --- | --- | --- |
| $10^{-5}$ | 0.00044 | 0.02084 | 0.02084 | 0.00000 |
| $10^{-4}$ | 0.00443 | 0.06441 | 0.06441 | 0.00000 |
| $10^{-3}$ | 0.04434 | 0.18956 | 0.18956 | 0.00000 |
| $10^{-2}$ | 0.44340 | 0.48012 | 0.48012 | 0.00000 |
| $10^{-1}$ | 4.43396 | 0.84063 | 0.84063 | 0.00000 |

⚠️ **The source labels this $m=20$, but the $q$ column is computed at $m=50$**: $10^{-4}/0.02255 =
0.00443$, and $0.02255 = 1.06/47$. At $m=20$ that row would read $q=0.00160$. The verification
stands either way — Bridge 1 is an identity in $q$ alone, and $q$ spans four orders of magnitude
here regardless — but the label is wrong and should say $m=50$, the then-default block size.

**The GLR step response matches theory.** 2000 seeds, real block-Spearman noise, jump
$0.2\to0.85$. Theory predicts consecutive post-jump blocks decay by $(1-K)=0.672$; measured
$0.678,\,0.652,\,0.690,\,0.690,\,0.657,\,0.735,\,0.626,\,0.683$.

**BOCPD recovers a jump.** Tracks $0.19\to0.85$ within two blocks with no reset. $P(r_t\le2)$ spikes
$0.0015\to0.53\to0.9999$ then returns to baseline; the MAP run length collapses to 1 exactly at the
changepoint.

**Marginal invariance holds.** A large purely-marginal shift ($x\mapsto5x+100$, $y\mapsto0.1y-50$,
both monotone) with the copula held exactly constant produced **zero alarms**; the CUSUM statistic
peaked at 3.19 against a threshold of 3.5, and the estimate held 0.468 against a true $\rho_S$ of
0.483. ⚠️ Run only on the rank channel, which passes by construction — R7.

### B.3 What the harness got wrong

Eight bugs, each a statement about what this setup can and cannot catch on its own.

| | bug | how it was caught |
| --- | --- | --- |
| 1 | BOCPD's alarm could never fire — $P(r_t{=}0)\equiv H$ under constant hazard, carrying no information | derived from the recursion, then confirmed empirically |
| 2 | BOCPD burn-in false alarms: with fewer than $c+2$ blocks of history *every* hypothesis satisfies $r\le c$ | **multi-seed** — the same $\approx2.1$ on nearly every seed, so systematic, not variance |
| 3 | `gaussian_mle` latched to NaN on heavy tails — the score carries $(1-\rho^2)^{-1}$, and a joint extreme pushes $\theta$ past $\tanh$ saturation | the cross-family grid; one seed in 20 poisoned the mean |
| 4 | `xiao_win` had never run — registered in `registry.hpp`, absent from both `keys` lists. No error, just no row | audit; now an invariant in `CLAUDE.md` |
| 5 | An unknown `COPULA` silently fell back to Gaussian — in a scripted grid a typo would mislabel a whole family | audit; `multiseed.cpp` now rejects unknown names up front |
| 6 | The `oscillate` overclaim — a comparison against untuned defaults | fair per-family tuning — R4 |
| 7 | Two lists that must agree, wrong in **both** directions: `xiao_win` had a branch with no entry; the three `_vff` keys had entries with no branch, so `make_estimator` returned `nullptr` and the harness segfaulted with no diagnostic | now checked both ways by `check_registry()` and `make check-registry` |
| 8 | $m=50$ broke the within-block-constant assumption on `oscillate` (period 400 — one block spans an eighth of a cycle) | swept; corrected to $m=20$ |

Bugs 2 and 3 are the argument for multi-seed as a **correctness tool** rather than an error-bar
formality: both look like plausible single-seed variance and are neither.

## C. Appendix — what did not survive its control

Claims this project made and then disproved itself. The **how** column carries as much weight as
the what: only six failure modes recur, and knowing which one a claim is exposed to is how the
next control gets designed.

| | claim | how it died | what killed it |
| --- | --- | --- | --- |
| R1 | Detect-and-restart costs accuracy | missing control | a no-op `reset()` — it was the *reset*, not the detection (→ §7) |
| R2 | Covariance inflation earns something on `oscillate` | missing control | a blind clock and a larger $\sigma_\omega^2$ both beat it |
| R7 | B.2 verified marginal invariance | missing control | run only on the channel that passes by construction; Xiao fails it catastrophically |
| R3 | Expected information wins all four scenarios | effect too small | 3 of the 4 gains were 0.9–1.3σ |
| R6 | `xiao_b40` is a free accuracy win | effect too small | 0.37σ and 0.06σ, for 3.7× the store |
| R4 | The Kalman filter wins `oscillate` | unfair comparison | tuned Xiao and Hermite beat its *best achievable* by 19% |
| R5 | §9.2 shows rank-PIT fixes the marginals | test had no power | Xiao cuts its bins at quantiles, so a monotone transform moves the cutpoints with the data and every point lands in the same bin — it passes automatically, whatever the marginals |
| R8 | Rank-PIT costs efficiency for Hermite, 0.0187 → 0.0274 | confounded comparison | `hermite_pit` is order 20 / burn-in 100, `hermite_ew` is order 10 / burn-in 50 — two differences, not one. Against a matched control (`hermite_ew20`) the ratio is 1.00 in all 20 cells. The gap was Hermite *order*, not the marginals. |
| R9 | The ρ-monitoring detectors are structurally blind to a shape change; no threshold can recover it | inferred, not measured | a hit rate at a calibrated threshold cannot distinguish "no signal" from "signal the threshold discards". Threshold-free AUC gives 0.52–0.55 against 0.641 for the score channel — not blind, but what they react to is a 10% rise in the reading's SD, not dependence (§7.1) |
| R10 | `kalman_bocpd` partially detects the shape change, 5/20 | missing control | on a matched stationary run it alarms 14/40 against 12/40 on the change. The apparent edge *is* its false-alarm rate. |

Read it as the credibility section: every headline number elsewhere is more believable *because*
this table exists.

## References

*stub*