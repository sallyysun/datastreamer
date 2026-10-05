# Online tail dependence under copula change — step 1: stationary validation

**Direction.** How should tail dependence be estimated online when the copula is subject to
abrupt structural change? Central design: switch copula family at **matched global dependence**,
so the change is in the *shape* (tails), not in rank association.

**Prior work this builds on** (branch `adaptive-memory`): `kalman_changepoint_results.md` §14.13
/ `copula_v1.md` §7.1 ran Gaussian → t(4) at fixed $\rho_S=0.58$. Every $\rho$-monitoring detector
was blind (1–2/20, AUC ≈ 0.53), coefficient monitoring scored 0/20, the `tail_cusum` monitor
($q=0.15$) scored 0/20, and only `gaussian_mle_cusum` had power (17/20, delay ≈ 700) — by
accident, via score-variance inflation. So H1 (global-dependence detectors miss shape change) is
already supported; what is new here is *estimating* tail dependence online, the memory × threshold
trade-off, and more transitions.

Harness: `experiments/tail_validate.cpp` (`make tail_validate && ./tail_validate 0.6 20`).

## Setup

- **Copula space, known uniform margins.** Rank-based margins are step 2.
- **Matched on $\rho_S$, not Kendall's $\tau$.** The DGP (`copula.hpp`) and every estimator in the
  repo target Spearman $\rho_S$; matching on $\tau$ would leave a $\rho_S$ step at the switch.
  Knob $d=0.6$ → target $\rho_S=0.58192$ (the §14.13 level). Families: Gaussian ($\rho=0.6$),
  Clayton ($\theta=1.419$), Gumbel ($\theta=1.712$), $t_4$ ($\rho=0.614$).
- **Samplers** reproduce `copula.hpp`'s formulas with the parameter solved once (the DGP class
  re-solves per draw: 225 µs/draw for $t$). Cross-checked against the DGP classes (1M draws,
  160k for $t$): $\chi_L(.05)$, $\chi_U(.05)$ and $\rho_S$ agree, 12 z-scores all $|z|\le2.1$.
  Achieved $\rho_S$ over 20M draws: 0.5814 / 0.5820 / 0.5822 / 0.5815 — matched to 0.001.
- **Targets.** $\lambda_L,\lambda_U$ in closed form; finite-threshold
  $\chi_L(q)=C(q,q)/q$, $\chi_U(q)=P(U>1-q,V>1-q)/q$ by 20M-draw Monte Carlo, checked against
  closed-form $C(q,q)$ for Clayton/Gumbel (agree to ≤ 0.0012).
- **Estimators** (standard definitions; the Frahm–Junker–Schmidt paper was not available in-session,
  so exact forms should be re-checked against it), exponentially forgotten with ESS $m$, normalised
  by accumulated weight (no startup bias):
  - **SEC** $\hat\lambda_L=\hat C(q,q)/q$;
  - **ratio** $\hat P(V\le q\mid U\le q)$ = joint / marginal corner count — what the rank-based
    empirical copula reduces to, since ranks make the marginal count exactly $qn$;
  - **LOG** $\hat\lambda_L=2-\log(1-2q+\hat C(q,q))/\log(1-q)$.
  Upper tail by $u\mapsto1-u$. $q\in\{.01,.02,.05,.10\}$, $m\in\{250,\dots,50000\}$, 20 seeds,
  300k ticks, scored after 150k.

## Results

### 1. The finite threshold is the dominant error, and memory cannot remove it

$\chi(q)-\lambda$ (the bias of a *perfect* estimator at threshold $q$):

| family, tail | $\lambda$ | $q=.01$ | $q=.02$ | $q=.05$ | $q=.10$ |
| --- | --- | --- | --- | --- | --- |
| Gaussian, both | 0 | **+0.19** | +0.23 | +0.31 | +0.39 |
| Clayton, lower | 0.613 | +0.000 | +0.002 | +0.004 | +0.009 |
| Clayton, upper | 0 | +0.02 | +0.05 | +0.11 | +0.21 |
| Gumbel, upper | 0.501 | +0.005 | +0.009 | +0.018 | +0.038 |
| Gumbel, lower | 0 | +0.10 | +0.14 | +0.23 | +0.32 |
| $t_4$, both | 0.324 | +0.04 | +0.06 | +0.09 | +0.13 |

- **A Gaussian copula looks tail-dependent at every usable threshold** — 0.19 even at $q=0.01$ —
  while its $\lambda$ is 0. Tail-*independent* sides are where the threshold bias is large;
  genuinely tail-dependent sides converge fast.
- So the advertised "$\lambda_L$: 0 → 0.32" of Gaussian → $t$ is, at $q=0.05$, an observable change
  of **0.31 → 0.41**: about a third of the limiting change.
- **Scoring rule adopted:** score each estimator against its own finite-$q$ target and report the
  gap to $\lambda$ separately. Scoring against $\lambda$ would mostly measure the table above.

### 2. Against their own target, all three estimators are unbiased; the ratio form is least noisy

Max $|\text{bias}|$ vs own target at $m\ge1000$: SEC 0.006, ratio 0.006, LOG 0.006. SD relative to
ratio: SEC 0.97–1.66 (median 1.26), LOG 0.99–1.77. The ratio form cancels the marginal-count
fluctuation that SEC and LOG carry with known margins. **Use the ratio (rank) form.**

### 3. Tail noise is exactly binomial thinning at rate $q$

$\mathrm{SD}^2\cdot m\cdot q\,/\,\chi(1-\chi)$ at $m=5000$: **0.89–1.08** across all families and
thresholds. So

$$
\operatorname{Var}\hat\chi(q)\approx\frac{\chi(1-\chi)}{q\,m}:
$$

a tail estimator is a dependence estimator on a stream thinned to rate $q$ — the same structure as
the async study's information-rate law. Prediction for the drift experiments: variance term
$\propto 1/(qm)$, so the optimal memory scales like $(q)^{-1/3}$ in the variance-dominated regime,
while the threshold bias (table 1) is, like async bias, untouched by memory.

### 4. Detectability: SNR of the matched-$\rho_S$ transitions

$|\Delta\chi(q)|/\max(\mathrm{SD}_\text{before},\mathrm{SD}_\text{after})$, ratio estimator:

| transition (tail) | $\Delta\lambda$ | best $q$ | SNR at $m=1000$ | $m$ for SNR ≈ 3 |
| --- | --- | --- | --- | --- |
| Gauss → Clayton (lower) | 0 → 0.61 | 0.10 | 4.7 | ~500 |
| Gauss → Gumbel (upper) | 0 → 0.50 | 0.05–0.10 | 3.0 | ~1000 |
| **Gauss → $t_4$ (lower)** | 0 → 0.32 | 0.05 | **1.5** | **~5000** |
| Clayton → Gumbel (lower) | 0.61 → 0 | 0.10 | 6.3 | ~250 |
| Clayton → Gumbel (upper) | 0 → 0.50 | 0.10 | 6.5 | ~250 |

- **For detection, a moderate threshold beats a small one.** SNR rises from $q=0.01$ to
  $q=0.05$–$0.10$ in every transition: $\Delta\chi$ shrinks slowly with $q$ while variance falls
  like $1/q$. Estimating $\lambda$ wants small $q$; detecting a change wants larger $q$. That is H3,
  measured.
- **Gaussian → $t$ is the hardest transition by a wide margin**, and it is the one §14.13 tested:
  SNR 3 needs memory ≈ 5000 ticks, but that experiment had only 2500 post-switch ticks. This
  explains `tail_cusum`'s 0/20 without any defect in the detector.
- One-shot SNR between two stationary estimates is an upper bound on what a sequential,
  ARL₀-calibrated detector achieves, not a detection rate.

## Step 2 — known changepoint, oracle restart vs best fixed memory

Harness: `experiments/tail_change.cpp` (`make tail_change && ./tail_change 40`, ~30 s). Shared
samplers/targets: `experiments/tail_common.hpp` (step 1 reproduces byte-identically through it).

**Design.** Matched-$\rho_S$ switch at $\tau=20{,}000$, run to $T=40{,}000$; MSE scored over
$[10{,}000, 40{,}000)$ (10k pre, 20k post) against $\chi_t(q)$. Ratio estimator only.
$q\in\{.01,.025,.05,.10\}$; memory grid $100\cdot2^{k/2}$, $k=0..18$ (100 → 51,200) plus cumulative;
40 seeds. Arms:
- **fixed** — EW at each memory, never reset;
- **oracle** — same estimator, *all* counts discarded at the true $\tau$ (holds its last value
  until the first post-change marginal-corner event); its best memory is cumulative in every cell;
- **bank** — `estimator_bank`'s device over 8 fixed arms (200 … 25,600), the only change being the
  score: a Bernoulli log-likelihood of the joint event given the marginal corner, scored before
  absorbing. Not tuned for tails.

Achieved $\rho_S$ across every switch: 0.5814–0.5822 on both sides.

### Result 1 — the oracle gate splits the transitions cleanly

Total MSE of the best fixed memory ÷ oracle restart (> 1 means restart helps), range over $q$:

| transition | tail | $\Delta\chi(q)$ | best fixed / oracle | bank / best fixed |
| --- | --- | --- | --- | --- |
| **Gauss → $t_4$** | both | 0.07–0.17 | **0.77–0.99** | 1.36–1.52 |
| Gauss → Gumbel | lower (0 → 0) | 0.07–0.09 | 0.95–1.10 | 1.28–1.49 |
| Gauss → Gumbel | upper | 0.15–0.32 | 1.61–1.93 | 1.15–1.27 |
| Gauss → Clayton | lower | 0.23–0.42 | 1.96–2.79 | 0.98–1.17 |
| Gauss → Clayton | upper (0 → 0) | 0.17–0.20 | 2.49–3.65 | 0.96–1.13 |
| Clayton → Gumbel | lower / upper | 0.31–0.51 | 2.08–4.39 | 0.87–1.14 |
| Gumbel → Clayton | lower / upper | 0.31–0.51 | 2.21–6.00 | 0.77–1.12 |

- **Gauss → $t_4$ — the transition with the dramatic asymptotic change (λ 0 → 0.32) — gains
  nothing from an oracle restart.** The best fixed memory *beats* it by 1–23 %, in both tails and at
  every threshold, and post-change-only MSE says the same (e.g. $q=.05$ lower: 0.00163 vs 0.00217).
  At finite $q$ the observable jump is 0.07–0.17, small enough that carrying the slightly-wrong old
  estimate is cheaper than re-learning from a handful of tail events. A detector — which can only
  approximate the oracle — has nothing to recover here *for estimation*.
- **Where the finite-$q$ jump is large (Clayton/Gumbel switches), restart is worth 1.6–6×.** That
  gap is the detector's maximum opportunity. It is not set by $\Delta\lambda$ alone: Gauss → Clayton
  *upper* ($\lambda$ 0 → 0, $\Delta\chi$ 0.17–0.20) gains 2.5–3.7× because the post-change $\chi$ is
  small and cheap to re-learn ($v=\chi(1-\chi)$ low).
- **The bank does not close the oracle gap.** It sits within 0.77–1.5× of the best fixed memory:
  slightly better on the largest jumps, 36–52 % worse on Gauss → $t$ and the small jumps, where its
  adaptivity is pure added noise. Untuned, but no sign it is a substitute for detection.
- **Recovery** (seed-mean estimate within 10 % of the jump): oracle 20–960 ticks; best fixed
  1,560–16,160; bank 620–8,660.

This is the third outcome the gate allowed for: **the oracle advantage exists for the easy
transitions and disappears for Gaussian → $t$** — the asymptotically dramatic change is not
observable quickly enough at finite thresholds to be worth restarting for.

### Result 2 — the memory law holds in absolute terms; the pure $q^{-1/2}$ scaling does not

For a jump of size $\Delta=\chi_b(q)-\chi_a(q)$, EW lag costs $\approx\Delta^2(m+1)/4$ integrated, and
noise $\approx V/(qm)$ with $V=W_\text{pre}\chi_a(1-\chi_a)+W_\text{post}\chi_b(1-\chi_b)$. So

$$
M^*=\sqrt{\frac{4V}{q\,\Delta(q)^2}} .
$$

- **Observed / predicted $M^*$: 0.92–1.09 in all 40 cells** — absolute, no anchoring, no fitted
  constant.
- **But the stated prediction $M^*\propto q^{-1/2}$ is falsified**: fitted exponents are −0.04 to
  −0.32. The $1/q$ (fewer tail events) is offset by $\Delta(q)$, which for tail-independent sides
  grows as $q$ shrinks; for Gauss → $t$ the two cancel almost exactly (−0.04 / −0.05). The
  correct statement is that memory is set by the effective information rate
  $q\,\Delta(q)^2/v$, not by $q$.
- (The cube-root form needs a *drift* in tail structure; it was not tested here.)

### Caveats

- Copula space, known margins; one $\rho_S$ level; $t$ with $\nu=4$.
- The oracle is a *hard* restart. A soft restart (keep old counts down-weighted) could beat best
  fixed even on Gauss → $t$; that is a better oracle to test before ruling detection out there.
- The Gauss → $t$ conclusion is about *estimating* $\chi(q)$. Whether *knowing* the copula changed
  is valuable in its own right (risk reporting) is a separate question the gate does not answer.

## Step 3a — Clayton–Gumbel mixture: feasibility at matched $\rho_S$

Model (Shamiri, Hamzah & Pirmoradian 2011): $C=\pi C_\text{Clayton}(\theta)+(1-\pi)C_\text{Gumbel}(\zeta)$,
sampled generatively (Clayton w.p. $\pi$, else Gumbel) with the step-1 samplers. Harness:
`experiments/tail_mixture.cpp` (`make tail_mixture && ./tail_mixture [d] [seed_base]`, ~8 s).
⚠️ The paper itself could not be read in-session (host blocked by the environment's network
policy); everything below is checked against *our* implementations, not the paper's text.

**Parameterisations.** `copula_family.hpp`: Clayton $C=(u^{-\theta}+v^{-\theta}-1)^{-1/\theta}$, $\theta>0$;
Gumbel $C=\exp\{-[(-\ln u)^\zeta+(-\ln v)^\zeta]^{1/\zeta}\}$, $\zeta\ge1$. These give
$\lambda_L=\pi\,2^{-1/\theta}$, $\lambda_U=(1-\pi)(2-2^{1/\zeta})$, as the handover states. Repo bounds
$\theta\le50$, $\zeta\le30$ are respected.

### What is linear in the mixture, and what is not

- **Linear (exact):** anything linear in $C$ — $C(q,q)$, hence $\chi_L(q),\chi_U(q)$; $\lambda_L,\lambda_U$;
  and **$\rho_S=12\iint C-3$**. So $\rho_S$ of the mixture *is* the weighted average of the
  components' $\rho_S$ — not an assumption, a property of Spearman's $\rho$. Checked: closed-form
  mixture $\chi$ vs 4M-draw MC agree to ≤ 0.008 (q = .01 upper) and typically ≤ 0.003; component
  $\rho_S$ by 80M draws: Clayton 0.58202 ± 0.00034, Gumbel 0.58231 ± 0.00037 (target 0.58192).
- **Not linear:** Kendall's $\tau=4E[C(U,V)]-1$ is quadratic in $C$. Off the equal-$\rho$ path it
  differs from the weighted average by ≈ 0.011–0.013 (e.g. 0.4191 vs 0.4297). This is the concrete
  reason to calibrate on $\rho_S$.

### The constraint surface

16,912 feasible $(\pi,\rho_C)$ grid points at $\rho_S=0.58192$ (all in `results/tail_mixture_feasible.csv`):

| $\lambda_L$ | $\lambda_U$ range | | $\chi_L(.05)$ | $\chi_U(.05)$ range |
| --- | --- | --- | --- | --- |
| 0.00–0.05 | 0.43–0.56 | | 0.20–0.25 | 0.49–0.52 |
| 0.15–0.20 | 0.33–0.38 | | 0.30–0.35 | 0.38–0.53 |
| 0.30–0.35 | 0.20–0.27 | | 0.40–0.45 | 0.28–0.55 |
| 0.45–0.50 | 0.06–0.14 | | 0.50–0.55 | 0.18–0.58 |
| 0.60–0.65 | 0.00–0.01 | | 0.60–0.65 | 0.11–0.17 |

- **The tails trade off almost one-for-one.** $\lambda_L+\lambda_U$ stays in a narrow band
  (≈ 0.50–0.61): fixing $\rho_S$ fixes roughly how much total tail dependence there is, and the
  mixture only chooses *which side* it sits on. Lower-heavy → upper-heavy is feasible; "both heavy"
  and "both light" are not.
- **Neither tail can be near zero unless the other is large.** At $\lambda_L<0.05$, $\lambda_U\ge0.43$.
  No Gaussian-like (both-light) regime exists in this family at this $\rho_S$.
- **The extreme corners are degenerate.** Max $\lambda_U=0.561$ comes from $\pi=0.42$ of a
  near-independent Clayton ($\theta=0.012$) plus a near-comonotone Gumbel ($\zeta=22$) — a mixture of
  noise and an almost-diagonal line. Feasible, but not a credible regime; avoid the corners.
- In $\chi(.05)$ space the band is wider (other parameter choices move the finite-$q$ value more
  than the limit), but the same lower-vs-upper trade-off holds.

### Recommended path: hold both components at $\rho_S=0.58192$, move only $\pi$

$\theta_0=1.4186$, $\zeta_0=1.7118$ (the step-1/2 Clayton and Gumbel). Then $\rho_S$ is matched
*exactly for every* $\pi$, $\lambda_L=0.613\pi$, $\lambda_U=0.501(1-\pi)$, and
$\Delta\chi(q)=\Delta\pi\,[\chi_C(q)-\chi_G(q)]$ — **the change magnitude is a single linear knob**.
It also runs close to the upper edge of the $\lambda$ envelope, and as a bonus Kendall's $\tau$
stays at 0.415 ± 0.001 along it (the two components have nearly equal $\tau$). The endpoints are
the step-2 Clayton ↔ Gumbel switch, which anchors the new results to the old.

MC-verified targets on the path ($q=.05$; all $q$ in `results/tail_mixture.txt`):

| $\pi$ | 1.00 | 0.90 | 0.75 | 0.60 | 0.50 | 0.40 | 0.25 | 0.10 | 0.00 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $\lambda_L$ / $\lambda_U$ | .61/.00 | .55/.05 | .46/.13 | .37/.20 | .31/.25 | .25/.30 | .15/.38 | .06/.45 | .00/.50 |
| $\chi_L(.05)$ | .617 | .577 | .518 | .460 | .420 | .381 | .322 | .263 | .224 |
| $\chi_U(.05)$ | .113 | .154 | .215 | .276 | .316 | .357 | .418 | .479 | .520 |

Symmetric switches $\pi_0\to1-\pi_0$ give $\Delta\chi(.05)$ ≈ 0.08 (0.6 → 0.4), 0.20 (0.75 → 0.25),
0.31–0.33 (0.9 → 0.1), 0.39–0.41 (1 → 0) — spanning both sides of the step-2 boundary where hard
restart went from losing (Gauss → $t$, $\Delta\chi$ 0.07–0.17) to winning.

## Step 3b — soft oracle restart: how much of the old regime to keep

Harness: `experiments/tail_soft.cpp` (`make tail_soft && ./tail_soft 40 8080`, ~10 s). DGP: the 3a
path, $\pi_0\to1-\pi_0$ for $\pi_0\in\{.55,.60,.70,.75,.90,1.0\}$ (six change sizes,
$\Delta\chi(q)$ 0.03–0.51), same timeline as step 2, $q\in\{.01,.02,.05,.10\}$, 40 seeds.

**What is reset.** The ratio estimator's entire state is two EW sums: $J$ (joint-corner events) and
$K$ (marginal-corner events), $\hat\chi=J/K$. At the true $\tau$: $J\leftarrow\gamma J$,
$K\leftarrow\gamma K$. $\gamma=1$ is no restart (= fixed memory), $\gamma=0$ the hard restart; in
between the old evidence is kept but aged by $\ln(1/\gamma)/\alpha$ ticks. Every $\gamma$ is swept
over the full memory grid; $\gamma\in\{0,.001,.0025,.005,.01,.025,.05,.1,.25,.5,.75,1\}$. A first
run on the coarse grid $\{0,.025,.05,.1,.25,.5,.75,1\}$ (seed base 6060) put most $\gamma^*$ on its
floor, so the grid was refined and the experiment **re-run on fresh seeds** (8080); the numbers below
are from the re-run. Scored against $\chi_t(q)$.

### Result 1 — partial forgetting beats both extremes, at every change size

Ranges over tail × $q$; $t$ is the paired-seed $t$-statistic of best-soft vs best-fixed:

| $\pi_0\to\pi_1$ | $\Delta\chi$ | $\gamma^*$ | soft / fixed | $t$ | hard / fixed | soft / hard | bank / fixed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| .55 → .45 | .03–.05 | .10–.25 | 0.80–0.95 | 1.0–5.0 | 2.2–3.4 | 0.26–0.37 | 1.9–2.8 |
| .60 → .40 | .06–.10 | .025–.10 | 0.57–0.78 | 3.2–8.7 | 1.1–1.9 | 0.41–0.53 | 1.4–1.8 |
| .70 → .30 | .12–.21 | .005–.025 | 0.39–0.58 | 5.8–13 | 0.56–0.92 | 0.63–0.69 | 1.2–1.4 |
| .75 → .25 | .15–.26 | .005–.025 | 0.33–0.55 | 7.7–16 | 0.44–0.79 | 0.69–0.76 | 1.1–1.4 |
| .90 → .10 | .24–.41 | .001–.01 | 0.24–0.44 | 12–28 | 0.27–0.52 | 0.82–0.91 | 0.94–1.2 |
| 1.0 → 0.0 | .31–.51 | .001–.005 | 0.20–0.41 | 15–27 | 0.21–0.45 | 0.87–0.97 | 0.86–1.1 |

- **The best soft restart beats the best fixed memory in all 48 cells** — by 5–20 % on the
  smallest change (all 8 cells the same sign, 4 of 8 individually at $t>2$, max 5.0) and by 2.4–5× on the
  largest.
- **The hard restart is the wrong default for small changes.** For $\Delta\pi\le0.2$ it is 1.1–3.4×
  *worse* than never restarting (this reproduces step 2's Gauss → $t$ result inside one family),
  yet a soft restart at the same moment is 1.9–3.8× better than it.
- **The handover's hypothesis is half right.** $\gamma^*$ falls monotonically with the change, and
  hard restart is near-optimal only for the largest ones (soft/hard → 0.87–0.97). But
  **"small change ⇒ $\gamma^*\approx1$" is falsified**: even $\Delta\chi\approx0.03$–0.05 wants
  $\gamma^*=0.1$–0.25. And $\gamma^*$ is not a function of $\Delta\chi$ alone — at fixed $\Delta\pi$ it
  falls as $q$ grows (e.g. $\pi_0=.70$ lower: 0.025 at $q=.01$, 0.01 at $q\ge.02$), because a larger $q$
  means more old evidence to discount.
- **The untuned bank is worse than fixed memory** almost everywhere (up to 2.8× on small changes);
  its continuous adaptation is not a substitute for knowing the change.
- At $\gamma^*$ the best post-change memory is cumulative or the longest in the grid: once the old
  evidence is correctly discounted, there is no further reason to forget (each regime is stationary
  here — see caveats).

### Result 2 — the right quantity is retained *events*, and it follows a shrinkage law

$\gamma$ is a fraction of whatever evidence happens to have accumulated
($E[K]=q\sum_{i<\tau}\ell^i$, e.g. $q\tau$ for cumulative memory), so it mixes the change with $q$ and
memory. The invariant is the number of old marginal-corner events kept,
$n_0=\gamma\,E[K_\tau]$. Treating them as a prior worth $n_0$ events centred on the old $\chi$, the
post-change estimate after $n$ new events has MSE $\approx(n_0^2\Delta^2+n\,v)/(n_0+n)^2$, minimised at

$$
n_0^*=\frac{v_\text{post}}{\Delta\chi(q)^2},\qquad v_\text{post}=\chi_\text{post}(1-\chi_\text{post}),
$$

independent of $n$. ⚠️ This was derived *after* seeing the coarse-grid run, so it is post hoc; the
refined grid on fresh seeds is the check:

- **$n_0^*$ / predicted: median 0.89, range 0.50–1.51; 47/48 cells within ×2** (the $\gamma$ grid's own
  resolution) and 39/48 within ×1.5. Log–log slope **1.03**, $R^2=0.98$, over $n_0$ from 0.2 to 266
  events. The prediction lies inside the seed-bootstrap range of $n_0^*$ (±1.5× slack) in 47/48.
- Equivalently, $\gamma^*\approx\min\bigl(1,1/Z^2\bigr)$ with
  $Z=\Delta\chi/\sqrt{v/E[K_\tau]}$ — the change **measured in standard errors of the old
  estimate**. That is the standardized signal the handover proposed, with the old regime's evidence
  as the denominator. $\gamma^*\approx1$ only when $Z\lesssim1$, i.e. when the change is smaller than the
  old estimate's own noise — a change nothing could detect.
- So the soft restart is Bayesian shrinkage: after a break, the old regime is worth about
  $v/\Delta\chi^2$ events of prior, never more, however much of it was observed.

### Caveats

- **An oracle.** It knows $\tau$ *and* is scored with the best $\gamma$; $n_0^*$ needs $\Delta\chi$, which a
  real system must estimate. These gaps are upper bounds on what a detector can recover.
- The post-reset optimum of "cumulative memory" holds because each regime is stationary and there is
  one change. Repeated changes would bound the useful memory again.
- Copula space, known margins, one $\rho_S$, one mixture path. The shrinkage law itself is
  model-free (binomial events), so it should transfer, but that is untested.
- Raw CSVs are in the gitignored `experiments/results/`; every number here regenerates in ~10 s
  from the committed harness and fixed seeds.

## What this sets up

- Use the ratio estimator; score against $\chi(q)$; report $\chi(q)-\lambda$ separately.
- ~~Step 2~~ done (above). Note the step-1 expectation that Gauss → $t$ would "gain most from
  restart" was wrong: it gains nothing.
- ~~Step 3b~~ done (above): soft restart dominates; $n_0^*=v_\text{post}/\Delta\chi^2$. Next: detection only for the transitions with a real oracle gap
  (Clayton/Gumbel switches, Gauss → Clayton/Gumbel on the dependent tail); rank-based margins
  (rank-PIT exists on `adaptive-memory`); a tail-structure drift to test the cube-root form.
- Only one dependence level ($\rho_S=0.58$) and $\nu=4$ so far.
