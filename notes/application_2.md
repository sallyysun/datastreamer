# Application track 2 — hedging & the covariance matrix

A second, stronger applied framing than pairs-trading alone: turn "estimate a dependence
parameter" into "maintain a live risk model".

## The link: $\Sigma = D\,R\,D$
Any covariance matrix splits as
$$\Sigma_t = D_t\,R_t\,D_t,\qquad D_t=\operatorname{diag}(\sigma_{1,t},\dots,\sigma_{d,t})$$

- marginal vols $D_t$ times a correlation matrix $R_t$ (the dependence). Sklar's theorem is the
same split at the distribution level (joint = marginals + copula). So the online dependence
estimator *is* a streaming estimator of $R_t$; pair it with online vols (Welford / EWMA variance)
to get a streaming $\Sigma_t$. And the copula carries **more** than $R_t$ 
- $\Sigma$ is only the
second-moment projection; the copula also captures **tail dependence / asymmetry** (probability of
joint crashes) that $\Sigma$ cannot express.

## Covariance-matrix applications
1. **Decoupled, dual-timescale $\Sigma_t$.** Copulas separate vol from dependence, so forget them at
   different rates — vol clusters fast (short memory), dependence drifts slowly (long memory). A
   single covariance EWMA (RiskMetrics) uses one decay for both and gets both wrong. Nonparametric,
   online analog of DCC-GARCH. (Same decoupling lesson as the `hermite_ew` cold-start.)
2. **Robust correlation.** Rank dependence is robust to fat tails/outliers; for elliptical copulas
   convert Kendall's $\tau$ via $R=\sin(\tfrac{\pi}{2}\tau)$ → a robust covariance not blown up by a
   single jump.
3. **PSD catch (real streaming challenge).** Pairwise rank-correlation converted elementwise is not
   guaranteed positive-semidefinite, but portfolio use requires PSD. Maintaining a **PSD,
   time-varying correlation matrix online** (incremental nearest-PSD projection / shrinkage) is an
   open problem for $d>2$ — a natural extension of the bivariate work.

## Hedging applications
1. **Dynamic minimum-variance hedge ratio.** Hold 1 unit of $S$, short $h$ units of a related $F$;
   the hedged position $S-hF$ has variance $\sigma_S^2 - 2h\operatorname{Cov}(S,F) + h^2\sigma_F^2$.
   Minimizing over $h$ gives
   $$h^{*}=\frac{\operatorname{Cov}(S,F)}{\operatorname{Var}(F)}=\rho\,\frac{\sigma_S}{\sigma_F},$$
   so the right amount to hedge is dictated by the two vols and the correlation $\rho$ (higher
   $\rho$ → $F$ tracks $S$ better → hedge more of it).

   *How the time-varying copula makes it adapt online.* Textbook hedging fits a single $\rho$ on a
   batch of history and freezes it, so $h^{*}$ is fixed — but real dependence drifts and jumps
   (calm-2019 $\rho$ ≠ crash-2020 $\rho$). A time-varying copula maintains $\rho_t$, re-estimated
   every tick by the streaming estimator, so
   $$h^{*}_t=\rho_t\,\frac{\sigma_{S,t}}{\sigma_{F,t}}$$
   is itself a live, updating quantity: as the estimator's $\rho_t$ moves (exactly what the oscillate
   / drift runs measure), the hedge re-sizes automatically with no re-fitting. E.g. $\rho$ climbing
   $0.4\to0.7$ → short more $F$ per unit $S$; $\rho$ falling → unwind part of the hedge. A stale
   batch $\rho$ gives a mis-sized hedge precisely when dependence has shifted (over/under-hedged when
   it matters most); streaming $\rho_t$ keeps it correctly sized in real time. The estimator *is* the
   engine driving the re-hedging. (Beyond linear $\rho$: item 2 uses the copula's conditional
   $h(y\mid x)=\partial C/\partial u$ for a state-dependent hedge that also captures stronger tail
   co-movement.)
2. **Conditional / tail-aware hedging (the copula's real edge).** Linear hedging uses one $\beta$
   for all states, but with tail dependence the calm-market hedge fails in a crash. The copula gives
   the hedge as a function of state via the conditional $h(y\mid x)=\partial C/\partial u$ — size
   protection against *joint* extreme moves $\Sigma$ says are negligible.
3. **Portfolio VaR / Expected Shortfall.** Tail risk needs the full joint loss distribution, not
   just $\Sigma$. Copula + marginals → portfolio loss → VaR/ES accounting for tail co-movement;
   time-varying copula → time-varying tail risk, tracked live.
4. **Basis risk in cross-hedging.** Residual basis risk *is* the dependence between target and
   proxy, including how it degrades under stress — exactly what the copula characterizes.

## Fit to the project
The streaming estimator produces $R_t$ / $\rho_t$ online; with online vols this is a **streaming,
nonparametric, tail-aware covariance/correlation matrix**, benchmarkable against DCC-GARCH
(parametric) and RiskMetrics EWMA (linear, single-timescale). Hedging sits on top: dynamic hedge
ratio (easy) + tail-conditional hedge (the novel bit). The $d>2$ streaming-PSD-correlation problem
is the clean scaling extension.

## Scope: we maintain dynamic dependence, not the whole matrix as one blob
$\Sigma_t$ has three free numbers for a pair ($\sigma_{S,t}^2,\sigma_{F,t}^2,\rho_t$). Of these the
**two vols are a solved, cheap, univariate problem** (EWMA vol / GARCH / Welford) and the
**dependence $\rho_t$ is the hard, under-studied, genuinely streaming part** — what this project
owns. So: maintain dynamic $\rho_t$ (for a book, dynamic $R_t$); run standard vol trackers
alongside; reassemble $\operatorname{Cov}(S,F)_t=\rho_t\sigma_{S,t}\sigma_{F,t}$, and hence $h^{*}_t$
and $\Sigma_t$, on demand. We *factor* the covariance matrix and specialize in the interesting
factor, rather than replacing it. Caveats: (i) $\rho$ (+vols) is all $\Sigma$ ever needs, but the
copula carries **more than $\rho$** (tails/asymmetry) that only matter for the tail-aware
hedging/VaR; (ii) scaling "dynamic $\rho$" → "dynamic $R_t$" is where the streaming-PSD challenge
bites.

### $\Sigma_t \leftrightarrow \operatorname{Cov}(S,F)$
$\operatorname{Cov}(S,F)$ is just one entry of $\Sigma_t$. For the pair $(S,F)$,
$$\Sigma_t=\begin{pmatrix}\sigma_{S,t}^2 & \rho_t\sigma_{S,t}\sigma_{F,t}\\ \rho_t\sigma_{S,t}\sigma_{F,t} & \sigma_{F,t}^2\end{pmatrix},$$
so the off-diagonal *is* $\operatorname{Cov}(S,F)_t$ and the hedge reads two entries:
$h^{*}_t=\Sigma_t[1,2]/\Sigma_t[2,2]=\rho_t\sigma_{S,t}/\sigma_{F,t}$. For $d$ assets the same entry
generalizes to $\Sigma_t[i,j]=\rho_{ij,t}\sigma_{i,t}\sigma_{j,t}$ — portfolio risk uses the full
grid, a single hedge uses one cell.

## Does the copula change how you estimate vols? No.
The copula changes how you estimate the **dependence** ($R_t$), not the vol *method*. Vols are a
marginal (per-series) quantity, and Sklar splits the joint as $F(s,f)=C(F_S(s),F_F(f))$ — marginals
and copula are separate, and $C$ is scale-free (defined on ranks/PIT). So vol estimation stays a
standard univariate problem. The one real subtlety is **ordering**: it's a two-stage pipeline —
(1) estimate & remove each marginal (de-mean, de-volatilize to residuals
$z_{i,t}=(x_{i,t}-\mu_{i,t})/\sigma_{i,t}$), (2) PIT/rank those residuals and feed the copula. So the
copula doesn't change the vol method but raises the **stakes**: leftover *time-varying* vol leaks
into the PIT and masquerades as spurious dependence change (the marginal-vs-dependence confound;
the `hermite_ew` cold-start). A constant vol misscaling is harmless (ranks unchanged), but the
*conditional* (time-varying) vol must be removed first. (Parametric non-Gaussian copulas could be
estimated jointly with the marginals for full efficiency, but the standard and streaming-friendly
choice is two-stage IFM — marginals first, copula second — which streaming forces anyway.)

## Baseline to beat: standard (RiskMetrics) EWMA — why popular, where it fails
$$\Sigma_t=\lambda\Sigma_{t-1}+(1-\lambda)r_t r_t^\top,\qquad \lambda=0.94\ (\text{daily}).$$

**Why popular (a genuinely strong baseline):**
1. **Captures vol clustering** — the dominant stylized fact — for ~1% of the effort.
2. **Truly streaming** — O(1) update/state, no optimization, no training.
3. **PSD for free** — a sum of outer products $r_t r_t^\top$ is PSD by construction, so it drops
   straight into portfolio optimization. (This is exactly what the pairwise-rank copula approach
   *loses* — a real EWMA advantage.)
4. **No overfitting, auditable, standardized** — nothing fit to data; $\lambda=0.94$ is a free
   industry default regulators trust.

**Where it fails (our thesis targets):**
1. **One decay, two timescales** — same $\lambda$ for variances and correlations; $\lambda=0.94$
   (half-life ~11 days) is fine for vol, too short for a stable correlation → noisy off-diagonals.
2. **Fixed $\lambda$ = one point on the efficiency$\leftrightarrow$adaptation curve** — can't be
   efficient when stationary *and* fast after a change; an adaptive-forgetting / changepoint
   estimator dominates it where $\rho$ jumps or drifts.
3. **Linear/elliptical only — no tail dependence.** $r_t r_t^\top$ is pure second-moment; it
   captures $\rho$ but nothing about *joint tails*, so it **underestimates joint-crash probability**
   — precisely when a risk model matters. The core copula critique.
4. **Non-robust to outliers** — squared/product returns let one extreme day move the estimate a lot
   and decay slowly (~2 weeks). Rank-based dependence is immune.
5. **No leverage effect, no mean reversion** — symmetric response to up/down moves (real vol rises
   more after drops), and being integrated (IGARCH) shocks never mean-revert in the forecast.

**Bottom line / pitch:** not "EWMA is wrong" — it gets the vols right and the dependence crude. The
contribution is the streaming *dependence layer* it lacks: forgets dependence on its own adaptive
timescale, and captures the *shape* of dependence (tails, asymmetry) a covariance-of-returns can't
see.

## Scaling to many assets: factor model + streaming-PSD
Don't estimate the full $d\times d$ stock covariance directly (noisy, PSD-hard). Split each return
into a factor part and idiosyncratic noise, $r_t = B_t f_t + \varepsilon_t$ ($B$ = $d\times K$
loadings, $f_t$ = $K\ll d$ factor returns, $\varepsilon_t$ = stock-specific), and **assemble** the
stock covariance from the small factor covariance:
$$\Sigma^{\text{stock}}_t = B_t\,\Sigma^{f}_t\,B_t^\top + D^{\varepsilon}_t.$$
All cross-stock co-movement flows through shared factors:
$\operatorname{Cov}(r_i,r_j)_t=\beta_i^\top\Sigma^f_t\beta_j$ ($i\neq j$); idiosyncratic risk stays
on the diagonal $D^\varepsilon_t$. This collapses $O(d^2)$ noisy pairwise correlations to a $K\times
K$ factor matrix + $d$ betas + $d$ idio vols, and is **PSD for free** if $\Sigma^f_t$ is PSD — the
clean answer to the streaming-PSD challenge (enforce PSD on the *small* $K\times K$ block, inherit
it on the big one).

Three time-varying pieces, each on its own timescale — the same factoring theme: (1) $\Sigma^f_t$
(the $K\times K$ factor covariance) is where the dynamic-dependence estimator lives — apply
$\Sigma=DRD$ *inside* it: dynamic factor correlations $R^f_t$ (copula, slow) $\times$ factor vols
(fast); small $K$, so even a full copula is cheap and PSD-easy. (2) $B_t$ (loadings drift) — a
streaming regression / Kalman per stock. (3) $D^\varepsilon_t$ — $d$ univariate vol trackers. So
"combine stock and factor covariance" = estimate dependence only at the **factor level** (low-dim,
where the methods shine), then **lift to stock level** through the betas. Tail-aware extension:
replace Gaussian $\Sigma^f_t$ with a **factor copula** (fat-tailed/asymmetric factor dependence) that
propagates joint-tail risk to all stocks sharing those factors.

## Residual dependence: sparse off-diagonal → hierarchical / vine copulas
The assembly assumes idiosyncratics are uncorrelated (diagonal $D^\varepsilon$). When they aren't
(same sector/supply chain), two fixes:

- **Sparse off-diagonal.** Allow a few nonzero residual links, $\Sigma^\varepsilon=D+S$ with $S$
  sparse — chosen by structure (block-by-sector) or data (**graphical lasso**). Put the sparsity in
  the **precision** matrix $\Theta=(\Sigma^\varepsilon)^{-1}$: zeros there = conditional
  independence, and it stays **PSD** (zeroing covariance entries directly can break PSD).
- **Recurse the copula/vol split on the residuals** (good instinct). $\varepsilon_t$ has its own
  $\Sigma^\varepsilon_t$; if non-diagonal, apply $\Sigma=DRD$/copula to it too — but the recursion
  **must reduce dimension each level**, else you're back to full $d\times d$. Which structure tells
  you how: residuals sharing a *common* movement (low-rank) → that movement **is a missing factor**,
  promote it (hierarchical factor model); residuals moving in *small clusters* (sparse/block) →
  a **copula per block**. Stop when residuals are diagonal/independent.

This "copula on residuals of a copula" is exactly a **vine copula** (pair-copula construction — a
cascade of low-dim copulas); **factor copulas** are the low-rank special case:
$$\text{full copula}\xrightarrow{\text{low-rank}}\text{factor copula}\xrightarrow{\text{+ sparse
residual blocks}}\text{hierarchical / vine copula}.$$
Tail payoff propagates: fat-tailed factor copula at the top + lower-tail-dependent block copulas on
residual clusters → joint-crash risk through both shared factors and within-sector links.
**Tradeoff:** each level adds parameters/noise and vines get expensive ($O(d^2)$ pair-copulas), so
the streaming sweet spot is **Level 1** — factor copula + sparse/block residual copulas — not a full
vine.

## Combining a direct and a factor estimate: shrinkage, ensemble, or composition
Two estimators of the covariance: **A = vols + copula** (direct $D R_t D$ on stocks — flexible,
tail-aware, but high-variance and PSD-hard) and **B = factor + idio** (structured, robust, PSD-easy,
but biased by its assumptions). Classic **bias-variance pair**, so combining beats either alone.

**Blend (shrinkage).** $\Sigma_t = w\,\Sigma^A_t + (1-w)\,\Sigma^B_t$. A convex combo of PSD matrices
is PSD, and shrinking the noisy direct estimate toward the structured one cuts variance for a little
bias — this is streaming, copula-flavored **Ledoit-Wolf**. Don't fix $w=0.5$:
- **Optimal (Ledoit-Wolf):** closed-form shrinkage intensity minimizing MSE, computed from data.
- **Adaptive $w_t$ (thesis tie-in):** lean on structured B **right after a change** (direct A is
  rebuilding/noisy; the factor model adapts fast through betas), shift back to A as stationary data
  accumulates. So $w_t$ is driven by the **same ESS / change-detection machinery** as the forgetting
  knob — the multi-timescale ensemble idea, blending *structure* (flexible vs parsimonious) instead
  of *memory*.
- **Bayesian:** factor model = prior, direct copula = likelihood; posterior weight emerges from
  relative precision. Ties to the Bayesian/BOCPD track.

**Key caveat — you can only average what's in the matrix.** Blending two *covariance matrices*
blends **second moments only**; tail dependence does **not** survive matrix averaging ($\Sigma$ has
no tails). So use two layers: blend $\Sigma$ for the linear/hedging use (PSD, cheap), and mix at the
**copula level** (mixture copula $C=wC^{\text{direct}}+(1-w)C^{\text{factor}}$, or a layered model)
for the tail-aware use (VaR/ES, tail hedging). One matrix average can't do both jobs.

**Weightless combination (often cleaner).** "Blend without a weight" has two good readings:
1. **Composition, not averaging** — let each estimator own a *disjoint* piece and **add** them:
   $\Sigma_t = B_t\Sigma^f_t B_t^\top + \Sigma^\varepsilon_t$ (factor model = systematic backbone;
   copula = residual dependence). No weight because they don't estimate the same thing — they
   partition into systematic + residual; still PSD; more expressive than averaging. Same at the
   distribution level: factor copula + residual copulas = hierarchical copula, not a weighted mix.
2. **Derive the weight, don't choose it** — precision (inverse-variance) weighting (the more certain
   estimate dominates automatically), Ledoit-Wolf, or Bayesian posterior. The weight exists but
   falls out of the uncertainties.
Prefer **composition** when the two cover *different* structure (systematic vs residual,
second-moment vs tail); only "two noisy estimates of the same matrix" needs a weight, and even then
derive it rather than hand-set 50/50.

# Combining temporal with vine — the dynamic / streaming vine

Two orthogonal axes: **vine** = cross-sectional dependence structure (which assets depend on which,
at one instant); **temporal** = how each dependence evolves and how you track it online. Static vines
are well-studied; a *streaming, adaptively-forgetting, change-aware* vine is the contribution.

## Core idea: one streaming estimator per vine edge
A vine factorizes the $d$-dim copula into a cascade of **pair-copulas**, one per edge $e$, each with
its own parameter $\theta_e$ (a $\rho$, Kendall's $\tau$, or a full bivariate copula). Temporal =
make every $\theta_e$ time-varying, $\{\theta_e\}\to\{\theta_{e,t}\}$, and track each with the
**existing bivariate online estimator** (Xiao/Hermite/parametric-$\rho$ + forgetting + change
detection). Vine = spatial skeleton; the estimator = temporal engine on every node. Modular because
every edge is bivariate → the estimators apply unchanged.

## Why the decomposition helps
1. **Reuse** — each edge is bivariate, so the existing estimators apply directly.
2. **Per-edge timescale** — each edge gets its own $\lambda_e$ (market-sector dependence drifts
   slowly; two names decouple fast). The adaptive-memory idea, now per edge.
3. **Localized change detection** — a per-edge detector localizes *where* a break hit (one sector
   decouples), not just that one occurred. More than a scalar detector gives.

## The hard part: the cascade (conditioning)
Pair-copulas beyond tree 1 act on **conditional** variables, transformed by **h-functions**
$h(u\mid v)=\partial C/\partial v$ built from the *lower* trees' copulas. Those lower copulas are
themselves time-varying, so the upper-tree pseudo-observations are non-stationary for two reasons
(marginal drift **and** lower-tree copula drift), and a change in a tree-1 edge **propagates upward**.
Process the trees **sequentially each tick**: update tree 1 → recompute streaming h-function
pseudo-obs with the current $\theta_{e,t}$ → update tree 2 → … Managing this cascade of time-varying
h-functions is the core technical challenge (closed-form for parametric pair-copulas, harder fully
nonparametric).

## Two ways to drive $\theta_{e,t}$
- **Observation-driven (cheap):** deterministic function of past data — the forgetting/EWMA
  estimators, or a GAS/DCC-style score update. What we have.
- **Parameter-driven (principled):** latent state filtered online (Kalman on
  $\operatorname{atanh}\rho_e$, or per-edge BOCPD) — heavier, gives uncertainty + clean jumps.

## Structure decision
The vine **tree structure** is data-chosen; for streaming, **fix the skeleton** from an initial
calibration and let only parameters drift (optionally re-fit structure on a slow schedule). Fixed
skeleton, flowing parameters — re-selecting the tree every tick is intractable.

## Tractable entry point
The full dynamic vine is heavy. The **factor-copula** special case tracks only the $K$ factor-level
dependences (plus betas) over time — $K$ streaming estimators, no deep cascade. That is the
**Level-1 streaming hierarchical copula**; the full dynamic vine is the general, expensive
generalization.

**Summary:** temporal $\times$ vine = attach one streaming dependence estimator (own forgetting +
changepoint detector) to each pair-copula edge, and process trees sequentially so the time-varying
h-functions cascade correctly — a **dynamic streaming vine** with per-edge adaptive memory and
localized change detection.

# Analysis / diagnostics on the estimated structure

The analysis layer on top of the estimated $\Sigma_t$ / copula. Through-line: everything static risk
analysis does with a covariance matrix, do it **dynamically (streaming)** and **tail-aware (copula)**
— that is the differentiator. All of it is *derived* from the $\Sigma_t$ / copula already estimated
online, so it becomes a live analytics layer (surface on the React/probe front-end), with the change
detector flagging *when* structure shifts and localizing *where*.

## 1. The dependence structure itself
- **Eigenstructure (PCA of $R_t$).** Largest eigenvalue = "market mode" (how much everything
  co-moves); a *rising* top eigenvalue = correlations converging = diversification breaking down
  (crisis signal). Its eigenvector = the co-movement direction.
- **Absorption ratio** (Kritzman): fraction of variance in the top few eigenvectors; high = fragile /
  systemic, and *spikes* have led drawdowns. Streaming it = a live fragility gauge.
- **Mean off-diagonal correlation** — one scalar for "how correlated is everything now"; balloons in
  stress.
- **Dependence network / MST** — graph of strongest links; clusters = sectors; watch it tighten and
  **rewire** through a regime change (per-edge change detection localizes which links moved).
- **Tail dependence (copula's unique bit).** Lower/upper tail-dependence $\lambda_L,\lambda_U$ per
  pair = probability of crashing *together*, invisible to linear correlation. Compare **body vs
  tail-implied correlation**: if the tail is far more dependent, linear diversification is an illusion
  exactly when it matters.
- **Dynamics** — plot $\rho_t$ / eigenvalues / absorption ratio as time series; segment regimes with
  the change detector.

## 2. Offsetting risks (portfolio risk decomposition)
Given weights $w$ and $\Sigma_t$ ($\sigma_p=\sqrt{w^\top\Sigma_t w}$):
- **Component risk contributions** $\text{CTR}_i = w_i(\Sigma_t w)_i/\sigma_p$ (sum to $\sigma_p$) —
  which positions *add* risk vs *offset* it; a negative CTR is a genuine hedge. Core "offsetting"
  analysis.
- **Diversification ratio** = (weighted-avg vol)/(portfolio vol) — how much risk the dependence
  structure cancels; near 1 = not actually diversified.
- **Effective number of bets** (PCA of the portfolio) — how many *independent* risk sources are held;
  if risk collapses onto one eigenvector, many positions are really one bet.
- **Tail caveat (copula).** Body-estimated offsets can vanish in the tail (correlations rush to 1 in a
  crash). Compute a **stressed/conditional covariance** (dependence conditional on large moves) or use
  tail dependence to check offsets *survive stress* — the tail-conditional hedge as a diagnostic.

## 3. How factors and stocks move together (attribution)
On $r_t=B_t f_t+\varepsilon_t$:
- **Exposures over time** — streaming regression → $B_t$; watch loadings *drift*.
- **Systematic vs idiosyncratic share** — each stock's $R^2$ (factor-explained variance); aggregate
  = how factor-driven the book is (rises in crises — everything becomes a factor trade).
- **Factor risk decomposition** — split portfolio variance into factor contributions ($B^\top w$ via
  $\Sigma^f_t$) + residual; reveals *which factors* you're actually exposed to (often surprising).
- **Factor co-movement** — $\Sigma^f_t$ itself (are value/momentum correlations shifting? crowding
  shows up here).
- **Residual diagnostic (model check)** — residual cross-correlation ≠ 0 ⇒ a **missing factor**
  (the "promote residual structure to a factor" loop). This analysis also *validates* the model.

