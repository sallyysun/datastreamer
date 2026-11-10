# Aligning the bank, Hermite and RMPL with Chapters 4–5: staged audit (2026-11-09)

**Platform:** Linux / libstdc++, branch `bank-vff-study`. Code changes are uncommitted, at Sally's
request.

**Diagnosis seeds:** 1–30, which are the tuning seeds, × 4 families.

**Seeds 42–101 are untouched,** so they remain a clean evaluation set for the final rerun.

**Scope:** the bank's marginals, Hermite's start-up and standardisation, RMPL's start-up gain, the
inputs the bank arms train on, definite bugs, and which Chapter 5 results apply to what.

The changes were made in stages. Each was checked on its own against a corrected baseline before
any combination.

## 1. Verified configuration (code as of this note)

| | Before (HEAD `33f51fd`) | Now (default) | Experiment switch |
|---|---|---|---|
| **Bank scoring margins** | `RankPit(500)`: sliding window, (R+1)/(n+2) | `EwMarginal`: EW empirical CDF, N_eff,m = 500, weighted GK with ε = 0.005, queried before the update | `BANK_MARGINAL=window`, `BANK_MARGINAL_NEFF` |
| **Bank scoring gate** | Scores once the margins hold 30 observations, including arms still outputting placeholders | Margins hold 30 observations **and** every arm's `ready()` is true | — |
| **Xiao arm inputs** | Probit of the window ranks; cut-points recalibrated on the first 100 of those | **Raw (X, Y)**; its own cut-points from the first 100 observations; EW count matrix (thesis 5.1.1) | `BANK_ARM_INPUTS=shared` (probit(Û) binned at Φ⁻¹(k/B)), `=composite` (old) |
| **Hermite arm inputs** | Probit of the window ranks, then EW standardisation | **Raw (X, Y)** with sequential EW standardisation | as above |
| **Hermite coefficient weight** | max(λ, 1/m); first observation not absorbed; λ = 0 used 1/n over n−1 terms | **Normalised EW**: w_m = 1/W_m with W_m = λ_th·W_{m−1} + 1 (equivalently λ/W with the code's λ); λ = 0 uses 1/m | `HERMITE_START=literal\|runningmean\|legacy` |
| **Hermite moments** | Mean seeded at X₁, variance seeded at 1, plain-λ updates, current observation included | Same normalised weights; Welford when λ = 0; current observation included (the `hermiter` convention) | `HERMITE_MOMENTS=predictive` |
| **Hermite output hold** | 0 until n > 50 (bank) or n > 100 (budget) | None (`burn_in = 0`); `ready()` covers start-up | — |
| **RMPL (bank arms)** | ExactNormal on the bank's u; gain max(g, 1/m) | ExactNormal on the shared Û; **gain g/(1−(1−g)^m)** (normalised) | `RMPL_START=max\|constant` |
| **RMPL (standalone / budget)** | Constant gain from r₀ = 0 | Normalised start-up gain (`startup = true` by default) | `Config::startup` |
| **Xiao VFF** | Its own `RankPit(500)` for the residual | Its own `EwMarginal` (raw inputs), or Φ(z) (shared inputs) | — |
| **Hermite VFF** | Surprise-ratio rule (unchanged) | Unchanged | — |
| **MPL VFF** | `ParametricCopulaMLE` (old estimator), cumulative standardisation of the window scores | Same estimator; it takes the shared Û (`normal_score_inputs`) unless composite | — |
| **Student-t ρ_S table** | Shared statics rebuilt in place: a **data race** (TSan: races and a corrupt allocation) | One immutable table per ν, built under a mutex, with a thread-local pointer | — |
| **Bank weights** | Hedge + fixed share; η = 1, δ = 10⁻⁴ (squared); δ = 3·10⁻⁴ (log); tuned on seeds 1–30 under the old margins | Unchanged (**not retuned yet**) | — |

Pre-update order (Algorithm 1), unchanged and verified:
1. Û, V̂ from the margins held through n−1.
2. Losses from the arms' estimates held through n−1.
3. Hedge, then fixed share.
4. The arms update.
5. The margins absorb (X_n, Y_n).

## 2. Definite bugs fixed (kept separate from the method changes)

1. **Hermite cumulative denominator.** λ = 0 used weight 1/n_ while only n_−1 terms had been
   absorbed, which shrinks Â by (n−1)/n. It now uses 1/m. This only affects cumulative arms and keys.
2. **Placeholder predictions were scored.** The bank started scoring at observation 31:
   - Hermite returned 0 until observation 51;
   - Xiao in composite mode returned 0 until observation 101;
   - RMPL returned r = 0 at observation 31.
   Banks now score only when every arm is `ready()`, and the regret and tuning probes replay the same
   gate.
   - **Effect on the existing banks: none.** All five arms of a single-estimator bank output the same
     placeholder 0, so their losses are equal and Hedge leaves the weights unchanged. The regression
     run reproduced HEAD's bank RMSE exactly.
   - It would matter for the mixed bank.
3. **Student-t table race:**
   - **Before:** under 8 threads, ThreadSanitizer reported data races and an allocation-size-too-big
     error.
   - **After:** clean, with 0 mismatches.
   - **Single-threaded runs** (`probe_thesis_bank`) were never affected.
4. **Memory accounting:** fixed earlier, in `budget_mpl.md` §24.

**Not fixed: the VFF comparators don't work as intended.**
- **MPL VFF** (`CopulaFit::vff_rate`) at the fast rate is worse than the 39-observation arm even on
  constant dependence (RMSE×1000 140 against 89). Three causes:
  - its score-drift statistic is smoothed at the adaptation rate itself;
  - λ is clamped to [0.90, 0.9999], which reaches outside the bank's memory range;
  - it is the old estimator, not RMPL.
- **Hermite VFF** barely leaves its starting memory: its RMSE equals the 199-observation arm's.
- **Thesis §5.2 is empty,** so no rule is defined for them to follow.
- **Recommendation:** one estimator-agnostic rule for all three. Xiao's block residual rule
  s = p̂_n − ρ̂_{n−1} has a known null variance and uses only the shared (Û, V̂) and ρ̂, so it applies
  unchanged to Hermite and RMPL. Clamp it to the bank's range, N_eff ∈ [39, 1999].
- **Until then, don't present the VFF comparisons as evidence about adaptation mechanisms.**

**Regression checks:**
- Xiao in raw mode, Hermite with `legacy = true` (λ > 0), and RMPL with constant and max start-up
  are **bit-identical** to HEAD.
- Full pipeline, old configuration (`BANK_MARGINAL=window BANK_ARM_INPUTS=composite
  HERMITE_START=legacy RMPL_START=max`), seeds 1–3: every fixed-arm, bank and uniform row is identical
  to HEAD. Only Xiao's VFF rows changed, because its internal window is now EW.

## 3. The EW marginal estimator: target, approximation, guarantee

**Target:** the EW empirical CDF F_n(x) = Σ_i a_{n,i} 1{X_i ≤ x}, with a_{n,i} ∝ λ_m^{n−i}. This is
Chapter 5's recursion applied to indicators.

**Pseudo-observation:** Û_n = (B_{n−1}(X_n) + w)/(W_{n−1} + 2w), computed before the update, where
B is the weight strictly below X_n and w the weight X_n is about to receive. Without forgetting this
is exactly (R+1)/(n+2): checked bit for bit against exact ranks.

**Approximation** (`src/estimators/dependence/ew_marginal.hpp`):
- **Weighted GK summary with forward decay:** observation i enters with weight λ_m^{−i}, so relative
  weights are exact and old tuples are never touched.
- **Rescaling:** weights are rescaled by 10⁻¹⁵⁰ when they would overflow.
- **Each tuple stores its own value's weight w.** The insert Δ is then g + Δ − w of the right
  neighbour, the weighted form of GK's "−1".
- **Compress** every ⌊1/(2ε)⌋ inserts at a cap of 2εW.
- **Queries** use rank-interval midpoints.

**Guarantee: empirical only.** Ordinary GK bounds don't transfer automatically, and published
weighted-GK results (ICDT 2023) assume specific weights. Validation against the exact EW ECDF
(`scratchpad/ewm_valid.cpp`; worth adding to the repo's tests):

| Stream | n | Max \|Û − exact\| | RMSE | Max tuples |
|---|---|---|---|---|
| N(0,1), N_eff 500 | 4·10⁵ (rescaling ×4) | 0.0046 | 0.0019 | 283 |
| N(0,1), N_eff 1999 | 10⁶ (rescaling ×2) | 0.0049 | 0.0022 | 284 |
| Heavy ties (7-point grid) | 2·10⁵ | 0.0036 | 0.0013 | 287 |
| Margin jumps every 50k | 4·10⁵ | 0.0043 | 0.0019 | 296 |
| Cauchy (extremes) | 2·10⁵ | 0.0046 | 0.0020 | 282 |
| No forgetting | 2·10⁵ | 0.0049 | 0.0023 | 271 |

- **Extreme queries** (±10³⁰⁰) are correct to 10⁻¹⁵. No non-finite or out-of-(0,1) values occurred.
- **Error** stays below ε = 0.005, about a tenth of the sampling SD of Û at N_eff = 500 (≈ 0.022).
- **Storage** stays flat at about 280–300 tuples (about 9 KB per margin) across 10⁶ observations.
  This is an observed bound, not a proved one.
- **Cost:** 0.3 µs per observation.

**ESS matching is not equivalence.** EW with N_eff = 500 has the same pointwise variance F(1−F)/N_eff
as a 500-window, but a different response to a change: a heavier tail and a faster initial response.
The paired run (§6) shows the consequences differ by sign across estimators and scenarios.

## 4. Hermite: start-up and standardisation

**Start-up variants** (`hstart.cpp`):
- Constant Gaussian copula, ρ_S = 0.582, order 10, 200 seeds.
- Prediction made before each update, as a bank scores it. RMSE×1000.

| Update weight (N_eff) | Variant | [1,50) | [50,200) | [200,1k) | [1k,5k) |
|---|---|---|---|---|---|
| 0.01 (199) | old code | 248 | 81 | 54 | 55 |
| | running mean max(λ, 1/m) | 249 | 73 | 53 | 55 |
| | **Ch. 5 literal: â₁ = h(X₁), then (1−λ)** | 426 | 203 | 58 | 55 |
| | **normalised Q/W** | 248 | 76 | 54 | 55 |
| 0.001 (1999) | old code | 248 | 77 | 39 | 30 |
| | running mean | 249 | 72 | 33 | 18 |
| | **Ch. 5 literal** | 557 | 466 | 290 | 86 |
| | **normalised Q/W** | 249 | 72 | 33 | 19 |
| 0.01, raw data N(5, 0.01²) | current-order standardisation | 242 | 78 | 54 | 55 |
| | predictive order | 266 | 79 | 54 | 55 |
| | old code | 522 | 520 | 318 | 55 |

**Findings:**
- **The literal Chapter 5 initialisation is unacceptable under slow forgetting.** X₁ keeps weight
  λ_th^{n−1}: 0.607 at n = 500 for λ_th = 0.999, so X₁ alone counts as about N_eff/2 observations.
  The defect is in the theoretical recursion as written, not only in the code.
- **Normalised exponential weighting** gives every observation weight ∝ λ_th^{n−i}/Σλ_th^{n−j} at
  every n:
  - it equals the running-mean rule's accuracy to within about 1;
  - it reduces to â₁ = h(X₁) at n = 1 and to constant gain 1−λ_th as n → ∞;
  - it is exactly the weighting the bias–variance analysis assumes (a_{n,i} summing to one).
- **The old code's long arms were hurt by its standardisation start-up,** not by the coefficient
  rule. On raw data with σ ≠ 1, the old code fails outright: RMSE 486 after 1000 observations. That
  is the live device's situation.
- **Clamping |ρ̂| ≥ 1 is negligible** (≤ 0.02%) for every variant except the literal one (0.18%). No
  numerical problems.
- **Standardisation order:** including the current observation (the `hermiter` convention) is better
  at start-up than predictive ordering (242 against 266 in [1,50)) and identical afterwards.
  - It is not a scoring leak: the bank scores before `update()`.
  - **Keep it.**

**In the bank** (paired against the old code, composite inputs, seeds 1–30):
- Normalised start-up improves the 1999-observation arm by −10.6 (drift) and −5.3 (jump) RMSE×1000,
  and leaves the banks within ±0.4.
- The literal start worsens the banks by +3 to +23 (worst on constant).

**What sequential standardisation means statistically:**
- Observation i enters the coefficient averages as h((X_i − μ̂_i)/σ̂_i), with moments that include
  X_i.
- So Â_n is a weighted average of Hermite functions of observations standardised by *different*,
  evolving moments. It is **not** the Hermite expansion of today's standardised history.
- This is the `hermiter` package's sequential standardisation. It is legitimate, but Section 4.4 has
  to define it.
- **Recommended formulation:**
  - m_n = m_{n−1} + γ_n(X_n − m_{n−1});
  - s²_n = (1−γ_n)(s²_{n−1} + γ_n(X_n − m_{n−1})²);
  - Z_n = (X_n − m_n)/s_n;
  - â_n = â_{n−1} + γ_n(h(Z_n) − â_{n−1});
  - γ_n = 1/W_n, W_n = λW_{n−1} + 1, W_0 = 0.
  - Coefficient updates start once two observations have been seen, since a scale needs two.

## 5. RMPL: the exact recursion and its relation to §5.1.3

**Implemented** (`recursive_mpl.hpp`, ConstantGain mode):
- z_i = Φ⁻¹(clip(Û_i, δ, 1−δ)), with δ = 10⁻³.
- s(r) = r/(1−r²) + [(1+r²)z₁z₂ − r(z₁² + z₂²)]/(1−r²)².
- I(r) = (1+r²)/(1−r²)².
- r_m = Π_{[−0.999, 0.999]}(r_{m−1} + γ_m I(r_{m−1})⁻¹ s(r_{m−1})), with r₀ = 0.
- The first update follows 30 marginal observations.
- ρ̂ = (6/π) arcsin(r/2).
- **Gains:**
  - constant: γ_m = g = 1 − λ_th;
  - old bank start-up: max(g, 1/m);
  - now: γ_m = g/(1 − (1−g)^m) = 1/W_m.

**Relation to the discounted pseudo-likelihood.** Let
L_n(r) = Σ_i a_{n,i} log c_r(Û_i, V̂_i), with normalised weights a_{n,i} = λ^{n−i}/W_n.
- Suppose r_{n−1} maximises L_{n−1}.
- Then ∇L_n(r_{n−1}) = (1/W_n)·s_n(r_{n−1}), because the λW_{n−1}/W_n·∇L_{n−1}(r_{n−1}) term
  vanishes.
- Replacing the Hessian by the expected information per observation gives the Fisher-scoring step
  r_n = r_{n−1} + (1/W_n) I(r_{n−1})⁻¹ s_n(r_{n−1}).
- **So the normalised gain is the one-step Fisher-scoring approximation to arg max L_n.**
- The constant gain 1 − λ is its n → ∞ limit. With r₀ = 0 it acts as a prior mass at r = 0 that
  decays like λ^n.
- The recursion does not exactly maximise L_n. The error comes from linearising and from r_{n−1}
  not being the exact maximiser. That is what Chapter 5 should say.

**Evidence** (bank arms, paired with the old max rule, seeds 1–30):
- The normalised gain improves the 1999-observation arm by −12.7 (drift) and −5.9 (jump).
- Constant gain is +106 worse on constant for that arm, from the shrinkage towards r₀ = 0.
- The banks are unchanged (±0.5).

## 6. Training inputs and scoring inputs

**Scoring**, for every arm and every bank:
- the shared EW marginal pseudo-observations (Û_n, V̂_n), computed before the update;
- p̂_n = 12(Û_n − ½)(V̂_n − ½) for squared loss;
- the arm's Gaussian MoM density at ρ̂ for log loss;
- the arm's own density for the own-loss bank.

**Training:**
- **RMPL:** the same (Û_n, V̂_n). It needs a marginal transform as part of its estimator.
- **Xiao and Hermite: raw (X_n, Y_n) with their own marginal procedures.** These are part of how
  each estimator constructs ρ̂ (Xiao's count-matrix ranks, Hermite's coefficient CDFs). The marginal
  information inside the Xiao and Hermite arms is therefore forgotten at the arm's own λ_k, as
  Sections 5.1.1 and 5.1.2 already imply.

**Evidence** (paired, seeds 1–30):
- Both alternatives beat the old composite pipeline: Hermite banks by up to −2.4, Xiao by about
  −0.5.
- Shared against native in the final setting: within ±1.2 everywhere.
- **The data don't force the choice.** The principle does: don't change the estimators. Native
  training is the default; `BANK_ARM_INPUTS=shared` is kept as a documented alternative.

**Regret:** the experts may be any predictors measurable with respect to F_{n−1} that are scored on
a common loss. Native training changes the experts, not the loss, so the conditions are unaffected.

## 7. Paired comparisons (RMSE×1000; seeds 1–30 × 4 families; mean ± SE)

**One factor at a time, against the corrected baseline R0** (window margins, composite inputs, old
Hermite, max-start RMPL):

| Factor | Main effect |
|---|---|
| EW margins, N_eff,m = 500 | Mostly within ±2. MPL arms +2 to +4 on constant and oscillate, −1 to −3 in jump recovery. Banks −1.6 to +2.4. |
| EW margins, N_eff,m = 1999 | Almost uniformly ≤ 0 (−0.2 to −1.2). Stationary margins favour long marginal memory. |
| Hermite normalised start | 1999-arm −10.6 (drift), −5.3 (jump); banks ±0.4 |
| Hermite literal start | 1999-arm +108 (constant); banks +3 to +23 |
| Hermite predictive moments | Like normalised; 39-arm slightly better (−0.4 to −0.7) |
| RMPL normalised gain | 1999-arm −12.7 (drift), −5.9 (jump); banks ±0.5 |
| RMPL constant gain | 1999-arm +106 (constant); banks +4 to +16 |
| Inputs: shared, against composite | Hermite banks −0.1 to −2.4; Xiao −0.2 to −0.6 |
| Inputs: native, against composite | Hermite −0.0 to −0.7; Xiao −0.2 to −0.7 |

**Combined final configuration F500r** (EW margins at 500, native Xiao/Hermite, normalised Hermite
and RMPL), against R0:
- Banks change by −1.9 to +2.5.
- Long arms improve by −7 to −12 (drift and jump).
- Bank jump recovery mostly improves: −0.5 to −4.0 in ticks 0–1500.
- MPL banks get slightly worse on constant and oscillate (+0.7 to +2.5), the cost of N_eff,m = 500
  against the window for RMPL's inputs. At N_eff,m = 1999 (F1999r) these become −1.0 to +1.0.

**Chapter 6 conclusions, recomputed:**

| Conclusion | R0 (baseline) | F500r | F1999r |
|---|---|---|---|
| Log bank vs hindsight-best fixed, drift (W/L/T of 12) | 0/12/0, +3.7 | 0/12/0, +4.1 | 0/12/0, +3.8 |
| Log bank vs hindsight-best fixed, jump | 12/0/0, −17.5 | 12/0/0, −17.7 | 12/0/0, −17.8 |
| Log vs squared bank (24 cells) | 19/0/5 | 19/0/5 | 20/0/4 |
| Squared bank vs uniform (24) | 24/0/0 | 24/0/0 | 24/0/0 |
| Regret violations, covered banks (log_mix, sq32, sq32_dstar) | 0/5760 each | 0 | 0 |
| Regret violations, uncovered (log_point, sq at η = 1) | 387, 402 | 299, 393 | 353, 392 |

- **Bank memory around the jump** (Σ_k w_k N_eff,k, mean over families) follows the same path in all
  three: about 1500–1800 before, about 140–350 at t = 2550, and about 120–350 by t = 3500.
- **No Chapter 6 conclusion changes.**

## 8. Decisions

1. **Shared EW marginal for scoring,** N_eff,m = 500 as the main setting and 1999 as sensitivity.
   - Weighted GK at ε = 0.005, accepted on the §3 evidence.
   - 1999 is marginally better in the simulations, because the margins are stationary. 500 is the
     more defensible default for the real-data application, where margins drift.
   - Report both. The difference is ≤ 2.5 RMSE×1000.
2. **Hermite:**
   - normalised EW weights for the coefficients and the moments;
   - sequential standardisation in the `hermiter` order;
   - every observation absorbed once a scale exists;
   - no output hold.
3. **RMPL:** normalised gain 1/W_m, everywhere.
4. **Bank inputs:** native training for Xiao and Hermite, shared for RMPL, shared for scoring.
5. **Scoring gate on `ready()`.**
6. **Retune η and δ** on seeds 1–30 under the final configuration, then rerun the final experiments
   on seeds 42–101. Hold out a fresh test set, for example seeds 200–259, if any further choice is
   made after looking at 42–101.

## 9. Remaining discrepancies and open items

- **VFF comparators** (§2): need a common rule; thesis §5.2 is empty.
- **η and δ** were tuned under window margins: retune (decision 6).
- **Budget study** (`probe_budget`): the Hermite and RMPL defaults changed (normalised start-up,
  `burn_in = 0`). Its numbers are stale until rerun.
- **Family and tail banks** (`family_bank.hpp`, `tail_*`, `estimator_bank.hpp`) still use
  `RankPit(500)`. They are outside this pass, and still need switching and rerunning.
- **Own-density scoring:**
  - Hermite's density is clipped at 0 and mixed with independence at weight 0.01;
  - Xiao's is a pooled histogram, whose margins are only approximately uniform;
  - these are proper densities on [0,1]², but not exact copula densities. Say so wherever log-loss
    regret is applied to them.
- **The log-loss bound covers the mixture of arm densities (log_mix).** The deployed "log bank"
  outputs Σ_k w_k ρ̂_k, and its Gaussian density at that average (log_point) is **not** covered.
- **The squared-loss bound needs η ≤ 1/32.** The tuned bank uses η = 1 and is not covered; it
  violates the bound in about 7% of runs.

## 10. Thesis statements to revise (Chapters 4–5)

1. **§4.2.1:** add the EW marginal estimator:
   - the target F_n;
   - Û_n = (B_{n−1}(X_n) + w)/(W_{n−1} + 2w), computed before the update;
   - the weighted-GK representation with forward decay and rescaling;
   - its accuracy as tested, not proved (§3 table).
   - Replace the sentence "Extending Gregory's linked summaries to exponential forgetting would
     require further investigation", since `budget_mpl.md` §23 now does it.
2. **§4.2.2 / 4.2.3** (empty):
   - pseudo-observations from the margins held through n−1;
   - the clip δ;
   - the score and information in §5;
   - projection to |r| ≤ 0.999;
   - the 30-observation warm-up;
   - ρ̂ = (6/π) arcsin(r/2).
3. **§4.3:** cut-points are the empirical quantiles of an initial calibration sample (100
   observations) and then fixed.
4. **§4.4.1–4.4.2:** define sequential standardisation (§4 formulation). State that Â_n averages
   Hermite functions of observations standardised by evolving moments.
5. **§5.1 (5.1.1):** introduce normalised weights a_{n,i} = λ^{n−i}/W_n, W_n = Σ_{j≤n} λ^{n−j}, as the
   weights every recursion uses:
   - S_n = S_{n−1} + (1/W_n)(Z_n − S_{n−1});
   - S₀ is irrelevant;
   - it reduces to (5.1.1) as n → ∞.
   - Xiao's ratio estimator is unaffected, since it is scale invariant.
6. **§5.1.2:** replace â₁ = h(X₁), â_n = λâ_{n−1} + (1−λ)h(X_n) with the normalised recursion, the
   same for Â. Give the reason: the literal form gives X₁ weight λ^{n−1}, i.e. about N_eff/2
   observations.
7. **§5.1.3:** complete it with the Fisher-scoring recursion and gain 1/W_n, the one-step derivation
   in §5, and the statement that the recursion approximates, but does not exactly maximise, L^EW_n.
8. **§5.1:** add one paragraph separating marginal forgetting (λ_m, the shared scoring margins) from
   dependence forgetting (λ_k). State that native Xiao and Hermite marginals forget at λ_k.
9. **§5.3 / Algorithm 1:**
   - **line 5:** Û_n, V̂_n from the shared EW marginal (λ_m);
   - **line 7:** score only arms that have produced a prediction;
   - **line 12:** "update E_k with (Û_n, V̂_n) (RMPL) or (X_n, Y_n) with its own marginal procedure
     (Xiao, Hermite)";
   - **line 13:** "update the marginal estimates with forgetting factor λ_m".
10. **§5.4 (Proposition 5.4.1 and §5.4.4):** keep them. State that they apply to a linear EW average
    of independent contributions, i.e. the ideal estimator mean_EW(Z_i) with *true* PITs.
    - The implemented estimators are nonlinear, and Xiao, Hermite and RMPL have no theory here.
    - Their inputs are built from estimated, overlapping marginal histories, so the contributions
      are dependent.
    - The bound is therefore an idealised guide. The ESS optimum is checked empirically in Chapter 6.
11. **§5.6.2 (Proposition 5.6.2):** ε_n now includes the summary error (observed at most ε per margin). Note
    that the shared margins are themselves estimated with forgetting.
12. **§5.7 (Corollaries 5.7.1–5.7.2):** these cover sq32 and log_mix only. Say explicitly that the
    tuned squared bank (η = 1) and the point-density log bank are outside the bounds. The empirical
    check confirms 0 violations for the covered banks and about 7% for the uncovered ones.

## 11. Reproduction

```bash
cd experiments
make -B probe_thesis_bank probe_bank_regret EIGEN=/usr
# baseline R0 and the final configuration F500r, seeds 1-30 (TAG names the output)
BANK_MARGINAL=window BANK_ARM_INPUTS=composite HERMITE_START=legacy RMPL_START=max \
  SEED_BASE=1 COPULA=gaussian COMPONENTS=xiao,hermite,mle TAG=diag_R0 ./probe_thesis_bank 30
SEED_BASE=1 COPULA=gaussian COMPONENTS=xiao,hermite,mle TAG=diag_F500r ./probe_thesis_bank 30   # defaults
BANK_MARGINAL_NEFF=1999 ... TAG=diag_F1999r ...
TAG=F500r COPULA=gaussian ./probe_bank_regret 1 30
python3 compare_diag.py R0 F500r F1999r          # paired deltas
python3 conclusions_diag.py R0 F500r F1999r      # Chapter 6 headline conclusions
```

The other tags (M1, M2, H1–H4, Rn, Rc, Ps, Pr, F500s) are listed in `scratchpad/diag*.sh`, with
their environment switches in the table in §1.

---

# Part II: common VFF, retuning, and the final evaluation (2026-11-09)

**Configuration frozen before evaluation:** fingerprint `6b773adcbc4d6527`, the sha256 of
`git diff` plus the new headers.

**What used which seeds:**
- **Development set, seeds 1–30:** every choice in this part (VFF design and rates, the η/δ check).
- **Evaluation set, seeds 42–101:** run once, with no method change afterwards.
- **Budget study:** its own validation seeds (1000–1039) and test seeds (42–141).
- **Labels:** dev results are marked **[dev]**, evaluation results **[confirmed]**.

## 12. Common VFF rule (`src/estimators/dependence/vff.hpp`)

**Rule:**
- One controller adapts the memory of the ideal estimator of §5.4.4, the normalised EW mean m_n of
  p̂_n = 12(Û_n − ½)(V̂_n − ½) on the shared EW margins.
- It runs stochastic gradient descent on that mean's prequential loss (p̂_n − m_{n−1})². This is
  Bodenham & Adams' (2017) adaptive forgetting factor.
- The derivative is carried recursively:
  - W_n = λW_{n−1} + 1;
  - Ẇ_n = λẆ_{n−1} + W_{n−1};
  - ṁ_n = ṁ_{n−1}(1 − 1/W_n) − (p̂_n − m_{n−1})Ẇ_n/W_n².
  - Checked against a finite difference to 10⁻⁶.
- The step is taken in θ = log N_eff and scaled by a running RMS of the gradient, which makes it
  scale-free.
- N_eff is clamped to [39, 1999], the bank's range.
- Each wrapped estimator's memory is set to the controller's N_eff before its update
  (`MemoryControlled::set_neff`):
  - Xiao: retention (N−1)/(N+1);
  - Hermite: update weight 2/(N+1), for the coefficients and the moments;
  - RMPL: gain 2/(N+1), with the normalised start-up.
- **The three estimators therefore follow the identical memory path.** Their VFF results differ only
  by how efficiently each uses it.
- **Order:** scored before the update, like a bank arm. Xiao and Hermite train natively, RMPL on the
  shared scores.

**Validation [dev], `probe_vff_common`, seeds 1–30 × 4 families.** Mean N_eff path:

| Scenario | Behaviour |
|---|---|
| constant | Grows towards the long end: about 1500 by t = 5000 at rate 0.02 |
| drift | Settles at **N_eff ≈ 280–300**. The analytic optimum for the ideal estimator, (2σ²/d²)^{1/3} with σ² ≈ 1 and d = 2.5·10⁻⁴, is about 315: an independent check of §5.4.3 |
| jump | Falls from about 1000 to about 50 within 150–200 ticks of the change, then recovers to the long end |
| oscillate | Stays short, about 50–60 |

**RMSE×1000 against the bank [dev]:**

| | constant | drift | jump | oscillate |
|---|---|---|---|---|
| Rate 0.02 | 30–33 | 63–69 | 89–97 | 182–188 |
| Rate 0.1 | 39–49 | 73–79 | 75–81 | 198–203 |
| Squared bank | 30–32 | 66–71 | 68–74 | 185–191 |

- **Rates fixed on dev:** slow = 0.02 (best on constant, drift and oscillate) and fast = 0.1 (best
  after jumps), as the harness's VFF pair.
- **The old comparators are retired:**
  - MPL `CopulaFit::vff_rate`: smoothing tied to the adaptation rate, a clamp outside the bank's
    range, and the old estimator;
  - Hermite surprise ratio: nearly stuck.

## 13. Retuning and the frozen configuration [dev]

**η, δ** (`probe_bank_tune 1 30`, the same objective and grid as `docs/chapter6_results.md`):
- **Squared bank:** optimum η = 1, δ = 10⁻⁴ (regret −3.6 against the hindsight-best fixed memory).
- **Log bank:** η = 1, δ = 3·10⁻⁴ (−6.8). (1, 10⁻⁴) gives −6.8 as well.
- **Every estimator picks η = 1.**
- **These equal the registry's values, so nothing changes.**

**No tuning needed:**
- **Family bank:** η = 1, δ = δ* (theory-prescribed).
- **Tail counters:** fixed memories.

**Own-density banks (`*_bank_own`):**
- An arm's own density is defined on the scale it trains on.
- Scored at the shared (Û, V̂), the arm must therefore train on the shared normal scores. There,
  Xiao's cells are exactly [k/B, (k+1)/B) and Hermite's density is on the probit scale.
- Under native training it would silently use the true N(0,1) margins: an oracle.
- `_own` keys now force shared training, and so do the tail probe's Xiao and Hermite density
  competitors.
- **The own-density bank was not retuned or rerun** (§6.5.3 is optional).

**Frozen configuration** (in addition to Part I §1):
- **VFF comparators:** the common rule at 0.02 and 0.1.
- **Shared EW margins everywhere,** including `FamilyBank`, `TailChi`, `TailCusum`, `EstimatorBank`,
  `RankScoreCusum`, and the probes' own margins.
- **Budget ablation:** `mpl_window` is replaced by `mpl_ew` (RMPL on EW margins, N_eff 500).
- **Not switched:** the estimator-internal `rank_pit_window` options (`xiao_pit`, `hermite_pit`,
  `gaussian_mle_pit`). They are used only by `memory_tradeoff.md`, which is not a final experiment.

## 14. Final results [confirmed] (seeds 42–101 × 4 families; paired against the pre-change results on the same seeds)

**Chapter 6 headline conclusions:**

| | Before (window, old pipeline) | Final, N_eff,m = 500 | Final, N_eff,m = 1999 |
|---|---|---|---|
| Log bank vs hindsight-best fixed, drift (W/L/T of 12) | 0/12/0, +3.8 | 0/12/0, +4.1 | 0/12/0, +3.8 |
| Log bank vs hindsight-best fixed, jump | 12/0/0, −17.9 | 12/0/0, −17.9 | 12/0/0, −17.8 |
| Log vs squared bank (24) | 21/0/3, −4.0 | 21/0/3, −3.7 | 22/0/2, −4.0 |
| Squared bank vs uniform (24) | 24/0/0, −19.3 | 24/0/0, −18.5 | 24/0/0, −18.7 |

**Bank memory around the jump** (Σ_k w_k N_eff,k, mean over families) is unchanged:
- about 1570–1860 at t = 2495;
- about 135–345 at t = 2550;
- about 50–110 at t = 2700;
- about 130–330 at t = 3500.

**Paired change, final500 minus before** (RMSE×1000):
- **Banks:** −1.7 to +2.3.
  - Xiao and Hermite banks mostly improve (−0.1 to −1.7).
  - MPL banks get slightly worse on constant (+2.2) and oscillate (+1.8). They also improve in jump
    recovery by −2 to −3.4 over ticks 0–1500.
- **Long arms:** −6.9 to −11.6 (drift and jump), from the normalised start-up.
- **N_eff,m 1999 against 500:** Xiao and Hermite arms are identical (native training); banks change
  by ±1.5; MPL arms by ±4 (better on constant and oscillate, slightly worse on drift).

**Final RMSE×1000** (mean over 60 seeds × 4 families, N_eff,m = 500):

| | 1999-arm | Squared bank | Log bank | Uniform | VFF slow | VFF fast |
|---|---|---|---|---|---|---|
| Xiao constant / drift / jump / oscillate | 19.9 / 175 / 196 / 317 | 29.4 / 70.6 / 71.2 / 184.1 | 29.2 / 68.3 / 65.6 / 184.1 | 55 / 86 / 94 / 235 | 29.4 / 67.5 / 94.5 / 181.0 | 44.7 / 77.2 / 77.6 / 196.6 |
| Hermite | 21.3 / 174 / 193 / 318 | 30.7 / 71.5 / 73.7 / 190.8 | 30.1 / 69.4 / 67.4 / 191.6 | 59 / 88 / 95 / 236 | 31.7 / 69.0 / 98.4 / 186.7 | 47.0 / 79.4 / 80.3 / 200.8 |
| RMPL | 22.8 / 180 / 195 / 318 | 31.1 / 66.5 / 68.2 / 188.1 | 30.6 / 64.7 / 63.9 / 186.6 | 49 / 80 / 90 / 237 | 29.5 / 63.4 / 90.4 / 184.8 | 38.4 / 73.9 / 74.2 / 202.3 |

**VFF against the log bank** (W/L/T over 12 estimator × family cells, paired):

| | constant | drift | jump | oscillate |
|---|---|---|---|---|
| Slow (0.02) | 3/3/6, +0.2 | 5/0/7, −0.8 | 0/12/0, +28.8 | 9/2/1, −3.3 |
| Fast (0.1) | 0/12/0, +13.4 | 0/12/0, +9.4 | 0/12/0, +11.8 | 0/11/1, +12.4 |

- **New result, replacing the old VFF comparison:** a well-defined VFF matches the bank under smooth
  change (constant, drift, oscillate).
- The bank's advantage is specific to abrupt changes. One adaptive memory must trade reaction speed
  against stationary variance; the bank holds all its memories at once.
- The old "bank beats VFF" result came largely from the broken comparators.

## 15. Regret bounds: assumptions against the implementation [confirmed numbers]

**What the implementation does** (`forgetting_bank.hpp`, `family_bank.hpp`, replayed by `probe_bank_regret`):
- **Prediction:** each arm's estimate is held through n−1. Arms that are not `ready()` are not scored.
- **Losses:** computed at the shared (Û_n, V̂_n), which are formed from the margins before the update.
- **Weights:** Hedge w̃_k ∝ w_{k,n−1} e^{−ηℓ_{k,n}}, then fixed share
  w_{k,n} = (1−δ)w̃_k + δ/K, from uniform initial weights.
- **Then** the arms update, then the margins.
- **This is Algorithm 1 exactly.** Expert advice may be adaptive and F_{n−1}-measurable, and the
  shared estimated inputs enter only through the common loss. So native training, shared margins and
  the readiness gate leave the bound's conditions intact.

| Bank (as replayed) | Prediction | η, δ | Covered by | Condition met? | Violations / runs [confirmed, N_eff,m = 500] | Mean regret/bound |
|---|---|---|---|---|---|---|
| `log_mix` | mixture Σ w_k c_k of the arms' Gaussian MoM densities | 1, 3·10⁻⁴ | Cor. 5.7.2 / switching (fixed-share) bound | Yes: mixture prediction, η = 1, finite losses | **0 / 11520** | −0.07 |
| `log_mix_hedge` | same | 1, 0 | Cor. 5.7.2 (static) | Yes | 0 / 11520 | 0.11 |
| `log_point` (the deployed log bank's reported output) | Gaussian density at Σ w_k ρ̂_k | 1, 3·10⁻⁴ | **None** | No: the density at the averaged ρ is not the mixture | 560 / 11520 (4.9%) | 0.19 |
| `sq` (deployed squared bank) | (p̂ − Σ w_k ρ̂_k)² | **1**, 10⁻⁴ | **None** | No: exp-concavity needs η ≤ 1/32 | 758 / 11520 (6.6%) | 0.34 |
| `sq32` | same | 1/32, 10⁻⁴ | Cor. 5.7.1 / switching bound ×32 | Yes: \|y − z\| ≤ 4 by (5.6.2) | 0 / 11520 | 0.03 |
| `sq32_dstar` | same | 1/32, δ* | same | Yes | 0 / 11520 | 0.03 |
| Family bank `log_mix` | mixture of the arms' own parametric copula densities | 1, δ* = 2.1·10⁻⁴ | Cor. 5.7.2 / switching bound | Yes: proper parametric densities | **0 / 3840** | 0.34 |
| Family bank `log_mix_hedge` | same | 1, 0 | Cor. 5.7.2 | Yes | 0 / 3840 | 0.12 |

The N_eff,m = 1999 run gives the same picture: 0 violations for every covered bank; 673 and 746
violations for log_point and sq.

**Qualifications for the thesis:**
1. **Prediction and estimate are different objects.** The log-loss guarantee is about the bank's
   *mixture density forecast*. The quantity reported in the RMSE tables, ρ̂_bank = Σ w_k ρ̂_k, is a
   different functional of the same weights. The regret bound says nothing directly about its RMSE.
   §5.7.3 should only link them under the stated conditions.
2. **The non-finite-loss guard** (a non-finite loss is replaced by the worst finite loss that tick)
   changes the loss sequence. Gaussian MoM densities with |r| ≤ 0.999 and u, v ∈ (0, 1) are always
   finite, so it never fires in the forgetting banks. For the family bank it would make the bound
   hold for the guarded losses.
3. **Own-density banks score with proper densities on [0,1]² that are not exact copulas:**
   - Hermite's series is clipped at 0 and mixed with independence at weight 0.01, so it does not
     integrate to 1 in general;
   - Xiao's pooled histogram has only approximately uniform margins.
   The algebraic regret bound still holds for these scores. Proposition 5.6.5's KL interpretation
   does not.
4. **The covered banks are not the tuned ones:**
   - the squared bank is tuned to η = 1, outside Cor. 5.7.1;
   - the reported log bank's ρ̂ is a point functional.
   Chapter 6 should therefore report their regret as empirical (the table above), separately from the
   covered configurations.

## 16. Changes needed in Chapters 4–6 (supersedes the Part I §10 list where they overlap)

**Chapter 4:**
- **4.2.1:**
  - The EW marginal estimator: target F_n; Û_n = (B_{n−1} + w)/(W_{n−1} + 2w); weighted GK with
    forward decay and rescaling; tested accuracy ≤ ε = 0.005; observed storage about 290 tuples.
  - "Matching ESS is not matching a window."
  - Replace the Gregory sentence (`budget_mpl.md` §23).
- **4.2.2–4.2.3:** the full RMPL recursion and the normalised gain (Part I §5).
- **4.3:** cut-points from a 100-observation calibration sample, then fixed.
- **4.4:** sequential standardisation (Part I §4), and what Â_n then estimates.

**Chapter 5:**
- **5.1:** normalised EW weights; marginal forgetting (λ_m, shared scoring margins) against dependence
  forgetting (λ_k); native Xiao and Hermite marginals forget at λ_k.
- **5.1.2:** the normalised recursion replaces â₁ = h(X₁).
- **5.1.3:** Fisher scoring with gain 1/W_n as the one-step approximation to arg max L^EW_n.
- **5.2 (currently empty):** define the common VFF. State that one controller drives every estimator
  and that its target is the §5.4.4 ideal estimator, so the VFF realises the online version of §5.4.3.
- **5.3, Algorithm 1:** shared EW margins (λ_m); score only arms with a prediction; update E_k with
  (Û, V̂) (RMPL) or (X, Y) natively (Xiao, Hermite); update the margins with λ_m.
- **5.4:** Proposition 5.4.1 and §5.4.4 are idealised: a linear EW average with true PITs and
  independent contributions. Fill the "todo: xiao, hermite, mpl" as "no theory; checked empirically
  (Chapter 6)".
- **5.6.2:** ε_n includes the summary error.
- **5.7:** say which banks are covered (the §15 table), and keep the bounds as stated.

**Chapter 6:**
- **6.1:**
  - Experimental design: the frozen configuration table (Part I §1 and §13).
  - Seed protocol: dev 1–30; evaluation 42–101; budget validation 1000–1039 and test 42–141.
  - Platform: Linux/libstdc++.
  - Scoring vs training inputs.
- **Main bank results:** use the final500 numbers (§14), with final1999 as sensitivity.
- **VFF:** replace every VFF statement with §14's. The bank wins only after abrupt changes; under
  smooth change the common VFF ties or slightly beats it.
- **Regret:** report the covered banks' bound checks and the uncovered banks' empirical regret
  separately (§15).
- **Family and tail sections:** §17.
- **Budget study:** §18.

## 17. Family bank and tail dependence [confirmed] (seeds 42–101; shared EW margins, N_eff,m = 500)

**Before vs final, paired on the same seeds:**

| | Before (window) | Final |
|---|---|---|
| Family bank vs log banks, ρ_S whole (W/L/T of 19: Xiao / Hermite / MPL) | 10/8/1, 11/6/2, 8/8/3 | 8/6/5, 11/6/2, 9/8/2 |
| Family bank vs best single arm, ρ_S whole | 5/14/0 | 5/14/0 |
| Excess log loss vs Xiao / Hermite / MPL log banks (19) | 15/2/2, 17/1/1, 16/3/0 | 15/3/1, 16/1/2, 15/3/1 |
| χ_L(0.05) vs EW ratio estimators M = 99/399/1999 (19) | 19/0/0 each | 19/0/0 each |
| χ_U(0.05) vs the same | 19/0/0 each | 19/0/0, 19/0/0, 18/0/1 |

**Shape-switch detection** (median ticks to 50% weight, all 60/60 seeds), before → final:

| Switch | Before | Final |
|---|---|---|
| Gauss → t | 180 | 174 |
| Clayton → Gumbel | 78 | 68 |
| Gauss → Clayton | 98 | 94 |

**Tail bank** (`probe_tail_bank`, 5 regimes). W/L/T of the family bank's whole-run χ RMSE against
each counter, over 20 cells:

| | Before | Final |
|---|---|---|
| `tail_chi_399` | 20/0/0 | 20/0/0 |
| `tail_chi_1999` | 16/4/0 | 16/4/0 |
| `tail_chi_4999` | 16/2/2 | 16/2/2 |
| Hermite own bank (provisional: not a normalised density, not retuned; excluded from conclusions) | 16/4/0 | 16/4/0 |

- **The losses are the Clayton–Gumbel mixture.** The family bank loses there to the long counters
  (+11 to +21), as before: the misspecification result stands.
- **Halfway crossing after a switch:**
  - family bank 51–180 ticks;
  - `tail_chi_399` 102–187;
  - `tail_chi_1999` 580–806;
  - `tail_chi_4999` 1353–1982.
- **The Xiao and Hermite density competitors** now train on the shared scores their densities are
  evaluated at. The Xiao ones improve markedly: Xiao M = 399 now crosses halfway in 100–600 ticks,
  where before it often never crossed. So that comparison is now fair.
- **The own-density banks in these tables** (`*_bank_own`) use the old (η, δ) = (0.25, 3·10⁻³) and
  now forced-shared arms. They were not retuned, so report them as secondary.

## 18. Memory-budget study, rerun [confirmed] (`./probe_budget sweep 100 40`: test 42–141, validation 1000–1039)

**What changed:** Hermite now uses normalised start-up and no output hold; RMPL uses the normalised
start-up gain; the window ablation is replaced by `mpl_ew`; memory accounting as in `budget_mpl.md` §24.

**Results:**
- **Winners:** DataSketches-MPL has the lowest validation-tuned test RMSE of {Xiao, Hermite, MPL} in
  **10/12 cells at 16 KiB** (was 9/12) and **9/12 at 64 and 256 KiB** (unchanged). Own-KLL MPL:
  6/12 at 16 KiB, 10/12 at 64 and 256 KiB.
- **Changes by estimator** (×10⁻²):
  - Xiao: unchanged;
  - Hermite: −0.02 on average (−0.06 to +0.03);
  - MPL: −0.02 to −0.03 on average (−0.15 to +0.08).
- **Marginal ablation** (MPL, test RMSE ×10⁻²): exact PIT, EW N_eff 500, KLL 64/256,
  DataSketches 64/256, CopulaFit. Full table in `results/budget_tables.txt`.
  - EW 500 is close to exact on Gaussian drift and jump (6.32 against 6.23; 7.89 against 7.87).
  - It is worse on constant (3.16 against 2.72).
  - It is *better* than exact on misspecified drift and jump (Clayton drift 5.21 against 5.64).
  - The likely cause: EW ranks cannot produce extreme scores (Û ∈ [≈1/500, 1 − 1/500]), which makes
    the Gaussian score robust when the copula is wrong. This is a side effect, not a reason to prefer
    EW in the budget study.
- **Timing** (`results/budget_timing.csv`): Hermite is about 30–40% slower per tick than before,
  from the moment bookkeeping; the others are within ±10%. Figures regenerated
  (`ch6_budget_*.png`).

## 19. Remaining discrepancies and not done

- **Own-density banks:** not retuned or rerun as a main experiment. Their tail-probe rows use the old
  (η, δ).
- **Real-data experiments (AAPL/GOOG):** not rerun in this pass. They still need the EW margins and
  the native/shared protocol.
- **Figures from the bank runs** (`plot_thesis_bank.py`, `plot_neff_optimum.py`, family and tail
  figures) still read the old untagged files.
  - Regenerate them from `results/thesis_bank_final500*`, and the family and tail files, which were
    overwritten in place by the final run. Point the scripts at the tagged files.
  - `docs/chapter6_results.md` and `notes/thesis/chapter6_draft.tex` still quote the old numbers.
- **Family-bank arms** (`CopulaFit`): Fisher scoring with accumulated information,
  H_n = λH_{n−1} + I(θ_{n−1}), H_0 = prior, θ_n = θ_{n−1} + s_n/H_n, with a step clamp. With constant
  information this is the gain 1/W_n plus a prior mass. It is consistent with §5.1.3's derivation,
  but it is not literally RMPL; describe it as such.
- **Budget study margins:** RMPL there uses KLL margins (cumulative, no forgetting); Xiao and Hermite
  use their own procedures. This is the matched-memory question, not the bank protocol, so it is
  intentionally different. State it in 6.1.
- **Memory-tradeoff study** (`memory_tradeoff.md`): still uses estimator-internal `rank_pit_window`.
  It is not a final experiment.
- **Live consumer** (`device_state.hpp`): uses the same estimator classes, so it inherits the Hermite
  fixes. Its `hermite_ew` key keeps `burn_in = 50`.

## 20. Reproduction (Part II)

```bash
cd experiments && make -B probe_thesis_bank probe_bank_regret probe_bank_tune probe_family_bank \
  probe_tail_bank probe_vff_common probe_budget EIGEN=/usr
# dev: VFF validation and tuning (seeds 1-30)
for f in gaussian clayton gumbel t; do SEED_BASE=1 COPULA=$f ./probe_vff_common 30; COPULA=$f ./probe_bank_tune 1 30; done
python3 tune_bank_table.py 1
# evaluation (seeds 42-101); scratchpad/final.sh runs all of it
for N in 500 1999; do for f in gaussian clayton gumbel t; do
  BANK_MARGINAL_NEFF=$N SEED_BASE=42 COPULA=$f COMPONENTS=xiao,hermite,mle TAG=final$N ./probe_thesis_bank 60
  BANK_MARGINAL_NEFF=$N COPULA=$f TAG=final$N ./probe_bank_regret 42 60; done; done
for f in gaussian clayton gumbel t; do COPULA=$f TAG=final500 ./probe_bank_regret 42 60 family; done
for c in gaussian clayton gumbel t gauss2t clayton2gumbel gauss2clayton; do COPULA=$c ./probe_family_bank 42 60; done
for r in gauss2clayton clayton2gumbel gauss2t mix gaussian; do ./probe_tail_bank $r 42 60; done
./probe_budget sweep 100 40 && ./probe_budget time
python3 conclusions_diag.py final500 final1999; python3 compare_diag.py final500 final1999
python3 family_bank_table.py 42; python3 tail_bank_table.py; python3 analyze_budget.py results/budget_sweep.csv
```

**Pre-change results** for the paired comparisons are in `experiments/results/pre_audit_2026-11-09/`.
All final tables are collected in `experiments/results/final_eval_2026-11-09.txt`. `results/` is
gitignored: force-add the `.txt` file when committing.

---

# Part III: review and reconciliation (2026-11-09, read-only; the frozen estimator is unchanged)

**Archive:** `audit_archive_2026-11-09/` and `.tar.gz`, with SHA-256 per file. Restore verified: a
clean clone of `33f51fd`, plus the patch and the untracked tarball, reproduces `src/`, the experiment
code and the notes.

**Code fingerprint:** `96f6727b67e98cf8`. No source file changed after the final runs began.

**Overwritten results:** listed in `MANIFEST.md`. The shape-switch and family-path files and the
budget mean paths have no paired "before".

Checks in this part are marked:
- **[verified]**: run or read against the code in this review;
- **[claim]**: not re-checked.

## 21. Definitive implementation (frozen configuration)

**Common to all streams:**
- N = 5000.
- Scenarios constant, drift, jump at 2500, and oscillate (period 400).
- Families Gaussian, Clayton, Gumbel, t₄. Data are the copula's normal scores.

### 21.1 Main rank-correlation banks (`<est>_bank`, `_bank_log`, `_bank_sq32`, `_bank_uniform`; `probe_thesis_bank`)

| | Xiao arms | Hermite arms | RMPL arms |
|---|---|---|---|
| **Training input** | raw (X, Y) | raw (X, Y) | z = Φ⁻¹(Û), Φ⁻¹(V̂) |
| **Scoring input** | shared (Û_n, V̂_n) | shared | shared |
| **Shared marginal** | EW ECDF, weighted GK ε = 0.005, N_eff,m = 500 (1999 sensitivity), λ_m = (N_eff,m − 1)/(N_eff,m + 1); Û = (B_{n−1} + w)/(W_{n−1} + 2w), before the update | same | same |
| **Own marginal handling** | cut-points: empirical quantiles of the first 100 raw observations, then fixed; ranks from the EW row and column totals (forget at the arm's λ) | sequential EW standardisation (below), forgets at the arm's λ | none beyond the shared one; ExactNormal returns u = Φ(z) = Û (Acklam error < 1.2·10⁻⁹); clip [10⁻³, 1 − 10⁻³] |
| **Forgetting (5 arms)** | retention λ ∈ {0.95, 0.98, 0.99, 0.995, 0.999} | update weight γ ∈ {0.05, 0.02, 0.01, 0.005, 0.001} | gain γ ∈ the same set |
| **Memory** | N_eff ∈ {39, 99, 199, 399, 1999} | same | same |
| **Initialisation and gain** | counts M_0 = 0. The 100 calibration observations are inserted in arrival order once the cut-points exist. EW weights (1−λ)λ^{n−i}; the ratio estimator is scale-invariant, so it equals normalised weighting | a = A = 0; coefficient gain w_m = γ/W_m, W_m = (1−γ)W_{m−1} + γ, W_0 = 0 (normalised) | r_0 = 0; g_m = γ/W_m (same W recursion, m = number of copula updates); projection r ∈ [−0.999, 0.999] |
| **Warm-up / ready** | ready once calibrated (after observation 100) | ready once one coefficient update has happened (observation 2) | first copula update at observation 31; ready afterwards |
| **Output** | weighted-rank Pearson correlation of the count matrix | ρ̂ = 12(u − z/2)ᵀA(v − z/2), u = W·a_X | ρ̂ = (6/π) arcsin(r/2) |

**Hermite sequential standardisation (raw inputs, `moments_step`)**, for observation n:
- c_n = γ/W^m_n, with W^m_n = (1−γ)W^m_{n−1} + γ and W^m_0 = 0;
- d = X_n − μ_{n−1};
- μ_n = μ_{n−1} + c_n d;
- σ²_n = (1−c_n)(σ²_{n−1} + c_n d²), with σ²_1 = 0;
- s_n = (X_n − μ_n)/√max(σ²_n, 10⁻¹²). This is the current-observation-included convention of
  `hermiter`.
- Coefficients update only once the moments hold at least two observations.
- **Safeguard:** only the 10⁻¹² variance floor.
- **Artefact:** as γ → 0 the first absorbed observation has s₂ = ±1 exactly, because μ₂ is the
  midpoint of X₁ and X₂ and σ₂ is half their distance. Normalised weighting removes its influence
  at rate 1/W_m.

**Bank (Algorithm 1)** [verified]:
- **Scoring gate:** score only when the margins hold at least 30 observations **and every arm is
  `ready()`**. The gate is bank-wide, not per arm.
- **First scored observation:** 101 for the Xiao banks, 31 for Hermite, 32 for RMPL.
- **Losses:**
  - squared: (p̂_n − ρ̂_{k,n−1})², with p̂_n = 12(Û_n − ½)(V̂_n − ½);
  - log: −log c^{Ga}_{g⁻¹(ρ̂_{k,n−1})}(Û_n, V̂_n), with g⁻¹ clamped to |r| ≤ 0.999.
- **Guard:** a non-finite loss becomes the tick's worst finite loss. It never fires for the Gaussian
  density.
- **Weights:** Hedge w̃ ∝ w e^{−ηℓ}, then fixed share (1−δ)w̃ + δ/K. Then the arms update, then the
  margins.
- **Output:** Σ_k w_k ρ̂_k after the update.
- **Settings (η, δ):**
  - squared: (1, 10⁻⁴);
  - log: (1, 3·10⁻⁴);
  - sq32: (1/32, 10⁻⁴);
  - uniform: (0, 0).
- **Evaluation:** RMSE and excess log loss over observation indices [500, 5000). Excess log loss is
  log c_true − log c^{Ga}_{g⁻¹(ρ̂)} at the **true** PIT.

### 21.2 VFF comparators (`vff.hpp`, rates 0.02 and 0.1)

The arm configuration is as in §21.1 at an initial N_eff of 199. The shared margins are owned by
the wrapper.

**Controller** (state θ, m, w, ṁ, ẇ, v, all 0 except θ₀ = log 199):
- It observes p̂_n once the wrapper's margins hold more than 30 observations, i.e. from observation
  32.
- **Step:**
  - e = p̂_n − m_{n−1};
  - g = −2e·ṁ_{n−1}·dλ/dθ, with dλ/dθ = 2N/(N+1)²;
  - v ← v + 0.01(g² − v);
  - θ ← clip(θ − r·g/√v, [log 39, log 1999]).
- **Then, with λ_n = (N_n − 1)/(N_n + 1):**
  - w_n = λ_n w_{n−1} + 1;
  - ẇ_n = λ_n ẇ_{n−1} + w_{n−1};
  - ṁ_n = ṁ_{n−1}(1 − 1/w_n) − e·ẇ_n/w_n²;
  - m_n = m_{n−1} + e/w_n.
- **The derivative recursion treats past λ as equal to the current one** (Bodenham & Adams'
  approximation). It is exact for constant λ: checked against a finite difference.

**Per observation:**
1. The prediction ρ̂_{n−1} is scored externally.
2. The margins transform (X_n, Y_n).
3. The controller steps.
4. `set_neff(N_n)`:
   - Xiao: λ = (N−1)/(N+1);
   - Hermite: γ = 2/(N+1), for the coefficients and the moments;
   - RMPL: γ = 2/(N+1).
5. The estimator updates.

**Normalised weights under time-varying memory** [verified to 10⁻¹²]:
- With W_n = (1−γ_n)W_{n−1} + γ_n and gain γ_n/W_n, observation i has weight
  γ_i Π_{j>i}(1−γ_j)/W_n, and these sum to one exactly.
- Xiao's weights (1−λ_i)Π_{j>i}λ_j are normalised implicitly by its ratio estimator.

### 21.3 Other studies

| Study | Training inputs | Scoring | Marginal | Arms / estimator | Loss, η, δ | Readiness and evaluation |
|---|---|---|---|---|---|---|
| **Own-density banks** (`*_bank_own`) | shared normal scores (forced) | shared (Û, V̂) | shared EW, N_eff,m = 500 | Xiao: cut-points Φ⁻¹(k/B); Hermite: no standardisation; RMPL as §21.1 | −log of the arm's own score; (0.25, 3·10⁻³), **old, not retuned** | bank-wide gate; **provisional** |
| **Family bank** (`family_bank`) | shared (Û, V̂), rotated by `base_uv` | shared | shared EW, 500 | 8 shapes × M ∈ {99, 399, 1999}; `CopulaFit` with expected information: φ₀ = 0 (Gauss, t) or log 0.5 (Clayton, Gumbel); H₀ = 1; H_n = λH_{n−1} + I_φ(φ_{n−1}); φ_n = φ_{n−1} + clip(s_φ/H_n, ±0.5); λ = 1 − 2/(M+1) | −log c_arm(Û, V̂) (proper parametric densities); η = 1, δ = δ* = 2.1·10⁻⁴ | margins ≥ 30 (no arm gate: arms predict from φ₀); evaluation [500, 5000) |
| **Tail counters** (`tail_chi_M`) | shared (Û, V̂) | none | shared EW, 500 | EW corner frequencies a, b with update weight 2/(M+1); χ̂ = b/a; M ∈ {399, 1999, 4999} | none | margins ≥ 30 |
| **Budget study** (`probe_budget`) | raw normal scores | none | Xiao: own cut-points (300 observations); Hermite: own standardisation; MPL: KLL or DataSketches (cumulative), ablation `mpl_ew` | single estimators, grid × gain g ∈ {0.005, …, 0.10}; Hermite and RMPL with normalised start-up | none | RMSE every 5th tick from 500; tuned on validation within the memory cap |

## 22. Corrected VFF interpretation

**Statement, replacing "ties under smooth change"** [verified]:
- **Slow VFF (0.02) against the log bank:**
  - constant: 3W/3L/6T;
  - drift: 5W/0L/7T, better in 5 of 12 cells and never worse;
  - oscillate: 9W/2L/1T, better in most cells;
  - jump: 0W/12L, +28.8.
- **Fast VFF (0.1):** worse in all but one cell.
- **So:** under smooth or periodic change the slow VFF is at least competitive with the bank and
  often better. After abrupt change it is clearly worse.

**Interpretation:**
- The controller imposes a **common forgetting schedule**. It is tuned to the prequential loss of the
  ideal EW-mean estimator, not to each estimator's own loss.
- Remaining differences between the VFF estimators therefore reflect:
  - each estimator's efficiency;
  - how well a schedule chosen for the ideal estimator suits that estimator;
  - the estimator's own internal forgetting: Xiao's calibration and frozen cut-points, Hermite's
    moments, RMPL's nonlinear recursion and start-up.
  They cannot be attributed to efficiency alone.
- **The drift memory** (N ≈ 280–300, against the formula's ≈ 315 for σ² ≈ 1) shows the controller
  approximately optimises the intended objective. It is **not an independent verification** of the
  theory: the controller and the formula target the same idealised loss under the same assumptions.

## 23. Predictive-density validity [verified by grid integration on [0,1]², 600²]

| Output | Scale | Construction | Integral | Status |
|---|---|---|---|---|
| Gaussian MoM, c^{Ga}_{g⁻¹(ρ̂)} | copula, (u, v) | closed form; \|r\| ≤ 0.999 | 1.0000 | **valid density** |
| Family-bank arm, c_θ (rotations) | copula | parametric; rotation Jacobian 1; parameter clamps | (parametric) | **valid density** |
| Xiao own (shared training) | (u, v) on [0,1]² | pooled 5×5 histogram, cells k/D exact; prior mass ∝ cell area: (n + a·du·dv)/(N + a)/(du·dv) | 1.000000 | **valid density on [0,1]²**; margins only approximately uniform, so not an exact copula |
| RMPL own | copula | Gaussian copula at clip(u, δ), clip(v, δ) | 1.00003 | valid up to the δ-clip (corner mass mis-assigned, O(δ)) |
| **Hermite own** | (u, v), via z = Φ⁻¹(u) | f_s = h(s_x)ᵀA h(s_y) on the standardised scale; Jacobians 1/(σ_xσ_y) and 1/(φ(z_x)φ(z_y)); **negative part clipped at 0**; mixed with independence at 0.01 | **1.071** | **not a density**: clipping adds mass and ∫f_s ≠ 1 under truncation. A diagnostic score only |

**Consequences:**
- **Hermite own-density "log loss" is not a proper scoring rule** applied to a forecast. Neither its
  excess log loss nor the own-density bank's regret can be read as predictive-density evidence.
- **Results affected by the own-density experiments not being rerun:**
  - the own-density bank rows of `ch6_family_summary.png` (a) and (b);
  - the Hermite own-bank rows of the tail tables;
  - the 6.5.3 own-density results in the superseded `chapter6_results.md`.
  - Valid comparisons need a normalised Hermite density (for example, renormalised on a grid) and a
    retune. That is an estimator change, so it was not done.
- **The main-bank log loss is unaffected.** It uses the valid Gaussian MoM density.

**"Oracle", clarified:**
- Training on data whose margins happen to be N(0,1) is not an oracle.
- The problem is **explicitly using Φ as the marginal CDF** when an arm trained on raw data is scored
  at a shared pseudo-observation:
  - Xiao maps u to its raw-scale cut-points through Φ;
  - Hermite treats Φ⁻¹(u) as its raw input.
- That evaluation is correct only because the simulated marginal is Φ: the known marginal law is
  used. Hence own-density arms train on the shared scores, where no marginal law is needed.

## 24. Weighting and RMPL, corrected

**Exact weights [verified]:**
- **Normalised scheme** (λ = retention):
  - a_{n,i} = λ^{n−i}/W_n, with W_n = (1−λ^n)/(1−λ);
  - ESS_n = (Σa)²/Σa² = [(1+λ)/(1−λ)]·(1−λⁿ)/(1+λⁿ) → N_eff = (1+λ)/(1−λ).
- **This is not the Chapter 5 first-observation initialisation.** That one has a_{n,1} = λ^{n−1} and
  a_{n,i} = (1−λ)λ^{n−i} for i ≥ 2, with ESS_n = 1/[λ^{2(n−1)} + ((1−λ)/(1+λ))(1 − λ^{2(n−1)})].

| λ, n | ESS normalised | ESS Ch5 literal | Weight on X₁ (normalised / literal) |
|---|---|---|---|
| 0.99, 100 | 92.4 | 7.1 | 0.006 / 0.370 |
| 0.999, 500 | 489.8 | 2.7 | 0.0015 / 0.607 |
| 0.999, 2000 | 1522.8 | 53.2 | 0.0002 / 0.135 |

The two coincide only as n → ∞.

**RMPL derivation, corrected:**
- With normalised weights,
  L_n(r) = (λW_{n−1}/W_n)·L_{n−1}(r) + (1/W_n)·ℓ_n(r), where ℓ_n(r) = log c_r(Û_n, V̂_n).
- So **∇L_n(r_{n−1}) = (λW_{n−1}/W_n)·∇L_{n−1}(r_{n−1}) + (1/W_n)·s_n(r_{n−1})**.
- The first term **vanishes only if r_{n−1} is an interior stationary point of L_{n−1}**. It does
  not vanish if r_{n−1} only approximates the maximiser, or sits at the projection boundary.
- The recursion **drops** that term, so the omission error accumulates. The recursion is a
  stochastic-approximation step, not exact maximisation.
- **Curvature:** −∇²L_n(r_{n−1}) = Σ_i a_{n,i}·(−∂²ℓ_i(r_{n−1})) is replaced by the Gaussian-model
  expected Fisher information I(r_{n−1}) = (1+r²)/(1−r²)². This is the same for every observation,
  so with weights summing to one the Hessian approximation is I(r_{n−1}).
- **This is Fisher scoring.** It is exact in expectation only under the Gaussian copula at
  r_{n−1}. Under misspecification (Clayton, Gumbel, t) the observed curvature differs, as in the
  sandwich form.
- **Result:** r_n = Π[r_{n−1} + (1/W_n)·I(r_{n−1})⁻¹·s_n(r_{n−1})], with 1/W_n = (1−λ)/(1−λⁿ).
- **Withdrawn:** the "prior mass at r = 0" interpretation of the constant gain, which was
  unsupported. The constant gain is simply the n → ∞ form, started from r₀ = 0.

## 25. Regret report, tightened [verified]

**Replay:** the probe's weight replay equals `ForgettingBank`'s weights exactly (max |Δw| = 0 over a
5000-tick Clayton jump run, for all three log banks). The replayed bank loss is the loss of the
**pre-update** prediction, using weights from n−1, and the aggregation is Hedge then fixed share, as
in Algorithm 1.

**Denominators:** 1440 runs = 60 seeds × 4 families × 2 scenarios (drift, jump) × 3 estimators. The
family bank has 480. **Each run is checked at 8 switch budgets** m ∈ {0, 1, 2, 3, 5, 10, 20, 50}, so
11,520 and 3,840 are repeated, dependent checks, not independent runs.

| Procedure | Theorem conditions | Runs exceeding at any m | Per-m checks |
|---|---|---|---|
| `log_mix` (mixture of Gaussian MoM densities), η = 1, δ = 3·10⁻⁴ | **covered** (mixable log loss, fixed share) | 0/1440 | 0 |
| `log_mix_hedge`, δ = 0 | covered, **m = 0 only** (bound infinite for m ≥ 1) | 0/1440 | 0 |
| `sq32`, η = 1/32 (\|y−z\| ≤ 4, so exp-concave) | **covered** (bound ×32) | 0/1440 | 0 |
| `sq32_dstar` | covered | 0/1440 | 0 |
| Family `log_mix`, η = 1, δ* | covered (proper densities) | 0/480 | 0 |
| `sq`, η = 1 (deployed squared bank) | **not covered** (η > 1/32) | **730/1440 (50.7%)** exceed the reference bound | 725 at m = 0, 29 at m = 1, 4 at m = 2 |
| `log_point` (Gaussian density at Σwρ̂) | **not covered** (not the mixture) | **333/1440 (23.1%)** | most at m ∈ {0, 2, 3} |

N_eff,m = 1999 gives the same picture: covered 0; sq 712 (49.4%); log_point 402 (27.9%).

**For the uncovered procedures** these are exceedances of a reference expression, not violations of
an applicable theorem.

**Corrected relative to Part II:** "about 5–7% violations" divided per-run counts by the per-check
denominator. Per run, the uncovered procedures exceed the reference in about a quarter (log_point)
and a half (sq) of runs.

## 26. Errata for Parts I and II

1. **Part II §12:** the drift-memory agreement was called "an independent check of §5.4.3". Withdrawn
   (§22).
2. **Part II §12 and §14:** "differ only by how efficiently each uses it" and "a well-defined VFF
   matches the bank under smooth change". Replaced by §22.
3. **Part I §5 and §8:** the normalised weighting was described as "the weights every recursion uses /
   reduces to (5.1.1)". It equals (5.1.1) with S₀ = 0 rescaled by 1/(1−λⁿ). It is **not** the
   Chapter 5 â₁ = h(X₁) scheme (§24). The RMPL derivation dropped the previous-gradient term without
   saying so, and the "prior mass" remark is withdrawn (§24).
4. **Part II §15 and the report:** per-check violation rates were presented as run rates. Corrected
   in §25.
5. **Part II §13 and §15:** "the algebraic regret bound still holds for these scores" for the Hermite
   own score. True as algebra, but the score is not a density (§23), so it supports no
   predictive-density claim.
6. **Part II §18:** "Û ∈ [≈1/500, 1 − 1/500]". Corrected to Û ≥ ≈1/252, i.e. |z| ≤ ≈2.65 at
   N_eff,m = 500. The extreme-score explanation of the EW-vs-exact ablation is a **hypothesis**. The
   old window ablation (|z| ≤ 2.88) gives mixed evidence and does not isolate it.
7. **Seeds 42–101:** held out from *this audit's* decisions only. They were used in earlier studies,
   including the pre-audit Chapter 6 results and a tuning check (`bank_tune_42`).
8. **Exact-PIT ablation:** a diagnostic, not a guaranteed upper bound. Ranks under misspecification
   can do better.

## 27. Verified versus not re-checked

**Verified in this review:**
- archive restore and checksums;
- no source change after the freeze;
- the bank-wide scoring gate, and first scored observations 101 / 31 / 32;
- the tuning and regret probes use the same gate (code, and an exact weight replay);
- density integrals;
- normalised weights under varying memory;
- exact weights and ESS formulas;
- regret denominators and per-run exceedances;
- figures regenerated from the final files.

**Verified earlier in the audit:** EW-marginal accuracy, theory checks for the Hermite and RMPL
recursions, regression against HEAD, the TSan race fix, the VFF derivative.

**Not re-checked:**
- the `hermiter` and `onlineNPCORR` parity claims (submodules absent);
- the KLL and DataSketches guarantees;
- the family-bank rotation formulas and χ(q) tables beyond existing tests (`check_thesis_bank`
  passes);
- that `probe_tail_bank`'s competitors match their registry keys beyond the configurations shown.

## 28. Thesis statements requiring revision (consolidated; supersedes Part I §10 and Part II §16)

**Chapter 4:**
- **4.2.1:**
  - Add the EW empirical marginal CDF: target, pre-update pseudo-observation, weighted-GK
    representation with forward decay and rescaling.
  - State its tested accuracy (≤ ε = 0.005 against the exact EW ECDF) and observed storage (about 290
    tuples). Say that weighted-GK guarantees are not claimed.
  - Replace the Gregory sentence.
- **4.2.2–4.2.3:** the RMPL recursion with
  - clip δ = 10⁻³;
  - 30-observation warm-up;
  - projection ±0.999;
  - gain 1/W_n;
  - ρ̂ = (6/π) arcsin(r/2);
  - the derivation of §24, keeping the dropped term and the Fisher-curvature approximation explicit.
- **4.3:** cut-points are the empirical quantiles of a calibration sample (100 observations in the
  banks, 300 in the budget study), then fixed. Ranks come from the EW row and column totals.
- **4.4:**
  - Define sequential standardisation (§21.1 equations, including the current-observation
    convention and the variance floor).
  - Â_n averages Hermite functions of observations standardised by *evolving* moments; it is not the
    expansion of today's standardised history.
  - Coefficient updates start at the second observation.

**Chapter 5:**
- **5.1:**
  - Define normalised weights a_{n,i} = λ^{n−i}/W_n and ESS_n (§24).
  - Separate marginal forgetting (λ_m, the shared scoring margins) from dependence forgetting (λ_k).
  - Native Xiao and Hermite marginal information forgets at λ_k.
- **5.1.2:** replace â₁ = h(X₁), â_n = λâ_{n−1} + (1−λ)h(X_n) with the normalised recursion (gain 1/W_n),
  and give the reason with the ESS comparison.
- **5.1.3:** complete it (§24): the exact normalised objective, the gradient identity with both
  terms, the condition under which the first term vanishes, Fisher scoring, and the projection.
- **5.2:**
  - Define the common VFF: §21.2 controller equations, initialisation, clipping, timing, and the
    per-estimator mapping.
  - Present it as a *common forgetting schedule* targeting the ideal estimator's loss.
  - State the derivative approximation for time-varying λ.
- **5.3, Algorithm 1:**
  - line 5: shared EW margins (λ_m);
  - line 7: score only when every arm is ready (bank-wide gate);
  - line 12: E_k is updated with (Û, V̂) (RMPL) or natively with (X, Y) (Xiao, Hermite);
  - line 13: the margins are updated with λ_m.
  - The bank's reported ρ̂ is Σ_k w_k ρ̂_k. The log-loss forecaster covered by the theory is the
    mixture density.
- **5.4:**
  - Proposition 5.4.1 and §5.4.4 apply to the ideal linear EW mean with true PITs and independent
    contributions.
  - The implemented estimators are nonlinear, and estimated shared margins make contributions
    dependent, so the result is an idealised guide.
  - Replace "todo: xiao, hermite, mpl" with that statement.
- **5.6.2:** ε_n includes the summary approximation error.
- **5.6.5:** the KL interpretation needs normalised densities. It does not apply to the Hermite own
  score.
- **5.7:** list the covered procedures:
  - mixture log loss, η = 1;
  - squared loss, η ≤ 1/32 (|y − z| ≤ 4);
  - the family-bank mixture.
  State that the deployed squared bank (η = 1) and the point-density log bank are outside the
  theorems.

**Chapter 6:**
- **6.1:**
  - the configuration tables (§21);
  - the seed protocol (dev 1–30; evaluation 42–101, held out from this audit but used in earlier
    studies; budget 1000–1039 and 42–141);
  - W/L/T orientation and criterion (2 paired SE over 60 seeds);
  - the scoring start (observations 101 / 31 / 32) and evaluation window [500, 5000);
  - Linux/libstdc++.
- **Main-bank results:** `docs/chapter6_results_final.md` (N_eff,m = 500, with 1999 as sensitivity).
- **VFF:** §22 wording.
- **Regret:** §25. Covered procedures are theorem checks; uncovered ones are empirical exceedance
  rates, per run, with the per-m breakdown.
- **Family and tail:** final tables. Own-density rows are provisional or removed.
- **Budget:** final tables; the exact-PIT ablation is a diagnostic, and the EW-vs-exact explanation a
  hypothesis.
- **Stale material:** remove or regenerate the figures listed in `chapter6_results_final.md`.

## 29. Additional experiments: needed or optional (none run in this review)

| Experiment | Purpose | Needed? | Cost |
|---|---|---|---|
| **Normalised Hermite density + own-density bank retune (seeds 1–30) and rerun (42–101)** | A valid density forecast for the own-density bank, the family-bank comparison and the tail rows. Requires renormalising the Hermite score (an estimator change), so it needs approval and a fresh holdout (e.g. 200–259) if chosen after viewing 42–101 | Needed if 6.5.3 or own-density rows stay in the thesis; otherwise drop them | about 15 min |
| `probe_bank_weights` (drift) | Regenerate `ch6_bank_weights_neff*.png` and `ch6_neff_optimum.png` panel (c) | Needed if those figures stay | about 2 min |
| `probe_bank_path`, `probe_family_path` | Regenerate the single-seed path figures (`ch6_bank_path*`, `ch6_family_path`) | Needed if those figures stay | a few min |
| `probe_ise` / ISE figures, `probe_tail_ci` | Regenerate the ISE and tail-CI figures | Only if kept | about 10–30 min |
| AAPL/GOOG with EW margins and native/shared protocol | Real-data section consistent with the final method | Needed for §6.7 | depends on the data pipeline; about 10 min compute |
| Exact-PIT with clipped scores (\|z\| ≤ 2.65) in the budget ablation | Test the extreme-score hypothesis | Optional (it only explains an ablation) | about 5 min |
| Family-bank shape-switch "before" | A paired before/after for the switch table | Not possible without re-running pre-audit code; aggregate comparison only | about 20 min with the restored pre-audit tree |

## 30. Reporting conventions and corrections (review of Part III)

All of these are applied in `docs/chapter6_results_final.md`, the results inventory for Chapter 6.

1. **Parenthesised values:** the unweighted mean across a row's cells of each cell's mean paired
   difference A − B, in units of 10⁻³ (RMSE of ρ_S). Negative values favour A.
   **W/L/T:** per cell, a paired difference beyond ±2 SE over 60 seeds, with no multiplicity
   correction. [verified: `conclusions_diag.py`, `family_bank_table.py`, `tail_bank_table.py`]
2. **Hindsight-best fixed memory and best single arm:** chosen once per cell, by mean RMSE across the
   60 seeds; per-seed differences are then taken against that arm. Not a per-seed oracle. [verified,
   same scripts]
3. **Fast VFF:** worse in 47 of 48 cells, with one tie (oscillation). This replaces "worse
   everywhere".
4. **Budget study:** "lowest in 10/12" is a ranking.
   - Conservative check: the DataSketches-MPL lead exceeds 2 SE with unpaired SEs in quadrature in
     7/12 cells (16 KiB) and 6/12 (64 and 256 KiB). Its two worse cells are also unresolved.
   - "All statistics are paired" does not apply to the budget study, which saves no per-seed output.
     The statement is qualified.
5. **Regret at m = 0:** for δ > 0 the check uses the fixed-share bound at zero switches,
   log K + (T−1)·log[1/(1−δ+δ/K)], not log K. Only δ = 0 Hedge reduces to log K.
   - Zero exceedances is implementation evidence. The guarantee comes from the assumptions. [verified:
     `probe_bank_regret.cpp::penalty`]
6. **Family bank:**
   - It is worse than its best single arm in 14/19 ρ_S cells, but improves density forecasts and
     represents shape and tail changes. Interpretation: useful for dependence *structure*, not a
     uniform improvement in rank correlation.
   - "Ticks to 50% weight" is the first crossing of the **new family's total weight** (summed over its
     three memory arms), per seed, median over seeds. It is a weight-crossing delay, not a calibrated
     detection delay: on the no-switch path some wrong family crossed 50% in 14/60 seeds.
     [verified: `probe_family_bank.cpp` lines 166–182]
7. **Provisional rows excluded from headline counts:**
   - Verified that the reported family W/L/T (against the log banks, the best arm, the ratio
     estimators) and the tail W/L/T (against the corner counters) contain no own-density rows. Each
     comparator is counted separately.
   - The §17 Hermite-own row is now marked provisional.
   - The stale ISE and real-data figures are excluded from conclusions.

## 31. Matched dependence paths: rerun (2026-11-10). The primary protocol for Chapter 6

**Why:** unrotated Clayton and Gumbel cannot represent negative dependence. On the original drift
($d$: −0.5 → 0.8) and oscillation (±0.6), the generator projected negative targets to independence,
so their true paths were flat at $\rho_S \approx 0$ over 31.6% of the evaluated drift (and half of
each oscillation cycle). Cross-family comparisons on those scenarios were therefore confounded.
Verified with 2·10⁵-draw Monte Carlo per point.

**Change:** `experiments/scenarios.hpp`, `SCENARIOS=matched`:
- drift $d = 0.2 \to 0.8$, linear in $d$ over observations 1..N;
- oscillation $d = 0.5 + 0.3\sin(2\pi t/400)$;
- constant and jump unchanged;
- the default (`original`) still reproduces the frozen outputs byte for byte (checked).

**Harness additions:**
- `TAG` outputs for the budget, tuning, family-bank, VFF, memory-sweep and bank-weights probes;
- per-seed budget output (`budget_seeds[_TAG].csv`);
- `BANK_LOG_DELTA` (default 3·10⁻⁴);
- `plot_neff_optimum.py` now follows `SCENARIOS`. It had the drift path hard-coded in three
  places, the same defect as `fit_mstar.py` (not used by any figure).

**Generator verification:** achieved $\rho_S$ equals the target (to 10⁻¹⁶). The sample Spearman
check is within ≤ 0.0032 for all four families over $d \in [0.2, 0.8]$.

**Development re-selection (seeds 1–30), before evaluation:**
- squared bank (1, 10⁻⁴), unchanged;
- **log bank (1, 10⁻⁴)**, changed from 3·10⁻⁴ (flat surface; −6.06 against −5.89);
- VFF 0.02 and 0.1, unchanged by the fixed rule in `select_vff_rates.py`. The rule reproduces the
  original choice on the original dev runs.

**Matched code fingerprint:** `fe4dc392d4f5e518`.

**Evaluation:** main banks and VFF ($N_{\mathrm{eff},m}$ = 500 and 1999), regret, family-bank
regret, family bank (4 base families), memory sweep, bank weights, budget (val/test, per-seed). Tail
bank and shape switches are unaffected and not rerun.

**Results:** see `docs/chapter6_results_matched.md`.
- **Unchanged qualitatively:** bank vs fixed memory, log vs squared (now 23/0/1), uniform, covered
  regret 0, family bank.
- **Changed:**
  - slow VFF: ties on drift (1/0/11); worse on constant (0/7/5);
  - point-density log bank exceeds the reference bound in 42.7% of runs (smaller $\delta$);
  - budget comparisons are now paired: 9/2/1 at every budget.
- **Optimal-memory check on the matched drift:** measured N\* 393 and 430, against predicted 426 and
  448.

**Original results are preserved:** untagged and `final*` files, `chapter6_results_final.md`,
`figures/`. Matched results are in `*matched*` files, `matched_eval_2026-11-10.txt` and
`figures/matched/`.

**Signed dependence:** the original drift for Gaussian and $t_4$ (in `final500`) serves as the
separate signed-dependence experiment.

## 32. Full-copula density prediction at constant ρ_S: audit and targeted run (2026-11-10)

**Question:** does modelling the full copula improve density (and tail) prediction when Spearman
estimation is already accurate?

**What existed** (`probe_family_bank`, seeds 42–101, the same streams for every method):
- Spearman RMSE for every bank and arm.
- Excess log loss at the true PIT for:
  - the family bank's **mixture** density;
  - the Gaussian log banks' **point** density (Gaussian at the combined ρ̂);
  - the own-density banks (provisional).
- χ(0.05) RMSE for the family bank and the EW ratio estimators only.
- Windows: pre, and 0–100, 100–500 and 500–1500 after the switch, and post.

Re-tabulated by `experiments/density_audit.py`.

**Missing, now added** (harness only, `DENSITY_AUDIT=1`, `TAG=density`, seeds 42–101, nothing tuned):
- `gauss_bank`: the family-bank machinery restricted to the Gaussian shape (memories 99, 399, 1999;
  η = 1; δ* = K/((K−1)(T−1)) = 3.02·10⁻⁴ for K = 3). It isolates "modelling the family" from the
  estimator and the weighting.
- `mle_bank_log_mix`: the MPL log bank's mixture of its arms' Gaussian densities; χ = Σ w_k χ_k, which
  is exact because χ is linear in C.
- `mle_bank_log_point`: its point density at the combined ρ̂, with χ of that Gaussian. It equals the
  existing `mle_bank_log` excess log loss exactly (checked).
- New regimes `gauss2t3` and `gauss2t8`: the true copula switches to t₃ or t₈ (truth density and tails
  use that ν) while the family bank's t arms assume ν = 4.

**Checks:**
- the default run reproduces the existing output exactly;
- the t₃ and t₈ generators reach the target ρ_S (sample Spearman within ≈2 MC SE).

Tables: `experiments/density_objectives.py` → `results/density_objectives_2026-11-10.txt`.

**Results [confirmed]** (family bank minus the Gaussian-only bank, paired, ×10⁻³; excess log loss in
10⁻³ nats per observation):

| Regime | Window | Spearman RMSE | Excess log loss | χ_L | χ_U |
|---|---|---|---|---|---|
| any Gaussian start | pre | +1.4 | +0.5 | +2.3 | +5.5 |
| Gauss → t₄ | post | +3.3 | **−30.7** | −56.6 | −57.8 |
| Gauss → t₃ (arms assume t₄) | post | +8.2 | **−49.5** | −78.1 | −79.9 |
| Gauss → t₈ (arms assume t₄) | post | +5.2 | **−3.9** | −4.3 | −3.2 (tie) |
| Clayton → Gumbel | pre (Clayton) | +1.3 (tie) | **−75.3** | −271.9 | −197.5 |
| Clayton → Gumbel | 0–100 after | +18.2 | **+46.4** | +164.8 | +78.4 |
| Clayton → Gumbel | post | +9.9 | **−23.8** | −37.7 | −132.4 |
| Gauss → Clayton | post | +12.9 | **−72.2** | −228.6 | −165.1 |

**Findings:**
- **Full-copula modelling greatly improves density and tail prediction after a structural change at
  constant ρ_S, while Spearman estimation gets slightly worse.** The objectives differ, and the gain
  is not visible in rank correlation.
- **When the truth is Gaussian there is a small cost of flexibility:** +0.5 in log loss, +2 to +6 in χ.
- **Right after Clayton → Gumbel the family bank is temporarily much worse,** because the weights
  still favour Clayton. That is the structural-adaptation lag.
- **Gaussian log bank, mixture vs point density:** the mixture is consistently slightly better (−0.2 to
  −1.3), so the family-vs-Gaussian conclusion holds for either.
- **t misspecification:** with t₃ truth, the t₄ arms still capture most of the gain. With t₈ truth
  (near Gaussian) the gain is small. The family bank does not need the exact ν to help, but the
  benefit scales with how far the truth is from Gaussian.
- **Three metrics, three roles:** Spearman RMSE (dependence strength), excess log loss (full density),
  χ RMSE (joint extremes). They can rank the same methods differently (above), and log loss does not
  replace a tail metric.

**KL projection of the Clayton–Gumbel mixture onto single families** (`experiments/kl_projection_mixture.cpp`,
numerical; components calibrated to ρ_S = 0.582):

| π (Clayton weight) | Mixture χ_L / χ_U | Best single family (KL*, nats) | Its χ_L / χ_U | Gaussian KL* |
|---|---|---|---|---|
| 0.75 | 0.518 / 0.215 | Clayton (0.013) | 0.573 / 0.105 | 0.037 |
| 0.50 | 0.420 / 0.316 | t₄ (0.012) | 0.393 / 0.393 | 0.017 |
| 0.25 | 0.322 / 0.418 | t₄ (0.009); Gumbel 0.011 | 0.393 / 0.393 | 0.014 |

- **No single family reproduces both tails.**
  - The Clayton-dominated mixture projects to Clayton: χ_U is halved and χ_L overstated.
  - Balanced or Gumbel-leaning mixtures project to the symmetric t₄, which equalises the tails.
  - Projected parameters also shift ρ_S (for example Clayton θ* gives ρ_S 0.54), so a KL projection
    does not preserve rank correlation.
- **This explains the mixture result in the tail study** (the family bank loses to long corner
  counters on χ). Its predictive density *is* a mixture Σ w_k c_k, but Hedge on log loss concentrates
  weight on the single best arm rather than learning mixture proportions. The arms are fixed shapes,
  so the bank's limit is roughly the best single-family projection.
- **Remedy:** an explicit mixture arm with a learned π. Methodology, not run.

## 33. Density follow-up: Gaussian comparators, best fixed arm, Gaussian floor, Clayton–Gumbel mixture (2026-11-10)

**Scope** (approved by Sally, 2026-11-10): A1–A3, R1b–R1c and R2 of the Chapter 2 session's request. No
estimator, registry config or bank setting changed; only outputs and one new regime in
`probe_family_bank` (`DENSITY_AUDIT=2`), an analysis script and a quadrature header.

**Configuration:**
- `SCENARIOS=matched BANK_LOG_DELTA=1e-4 DENSITY_AUDIT=2 TAG=density2 ./probe_family_bank <s> 15`,
  s ∈ {42, 57, 72, 87} (seeds 42–101), for COPULA ∈ {gaussian, clayton, gumbel, t} (constant, drift,
  jump, oscillate) and {gauss2t, clayton2gumbel, gauss2clayton, claygum_mix} (constant ρ_S of d = 0.6,
  switch at 2500); T = 5000, burn 500.
- Code fingerprint `54185f0f6c977037` (sha256 of the git tree `HEAD:src` plus `probe_family_bank.cpp`,
  `gauss_floor.hpp`, `scenarios.hpp`, `copula_source.hpp`, `Makefile`); `src/` unmodified.
- Linux x86_64, libstdc++, clang++ 18 `-O2 -std=c++20`.
- Tables: `experiments/density_followup.py check|a1|a2|r1|r2` →
  `experiments/results/density_followup_2026-11-10.txt`. Units ×10⁻³; mean (SE) of paired per-seed
  differences over 60 seeds; * marks |mean| > 2 SE.

**Verification:**
- Base families: all 95,040 rows of the matched run are reproduced exactly.
- Switches: every row of the §32 run is reproduced exactly except the log-bank rows. §32 ran the log
  bank at δ = 3·10⁻⁴; this run uses the matched δ = 10⁻⁴. The shifts are small: Gaussian mixture or
  single Gaussian excess changes by −0.02 to −0.72, ρ_S RMSE by −0.9 to −4.0.
- One R2 entry was recomputed independently from the raw CSV and matches.

**Density forecasts compared** (all on the same seed, tick and true-PIT point):

| Label | What it is |
|---|---|
| family-bank mixture | `FamilyBank::log_density`, Σ_k w_k c_k over 24 arms (8 shapes × M 99/399/1999) |
| single Gaussian | MPL log bank: one Gaussian copula at the combined ρ̂ (`mle_bank_log`) |
| Gaussian mixture (log w) | MPL log bank's own weights and arms, Σ_k w_k c_G(2 sin(πρ̂_k/6)) |
| Gaussian mixture (own w) | `mle_bank_own`: own-density weights, η 0.25, δ 3·10⁻³ (old, not retuned; provisional) |
| Gaussian-only family bank | family-bank machinery with the Gaussian shape only (`gauss_bank`) |
| best fixed arm | the family-bank arm with the lowest mean whole-window excess, chosen per cell in hindsight |
| Gaussian floor | KL(c_true ‖ c_G(r*)), r* = E[Z₁Z₂]; expected D(t) and realised log c_true − log c_G(r*) |

**Floor method (R1c):**
- The KL-closest Gaussian copula has r* = E_c[Z₁Z₂]: the derivative of E log c_G(r) vanishes at r = E[Z₁Z₂]
  because E Z₁² = E Z₂² = 1.
- Hence D = E_c log c + ½ log(1 − r*²). Computed by trapezoid quadrature in normal-score space (401²,
  |z| ≤ 7), tabulated on a 0.01 grid in d and interpolated.
- **Checks:**
  - Gaussian: 0;
  - t₄ at ρ_S 0.582: 0.0339 (`probe_kl_shape` 0.034);
  - mixture r* 0.597 → 0.602 and D 0.0372 → 0.0138 (matching the relayed reference and §32's KL table);
  - n = 201/401/801 agree to 5 digits;
  - Monte Carlo with 2·10⁷ draws: Clayton 0.07813 (SE 0.00009) vs 0.07823, Gumbel 0.02894 vs 0.02902.
- **The realised floor on the probe's streams agrees with D** (z across 49 cell-windows: mean −0.5,
  SD 0.96).
  - Exception: Clayton truth, where it is about 2·10⁻³ below D (z down to −2.9). Those cells share seeds,
    and the log ratio is right-skewed, so this is attributed to sampling.
  - The decomposition uses the realised floor (same observations), so it holds exactly either way.

**A1: family bank minus the Gaussian comparators** (19 cells; W/L/T by the 2-SE rule; *r* = per-seed
correlation of Δρ_S RMSE and Δexcess). Excess wins mean better (lower) density loss.

| Comparator | Window | ρ_S RMSE W/L/T | Excess W/L/T | r median [range] |
|---|---|---|---|---|
| single Gaussian | whole (19) | 8/8/3 | 15/4/0 | 0.13 [−0.38, 0.67] |
| single Gaussian | post (7) | 2/4/1 | 6/1/0 | −0.08 [−0.30, 0.27] |
| single Gaussian | 0–100 after (7) | 1/5/1 | 5/2/0 | 0.28 [−0.60, 0.59] |
| Gaussian mixture (own w) | whole (19) | 15/4/0 | 17/2/0 | 0.18 [−0.33, 0.45] |
| Gaussian mixture (own w) | post (7) | 4/2/1 | 6/1/0 | −0.03 [−0.33, 0.28] |

- **The four excess losses against the single Gaussian** are all Gaussian-truth cells: constant, drift,
  jump and oscillate, by +0.2 to +9.7.
- **Within a cell the two differences are nearly uncorrelated across seeds.** Better density prediction
  on a seed says little about better ρ_S on that seed.

**A2** (identical to the matched evaluation):
- Gauss → t₄: t50 174 [130, 247].
- Clayton → Gumbel: t50 68 [45, 143].
- Gauss → Clayton: t50 94 [80, 124].
- 60/60 seeds reach 50% and 90% in each switch.
- Null paths, seeds with any false 50% crossing: Gaussian 14/60, Clayton 2/60, Gumbel 9/60, t 6/60.

**A3:** `hermite_bank_own` is excluded from every density table here. Its own score integrates to
about 1.07 over [0,1]², so it is not a density, and its log loss is not comparable.

**R1b: structural advantage vs adaptive weighting** (family-bank mixture minus the best fixed arm):

| Situation | FB − best fixed arm (excess) | Reading |
|---|---|---|
| Stationary, true family in the bank (constant, drift; 8 cells) | +0.3 to +0.9 | The cost of adaptivity is small: the bank is within 1·10⁻³ of the best fixed (true-family) arm. |
| Jump in ρ_S, whole window (4 cells) | −0.2 (tie, Gaussian), −0.8 to −1.2 | Slightly better than any fixed arm. |
| Jump, 0–100 after | +18 to +61, except Gumbel −72 | The best arm is usually the M = 99 arm, which adapts faster. |
| Family switches, whole window | −7.7 (Gauss → t), −41.0 (Clayton → Gumbel), −24.9 (Gauss → Clayton) | No single fixed arm is good on both sides of a switch. |
| Family switches, post | +2.6 / −18.9 / +3.5 | After switching, the bank pays a few 10⁻³ against the hindsight-best long-memory arm. Against Clayton → Gumbel's best compromise arm (t₄, M 1999) it wins. |

- **Most of the family bank's advantage over the Gaussian forecasts is structural.** The arm with the
  true family, used alone, is already within 1·10⁻³ of the bank. Hedge's weighting adds the ability to
  change family, which no fixed arm has.

**R1c: decomposition of the Gaussian forecasts' excess:**
- **Under non-Gaussian truth, the floor is almost all of it** (×10⁻³, D per tick):
  - Clayton 70–78, Gumbel 25–29, t₄ 32.6–33.9;
  - jump post: Clayton 177, Gumbel 50, t 41.
- **Estimation is +1 to +3.5 in stationary windows.** Over the oscillation's whole window it is +23 to +36
  (tracking lag). After a jump in ρ_S it is +12 to +17 over the post window and +16 to +27 at 100–500.
- **Mixture vs single Gaussian:** the log-weight Gaussian mixture is 0.0–4.5 better than the single
  Gaussian in whole and post windows (up to 12.9 in t-jump 0–100). The own-weight mixture is worse in stationary cells (old η, δ).
- **Interpretation:** no Gaussian forecaster, however well weighted, can close the gap to a
  non-Gaussian truth. The family bank's −23 to −178 (non-Gaussian cells, whole or post) comes almost entirely from removing the floor.

**R2: Clayton–Gumbel mixture** (π(Clayton) 0.75 → 0.25 at 2500, ρ_S constant 0.582; the truth is not in the
family bank). Reported as found, without a prior favourite. Excess log loss by window:

| Window | Floor D | Family mix | Single Gauss | Gauss mix (log w) | Gauss-only FB | Best fixed arm (t₄, 1999) | FB − single | FB − best arm |
|---|---|---|---|---|---|---|---|---|
| whole | 24.2 | 11.7 | 24.8 | 24.6 | 24.5 | 19.0 | −13.0 (0.6) | −7.3 (0.5) |
| pre | 37.2 | 13.9 | 37.5 | 37.2 | 37.0 | 30.9 | −23.6 (1.1) | −17.0 (1.1) |
| 0–100 after | 13.8 | 33.7 | 14.2 | 14.1 | 14.0 | 6.7 | +19.5 (3.7) | +27.0 (2.7) |
| 100–500 after | 13.8 | 9.4 | 14.5 | 14.3 | 14.4 | 9.0 | −5.2 (1.4) | +0.4 (0.7) |
| post | 13.8 | 10.1 | 14.6 | 14.5 | 14.5 | 9.5 | −4.6 (0.6) | +0.6 (0.2) |

ρ_S RMSE, family bank minus comparator:
- MPL log bank: whole +4.6 (1.4), pre +6.5, 0–100 +13.6, post +2.5 (tie);
- best arm: +9 to +16 throughout.

χ RMSE at q = 0.05, family bank minus EW counter:

| Window | χ_L vs M 99 / 399 / 1999 | χ_U vs M 99 / 399 / 1999 |
|---|---|---|
| whole | −141 / −31 / −3 (tie) | −127 / −22 / +3 (tie) |
| pre | −148 / −32 / +14 | −85 / +13 / +53 |
| 0–100 after | −83 / tie / tie | −44 / tie / tie |
| post | −135 / −29 / −12 | −164 / −53 / −36 |

- **Tails at q = 0.10** follow the same pattern.
- **Gaussian forecasts' χ_L RMSE** (post): 14, against the family bank's 78. The Gaussian χ_L(0.05) at
  r ≈ 0.6 happens to sit near the post-switch truth 0.322. Their χ_U is 95, against 62.

Shape weights (seed means):
- Pre: Clayton 0.81, t 0.13.
- Post: t 0.64, Gumbel 0.21, Gaussian 0.12.
- These match the KL projections in §32 (Clayton best at π = 0.75, t₄ at π = 0.25).
- Gumbel's summed weight crosses 50% at some point after the switch in 52/60 seeds (median 492 ticks),
  but its mean post-switch weight is 0.21. The crossing is often transient.

**R2 findings:**
- **Density:** the family-bank mixture beats every Gaussian forecast before and after the change. It
  loses for about 100 ticks after the change, the same structural lag as Clayton → Gumbel.
  - It beats the best fixed arm over the whole window, mainly before the change.
  - After the change it is level with or slightly behind t₄ at M 1999 (+0.6 post).
- **Tails:** before the change, the family bank's χ_U is worse than the long counters' (+13 against
  M 399, +53 against M 1999). It has projected onto Clayton, which understates the upper tail (§32: χ_U
  0.105 vs the truth's 0.215).
  - After the change it is better than every counter in both tails. The t₄ projection
    (χ_L = χ_U = 0.393) sits between the truth's 0.322 and 0.418.
- **Does better density prediction come with worse tail estimation?** Partly, and only before the change:
  - pre: the family bank has the best density but worse χ_U than long counters (and worse χ_L than
    M 1999);
  - post: better on both.
  - So a better density does not imply better tails when the truth is outside the bank. What
    determines the tails is which single family the bank projects onto.
- **ρ_S:** the family bank is worse than the Gaussian log bank before and right after the change, and
  level after. Same direction as the switch regimes.

**Caveats:**
- The own-weight Gaussian mixture uses the old (η, δ) and is provisional.
- Each best fixed arm is chosen in hindsight per cell, so it is an oracle comparator.
- One ρ_S level and one π pair. ν mismatch, other π and real data are out of scope (§32 has t₃/t₈).
