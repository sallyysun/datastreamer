# Chapter 6 evaluation metrics: audit, feasibility and protocol (2026-11-10)

§§1–7 are the read-only audit; §§8–11 are Step 1 (self-checks); §12 records the settled framework and
decisions; §13 is Step 2 (Hellinger). No estimator code was changed.

**Settled framework (2026-11-10, Sally):**

| Metric | Purpose | Priority |
|---|---|---|
| Spearman RMSE | Accuracy of estimated dependence strength | Primary |
| Excess copula log loss | Accuracy of predictive copula densities against the known truth | Primary (the full-copula density metric) |
| Squared Hellinger distance | Overall discrepancy between estimated and true copula densities | Secondary (a check, not a replacement) |
| Tail RMSE (χ_L, χ_U) | Lower- and upper-tail probabilities | Primary in tail experiments |
| Recovery time, regret, memory, runtime | Adaptation and computation | Experiment-specific |

Density ISE is dropped from the experimental results; its finiteness analysis (§3, §8D) is kept as a
technical note.

**Question:** distinguish accurate estimation of dependence *strength* (ρ_S) from accurate estimation of
the full dependence *structure* (the copula density).

**Ground rules:**
- Chapter 6 evaluates against the simulation truth, not against the Chapter 5 online weight-update losses.
- A ρ_S estimate does not define a copula density. A Gaussian copula built from a Xiao or Hermite ρ̂ is
  a *working-model* density, not that estimator's own.

---

## 1. Audit: which estimators produce a valid predictive copula density

"Valid" means non-negative, integrates to 1 on [0,1]², and has an explicit formula. "Copula" means, in
addition, exactly uniform margins.

| Estimator (as run in Ch. 6) | Own density in code | Formula | Non-negative | Normalised | Exact copula | Status for density metrics |
|---|---|---|---|---|---|---|
| **RMPL** (fixed memory, VFF inner, budget MPL with KLL margins) | `RecursiveMPL::log_predictive` | Gaussian copula c_{r̂}(u,v) at its fitted r̂ | yes | yes, except for the scoring guard below | yes | **Valid.** Evaluate the unclamped c_{r̂}. `log_predictive` clamps u, v to [δ, 1−δ] (δ = 10⁻³) as a scoring guard, and the clamped function is not exactly normalised. |
| **Xiao** (raw inputs, calibrated cut-points) | `XiaoSpearmanEstimator::log_predictive` | Histogram on its calibrated cells pooled to D × D (D = 5; D = B when B is not divisible by 5); 0.5 prior pseudo-observations per pooled cell, spread by area | yes (min ≥ 0.05 in the fixtures) | yes, exactly by construction | **no**: margins deviate from uniform by up to 0.07–0.48 (L1 0.03–0.21) in the §8 fixtures | **A valid bivariate density on [0,1]², not a copula density.** Piecewise constant and coarse. Cell edges are Φ(cut-point), which is exact only under N(0,1) margins. Not to be presented as a copula. |
| **Hermite** | `HermiteEstimator::log_predictive` | (1−e)·max(f̂, 0)/(sd_x sd_y φ(z_x) φ(z_y)) + e, with f̂ = h(s_x)ᵀA h(s_y) and e = 0.01 | only through the clip: the series f̂ is negative on 2–19% of the square (§8) | **no**: 1.02–1.25 in the §8 fixtures (1.01–1.18 earlier) | **no**: margin L1 deviation 0.05–0.51; unbounded near the boundary (margin values up to ~10⁵ at \|z\| = 8) | **Not a density as output, and not a copula.** Renormalising in evaluation code fixes the mass only: margins stay non-uniform (L1 0.06–0.40) and the boundary blow-up remains. Excluded from density metrics. |
| **ForgettingBank, `Loss::Squared` / `Loss::Log`** | `log_density` | Gaussian copula at the **combined** ρ̂ = Σ w_k ρ̂_k | yes | yes | yes | **Valid density, but a working model.** Legitimate as the "single Gaussian" forecast for RMPL banks. For Xiao/Hermite banks it is not their own density and must not be labelled so. |
| Same banks, mixture form | not in code (the density follow-up computed it for the MPL log bank) | Σ_k w_k ĉ_k with each arm's own density ĉ_k | yes | yes, if every ĉ_k is | RMPL: yes (mixture of copulas). Xiao: no | RMPL banks: valid now. Xiao banks: valid after the evaluation code is written. Hermite banks: valid only after per-arm normalisation. |
| **ForgettingBank, `Loss::Own`** (`*_bank_own`) | `log_density` | Σ_k w_k ĉ_k (own densities) | yes | RMPL and Xiao yes; **Hermite no** | as above | Provisional: old (η, δ), not retuned. `hermite_bank_own` is excluded. |
| **Common VFF** (`CommonVff`) | none: the wrapper does not expose the inner estimator's `PredictiveDensity` | the inner estimator's density at the controller's memory | as inner | as inner | as inner | Needs a **harness accessor** to the inner estimator. No algorithm change. |
| **FamilyBank** | `log_density` | Σ_k w_k c_k over 24 parametric arms (8 shapes × M ∈ {99, 399, 1999}) | yes | yes | **yes** (a mixture of copulas is a copula) | **Valid.** Arm densities are closed form. |
| Family-bank arms, Gaussian-only bank | `arm_log_density`, `gauss_bank` | parametric c_{θ̂_k} | yes | yes | yes | Valid. |

**Not densities:** EW corner counters (χ only), VFF controllers, batch Spearman comparators.

---

## 2. Audit: which metrics already exist

| Metric | Where | Definition in code | Prequential? | PIT | Matched / final config? |
|---|---|---|---|---|---|
| ρ_S RMSE | every probe | √(mean_t (ρ̂_{t−1} − ρ_S(t))²) per seed and window | yes (estimate held before tick t) | none needed; estimators use their own inputs | **yes**: all E1–E5 outputs |
| Excess log loss | `probe_thesis_bank` | log c_t(U_t,V_t) − log c^{Ga}_{g⁻¹(ρ̂_{t−1})}(U_t,V_t): Gaussian **method-of-moments density at each method's ρ̂**, for every method including Xiao and Hermite | yes | **true** U_t = Φ(X_t) | yes, but for Xiao/Hermite it is a working-model density, so do not present it as their own |
| Excess log loss | `probe_family_bank` (`density`, `density2`) | the same formula with: family-bank mixture; single Gaussian and Gaussian mixture (MPL log bank); own banks; 24 arms; Gaussian-only bank; floor | yes | true | **yes** (`density2`, seeds 42–101, matched) |
| Excess log loss (own) | `probe_bank_tune` | own-density banks, `excess_own_x1000` | yes | true | dev seeds only (tuning) |
| Density ISE, normal-score scale | `probe_ise` | ∫∫(ĉ−c)² φ(z_u)φ(z_v) du dv on a 32 × 32 midpoint grid, checkpoints every 100 ticks | evaluated after updates up to the checkpoint | grid in u; ĉ in the estimator's own coordinates | **no**: pre-audit, stale (figures `ch6_ise_*`), uses the unnormalised Hermite |
| Copula CDF ISE | `probe_ise`, `probe_batch_ise` | ∫∫(Ĉ−C)², midpoint 32² or Gauss–Legendre 12², every 50/100 ticks | as above | as above | **no**: pre-audit, stale |
| Squared Hellinger | — | — | — | — | **missing everywhere** |
| Gaussian floor D(t) | `probe_family_bank` (`DENSITY_AUDIT=2`) | KL(c_t ‖ c_{r*}) by quadrature | per tick | — | yes (`density2`) |
| χ_L, χ_U RMSE; t50/t90; regret; memory; update time | family/tail/regret/budget probes | as documented | — | true PIT for the counters | yes |

---

## 3. Is density ISE finite?

**Gaussian copula.** ∫∫c_r² du dv = 1/(1−r²), finite for |r| < 1. So ISE between Gaussian copulas, or
between a Gaussian truth and any bounded estimate (Xiao's histogram), is finite.

**Tail-dependent copulas (Clayton lower, Gumbel upper, t both):**
- Near a tail-dependent corner the density is homogeneous of degree −1: c(su, sv) ≈ s⁻¹ g(u, v) as s → 0.
  For Clayton, c ≈ (1+θ)(uv)^{−1−θ}(u^{−θ} + v^{−θ})^{−2−1/θ}; on the diagonal c(s,s) ≈ (1+θ)2^{−2−1/θ}/s.
- Then c² is homogeneous of degree −2 in two dimensions, so ∫∫c² diverges logarithmically, for every θ
  (every tested ρ_S).
- Write ISE(c, ĉ) = ∫∫c² − 2∫∫cĉ + ∫∫ĉ². It is finite only if the leading corner terms cancel, i.e. ĉ has
  exactly the same tail profile g. Any estimate with a different tail parameter, or no tail
  dependence at all, gives **ISE = +∞**.

**Numerical confirmation** (quadrature in normal-score space over [ε, 1−ε]², ρ_S = 0.582, the d = 0.6
level used throughout). Scratch program `ise_feas.cpp` in the session scratchpad, built on
`experiments/gauss_floor.hpp`:

| ∫∫_{[ε,1−ε]²} c² | ε = 10⁻² | 10⁻³ | 10⁻⁴ | 10⁻⁶ | Limit |
|---|---|---|---|---|---|
| Gaussian (r = 0.600) | 1.323 | 1.485 | 1.537 | 1.560 | 1/(1−r²) = 1.5625 ✓ |
| Clayton (θ = 1.419) | 1.609 | 2.141 | 2.647 | 3.656 | ∞ (≈ +0.5 per decade of ε) |
| Gumbel (θ = 1.712) | 1.468 | 1.832 | 2.152 | 2.776 | ∞ |
| t₄ (r = 0.614) | 1.477 | 1.804 | 2.074 | 2.578 | ∞ |

| Truncated ISE(c, ĉ): truth vs the same family with ρ_S 5% higher | ε = 10⁻² | 10⁻³ | 10⁻⁴ | 10⁻⁶ |
|---|---|---|---|---|
| Gaussian | 0.0021 | 0.0032 | 0.0039 | 0.0046 (converging) |
| Clayton | 0.0052 | 0.0088 | 0.0125 | 0.0197 (growing like log 1/ε) |
| Gumbel | 0.0028 | 0.0044 | 0.0058 | 0.0085 |
| t₄ | 0.0024 | 0.0037 | 0.0050 | 0.0075 |

**Gaussian truth vs the family bank:**
- Fixed share keeps every arm, including the tail-dependent ones, at weight ≥ δ/K > 0, so ∫∫ĉ² = ∞.
- Demonstration: truth Gaussian, ĉ = 0.99·Gaussian + 0.01·Clayton. Truncated ISE grows 0.00002 →
  0.00010 → 0.00019 → 0.00029 → 0.00039 for ε = 10⁻² … 10⁻¹⁰. Each decade of ε adds the same amount, so
  the integral diverges, slowly.

**Where full copula-scale ISE is finite** among the tested configurations: Gaussian truth against
RMPL, Gaussian mixtures and Xiao. Hermite is finite if the standardised scale is not too wide
(roughly ŝ² < 2). It is **infinite** for every Clayton, Gumbel and t cell, every shape switch, the mixture
regime, and the family bank in every cell.

**Recommendation (revised in §8–9):** keep full-domain copula ISE as one of the four measures. Report its value where it is finite (Gaussian truth with a Gaussian-type estimate) and "∞ (not defined)" elsewhere. Never replace it silently. Two precisely defined quantities can appear *separately*, labelled, if wanted:

- **(a) Normal-score ISE (optional supplement):**
  ISE_z = ∫∫_{ℝ²}(f̂ − f)² dx dy = ∫∫_{[0,1]²}(ĉ − c)² φ(Φ⁻¹u) φ(Φ⁻¹v) du dv,
  where f is the density of (Φ⁻¹U, Φ⁻¹V).
  - The weight vanishes like s·√(2 log 1/s) at the corners, which kills the s⁻² singularity, so ISE_z
    is finite for every tested pair.
  - It is what the stale `probe_ise` already computes.
  - It is scale-dependent (it changes with the marginal scale chosen) and weights the centre heavily.
- **(b) Truncated ISE on [ε, 1−ε]²** with ε stated (e.g. 0.01). Defined, but its value moves strongly
  with ε in tail-dependent cells (above), so cross-family comparisons depend on an arbitrary choice.
  If used, call it "interior ISE (ε = 0.01)", never "ISE".

**Stability comparison** (normal-score quadrature on |z| ≤ 8, 401²; same comparisons as above):

| Pair | ISE_z | Squared Hellinger H² | KL(c ‖ ĉ) |
|---|---|---|---|
| Gaussian vs 5% off | 0.000096 | 0.000374 | 0.00153 |
| Clayton vs 5% off | 0.000178 | 0.000480 | 0.00196 |
| Gumbel vs 5% off | 0.000118 | 0.000335 | 0.00136 |
| t₄ vs 5% off | 0.000108 | 0.000272 | 0.00110 |
| Clayton truth vs Gaussian at the same ρ_S | 0.00352 | 0.0203 | 0.0783 (≈ the floor 0.0782 ✓) |

Check: the Gaussian row agrees with the closed forms, H² = 0.000374 and KL = 0.001532.

- **Squared Hellinger**, H² = 1 − ∫∫√(c ĉ):
  - bounded in [0, 1], symmetric;
  - **invariant to monotone marginal transforms**, so it is the same on the u-scale and the z-scale;
  - the integrand √(cĉ) is integrable whenever both are densities;
  - normal-score quadrature converges quickly;
  - closed form for two Gaussian copulas, a free check: H² = 1 − (1−r²)^{1/4}(1−s²)^{1/4}/√(1 − ((r+s)/2)²).
  - **Most stable.**
- **KL / expected excess log loss:**
  - invariant to transforms, finite for every tested pair because every estimator's density is
    bounded away from 0 (Xiao prior, Hermite floor e, parametric arms > 0);
  - asymmetric and tail-sensitive: it heavily penalises ĉ ≪ c in the corners;
  - its prequential (sample) version is right-skewed. In `density2`, the realised Clayton floor is
    about 2.5 SE below the quadrature value.
- **ISE_z:**
  - finite, but scale-dependent;
  - values are small and dominated by the centre, so it is the least sensitive to tails;
  - complementary to KL, not a substitute.

---

## 4. Common evaluation protocol

Notation:
- Seeds s = 42…101 (60), ticks t = 1…5000, burn 500.
- Windows: whole [500, 5000); pre [500, 2500); 0–100, 100–500 and 500–1500 after a change; post [2500, 5000).
- c_t is the true copula at tick t. ĉ_{t−1} is the estimator's density **held before** observation t.
- True PIT: U_t = Φ(X_t), V_t = Φ(Y_t) (the DGP margins are N(0,1)).

| Metric | Formula, per seed and window w | Estimator output | Before/after obs t | PIT | Computation | Uncertainty |
|---|---|---|---|---|---|---|
| **ρ_S RMSE** | √( \|w\|⁻¹ Σ_{t∈w}(ρ̂_{t−1} − ρ_S(t))² ) | ρ̂ | before | none | exact | paired per-seed difference A − B, mean ± SE (sd/√60); W/L/T at 2 SE per cell; no multiplicity correction (existing convention) |
| **Excess log loss** | \|w\|⁻¹ Σ_{t∈w}[log c_t(U_t,V_t) − log ĉ_{t−1}(U_t,V_t)]; its conditional expectation is KL(c_t ‖ ĉ_{t−1}) | a **valid normalised** density ĉ | before | **true** | the stream itself (Monte Carlo); no integration | as above |
| **Squared Hellinger** | \|C_w\|⁻¹ Σ_{t∈C_w}[1 − ∫∫√(c_t ĉ_{t−1}) du dv] over checkpoints C_w = {t ∈ w : (t − 500) mod Δ = 0}, Δ = 50 | valid normalised ĉ as a function | before | none (densities compared as functions on [0,1]²) | tensor trapezoid in normal-score space, \|z\| ≤ 7, n = 201 per axis (verified to 5 digits for the floor). Xiao's piecewise-constant ĉ: integrate cell by cell, or confirm grid convergence | as above |
| **Normal-score ISE** (if kept) | \|C_w\|⁻¹ Σ_{t∈C_w} ∫∫(ĉ_{t−1} − c_t)² φ(Φ⁻¹u) φ(Φ⁻¹v) du dv | as above | before | none | same grid and checkpoints | as above |

**Conventions to state once in the chapter:**
- **Coordinates.** Each estimator's density is expressed for its own pseudo-observations (shared EW
  margins, or its own standardisation). Comparing it with c_t at the true PIT, or as a function on
  [0,1]², charges the estimator for marginal error as well. This is the same convention as §32–33 of
  `ew_marginals.md`.
- **Valid densities only.** The single Gaussian, and the Gaussian at ρ̂ for Xiao/Hermite, are reported
  only as labelled working-model forecasts.
- **Averaging.** Average over ticks within a seed, then over seeds. Pair by seed: the same streams are
  shared by every method.

---

## 5. Mapping onto E1–E5, and what is missing

Key:
- ✅ available on matched/final outputs;
- 🟡 needs evaluation code only (no estimator change; at most a harness accessor);
- 🔴 needs an estimator change;
- — not recommended for that experiment.

| Experiment | Estimators | ρ_S RMSE | Excess log loss | H² | ISE_z | Experiment-specific (kept) |
|---|---|---|---|---|---|---|
| **E1** budgets | RMPL (DS-KLL margins), Xiao, Hermite | ✅ (`budget_seeds_matched`, paired) | — (optional 🟡: RMPL and Xiao own densities; Hermite after normalisation) | — | — | memory, update time, recovery |
| **E2** fixed forgetting over N_eff | Xiao, Hermite, RMPL fixed memories | ✅ | RMPL ✅ (`thesis_bank`: the Gaussian at ρ̂ is its own family, up to the clamp). Xiao 🟡 (own histogram). Hermite 🟡 (normalise) | 🟡 | 🟡 / — | N*(t), optimum memory |
| **E3** none vs fixed vs VFF vs bank | same three; common VFF; log bank | ✅ | RMPL fixed/bank ✅ (single Gaussian; mixture via the `density2` code). VFF 🟡 (accessor). Xiao/Hermite 🟡 | 🟡 | — | recovery windows, weight paths |
| **E4** Hedge vs fixed share; squared vs log | banks of all three | ✅ (sq vs log); δ = 0 vs δ > 0 at equal η on eval seeds needs a **run** (no code) | MPL log bank ✅ (single and mixture, `density2`). Squared bank 🟡 (same two constructions). Xiao/Hermite banks 🟡 | 🟡 (secondary) | — | regret ✅, weights ✅ |
| **E5** family bank, shape changes, tails | family bank, arms, Gaussian forecasts | ✅ | ✅ (`density2`: mixture, single and mixture Gaussian, best arm, floor) | 🟡 (closed-form arms) | 🟡 | χ RMSE ✅, t50/t90 ✅, false switches ✅ |

**No 🔴 cells.** Every recommended metric can be added without changing an estimator. The Hermite
correction and the VFF accessor live in evaluation and harness code. Existing outputs that must be
**relabelled, not rerun**: the `thesis_bank` excess log loss for Xiao and Hermite methods is a
Gaussian working-model forecast.

---

## 6. Minimal implementation and verification plan (needs authorisation; nothing done yet)

1. **`experiments/density_metrics.hpp`** (evaluation only): normal-score tensor grid (n = 201, |z| ≤ 7);
   H² and ISE_z for callables ĉ(u,v), c(u,v); Hermite normalisation Z = ∫∫max(f̂,0) from `log_predictive`.
   - **Checks before any run:**
     - closed-form Gaussian–Gaussian H² and KL to 10⁻⁵;
     - ∫∫ĉ = 1 ± 10⁻⁴ for Xiao, normalised Hermite and the mixtures;
     - grid convergence n = 101/201/401;
     - KL by quadrature vs prequential excess at a fixed state.
2. **E5 first** (cheapest, mostly done): `probe_family_bank` with `DENSITY_AUDIT=3` adds H² (and ISE_z)
   at checkpoints for the family-bank mixture, single and mixture Gaussian, best arm, and Gaussian-only
   bank. Reuse the `density2` configuration. It must reproduce every `density2` row exactly.
3. **E2–E4**: `probe_thesis_bank`, evaluation-only block, behind a switch:
   - own densities for fixed memories (RMPL Gaussian at r̂, Xiao histogram, normalised Hermite);
   - bank mixtures Σ w_k ĉ_k from the existing weights;
   - the squared bank's two Gaussian constructions;
   - VFF via an inner-estimator accessor;
   - excess log loss at the true PIT, plus H² at checkpoints.
   - Must reproduce the matched `thesis_bank` rows exactly.
   - Cost driver: the t density and the Hermite normalisation at each checkpoint. Use Δ = 100 for E2–E4
     if run time is a problem; the checkpoint count barely affects paired SEs.
4. **E4 δ comparison**: η fixed, δ ∈ {0, matched δ}, eval seeds. A run, no code.
5. **E1**: ρ_S RMSE, memory and time only, unless a density result is specifically wanted.

---

## 7. Recommended Chapter 6 presentation (concise)

- **E1–E4: ρ_S RMSE is the primary metric** (paired W/L/T plus mean differences), with the
  experiment-specific measures.
  - One sentence states that weight-update losses (Ch. 5) are training objectives, and every method
    is evaluated on the same truth-based metric.
- **One "full-structure" subsection**, mainly in E5, with **two** density metrics:
  - **excess log loss**: prequential, a proper scoring rule, equal to KL(c_t ‖ ĉ) in expectation,
    tail-sensitive;
  - **squared Hellinger**: truth-based, bounded, symmetric, scale-free.
  - Both computed only for valid densities: family bank, RMPL and the Gaussian forecasts, Xiao's
    histogram, and Hermite after verified normalisation.
  - The Gaussian floor D(t) explains why no Gaussian forecaster can match a non-Gaussian truth.
- **χ RMSE** stays the tail metric. Log loss does not replace it (§33: the mixture regime).
- **ISE:**
  - kept as the third measure: numbers where finite, "∞ (not defined)" where not (§9);
  - one paragraph explains the divergence (tail-dependent copulas; the family bank under fixed share);
  - ISE_z appears only as a labelled supplement, if at all.
- **Footnotes:**
  - `hermite_bank_own` is excluded (unnormalised);
  - Gaussian-at-ρ̂ densities for Xiao and Hermite are working-model forecasts, not their own.

---

## 8. Step 1: evaluation header and self-checks (2026-11-10)

**Code (evaluation only, uncommitted):**
- `experiments/density_metrics.hpp`: normal-score tensor grid, mass, H², copula-scale ISE (full grid or
  interior square), ISE_z, KL, margin check, and Gaussian closed forms.
- `experiments/check_density_metrics.cpp`: the checks below.

**What was run:**
- Parts A–D are deterministic quadratures of known densities.
- Part E builds one fixed Xiao or Hermite state per case from a single fixed stream (seed 42, 2000 ticks,
  the bank arms' configurations, raw inputs). This is a unit-test fixture of the *form* of each density
  output, not an experiment; no results are drawn from it.
- No estimator was changed.

### A. Quadrature against closed forms (two Gaussian copulas)

- Closed forms: int c_r c_s = 1/(1−rs) (Mehler), H², KL.
- Pair (0.600, 0.629): closed H² 0.00037391, KL 0.00153195, ISE 0.00480858.
  - At L = 8 every n from 101 to 801 gives H² and KL to all 8 printed digits, ISE 0.00480849, and mass
    1.00000000.
- Pair (0.2, 0.8): H² 0.11465446 and KL 0.84291871 match at every n.
  - ISE converges to 1.437826 against the closed form 1.438492. The gap (5·10⁻⁴) is the tail beyond
    \|z\| = 8, which for a strongly dependent Gaussian decays only like exp(−0.05 z²).
- At L = 6 the ISE is short by 0.017 for the same reason.

### B. Mass of the true densities

1.000000 for all four families at d = 0.2, 0.6 and 0.8 (L = 8, n = 801).

### C. Squared Hellinger convergence, including the tail corners

Definition: H²_t = 1 − ∫∫ √( c_t(u,v) · ĉ_{t|t−1}(u,v) ) du dv. Here c_t is the true copula at tick t and
ĉ_{t|t−1} is the forecast held *before* observation t: the same object the excess log loss scores at t.

| Pair | H², n = 101…801, L = 6…8 | Share of the integrand outside max(\|x\|, \|y\|) > 4 |
|---|---|---|
| Clayton vs Clayton (ρ_S +5%) | 0.00048038 (identical in all 12 settings) | 1.0·10⁻⁴ |
| Clayton vs Gaussian (same ρ_S) | 0.02032573 | 5.1·10⁻⁵ |
| Gumbel vs Gumbel (+5%) | 0.00033468 | 1.1·10⁻⁴ |
| t₄ vs Gaussian (same ρ_S) | 0.00681244 | 8.1·10⁻⁵ |
| mixture (0.75 C + 0.25 Gu) vs Clayton | 0.00341604 (L = 6 differs by 1·10⁻⁸) | 9.2·10⁻⁵ |
| Gaussian vs 0.97 G + 0.01 C/Gu/t | 0.00000798 | 1.2·10⁻⁴ |

- **The corners are under control.** √(cĉ) ≤ (c + ĉ)/2, so the omitted region \|z\| > 8 contributes at most
  the probability mass there, below 2Φ(−8) ≈ 10⁻¹⁵.
- **L is capped at 8.** In double precision Φ(z) rounds to 1 for z > 8.29, so densities cannot be evaluated
  there. L = 9 returned NaN, which is the reason for the cap.

### E. Xiao and Hermite outputs: the three constraints checked separately

Grid L = 8; n = 201, 401 and 801 agree. Xiao's mass converges only at first order (a step function):
1.0096 → 1.0023 → 1.0006 for its short arm. Fixtures run on Gaussian and Clayton data, d = 0.6.

| Fixture | min ĉ | ∫∫ĉ | Margin max \|m(u) − 1\| | Margin L1 | Series clipped (area) | Renormalised: ∫∫ / margin L1 |
|---|---|---|---|---|---|---|
| Xiao, Gaussian, λ 0.95 | 0.417 | 1.000 (exact; quadrature 1.0006) | 0.241 | 0.181 | — | — |
| Xiao, Gaussian, λ 0.999 | 0.137 | 1.000 | 0.071 | 0.030 | — | — |
| Xiao, Clayton, λ 0.95 | 0.385 | 1.000 | 0.478 | 0.210 | — | — |
| Xiao, Clayton, λ 0.999 | 0.050 | 1.000 | 0.073 | 0.040 | — | — |
| Hermite, Gaussian, λ 0.05 | 0.010 (floor) | **1.236** | 1.2·10⁴ | 0.511 | 18.4% | 1.000 / 0.403 |
| Hermite, Gaussian, λ 0.001 | 0.010 | **1.023** | 5.9·10⁴ | 0.050 | 2.3% | 1.000 / 0.058 |
| Hermite, Clayton, λ 0.05 | 0.010 | **1.245** | 6.6·10⁵ | 0.478 | 18.8% | 1.000 / 0.374 |
| Hermite, Clayton, λ 0.001 | 0.010 | **1.037** | 1.2·10⁵ | 0.068 | 5.2% | 1.000 / 0.070 |

**Conclusions:**
- **Xiao:** non-negative, normalised, **not uniform margins**. It is a valid bivariate density on [0,1]²,
  not a copula density.
- **Hermite:**
  - non-negative only because of the clip (the series is negative on 2–19% of the square);
  - **not normalised**;
  - margins not uniform, and the density is unbounded near the boundary. Its huge maximum margin values
    sit at \|z\| ≈ 8, where no data fall.
  - Renormalisation fixes only the total mass. **It does not give a copula density.**
- **Neither estimator was modified.** Their outputs must not be presented as copula densities.

### D. Full-domain ISE for the actual true and estimated pairs

**Method:**
- Compute the interior ISE on [ε, 1−ε]² for ε = 10⁻², 10⁻⁴, …, 10⁻¹⁰ (equal steps in log ε).
- A log-divergent integral gains a constant amount per step (step ratio ≈ 1). A convergent one gains less
  each step.
- Gaussian–Gaussian pairs are checked against the closed form.
- Theory: a tail-dependent copula has c(su, sv) ≈ s⁻¹g(u, v) at its corner. So (c − ĉ)² is not integrable
  unless ĉ has exactly the same tail profile g.

| Truth | Estimate (stands for) | Interior ISE, ε = 10⁻² → 10⁻¹⁰ | Step ratios | Full-domain ISE |
|---|---|---|---|---|
| Gaussian d = 0.2 / 0.6 / 0.8 | Gaussian, ρ_S +5% (RMPL, single Gaussian) | 0.6: 0.0021 → 0.0048 | 0.27, 0.22 | **finite**; closed forms 0.000118 / 0.004809 / 0.066385 (d = 0.8 converges slowly in ε) |
| Gaussian | Gaussian mixture (log-bank style) | 0.00029 → 0.00069 | 0.30, 0.24 | **finite**: a mixture of Gaussians; closed form Σ w_k w_l/(1 − r_k r_l) |
| Gaussian | Xiao histogram (fixture) | 0.087 → 0.236 | 0.16, 0.29 | **finite** (ĉ bounded) |
| Gaussian | family-bank-like mixture (0.97 main + 0.01 each of Clayton, Gumbel, t₄) | 0.0023 → 0.0078 | 0.55, 0.66 | **infinite**. D2: the lower corner alone, 0.99 G + 0.01 C, gains a constant 5.0·10⁻⁵ per decade of ε down to ε = 10⁻⁸⁰. That equals 0.01² × Clayton's own ≈ 0.5 per decade. |
| Gaussian | Hermite renormalised (fixture) | 0.025 → 0.18 → 20 → 72 → 89 | irregular | **not usable**: dominated by the boundary blow-up at \|z\| > 4, where there are no data. Finite only if each standardised scale has ŝ² < 2. |
| Clayton d = 0.2 / 0.6 | Clayton, ρ_S +5% (best arm) | 0.6: +0.00727 per step, constant | 1.00, 1.00 | **infinite** |
| Clayton | Gaussian at r* (single Gaussian) | +0.76, 0.93, 0.98, 1.00 | 1.06, 1.02 | **infinite** |
| Clayton | family-bank-like mixture | +0.0052 per step | 1.00, 1.00 | **infinite** |
| Clayton | Xiao histogram | +1.0 per step | 1.00, 1.00 | **infinite** |
| Clayton | Hermite renormalised | 0.03 → 10,246 | irregular | **infinite or unusable** |
| Gumbel | Gumbel +5% / Gaussian at r* | constant increments | 0.99–1.11 | **infinite** |
| t₄ | t₄ +5% / Gaussian at r* | constant increments | 1.00–1.21 | **infinite** |
| Clayton–Gumbel mixture (π 0.75) | Clayton / t₄ / Gaussian at r* | constant increments | 1.00–1.10 | **infinite** |

**What the checks support:** for **the pairs tested** (and the estimator types they stand for),
full-domain copula ISE is finite only when the truth is Gaussian and the estimate has no
tail-dependent component: a Gaussian, a Gaussian mixture, or Xiao's histogram. It diverged:
- for every tested estimate when the truth is Clayton, Gumbel, t₄ or the mixture;
- for the family-bank-like mixture against a Gaussian truth.

**Qualification:**
- Divergence of ∫∫c² or ∫∫ĉ² on their own does **not** imply that ∫∫(c − ĉ)² diverges for every pair.
- The leading corner terms cancel when ĉ has the same tail singularity as c: the same profile g, with an
  error that vanishes fast enough at the corner. Examples are ĉ = c near the corner, or a mixture that
  reproduces c's tail component exactly.
- So the statement is about the configurations the estimators actually produce, where the tail
  parameters differ from the truth's: estimated θ̂ ≠ θ, a different family, or fixed-share mixtures. It is
  not a theorem about all pairs.
- This does not change the decision to omit ISE.

Hermite's ISE is not usable in any cell.

---

## 9. Four accuracy measures: availability (ISE marked, not replaced)

Key:
- ✅ defined and computable.
- **∞**: the full-domain ISE is infinite, so the cell is reported as "∞ (not defined)", never as a
  number and never replaced silently.
- ✗: no valid density.
- (✓): the estimator's density is a valid bivariate density but not a copula; report it only if labelled
  as such.

Rows are estimators; columns are true copulas.

**Spearman RMSE:** ✅ for every estimator and every truth.

**Density ISE (full domain):**

| Estimator | Gaussian truth | Clayton / Gumbel / t₄ / switches / mixture |
|---|---|---|
| RMPL; single Gaussian | ✅ | ∞ |
| Gaussian mixture (log w); Gaussian-only bank | ✅ | ∞ |
| Family bank | ∞ | ∞ |
| Best fixed arm | ✅ if it is a Gaussian arm, else ∞ | ∞ |
| Xiao | (✓) finite, not a copula | ∞ |
| Hermite | ✗ | ✗ |

**Squared Hellinger and excess log loss** (the same in every truth column):

| Estimator | H² | Excess log loss |
|---|---|---|
| RMPL; single Gaussian | ✅ | ✅ |
| Gaussian mixture (log w); Gaussian-only bank | ✅ | ✅ |
| Family bank | ✅ | ✅ |
| Best fixed arm | ✅ | ✅ |
| Xiao | (✓) not a copula | (✓) not a copula |
| Hermite | ✗ (not normalised; renormalised is still not a copula) | ✗ |

**Bank forecasts built from Xiao or Hermite ρ̂** (Gaussian at ρ̂): valid Gaussian copulas, but **working
models**. They can be scored only under that label, not as the estimators' own densities.

**Normal-score ISE_z** remains available as a separate, labelled quantity if wanted. It is never
substituted for the ISE column.

---

## 10. Online log loss (training) vs truth-based evaluation

| | Chapter 5 online bank weights (training) | Chapter 6 evaluation |
|---|---|---|
| Purpose | Update weights w_{k,t} | Measure accuracy against the simulation truth |
| Data | Observations only | Observations **and** the known c_t, ρ_S(t) |
| Point | The estimator's own pseudo-observation (Û_t, V̂_t) from the shared EW margins | The true PIT (U_t, V_t) = (Φ(X_t), Φ(Y_t)), or the whole square |
| Quantity | ℓ_{k,t} = −log ĉ_{k,t\|t−1}(Û_t, V̂_t) (log bank: the Gaussian at ρ̂_k; own bank: the arm's own density); squared bank (p_t − ρ̂_k)², p_t = 12(Û−½)(V̂−½) | Excess log loss log c_t(U_t,V_t) − log ĉ_{t\|t−1}(U_t,V_t); H²(c_t, ĉ_{t\|t−1}); ISE where finite; ρ_S RMSE |
| Normalisation | Not required to *compute* −log ĉ: any positive function can be plugged in | Not required to compute the number either; **required to interpret it**: see below |
| What it rewards | Relative predictive fit of the arms on the data seen | Distance to the truth |

**Normalisation: computing vs interpreting.**
- Evaluating −log ĉ(u, v) numerically needs only ĉ > 0. An unnormalised ĉ (e.g. Hermite's, mass 1.02–1.25)
  still returns a number.
- Interpreting the result does need a valid density, ∫∫ĉ = 1 and ĉ ≥ 0, in two places:
  - **Strict propriety.** The log score is strictly proper only over normalised densities. Over
    unnormalised functions it can be gamed: scaling ĉ by a > 1 lowers the loss by log a everywhere.
  - **KL.** The expected excess loss E_c[log c − log ĉ] equals KL(c ‖ ĉ) ≥ 0 only when ĉ is a density. For
    ĉ = a·q it equals KL(c ‖ q) − log a, which can be negative.
- So excess log loss is reported only for valid densities. For Chapter 6's comparisons, that means valid
  copula densities.
- In Chapter 5 the weight update uses log-loss *differences between arms*. A common unnormalised factor
  would cancel there, but arm-specific factors (Hermite's) would not.

The two log losses differ in three ways: the evaluation point (Û vs U), the reference (none vs log c_t),
and the density scored (an arm vs the forecast actually issued).
- A bank trained by log loss is **not** thereby evaluated by log loss.
- Every method, whatever its training loss, is evaluated by the same truth-based metrics.

---

## 11. Proposed next steps (awaiting authorisation)

1. **Step 2 (E5):** add H² at checkpoints (Δ = 50) to `probe_family_bank` for:
   - the family-bank mixture;
   - the single Gaussian and Gaussian mixture;
   - the best fixed arm;
   - the Gaussian-only bank.

   Full-domain ISE is written only for the cells where §9 says finite (Gaussian truth with
   Gaussian-type estimates); every other cell gets "∞". It must reproduce every `density2` row exactly.
   The ρ_S RMSE and excess log loss are already there.
2. **Step 3 (E2–E4):** density metrics only for **copula** forecasts: RMPL arms, banks and VFF (via an
   accessor); the single Gaussian and Gaussian mixture of the squared and log banks. Xiao and Hermite
   get ρ_S RMSE only, with the §8 footnote.
   - Optional, needs your decision: Xiao's histogram as a labelled "bivariate density, not a copula".
   - If included, it needs cell-exact integration, because the grid error is about 6·10⁻⁴.
3. **Step 4:** the E4 δ comparison (a run; no code).

**Decisions for you:**
- (a) Is Xiao's non-copula histogram reported at all?
- (b) Should the normal-score ISE_z appear as a separate labelled supplement, or not at all?

---

## 12. Decisions on Step 1 (2026-11-10, Sally)

1. **Xiao's histogram** is not included as a copula density in the main results: its non-uniform margins
   prevent a like-for-like copula-density comparison.
2. **Hermite's density** is not repaired for these experiments. Xiao and Hermite are evaluated by
   Spearman RMSE.
3. **Ordinary density ISE** is dropped from the main results. Normal-score ISE is not added. The
   finiteness analysis (§3, §8D, with the §9 qualification) stays in these notes.
4. **The seed-42 fixture** in `check_density_metrics.cpp` stays. It is a validation test, not a research
   experiment.
5. **Clarification** (in §10): computing −log ĉ needs no normalisation. Interpreting log loss as a strictly
   proper score, and its expectation as KL, requires valid densities.
6. **Qualification** (in §8D): divergence of individual squared densities does not imply divergence for
   every pair; matching tail singularities can cancel.
7. **Excess log loss is the primary full-copula metric.** Hellinger is secondary: it tests whether the
   log-loss findings also hold under a symmetric, bounded discrepancy.
8. **The oscillation scenario** is excluded from the final Chapter 6 results and summary counts.

---

## 13. Step 2: squared Hellinger in the family-bank density evaluation (2026-11-11)

**What was added (evaluation only):**
- `probe_family_bank` with `DENSITY_AUDIT=3` writes H²(c_t, ĉ_{t|t−1}) = 1 − ∫∫√(c_t ĉ_{t|t−1}) for the
  forecasts that the excess log loss already scores at the same tick, before the update:
  - the family-bank mixture;
  - the single Gaussian at the combined ρ̂ (`mle_bank_log`'s own `log_density`);
  - the Gaussian mixture under the log bank's weights;
  - every one of the 24 arms.
- **Quadrature:** normal-score tensor trapezoid, n = 81, L = 8. Absolute error ≤ 8·10⁻⁹ against n = 801 for
  family-bank-like mixtures, t, Gumbel and the mixture (`check_density_metrics F`).
- **Checkpoints:**
  - every 50 ticks from tick 500;
  - in jump and switch scenarios, also every 10 ticks over [2500, 2600) — these count only in the 0–100
    window;
  - every other window averages the 50-tick checkpoints, so time is weighted uniformly.
- A first pass averaged all checkpoints in the whole and post windows, which over-weighted early recovery
  (about 10% of the weight for 2% of the ticks). It was corrected and the affected regimes rerun. In that
  pass, the jump whole-window and mixture post-window disagreements were artefacts; they disappeared.
- **Configuration:** seeds 42–101, matched paths, log-bank δ = 10⁻⁴, oscillation not run (`FB_SCENARIOS`).
  - Fingerprints `b80462780a3ff7b8` (constant, drift) and `f7d6dc6f247b8fb5` (jump, switches, mixture).
  - Commands: `results/density_hellinger_2026-11-11.txt`.
- **Verification:** all 329,040 existing ρ_S-RMSE and excess-log-loss rows of `density2` (the scenarios
  run) are reproduced exactly (max |diff| = 0). There are no missing H² values.

**Best fixed arm:** selected in hindsight by whole-window excess log loss. In all 16 cells the
H²-selected arm is the same arm, so a single oracle serves every metric.

**Results**

All values are ×10⁻³ and are family-bank mixture (FB) minus comparator: mean over 60 seeds, every entry
beyond 2 SE unless marked (tie).

| Regime | Window | FB − single Gaussian: excess | FB − single Gaussian: H² | FB − best fixed arm: excess | FB − best fixed arm: H² |
|---|---|---|---|---|---|
| Gaussian, constant / drift | whole | +0.2 / +0.6 | +0.1 / +0.2 | +0.7 / +0.9 | +0.2 / +0.3 |
| Gaussian, jump | post | +2.1 | +1.0 | +2.3 | +0.7 |
| Clayton, constant / drift | whole | −76.3 / −68.3 | −20.0 / −17.8 | +0.6 / +0.3 | +0.1 / +0.1 |
| Clayton, jump | post | −178.2 | −44.2 | +1.4 | +0.1 (tie) |
| Gumbel, constant / drift | whole | −29.3 / −26.1 | −7.1 / −6.3 | +0.5 / +0.4 | +0.1 / +0.1 |
| Gumbel, jump | post | −53.4 | −12.1 | −1.3 | −0.6 |
| t₄, constant / drift | whole | −34.6 / −34.1 | −6.5 / −6.9 | +0.4 / +0.3 | +0.1 / +0.1 |
| t₄, jump | post | −46.3 | −7.7 | +1.9 | +0.3 |
| Gauss → t₄ | pre | +0.3 | +0.1 (tie) | −20.7 | −5.9 |
| Gauss → t₄ | 0–100 after | −12.3 | −0.9 | +24.7 | +5.2 |
| Gauss → t₄ | post | −31.2 | −5.8 | +2.6 | +0.7 |
| Clayton → Gumbel | pre | −75.7 | −19.9 | −68.6 | −18.5 |
| Clayton → Gumbel | 0–100 after | **+46.3** | **+12.8** | +53.6 | +13.7 |
| Clayton → Gumbel | post | −23.8 | −5.5 | −18.9 | −4.6 |
| Gauss → Clayton | pre | +0.3 | +0.1 (tie) | −60.3 | −14.9 |
| Gauss → Clayton | 0–100 after | −17.5 | −3.0 | +50.1 | +12.8 |
| Gauss → Clayton | post | −72.3 | −18.5 | +3.5 | +1.0 |
| Clayton–Gumbel mixture | pre | −23.6 | −6.2 | −17.0 | −4.9 |
| Clayton–Gumbel mixture | 0–100 after | **+19.5** | **+6.0** | +27.0 | +7.0 |
| Clayton–Gumbel mixture | post | −4.6 | −0.8 | +0.6 | +0.1 |
| Clayton–Gumbel mixture | whole | −13.0 | −3.2 | −7.3 | −2.1 |

- **The Gaussian mixture behaves like the single Gaussian** under both metrics. It is never worse, and at most
  1.6 better in H² and 12.9 better in excess (both maxima: t₄ jump, 0–100 after).
- **Excess log loss and H² agree:** across all 144 cell × window × comparator triples (16 cells; whole, pre,
  0–100, 100–500 and post where defined; comparators single Gaussian, Gaussian mixture, best arm), the
  2-SE verdict (W/L/T) matches in 138.
- **The 6 disagreements are all "significant vs tie", never opposite signs:**
  - Gauss → t₄ and Gauss → Clayton, pre-change, against both Gaussian forecasts (4 triples). With Gaussian
    truth, the family bank's small cost of flexibility is significant in excess (+0.2 to +0.3) but not in
    H² (+0.1).
  - Clayton jump, post, against the best arm: excess +1.4, H² +0.1 (tie).
  - Gumbel jump, pre, against the best arm: excess −0.2, H² −0.0 (tie).
- **Per-seed correlation** of the excess and H² differences (FB minus each Gaussian forecast): median
  0.37, IQR [0.11, 0.49], range [−0.15, 0.91]. The aggregate verdicts agree, but a seed that gains more
  in log loss does not reliably gain more in H².

**Interpretation:**
- **The main log-loss findings hold under the symmetric, bounded H²:**
  - the family bank is far closer to the truth than every Gaussian forecast whenever the truth is
    non-Gaussian (every non-Gaussian cell and window, except immediately after a family change);
  - it pays a small cost when the truth is Gaussian;
  - it is temporarily worse in the first 100 ticks after Clayton → Gumbel and in the mixture regime (the
    structural-adaptation lag);
  - it is within a few 10⁻³ of the hindsight-best fixed arm in steady state, and better than any fixed
    arm across a family change.
- **H² differences are smaller than the excess differences:** typically 3–6×, from about 2× (Gaussian truth)
  up to about 14× (Gauss → t₄, 0–100 after). H² is bounded and
  downweights the tails, where the log score penalises a Gaussian forecast of a tail-dependent truth
  heavily (the KL floor). The ranking does not change.
- **H² rarely resolves the smallest effects** that excess log loss resolves: the cost of flexibility under a
  Gaussian truth, and the steady-state gap to the best arm after a jump. These are the only
  disagreements.

---

## 14. Step 3: density accuracy of the MPL forgetting-factor banks (2026-11-11)

**Question:** does adaptive weighting of forgetting factors improve copula-density prediction, and do the
ρ_S-RMSE gains carry over to excess log loss and H²?

**Harness (evaluation only).** `probe_thesis_bank` with `DENSITY_EVAL=1` (mle component) writes
`results/thesis_bank_density_<TAG>[_fam].csv`. The main CSV is unchanged.
- **Methods:**
  - fixed arms, N_eff 39 / 99 / 199 / 399 / 1999;
  - **no forgetting**: the cumulative arm of `mlecum_bank`, gain → 1/m (a running-mean MPL) on the same
    shared EW margins. Those margins still forget at N_eff,m = 500.
  - squared bank (η 1, δ 10⁻⁴), log bank (η 1, δ 10⁻⁴) and uniform weights, each as two forecasts:
    - **single**: the Gaussian at the bank's combined ρ̂, r̂ = 2 sin(πρ̂/6);
    - **mix**: Σ_k w_{k,t−1} c_{r̂_{k,t−1}};
  - VFF slow (0.02) and fast (0.1).
- **What is scored.** Every forecast uses the weights and arm estimates held before tick t:
  - excess log loss at the true PIT, against the true density;
  - H² against the true copula, by normal-score quadrature (n = 81, L = 8), every 50 ticks plus every 10
    over [2500, 2600) on the jump path, counted only in the 0–100 window.
- **Densities.** All are valid copula densities: Gaussian copulas, or convex mixtures of them.
- **Configuration:**
  - families Gaussian, Clayton, Gumbel, t₄; scenarios constant, drift and jump (oscillation not run);
  - matched paths; seeds 42–101; log-bank δ = 10⁻⁴; N_eff,m = 500;
  - fingerprint `25c752ab1449e854`;
  - commands and full tables: `experiments/results/density_forgetting_banks_2026-11-11.txt`.
- **Verification:**
  - all 29,040 MPL rows of `thesis_bank_matched500` (four families; constant, drift, jump) are reproduced
    exactly (ρ_S RMSE and excess, max |diff| = 0);
  - the density file agrees with the main CSV on every shared row;
  - no H² values are missing.
- **Best fixed arm:** chosen per cell and **per metric** in hindsight (lowest whole-window mean).
  - Constant: N_eff 1999 under all three metrics.
  - Drift: 399 under all three.
  - Jump: 199 (RMSE; excess except for the Gaussian, where it is 99) and 99 (H²).

### Summary

W/L/T of A vs B at 2 SE. "Whole" counts all 12 cells; "after 0–100" and "post" count the 4 jump cells.

| Comparison | Window | ρ_S RMSE | Excess log loss | H² |
|---|---|---|---|---|
| Log vs squared bank (single) | whole | 10/0/2 | 11/0/1 | 11/0/1 |
| | post | 3/0/1 | 4/0/0 | 4/0/0 |
| Log vs squared bank (mix) | whole | 10/0/2 | 9/0/3 | 10/0/2 |
| Log bank vs best fixed arm (single) | whole | 4/8/0 | 4/8/0 | 4/8/0 |
| | after 0–100 | 4/0/0 | 4/0/0 | 2/0/2 |
| | post | 4/0/0 | 4/0/0 | 4/0/0 |
| Log bank vs best fixed arm (mix) | whole | 4/8/0 | 4/8/0 | 4/8/0 |
| Squared bank vs best fixed arm | whole | 4/8/0 | 1/9/2 | 4/8/0 |
| Log bank vs no forgetting | whole | 8/4/0 | 8/4/0 | 8/4/0 |
| Log bank vs VFF slow | whole | 5/1/6 | 5/2/5 | 7/0/5 |
| | post | 4/0/0 | 4/0/0 | 4/0/0 |
| Log bank vs VFF fast | whole | 12/0/0 | 8/0/4 | 11/0/1 |
| | post | 3/0/1 | **0/3/1** | 3/0/1 |
| Mix vs single, log bank | whole | 0/0/12 (identical ρ̂) | 10/0/2 | 12/0/0 |
| Mix vs single, squared bank | whole | 0/0/12 | 10/0/2 | 11/0/1 |
| Uniform vs log bank | whole | 0/12/0 | 0/12/0 | 0/12/0 |

The 4/8/0 "whole" rows against the best fixed arm split by scenario: wins in all 4 jump cells, losses in
all 8 constant and drift cells.

### Answers (×10⁻³; excess in nats per observation)

1. **Log vs squared weighting: yes, log is better for density prediction too.**
   - It wins in 11 of 12 cells on both excess and H² (single forecasts). The tie is t₄ constant.
   - Steady-state gains are small (excess −0.2 to −0.6). They are largest after a jump: post −3.0 to −5.3,
     H² −0.4 to −0.9.
   - With mixture forecasts the gap narrows (9/0/3 and 10/0/2), because mixing helps the squared bank more.

2. **Adaptive weighting vs fixed memory: it depends on the scenario, the same pattern as for RMSE.**
   - **Constant and drift:** the bank loses to the hindsight-best fixed arm in every cell (8/8). The cost is
     small in density terms: excess +0.4 to +1.0, H² +0.1 (RMSE +3.7 to +8.6).
   - **Jumps:** the bank beats the best fixed arm over the whole run (excess −1.3 to −2.3 single, −2.2 to
     −4.7 mix; H² −0.6 to −0.9).
     - The gain is concentrated in the first 100 ticks after the jump: excess −10 to −80 single, up to −92
       mix.
     - Post-change it also wins: excess −0.5 to −2.6, H² −0.3 to −0.7.
     - **Exception:** at 100–500 after the jump, the bank is worse than the short best arm in excess for the
       Gaussian (+8.5) and Clayton (+4.1), and worse in H² for Gumbel and t₄ (+1.1 to +2.1). The bank is
       still re-concentrating its weights there.
   - **No forgetting:**
     - wins only on the constant path, as it should (excess +0.5 to +1.2 in its favour);
     - loses heavily on drift (bank −32 to −38 excess) and jumps (−105 to −129 excess over the whole run;
       −189 to −234 post).

3. **Single Gaussian vs Gaussian mixture.** Mixing never changes ρ̂, and it improves density prediction
   modestly but consistently: excess 10/0/2, H² 12/0/0 for the log bank.
   - Stationary gains are tiny (0.0 to −0.7).
   - Gains are largest in the 0–100 window after a jump (−3.2 to −12.8 log bank, −3.2 to −16.0 squared bank)
     and post-change under misspecified truth (−0.9 to −4.4). The weights are spread over arms there, and
     the mixture hedges.
   - The gain is larger for the squared bank than for the log bank.
   - Uniform mixing: H² 11/1/0; its one loss is Gaussian drift.

4. **VFF.**
   - **Slow VFF** is level with the log bank on constant and drift: ties, or at most ±0.3 excess.
     - It loses heavily after jumps: post excess +9 to +11 in the bank's favour, H² +2.7 to +3.1.
     - On drift under Clayton and t₄, it is slightly better than the single-Gaussian bank in excess (+0.1,
       +0.3) but slightly worse in H² (−0.0, −0.1).
   - **Fast VFF** is worse than the bank on constant and drift in every metric.
     - After a jump it recovers faster in the 100–500 window: bank +4.6 to +8.2 excess, +0.7 to +1.9 H².
     - Post-change, excess slightly favours fast VFF in Clayton, Gumbel and t₄ (+1.0 to +2.0), while H²
       favours the bank in Clayton and Gumbel (−0.2) and ties in t₄ — the clearest metric disagreement.
   - **The pattern of earlier findings holds:** VFF is competitive under smooth change and weaker after
     abrupt jumps. Under density metrics, fast VFF's quicker mid-recovery shows up.

5. **Does Hellinger confirm the log-loss conclusions?** In direction, yes, for every headline comparison
   above (log > squared, the bank's jump advantage, the steady-state cost, mixing helps, uniform is worst).
   - Verdict agreement: 324 of 364 comparison × cell × window entries.
   - **Disagreements, part 1: the same sign, but only one metric reaches 2 SE.** Most fall in the 0–100
     window, where H² has only 10 checkpoints per seed and is less powered.
   - **Disagreements, part 2: a handful of opposite-sign cases, all small** (|excess| ≤ 1.2, |H²| ≤ 0.2 in
     the whole and post windows) and all under non-Gaussian truth.
     - Cases: VFF slow vs bank on Clayton/t₄ drift; VFF fast vs bank post-jump; squared bank vs best arm on
       the Clayton jump; log mix vs best arm at 100–500 on t₄.
   - **Why opposite signs can happen:** under misspecification the KL-best and Hellinger-best Gaussians
     differ (quadrature, d = 0.6):
     - Clayton r* = 0.595 vs r_H = 0.626;
     - Gumbel 0.605 vs 0.616;
     - t₄ 0.606 vs 0.640;
     - the Gaussian at the true ρ_S is r = 0.600.
     So a forecast whose r̂ sits slightly higher can win on H² and lose on excess.
   - **Interpretation:** the residual differences between Gaussian forecasts under a non-Gaussian truth are
     small next to the floor (e.g. Clayton post excess ≈ 189 against a floor ≈ 177). Their sign then
     depends on which Gaussian each metric prefers.

**Caveats:**
- The best fixed arm is a per-cell hindsight oracle.
- The 0–100 H² averages rest on 10 checkpoints per seed.
- Gaussian density misspecification under Clayton, Gumbel and t₄ is intended (memory vs family).
- The no-forgetting baseline still uses EW margins (N_eff,m = 500).

---

## 15. Settled framework and retained findings (2026-11-11, Sally)

**Metric hierarchy (final):**

| Metric | Role |
|---|---|
| Spearman RMSE | Primary for dependence-strength estimation |
| Excess logarithmic loss | Primary for predictive copula-density accuracy |
| Squared Hellinger | Secondary validation |
| Tail RMSE (χ_L, χ_U) | Primary for tail experiments |
| Recovery time, regret, memory, runtime | Experiment-specific |

**Scope decisions:**
- No further density evaluation for E1.
- Xiao and Hermite are evaluated primarily through Spearman RMSE.
- Density ISE is dropped; its finiteness analysis stays here as a technical note.

**Retained Step 3 findings:**
1. **Log vs squared weighting:** the log bank improves excess log loss and H² in 11/12 cells, with one tie
   under each metric.
2. **Adaptive vs fixed forgetting:** the bank beats the hindsight-best fixed memory in all 4 jump cells and
   loses in all 8 constant/drift cells. Its advantage is concentrated around abrupt changes.
3. **Density mixing:** the Gaussian-mixture forecast improves excess log loss in 10/12 cells and H² in 12/12,
   relative to the single Gaussian at the bank's combined ρ̂.
4. **VFF:** slow VFF is competitive under smooth change; the bank is better after jumps.
5. **Metric agreement:** 324/364. Some disagreements under misspecification are expected, because the
   KL-optimal and Hellinger-optimal Gaussian parameters differ.

**Next and last experiment:** Step 4, Hedge (δ = 0) vs fixed share (δ > 0). After that, consolidate the
results; no new experiments.

---

## 16. Step 4: plain Hedge (δ = 0) vs fixed share (δ = 10⁻⁴), MPL log bank (2026-11-11)

**Design (controlled):**
- MPL log bank at η = 1, with the same arms, shared margins, initial weights, forecast constructions and
  input streams.
- **Only δ differs.** Fixed share is Step 3's `density4` run (the tuned δ = 10⁻⁴). Hedge is the same command
  with `BANK_LOG_DELTA=0`.
- Families Gaussian, Clayton, Gumbel and t₄; scenarios constant, drift and jump; seeds 42–101; oscillation
  excluded.
- **Principal density forecast:** the Gaussian mixture Σ_k w_{k,t−1} c_{r̂_{k,t−1}}. The single Gaussian at the
  combined ρ̂ is a secondary diagnostic.
- **Reuse:** no evaluation-seed δ = 0 output existed. The tuning grid is on the development seeds; the
  regret probe has δ = 0 but no accuracy metrics.

**Verification:**
- In every family, every method other than the log bank is identical between the two runs (density file
  and main CSV, max |diff| = 0).
- The fixed-share run itself reproduces `thesis_bank_matched500` (§14).
- Fingerprint `25c752ab1449e854`, the same code as Step 3.
- Tables: `experiments/results/hedge_vs_fixedshare_2026-11-11.txt`.
- Figures: `notes/thesis/figures/matched/ch6_hedge_vs_fixedshare.png` (Gaussian, Clayton) and `_all.png`.

### Results

Fixed share minus Hedge, ×10⁻³, mean (SE) over 60 seeds; negative favours fixed share. Excess and H²
are for the mixture forecast.

| Cell | Window | ρ_S RMSE | Excess log loss | H² |
|---|---|---|---|---|
| Constant (4 families) | whole | +0.4 to +2.8 (3 L, 1 T) | +0.0 to +0.1 (3 L, 1 T) | +0.0 (1 L, 3 T) |
| Drift (4 families) | whole | +0.5 to +1.7 (4 L) | +0.1 (4 L) | −0.0 to +0.0 (1 W, 2 L, 1 T) |
| Jump, Gaussian | 0–100 after | −119.3 (5.1) | −113.0 (4.6) | −35.5 (1.4) |
| Jump, Clayton | 0–100 after | −120.1 (4.5) | −90.4 (4.5) | −31.6 (1.1) |
| Jump, Gumbel | 0–100 after | −124.0 (4.7) | −116.8 (4.4) | −34.5 (1.1) |
| Jump, t₄ | 0–100 after | −136.2 (5.5) | −140.0 (7.2) | −37.3 (1.4) |
| Jump (4 families) | 100–500 after | +5.8 to +6.3, or tie (2 L, 2 T) | +1.1 to +7.5 (3 L, 1 T) | +0.4 to −2.4 (2 W, 2 T) |
| Jump (4 families) | 500–1500 after | −3.8 to +4.2 (2 W, 1 L, 1 T) | −0.2 to −4.8 (3 W, 1 T) | (2 W, 2 T) |
| Jump (4 families) | post [2500, 5000) | −22.6 to −23.6 (4 W) | −4.6 to −9.0 (4 W) | −1.6 to −1.8 (4 W) |
| Jump (4 families) | whole | −14.8 to −15.7 (4 W) | −2.6 to −5.0 (4 W) | −0.9 to −1.0 (4 W) |

For scale, Hedge's levels in the first 100 ticks after the jump are RMSE 445–476, excess 235–383 and
H² 80–119.

**Weights and weight-averaged memory** (seed means; jump scenario; t₄ in brackets where it differs):

| | Hedge | Fixed share |
|---|---|---|
| Weight on N_eff 39 just before the jump | 0.0000 | 0.002–0.003 |
| Largest weight on N_eff 39 after the jump | ≤ 0.02 | 0.92–0.99 |
| Memory at t = 2600 | 133–305 | 40–78 |
| Memory at t = 3000 | 99–115 | 65–148 |
| Memory at t = 3995 | 102–156 | 221–387 |
| Weight on N_eff ≥ 199 at t = 3995 | 0.03–0.57 | 0.83–0.95 |

### Answers

1. **After abrupt changes, fixed share beats Hedge under all three metrics in every family.** This holds
   over the whole jump run and the post-change period, and is largest in the first 100 ticks: excess −90 to
   −140, H² −32 to −37, ρ_S RMSE −119 to −136. The single-Gaussian diagnostic gives the same verdicts.
2. **The cost under stationarity and drift is very small and mostly significant:** ρ_S RMSE +0.4 to +2.8,
   excess +0.0 to +0.1, H² essentially 0. It is the price of the δ/K mass kept on arms that are wrong in
   steady state.
3. **Yes, fixed share restores weight to short memory and then shifts back.**
   - Under Hedge, the 39- and 99-tick arms have weight ≈ 0 (underflow-level) after 2000 stationary ticks.
     After the jump Hedge moves to the 99-tick arm, which keeps some weight, but never reaches the 39-tick
     arm (≤ 0.02).
   - Hedge then **stays at short memory**: memory ≈ 100 at t = 4000 for the Gaussian, 123–156 otherwise.
     The long arms accumulated large losses during the transient and cannot recover in finite time.
   - Fixed share puts 0.92–0.99 of the weight on the 39-tick arm within about 50 ticks. It then moves
     progressively back to 99, 199 and 399 (memory 221–387 by t = 4000), as the regime stabilises.
4. **The gain is concentrated, and also sustained:**
   - **Concentrated:** most of the gain is in the first 100 ticks.
   - **A partial reversal at 100–500 ticks:** Hedge, parked at N_eff 99, is slightly better there in excess
     (+1.1 to +7.5) and ρ_S. Fixed share's very short memory is still noisy as it climbs back. H² does not
     show this reversal (2 W, 2 T).
   - **Sustained:** fixed share then wins again at 500–1500 and over the whole post period (excess −4.6 to
     −9.0), because it returns to longer memory while Hedge stays short.
5. **Hellinger supports the log-loss findings** for every headline (after-jump, post, whole-run wins; the
   small stationary cost).
   - The one divergence is the 100–500 window: excess favours Hedge (3 L, 1 T), while H² favours fixed share
     or ties (2 W, 2 T).
   - This matches the Step 3 pattern: small residual differences between Gaussian forecasts, with different
     KL- and H²-optimal parameters.

**For Chapter 6:**
- The fixed-share term is what lets the bank use its short-memory arms after a change. Without it, Hedge
  reacts more slowly and then fails to return to long memory.
- Fixed share's advantage is large and immediate after abrupt changes, persists over the post-change
  period, and costs almost nothing when dependence is stable.

**Retained Step 4 conclusions (Sally, 2026-11-11):**
1. Fixed share improves ρ_S RMSE, excess log loss and H² after abrupt changes in all four families.
2. Its largest advantage is in the first 100 ticks after the jump, when it restores weight to the
   shortest-memory arm.
3. As dependence stabilises, it transfers weight back to longer-memory arms, reducing estimation variance.
4. Plain Hedge struggles to recover weight on previously discarded arms.
5. Fixed share incurs a small cost under constant dependence and gradual drift.
6. **Qualification:** in the 100–500-tick recovery window Hedge is sometimes better under excess log loss.
   Fixed share does not dominate every post-change window.

Attribution: this is a controlled comparison (only δ differs), so the differences are attributable to the
fixed-share redistribution, under the tested conditions.

---

## 17. E5 scope decision (2026-11-11, Sally)

**E5 question:** does accurate estimation of Spearman's ρ imply accurate copula-density prediction, and
does allowing multiple copula families improve prediction when the dependence structure changes?

**What the existing results can and cannot support:**
- They support the distinction between memory adaptation (forgetting horizon; the Gaussian forgetting-factor
  bank) and family flexibility (dependence-structure misspecification; the family bank).
- They do not support a claim that the two contributions are isolated or quantified, or that they don't
  interact.

**Ablation:** the fixed-memory family bank (8 shapes × one memory) is deferred, not run.

**E2:** the optimal-memory check (Xiao and Hermite, Gaussian drift) is stated as a scope limitation.
