# Chapter 6: central results (consolidated, 2026-11-12)

**What this file is:** the single reference for writing Chapter 6. Every number comes from the final result tables,
listed in the Sources section at the end. Superseded results are listed in §8 and must not be quoted.

**Contents:**
- §0 Protocol and evaluation framework
- §1 E1 memory budgets
- §2 E2 fixed forgetting
- §3 E3 adaptive forgetting
- §4 E4 scoring loss and reweighting
- §5 E5 full-copula accuracy
- §6 Figures
- §7 Limitations and open items
- §8 Superseded results
- Sources

---

## 0. Protocol and evaluation framework

### 0.1 Simulated streams

**Streams:**
- T = 5000 pairs per stream, burn-in 500.
- Standard normal margins.
- Copula: Gaussian, Clayton, Gumbel or Student-t₄.

**Dependence paths** (matched: the true ρ_S path is identical for every family):

| Path | Target |
|---|---|
| constant | d = 0.6 (ρ_S = 0.582) |
| drift | d 0.2 → 0.8 (ρ_S 0.191 → 0.786) |
| jump | d 0.2 → 0.8 at t = 2500 |

The target is ρ_S = (6/π) arcsin(d/2), and each family's parameter is obtained by the method of moments.

**E5 shape regimes** (ρ_S fixed at 0.582, change at t = 2500):
- Gaussian → t₄;
- Clayton → Gumbel;
- Gaussian → Clayton;
- Clayton–Gumbel mixture, π(Clayton) 0.75 → 0.25. The truth is outside the family bank.

**Oscillation** is excluded from all final results.

**Seeds:**
- development 1–30 (all tuning);
- evaluation 42–101 (60 seeds);
- E1: validation 1000–1039, test 42–141 (100 seeds).

**Comparison:** paired per-seed differences, mean ± SE. W/L/T means beyond ±2 paired SE per cell, with no
multiplicity correction. The hindsight-best fixed memory is an **oracle**, chosen per cell (and per metric) after
seeing the results.

### 0.2 Metrics (settled hierarchy)

| Metric | Definition | Role |
|---|---|---|
| **Spearman RMSE** | √( mean_t (ρ̂_{S,t−1} − ρ_{S,t})² ), the forecast held before t | **Primary:** dependence strength (all experiments) |
| **Excess log loss** | mean_t [log c_t(U_t,V_t) − log ĉ_{t\|t−1}(U_t,V_t)] at the **true** PIT (U_t, V_t) = (Φ(X_t), Φ(Y_t)); nats per observation | **Primary:** predictive copula density. For a valid density its expectation is KL(c_t ‖ ĉ) ≥ 0, a strictly proper score |
| **Squared Hellinger** | H²_t = 1 − ∫∫ √(c_t ĉ_{t\|t−1}) du dv ∈ [0, 1], pre-observation forecast, evaluated every 50 ticks (every 10 in the first 100 after a change), windows weighted uniformly in time | **Secondary:** validates the density findings with a bounded, symmetric measure |
| **Tail RMSE** | RMSE of χ_L(q) = C(q,q)/q and χ_U(q) = [1 − 2(1−q) + C(1−q,1−q)]/q at q = 0.05 | **Primary** in tail experiments |
| Recovery, regret, memory, runtime | as defined per experiment | Experiment-specific |

**Hellinger numerics:**
- Normal-score tensor trapezoid, 81 nodes per axis on [−8, 8]².
- Error below 10⁻⁸, checked against the closed forms for Gaussian pairs and against 801 nodes for tail-dependent pairs.
- The tail beyond |z| = 8 contributes less than 10⁻¹⁵.

**Which forecasts get density metrics:**
- recursive MPL arms (their fitted Gaussian copula);
- MPL banks: a **single Gaussian** at the combined ρ̂, and a **Gaussian mixture** Σ w_k c_{r̂_k} with the same weights;
- the family bank (a mixture of 24 parametric copulas).

**Not evaluated as copula densities** (ρ_S RMSE only):
- **Xiao:** its histogram integrates to 1 but has non-uniform margins (deviations 0.07–0.48).
- **Hermite:** its series density is negative on 2–19% of the square and integrates to 1.02–1.25; renormalising
  does not make the margins uniform.

**Online losses vs evaluation:**
- Bank weights are *trained* by online losses: squared (p_t − ρ̂_k)² or log −log c_G(Û_t, V̂_t). These are computed
  at the estimator's own pseudo-observations, without the truth.
- Every method is *evaluated* by the truth-based metrics above, whatever its training loss.

**Normalisation:** computing −log ĉ needs only ĉ > 0. Interpreting it as a proper score, or its mean as a KL
divergence, needs a valid normalised density.

### 0.3 Why density ISE is not used

1. **It diverges for the copulas we test.**
   - Clayton, Gumbel and t₄ densities behave like 1/s near their tail-dependent corners, so ∫∫c² = ∞.
   - For every tested pair with a tail-dependent truth, the ISE grows by a constant amount per decade of the
     truncation level ε, which means it diverges. Examples (interior ISE, step per two decades of ε):
     - Clayton vs Clayton with ρ_S +5%: +0.0073;
     - Clayton vs Gaussian: ≈ 1.0.
   - It also diverges against the **family bank** even when the truth is Gaussian. Fixed share keeps every
     tail-dependent arm at a positive weight: 0.99 Gaussian + 0.01 Clayton adds a constant 5.0·10⁻⁵ per decade,
     checked down to ε = 10⁻⁸⁰.
   - It is finite only for a Gaussian truth with a Gaussian-type estimate (closed form 1/(1−r²) − 2/(1−rs) + 1/(1−s²)).
   - **Qualification:** this is a statement about the tested pairs. An estimate with exactly matching tail
     singularities would give a finite ISE.
2. **Hermite makes it unusable.** Its density blows up near the boundary, where there are no data.
3. **It is scale-dependent.** Truncated ISE depends heavily on the arbitrary ε, and normal-score ISE changes with the
   choice of coordinates.
4. **Excess log loss and Hellinger already answer the density question.** Both are finite for every tested pair,
   invariant to transformations of the margins, and numerically stable.

**Decision (Sally, 2026-11-11):** drop density ISE from the results, and don't use normal-score ISE. The finiteness
analysis stays as a technical note (`notes/studies/chapter6_metrics.md` §3, §8D).

### 0.4 Controlled vs separately tuned comparisons

- **Controlled** (one thing differs):
  - Hedge vs fixed share (δ only);
  - Gaussian mixture vs single Gaussian (forecast construction only);
  - Gaussian-only vs family bank (set of shapes only);
  - fixed arms, no forgetting and the banks (the same arms and streams).
- **Separately tuned** (whole procedures compared):
  - log vs squared bank;
  - bank vs VFF;
  - the E1 estimators.

---

## 1. E1: estimation under memory budgets

**Question:** how do Xiao, Hermite and recursive MPL compare in accuracy and cost under limited retained memory?

**Design:**
- Single estimators, each tuned on validation seeds within 16, 64 or 256 KiB of measured state.
- Tuned quantities: approximation size (Xiao bins, Hermite order, MPL KLL sketch k) and gain.
- Gain grid: {no forgetting, 0.0005, 0.001, 0.002, 0.005, …, 0.10}.
- Test on 100 seeds. MPL uses cumulative DataSketches-KLL margins.
- This is an offline benchmark: the gain is tuned per scenario.

**Headline:** MPL vs the better of Xiao/Hermite is **6/6/0 at every budget**. The pattern depends on the regime.

Test RMSE of ρ_S, ×10⁻², 64 KiB (verdicts identical at 16 and 256 KiB). The last column is MPL minus the better
of Xiao/Hermite, mean ± SE.

| Path | Copula | Gain | Xiao | Hermite | MPL | MPL − better |
|---|---|---|---|---|---|---|
| constant | Gaussian | none | **1.53** | 1.58 | 1.71 | +0.18 ± 0.08 L |
| | Clayton | none | **1.51** | 1.53 | 1.81 | +0.30 ± 0.09 L |
| | Gumbel | none | 1.51 | **1.49** | 1.67 | +0.18 ± 0.06 L |
| | t₄ | none | **1.64** | 1.79 | 1.81 | +0.17 ± 0.07 L |
| drift | Gaussian | 0.005 | 4.62 | 4.64 | **4.15** | −0.46 ± 0.04 W |
| | Clayton | 0.005 | 4.65 | 4.69 | **4.47** | −0.18 ± 0.06 W |
| | Gumbel | 0.005 | 4.63 | 4.66 | **3.98** | −0.65 ± 0.05 W |
| | t₄ | 0.005 | 4.86 | 4.88 | **4.65** | −0.20 ± 0.05 W |
| jump | Gaussian | 0.01 | 8.19 | 8.25 | **7.91** | −0.28 ± 0.04 W |
| | Clayton | 0.01 | **8.29** | 8.36 | 8.49 | +0.20 ± 0.04 L |
| | Gumbel | 0.01 | 8.31 | 8.36 | **8.12** | −0.19 ± 0.04 W |
| | t₄ | 0.01 | **8.52** | 8.54 | 8.63 | +0.11 ± 0.04 L |

**Findings:**
- **Constant:** every estimator chooses no forgetting, and Xiao (or Hermite) is lowest.
  - MPL's extra error comes substantially from its KLL margins. With exact PIT it would reach 1.23, 1.50, 1.34 and
    1.67 (Gaussian, Clayton, Gumbel, t₄): below Xiao for Gaussian and Gumbel, level for Clayton and t₄. This is a
    diagnostic only; exact PIT doesn't fit the budget.
- **Drift:** MPL is lowest in all four families, Gumbel included. So the advantage isn't limited to the correctly
  specified case.
- **Jump:** split by family. The differences are small, about 1–4%. Don't attribute this to misspecification alone,
  since Gumbel is also non-Gaussian.
- **Memory:** tuned accuracy improves by at most 0.12 from 16 to 256 KiB. The 16 KiB configurations already sit on
  each method's error floor. That floor is not pure sampling variance: MPL's exact-PIT gap persists at k = 1024.
- **Runtime** (update + estimate, single core) at the 16 KiB configurations:
  - Xiao 0.6–1.1 µs (20–30 bins);
  - Hermite 0.7–1.3 µs (order 10–20);
  - MPL 1.3–2.0 µs (k = 256–512).
  - MPL is not the cheapest. Larger budgets select bigger summaries (up to 14.6 µs for Xiao) for almost no accuracy.
- **Hand-written vs DataSketches KLL:** the same accuracy at the same k (from k = 128); the hand-written one uses
  4–6× more memory. This is an implementation detail for the appendix.

**Supersedes:** the 9/2/1 "MPL wins" headline, which came from a gain grid truncated at g ≥ 0.005. Every constant and
drift pick had sat on that edge.

---

## 2. E2: fixed exponential forgetting

**Question:** how does effective memory trade variance against tracking bias, and does the approximate optimal
memory describe the observed one?

**Optimal memory on drift** (Gaussian copula; Xiao and Hermite; 40 seeds; 20 memories from 25 to 30,000, steps of
about 1.45×; optimum by log-quadratic interpolation):

| | Measured N\* | Predicted (2σ²/d²)^{1/3} | Prediction − measured |
|---|---|---|---|
| Xiao | 393 | 426 | +8.4% |
| Hermite | 430 | 448 | +4.2% |

- **Inputs:** σ² is the median of variance × N_eff over 50–2000 (0.55 for Xiao, 0.64 for Hermite); d is the RMS of
  dρ_S/dt along the matched path (1.19·10⁻⁴ per tick).
- **The U-shape:** squared bias rises and variance falls with N_eff, and the MSE is U-shaped.
- **Scope and caveats:** Gaussian drift, Xiao and Hermite only. σ² comes from the same sweep, so this is a consistency
  check, not an independent test.

**Best fixed memory** (MPL arms, the same under all three metrics and all four families):

| Path | Best N_eff |
|---|---|
| constant | 1999 |
| drift | 399 |
| jump | 199 (99 for some density metrics) |

On the jump path, N_eff 39 is best in the first 100 ticks after the change, and long memories are best before it.

---

## 3. E3: adaptive forgetting

**Question:** can adaptive weighting of forgetting factors beat fixed, no and variable forgetting, and under which
changes?

**Log bank vs comparators, ρ_S RMSE** (12 cells = 4 families × Xiao, Hermite, MPL). Cells read W/L/T (mean
difference ×10⁻³); the first row is from the bank's side, the VFF rows from VFF's side.

| Comparison | constant | drift | jump |
|---|---|---|---|
| Log bank vs hindsight-best fixed memory | **0/12/0 (+6.3)** | **0/12/0 (+4.6)** | **12/0/0 (−16.6)** |
| Slow VFF vs log bank | 0/7/5 (+2.6) | 1/0/11 (−0.2) | 0/12/0 (+27.6) |
| Fast VFF vs log bank | 0/12/0 (+15.8) | 0/12/0 (+6.5) | 0/12/0 (+10.5) |

**MPL, averaged over the four families** (RMSE ×10⁻³; the figure's numbers):

| | Log bank | Best fixed (oracle) | Slow VFF | Bank − best (±2 SE) | Bank − VFF (±2 SE) |
|---|---|---|---|---|---|
| Constant | 28.5 | **22.8** | 29.5 | +5.7 ± 0.5 | −1.0 ± 0.6 |
| Drift | 43.6 | **38.6** | 43.0 | +5.0 ± 0.4 | +0.5 ± 0.5 |
| Jump | **65.1** | 81.6 | 90.4 | −16.5 ± 0.7 | −25.3 ± 1.2 |

**No forgetting** (MPL, the cumulative arm on the same margins):
- It beats the bank only on constant paths: bank − no forgetting RMSE +5.5 to +12.6 ×10⁻³.
- It loses heavily on drift (−124 to −137) and jump (−244 to −256 over the whole run).

**Density metrics** (MPL bank, 12 cells; Step 3):
- Bank vs best fixed memory is **4/8/0 under RMSE, excess log loss and Hellinger alike**: wins in the 4 jump cells,
  losses in the 8 constant and drift cells.
- Steady-state cost: excess +0.4 to +1.0 ×10⁻³.
- Jump gain: −1.3 to −2.3 over the whole run, mostly in the first 100 ticks (−10 to −80).
- Exception: at 100–500 after the jump, a short fixed memory is sometimes better.

**Interpretation:**
- The bank's advantage is robustness to abrupt change.
- Under constant or slowly drifting dependence, the oracle fixed memory is better, and slow VFF matches the bank.

---

## 4. E4: scoring loss and reweighting

### 4.1 Log vs squared loss (separately tuned procedures)

| Comparison | Cells | ρ_S RMSE | Excess log loss | Hellinger |
|---|---|---|---|---|
| Log vs squared bank | 24 (drift + jump × 3 est. × 4 fam.) | **23/0/1** (−3.8) | — | — |
| Log vs squared bank (MPL, single Gaussian) | 12 (MPL × 4 fam. × 3 paths) | 10/0/2 | **11/0/1** | **11/0/1** |
| Squared bank vs uniform weights | 24 | **24/0/0** (−16.3) | — | — |
| Gaussian mixture vs single Gaussian (log bank) | 12 | identical ρ̂ | **10/0/2** | **12/0/0** |

- The log bank's density gains are largest after jumps: post excess −3.0 to −5.3 ×10⁻³.
- Mixing helps most right after jumps, and more for the squared bank.

### 4.2 Hedge (δ = 0) vs fixed share (δ = 10⁻⁴), controlled

Only δ changes (MPL log bank, η = 1). Fixed share minus Hedge, ×10⁻³, 60 seeds:

| Window | ρ_S RMSE | Excess log loss (mixture) | Hellinger (mixture) |
|---|---|---|---|
| Jump, first 100 after (4/4 W) | −119 to −136 | −90 to −140 | −32 to −37 |
| Jump, 100–500 after | +5.8 to +6.3, or tie | **+1.1 to +7.5 (3 L, 1 T)** | 2 W, 2 T |
| Jump, post (4/4 W) | −23 to −24 | −4.6 to −9.0 | −1.6 to −1.8 |
| Constant / drift (cost) | +0.4 to +2.8 | +0.0 to +0.1 | ≈ 0 |

**Mechanism** (weights, jump path; weighted memory = Σ_k w_k N_eff,k, an average over the arms, not an effective
sample size):
- **Hedge:** the short arms' weights decay to about 0 during the stable period. After the jump it moves to N_eff 99,
  never uses the 39-tick arm (maximum weight ≤ 0.02), and stays short (memory 102–156 at t = 4000).
- **Fixed share:** it puts 0.92–0.99 of the weight on the 39-tick arm within about 50 ticks, then returns through
  99/199/399 (memory 221–387 at t = 4000).

**Conclusion:**
- Fixed share recovers discarded short-memory arms after a change and returns to long memory as dependence
  stabilises, at a small steady-state cost.
- It does **not** dominate every post-change window: Hedge is sometimes better on excess log loss at 100–500 ticks.

### 4.3 Regret

- **Covered procedures** never exceed their bounds (0/1440 runs; family bank 0/480). These are the density-mixture
  log bank, squared loss at η = 1/32, and the family mixture.
- **Uncovered procedures** exceed the reference expression in some runs, which is not a theorem violation: squared
  bank at η = 1 in 50.8% of runs, point-density log bank in 42.7%.

---

## 5. E5: full-copula accuracy

**Question:** does accurate ρ_S estimation imply accurate copula-density prediction, and does allowing multiple
copula families help when the dependence structure changes?

**Comparison:** family bank vs the single Gaussian at the MPL log bank's combined ρ̂ (the Gaussian mixture is within
0.5).

**Whole run, family bank minus single Gaussian** (×10⁻³ nats per observation; Hellinger ×10⁻³):

| Truth | Path | Excess log loss | Hellinger |
|---|---|---|---|
| Gaussian | constant / drift / jump | **+0.2 / +0.6 / +1.5** (small cost) | +0.1 / +0.2 / +0.7 |
| Clayton | constant / drift / jump | −76.3 / −68.3 / −102.5 | −20.0 / −17.8 / −25.4 |
| Gumbel | constant / drift / jump | −29.3 / −26.1 / −32.4 | −7.1 / −6.3 / −7.4 |
| t₄ | constant / drift / jump | −34.6 / −34.1 / −37.4 | −6.5 / −6.9 / −6.9 |

**Family changes at constant ρ_S, by window** (same sign convention). Each cell reads excess log loss / Hellinger:

| Change | before | first 100 after | 100–500 after | all after |
|---|---|---|---|---|
| Gaussian → t₄ | +0.3 / +0.1 | −12.3 / −0.9 | −27.1 / −4.6 | −31.2 / −5.8 |
| Clayton → Gumbel | −75.7 / −19.9 | **+46.3 / +12.8** | −23.8 / −5.6 | −23.8 / −5.5 |
| Gaussian → Clayton | +0.3 / +0.1 | −17.5 / −3.0 | −69.1 / −17.6 | −72.3 / −18.5 |
| Clayton–Gumbel mixture | −23.6 / −6.2 | **+19.5 / +6.0** | −5.2 / −0.9 | −4.6 / −0.8 |

**Why Gaussian forecasts fail under non-Gaussian truth:**
- **The misspecification floor** D = KL(c_true ‖ best Gaussian), with best Gaussian r\* = E[Z₁Z₂]:
  Clayton 70–78, Gumbel 25–29, t₄ 32.6–33.9 ×10⁻³.
- The Gaussian forecasts' excess is mostly this floor. Estimation error adds +1 to +3.5 in steady state.
- The family bank removes most of the excess. Wording: "most", not "the entire floor".

**Accurate ρ_S does not imply an accurate density:**
- The family bank does not improve ρ_S: against its own best single arm it is **4/11/0**; against the MPL log bank
  7/5/3 (15 cells, no oscillation).
- Its density and tail gains come from modelling the structure, not dependence strength.

**Structure vs adaptation:**
- In steady state, the arm with the true family is within +0.3 to +0.9 ×10⁻³ of the bank. Most of the gain over the
  Gaussian forecasts comes from **having the right family available**.
- Across family changes the bank beats every fixed arm over the whole run (−7.7, −41.0, −24.9).
- **Do not claim** to have isolated the memory and family contributions, or ruled out an interaction. The
  fixed-memory family-bank ablation was deferred.

**Tails** (χ(0.05) RMSE against exponentially weighted corner counters, memories 99/399/1999):
- **15/0/0** for both χ_L and χ_U at every memory.
- **Mixture limitation:** before the change the family bank projects onto Clayton, and its χ_U is worse than the
  long counters' (+13 vs memory 399, +53 vs 1999). After the change it beats all counters.
- So **a better density does not guarantee better tail estimation** when the truth is outside the bank.

**Metric agreement:**
- Excess log loss and Hellinger give the same 2-SE verdict in **138/144** family-bank comparisons, with no sign
  reversals.
- In the forgetting-bank comparisons they agree in **324/364**. The few opposite-sign cases there are small and
  occur under misspecification, because the KL-best and Hellinger-best Gaussians differ (Clayton r\* 0.595 vs
  r_H 0.626).
- Hellinger differences are typically 3–6× smaller: it is bounded and less tail-sensitive.

---

## 6. Figures

Folder: `notes/thesis/figures/matched/`. Script: `experiments/plot_ch6_core.py`. Final style: **grouped columns**,
pending Sally's confirmation for E5.

| Use | File | Message |
|---|---|---|
| E1 main | `ch6_fig_e1_columns.png` | Xiao/Hermite lower under constant, MPL under drift, jump split (small) |
| E2 main | `ch6_fig_e2.png` | U-shaped MSE in N_eff; measured vs predicted optimum |
| E3 main | `ch6_fig_e3_columns.png` | Oracle fixed memory lower under constant and drift; bank lower after jumps; slow VFF ≈ bank when smooth |
| E4 main | `ch6_fig_e4.png` | Fixed share jumps to N_eff 39, then returns to long memory; Hedge stuck near 100 |
| E5 main | `ch6_fig_e5.png` (to be redone in column style) | Family bank removes most of the excess under non-Gaussian truth; small Gaussian cost; post-switch penalty |
| E5 (limitation) | `ch6_fig_e6.png` | Mixture: best density, but a biased upper tail before the change |
| Appendix | `ch6_budget_memory_all.png`, `ch6_budget_runtime.png`, `ch6_budget_rmse_vs_k.png`, `ch6_hedge_vs_fixedshare_all.png` | Supporting detail |
| Drop | `ch6_family_summary`, `ch6_bank_weights_neff`, `ch6_bank_weights`, `ch6_budget_recovery`, `ch6_bank_recovery`, `ch6_bank_path*`, `*_ownkll`, `*_ds_vs_own` | Outdated or superseded |

---

## 7. Limitations and open items

**Limitations to state:**
- E1 tuning is offline (per scenario).
- E2 is Gaussian drift only.
- The best fixed memory is an oracle.
- One ρ_S level per path; four families.
- The Linux/libstdc++ streams are platform-specific.

**Thesis text:**
- Three methods used in Chapter 6 are not yet described in Chapters 4–5: the EW marginal, the common VFF controller
  and the family bank.
- The current §6.1 describes the unused GBM simulator; the draft `docs/chapter6_drafts/6_1_experimental_design.tex`
  replaces it.
- The family-bank δ is analytic (δ\* = Km/((K−1)(T−1))), not tuned.

**Drafts:**
- 6.1 and 6.2 are drafted (`docs/chapter6_drafts/`). Their `\ref` labels are placeholders.
- 6.3 onward are to write.

**Scripts:** `plot_neff_optimum.py`, `plot_thesis_bank.py`, `plot_bank_compare.py` and `plot_family_summary.py`
default to superseded untagged files.

---

## 8. Superseded results (do not quote)

| Superseded result | Replaced by |
|---|---|
| E1 "MPL lowest 9/2/1" (gain grid g ≥ 0.005) | 6/6/0 (§1) |
| "MPL ≈ 0.4–0.9 µs per update" (small k) | 1.3–2.0 µs at the selected configurations |
| Family-bank 19-cell counts including oscillation (5/14/0, χ 19/0/0, 9/5/5, 12/2/5, 8/7/4) | 15-cell counts in §5 |
| Excess-log-loss counts against the Xiao/Hermite log banks | none: their density is a Gaussian built from ρ̂, a working model |
| Own-density banks, `hermite_bank_own` | invalid or provisional |
| ISE results and figures | dropped (§0.3) |
| `chapter6_results.md`, `chapter6_results_final.md` (original paths) | the matched results in this file |
| Bank figures using δ = 3·10⁻⁴, and single-seed path figures | not used |

---

## Sources

- **Result tables** (`experiments/results/`):

| Table | Covers |
|---|---|
| `budget_gainext_2026-11-11.txt` | E1 |
| `matched_eval_2026-11-10.txt` | E3 RMSE, VFF, regret |
| `density_forgetting_banks_2026-11-11.txt` | E3/E4 density |
| `hedge_vs_fixedshare_2026-11-11.txt` | E4 |
| `density_followup_2026-11-10.txt` | E5 excess log loss, floor, mixture |
| `density_hellinger_2026-11-11.txt` | E5 Hellinger |

- **Notes:**
  - `notes/studies/chapter6_metrics.md` §1–17 (metrics, ISE analysis, Steps 1–4, decisions);
  - `notes/studies/budget_mpl.md` §25 (E1 rerun);
  - `notes/studies/ew_marginals.md` §31–33 (matched protocol, density follow-up).
- **Other docs:**
  - `docs/chapter6_results_matched.md` (inventory);
  - `docs/chapter6_outline.md` (writing outline);
  - `docs/chapter6_visual_plan.md`, `docs/chapter6_figure_review.md`.
