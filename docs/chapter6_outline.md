# Chapter 6 consolidated results outline (2026-11-11)

**Sources:**
- `docs/chapter6_results_matched.md`;
- `notes/studies/chapter6_metrics.md` §12–16;
- the tables in `experiments/results/`:
  - `matched_eval_2026-11-10.txt`;
  - `density_followup_2026-11-10.txt`;
  - `density_hellinger_2026-11-11.txt`;
  - `density_forgetting_banks_2026-11-11.txt`;
  - `hedge_vs_fixedshare_2026-11-11.txt`.
- Counts marked † were recomputed for this outline from the existing per-seed CSVs, without oscillation.
- No new simulations were run.

**Protocol for every experiment unless stated:**
- Paths: matched dependence paths (`SCENARIOS=matched`), so the true ρ_S path is identical across families.
  - constant d = 0.6 (ρ_S 0.582);
  - drift d 0.2 → 0.8 (ρ_S 0.191 → 0.786);
  - jump 0.2 → 0.8 at t = 2500.
- T = 5000, burn-in 500. Seeds 42–101 (budget: test 42–141, validation 1000–1039).
- Margins: shared EW margins, N_eff,m = 500. Log-bank δ = 10⁻⁴; squared bank (η, δ) = (1, 10⁻⁴); VFF rates 0.02 / 0.1. All selected on development seeds 1–30.
- W/L/T: paired per-seed difference beyond 2 SE, per cell, no multiplicity correction.
- **Oscillation is excluded.**

**Metric hierarchy:**
- ρ_S RMSE: primary, dependence strength.
- Excess copula log loss at the true PIT, using the pre-observation forecast: primary, density.
- Squared Hellinger: secondary.
- χ_L / χ_U RMSE: primary in tail experiments.
- Recovery, regret, memory, runtime: experiment-specific.

**Training vs evaluation (state once, in 6.1):**
- Bank weights are updated by *online* losses on the estimator's own pseudo-observations (Û, V̂) and their own forecasts. No truth is used.
- Chapter 6 *evaluates* every method against the simulation truth with the same metrics, whatever its training loss.

---

## E1. Estimators under memory constraints (Xiao, Hermite, recursive MPL)

> **UPDATED 2026-11-11 (extended gain grid, `budget_mpl.md` §25): this supersedes the 9/2/1 headline below.**
> Gains were validation-tuned over {cumulative, 0.0005, 0.001, 0.002, 0.005, …, 0.10} for constant and drift cells.
> - **DS-MPL vs the better of Xiao and Hermite: 6/6/0 at every budget.**
>   - **Constant:** Xiao is best in all 4 cells, all estimators cumulative. DS-MPL is worse by +0.15 to +0.30 ×10⁻² because
>     of its KLL margins; with exact PIT it would win.
>   - **Drift:** DS-MPL wins all 4.
>   - **Jump:** DS-MPL wins Gaussian and Gumbel, loses Clayton and t₄.
> - **Memory:** tuned accuracy is nearly flat across budgets. The "plateau" reflects variance-limited estimation, not
>   approximation error.
> - **Rewrite E1 around this:** the best estimator depends on whether forgetting is needed. Without it, cheap cumulative
>   counting wins. With finite memory, the parametric MPL is most efficient unless its copula is misspecified after an
>   abrupt change.

**Design:**
- Single estimators. Each method's approximation parameter (Xiao bins B, Hermite order N, KLL sketch k) and gain are tuned on validation seeds within each memory cap of 16, 64 or 256 KiB.
- Test on 100 seeds. Paired per-seed comparisons.
- MPL uses cumulative KLL margins (Apache DataSketches): a matched-memory question, not the bank protocol.
- This is a comparison between **separately tuned** procedures.

**Evidence** (`matched_eval` "Budget"; ×10⁻² RMSE):

| Budget | DataSketches-MPL vs the better of Xiao and Hermite (W/L/T) | Lowest observed RMSE |
|---|---|---|
| 16 KiB | 9/2/1 | 10/12 cells |
| 64 KiB | 9/2/1 | 9/12 |
| 256 KiB | 9/2/1 | 9/12 |

- **Losses:** Clayton jump (+0.20 ± 0.09) and t₄ jump (+0.12 ± 0.09). **Tie:** t₄ constant.
- **Drift:** DataSketches-MPL leads in all four cells, e.g. Gaussian 4.16 vs 4.66.
- **Runtime per tick** (`budget_timing_matched.csv`, mean including estimate()):
  - Xiao 0.27–27.5 µs (B 10 → 180);
  - Hermite 0.47–28.4 µs (N 5 → 128);
  - DataSketches-MPL about 0.43–0.91 µs over k 8–128.
- **Memory** grows with B² (Xiao), N² (Hermite) and slowly in k (KLL).
- **Hand-written KLL vs DataSketches:** the same accuracy at equal k from k = 128, and 4–6× more memory (`ch6_budget_rmse_vs_k.png`).

**Interpretation:**
- The parametric MPL reaches lower RMSE than the nonparametric summaries at every budget in most cells, at near-constant cost per tick.
- It loses when the Gaussian working model is misspecified after a jump (Clayton, t₄).
- Xiao and Hermite RMSE plateau above about 64 KiB because their approximation error becomes negligible. Varying B and N did not change it (B = 30 vs 180: 3.49 vs 3.48; Hermite N = 128: 3.51).

**Figures:**
- `matched/ch6_budget_memory.png` (Gaussian and Clayton; `_all` in the appendix);
- `ch6_budget_runtime.png`, `ch6_budget_recovery.png`, `ch6_budget_rmse_vs_k.png`.

**Check against the tuning audit:**
- The rankings above are from the post-audit rerun (`ew_marginals.md` §18: normalised start-ups, corrected memory accounting `budget_mpl.md` §24), redone on matched paths with per-seed output. Constant and jump cells are identical to the original-path rerun.
- Quote these, not the pre-audit "10/12" from `chapter6_results.md`.
- The exact-PIT ablation is a diagnostic only (EW margins beat it on misspecified drift and jump; explanation unproven).

## E2. Fixed exponential forgetting (the bias–variance trade-off)

**Evidence:**
- **Measured optimal memory on Gaussian drift:** N* = 393 (Xiao) and 430 (Hermite). The approximation (2σ²/d²)^{1/3} with d = 1.19·10⁻⁴ per tick gives 426 and 448 (`matched/ch6_neff_optimum.png`).
- This is consistency with the bias–variance model on this path, not an independent proof.
- **Best fixed MPL memory** (Step 3, chosen in hindsight per cell): N_eff 1999 constant, 399 drift, 199 jump (99 for some density metrics). This is the same ordering in all four families and under all three metrics.

**Interpretation:**
- Long memory is efficient when dependence is stable.
- Gradual drift needs an intermediate memory that the theory predicts well.
- Abrupt change needs short memory. No single λ is good everywhere, which motivates E3.

**Figures:** `ch6_neff_optimum.png`, `matched/ch6_bank_weights_neff.png` (the bank's weighted memory vs N*(t)).

**Gap:** N* was measured for Xiao and Hermite on Gaussian drift only; MPL and the other families are not shown.

## E3. Adaptive forgetting: none vs fixed vs VFF vs bank

**Evidence, ρ_S RMSE** (12 cells = 4 families × Xiao, Hermite, MPL; †constant row recomputed from `thesis_bank_matched500`):

| Log bank vs | constant | drift | jump |
|---|---|---|---|
| Hindsight-best fixed memory | †0/12/0 (+6.3) | 0/12/0 (+4.6) | 12/0/0 (−16.6) |
| VFF slow (A = VFF) | 0/7/5 (+2.6) | 1/0/11 (−0.2) | 0/12/0 (+27.6) |
| VFF fast (A = VFF) | 0/12/0 (+15.8) | 0/12/0 (+6.5) | 0/12/0 (+10.5) |

**Density evaluation (MPL only, Step 3; 12 cells = 4 families × 3 scenarios):**
- **Log bank vs best fixed memory:**
  - 4/8/0 under all three metrics: wins in all 4 jump cells, losses in all 8 constant/drift cells;
  - steady-state cost: excess +0.4 to +1.0 ×10⁻³;
  - jump gain mainly in the first 100 ticks after the change.
- **No forgetting** (cumulative MPL on the same margins):
  - beats the bank only on constant paths;
  - loses heavily on drift (excess −32 to −38 in the bank's favour) and after jumps (−189 to −234 post).
- **VFF:**
  - slow VFF is level with the bank on smooth paths (ties or ±0.3 excess) and much worse after jumps;
  - fast VFF is worse in steady state but recovers faster 100–500 ticks after a jump.

**Interpretation:**
- Adaptive banks improve abrupt-change performance.
- They do **not** beat the hindsight-best fixed memory under constant dependence or gradual drift. That comparator is an oracle chosen per cell; the bank does not know the scenario.
- Slow VFF matches the bank under smooth change. The bank's advantage is robustness to abrupt change.

**Figures:**
- `matched/ch6_bank_compare.png`, `ch6_bank_forest.png`, `ch6_bank_recovery.png`, `ch6_bank_weights.png`;
- check that they exclude oscillation (open item 3).

## E4. Choice of loss and bank reweighting

**(a) Log vs squared weighting** (separately tuned η and δ, but the same arms and streams):
- **ρ_S RMSE:** 23/0/1 over 24 cells (drift + jump × 4 families × 3 estimators), mean −3.8 ×10⁻³.
- **Squared vs uniform weights:** 24/0/0 (−16.3).
- **MPL density (Step 3, 12 cells):**
  - single-Gaussian forecasts: excess 11/0/1, H² 11/0/1, RMSE 10/0/2;
  - gains are largest after jumps (post excess −3.0 to −5.3 ×10⁻³).
- **Density mixing** (same weights, mixture vs single Gaussian at the combined ρ̂; controlled):
  - excess 10/0/2, H² 12/0/0;
  - largest just after jumps;
  - it helps the squared bank more than the log bank.

**(b) Plain Hedge vs fixed share** (**controlled**: only δ differs; MPL log bank, η = 1):

| Fixed share − Hedge, ×10⁻³ | ρ_S RMSE | Excess (mixture) | H² (mixture) |
|---|---|---|---|
| Jump, 0–100 after (4/4 W) | −119 to −136 | −90 to −140 | −32 to −37 |
| Jump, post (4/4 W) | −23 to −24 | −4.6 to −9.0 | −1.6 to −1.8 |
| Constant / drift (cost) | +0.4 to +2.8 | +0.0 to +0.1 | ≈ 0 |
| Jump, 100–500 after | +5.8 to +6.3 or tie | +1.1 to +7.5 (3 L, 1 T) | 2 W, 2 T |

- **Mechanism** (seed-mean weights, `ch6_hedge_vs_fixedshare.png`):
  - Hedge's short arms decay to ≈ 0 weight during the stable period.
  - After the jump Hedge never uses the 39-tick arm (max weight ≤ 0.02) and stays at short memory (≈ 100–156 at t = 4000).
  - Fixed share moves 0.92–0.99 of the weight to the 39-tick arm within about 50 ticks, then back to 199–399 (memory 221–387 at t = 4000).
- **Qualification:** in the 100–500-tick recovery window Hedge is sometimes better under excess log loss. Fixed share does not dominate every post-change window.

**(c) Regret checks:**
- 0 exceedances for the covered procedures (density-mixture log bank with δ = 10⁻⁴ and δ = 0, squared loss at η = 1/32, and the family-bank mixture).
- The uncovered squared bank (η = 1) and point-density log bank exceed the reference expression in 50.8% and 42.7% of runs. These are not theorem violations.

## E5. Full-copula accuracy (family bank vs Gaussian forecasts; tails)

**Question (narrowed, Sally 2026-11-11):** does accurate estimation of Spearman's ρ imply accurate copula-density
prediction, and does allowing multiple copula families improve prediction when the dependence structure changes?
- Memory adaptation addresses changes in the appropriate forgetting horizon (E3/E4).
- Copula-family flexibility addresses misspecification of the dependence structure (E5).
- **Do not claim** to have isolated or quantified their independent contributions, or that they do not interact.
- The fixed-memory family-bank ablation is deferred. Run it only if a central claim comes to need it.

**Evidence.** All excess and H² values are ×10⁻³, family-bank mixture minus comparator.

**Against the MPL log bank's Gaussian forecasts** (single Gaussian at the combined ρ̂, or the Gaussian mixture):
- **Non-Gaussian truth:** excess −23 to −178.
- **Gaussian truth:** a small cost, +0.2 to +2.
- **Adaptation lag in the first 100 ticks after a family change:** Clayton → Gumbel +46.3, mixture +19.5.
- **Hellinger agrees in 138/144 verdicts,** with no sign reversals and differences 3–6× smaller. Examples (excess / H²):
  - Clayton constant: −76.3 / −20.0;
  - Gauss → Clayton post: −72.3 / −18.5;
  - mixture post: −4.6 / −0.8.
- **Gaussian KL floor D(t)** (expected excess of the KL-best Gaussian): Clayton 70–78, Gumbel 25–29, t₄ about 33. Almost all of the Gaussian forecasts' excess is this floor; estimation error adds +1 to +3.5 in steady state.

**ρ_S RMSE, 15 cells**† (4 families × 3 scenarios + 3 switches):
- against the MPL log bank: 7/5/3;
- against its own best single arm: **4/11/0**.
- So full-copula modelling does not improve rank-correlation accuracy.

**Tails, χ(0.05) RMSE against EW corner counters** (M = 99 / 399 / 1999):
- **15 cells:** 15/0/0 for χ_L and for χ_U at every M.†
- **Mixture regime** (the truth is not in the bank):
  - before the change the family bank is worse than the long counters for χ_U (+13 vs M = 399, +53 vs M = 1999). It projects onto Clayton.
  - after the change it beats all counters in both tails.
- **Tail bank** (separate probe, constant-ρ_S regimes, original protocol): against counters M = 399 / 1999 / 4999: 20/0/0, 16/4/0, 16/2/2. The losses are the Clayton–Gumbel mixture.

**Separating the two banks' contributions:**
- **Family coverage:**
  - **Gaussian-only bank** (same machinery and memories, Gaussian shape only): the family bank is far better after shape changes (excess post −24 to −72 for Gauss → t₄, Clayton → Gumbel and Gauss → Clayton; −50 for t₃; only −4 for the near-Gaussian t₈) and costs +0.5 when the truth is Gaussian.
  - **Best fixed family-and-memory arm:**
    - in steady state the arm with the true family is within 1·10⁻³ of the bank, so most of the gain over Gaussian forecasts is **structural** (the right family is in the bank);
    - across a family change the bank beats every fixed arm (−7.7 to −41.0 whole-window excess), which is what adaptive weighting adds.
- **Adaptive memory** (the forgetting-factor bank, E3/E4): the gain is around abrupt changes in ρ_S, with a steady-state cost against the best fixed memory.

**Interpretation:** full-copula modelling matters for density and tails, not for ρ_S. Its value depends on covering the true family.

**Figures:**
- `matched/ch6_mixture_regime.png`, `ch6_tail_chi.png`;
- `ch6_family_summary.png`, which needs regenerating (open item 4).

---

## Controlled vs separately tuned comparisons (for the text)

- **Controlled** (only one thing differs):
  - Hedge vs fixed share (δ);
  - mixture vs single Gaussian (forecast construction);
  - Gaussian-only vs family bank (shape set);
  - fixed arms, no forgetting and the banks share arms and streams.
- **Separately tuned:**
  - log vs squared bank (each with its own dev-selected η and δ);
  - bank vs VFF (VFF rates by a fixed rule);
  - E1 estimators (each tuned within the memory cap on validation seeds).

## Negative findings and trade-offs to keep

- Every bank loses to the hindsight-best fixed memory under constant dependence and drift.
- Fixed share costs a little in steady state, and Hedge is sometimes better 100–500 ticks after a jump.
- Slow VFF matches the bank under smooth change.
- The family bank does not improve ρ_S RMSE (4/11/0 against its best arm). It lags about 100 ticks after a family change and pays a small cost under a Gaussian truth.
- Under misspecification no Gaussian forecast can beat the KL floor.
- DataSketches-MPL loses to Xiao or Hermite in two misspecified jump cells.

## Unresolved inconsistencies and open items

1. **Counts that included oscillation must not be quoted.** These include the family bank's 19-cell counts (best arm 5/14/0, log banks 9/5/5, 12/2/5, 8/7/4, χ 19/0/0) and the VFF oscillation column. Use the † 15-cell recounts above.
2. **Do not quote the excess-log-loss counts against the *Xiao/Hermite* log banks** (16/2/1 etc.). Their density is a Gaussian at ρ̂, a working model, not their own. For E5 use only the MPL log bank's single and mixture forecasts.
3. **Check the matched bank figures for oscillation panels** before use. `ch6_bank_*` came from runs that included oscillation; regenerate without it if present.
4. **`ch6_family_summary.png`** includes provisional own-density banks (`hermite_bank_own` is invalid) and oscillation. Regenerate it or drop it.
5. **Not rerun:** the single-seed path figures (`ch6_bank_path*`, `ch6_family_path`) and the AAPL/GOOG real-data study.
6. **Cell sets differ between results:** E4(a) RMSE uses 24 cells (drift + jump, three estimators); the density results use 12 MPL cells (constant + drift + jump). State the cell set beside every count.
7. **The tail bank** ran on the original protocol. Its constant-ρ_S regimes are unaffected by the matched paths, but say so.
8. **E2's N\*** covers only Gaussian drift with Xiao and Hermite.
