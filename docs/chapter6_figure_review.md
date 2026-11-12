# Chapter 6 figure review (2026-11-11), for Codex

**Scope:** every figure in `notes/thesis/figures/matched/` that Chapter 6 could use. Each was inspected visually against
the final results (`docs/chapter6_outline.md`, `notes/studies/chapter6_metrics.md`, `budget_mpl.md` §25).

**Verdicts:**
- **MAIN:** keep in the main text.
- **APPX:** move to the appendix.
- **FIX:** regenerate or restyle before use.
- **DROP:** superseded.

**Ground rules for every regenerated figure:**
- matched protocol only; no oscillation;
- no `*_bank_own` / Hermite own-density rows;
- no density comparisons against the Xiao/Hermite "Gaussian at ρ̂" forecasts;
- label memories as N_eff;
- state the cell set and whether error bars are paired.

## E1: memory budgets (§6.2)

| Figure | Intended message | What the plot shows | Verdict and fixes |
|---|---|---|---|
| `ch6_budget_paired.png` | No estimator dominates. MPL wins under drift and on Gaussian/Gumbel jumps; Xiao (or Hermite) wins under constant dependence and Clayton/t₄ jumps. 6/6/0 at every budget. | Horizontal dot plot, 12 rows (4 families × constant/drift/jump, grouped by path). x = MPL minus the better of Xiao/Hermite in test RMSE (×10⁻², paired mean ± 2 SE, 100 seeds); three dots per row = 16/64/256 KiB. Constant rows all right of 0 (+0.15 to +0.30); drift rows all left (−0.17 to −0.67); jump split (Gaussian and Gumbel left, Clayton and t₄ right). Dots for the three budgets nearly coincide. | **MAIN.** Clear, and it tells the E1 story alone. |
| `ch6_budget_memory.png` | Accuracy against retained memory. Once tuned, accuracy barely improves beyond 16 KiB, and the curves show where each method stops gaining. | 2 × 3 panels (rows Gaussian, Clayton; columns constant, drift, jump). x = measured peak state (KiB, log); y = test RMSE ×10⁻². One curve per estimator (each point one approximation setting at its validation-best gain), unpaired ±2 SE; dotted verticals at 16/64/256 KiB; dashed line = MPL with exact PIT (reference). Constant: flat curves near 1.5 (Xiao/Hermite) and 1.7 (MPL), exact PIT 1.23/1.50. Drift and jump: MPL drops steeply up to about 10 KiB, then flat below Xiao/Hermite (Gaussian), or level/slightly above them (Clayton jump). | **MAIN** (or APPX if space is tight). Unpaired bars look overlapping; the caption must point to the paired figure and table. Long subtitle. |
| `ch6_budget_memory_all.png` | As above, all four families. | Same layout, 4 × 3 panels. | **APPX.** |
| `ch6_budget_runtime.png` | Cost: at the configurations validation selects, all three cost about 1–2 µs per observation, and MPL is not the cheapest. | 3 panels (Gaussian constant/drift/jump). x = time per tick (µs, log); y = test RMSE; curves labelled by B/N/k at the ends. Xiao reaches its floor at about 1 µs; MPL needs k ≥ 256 (≈ 1.3–2 µs) to reach its lower floor on drift and jump; under constant MPL sits above Xiao at every cost. | **APPX.** Gaussian only; legend overlaps the constant curve; title lacks "validation-best gain". |
| `ch6_budget_recovery.png` | Tracking after the jump. | 2 panels (Gaussian, Clayton jump): mean ρ̂_S paths of the three estimators around t = 2500 against the truth. The configurations were tuned on post-change RMSE, so g = 0.05, a different rule from the rest of E1. The three lines overlap almost completely. | **DROP** (or APPX with the selection rule stated). Low information and an inconsistent selection rule. |
| `ch6_budget_rmse_vs_k.png` | Implementation check: the hand-written KLL and DataSketches give the same accuracy at the same k from k = 128; the hand-written one uses 4–6× more memory. | Top 2 × 3 panels: RMSE against k (log₂) for the two KLL versions plus the exact-PIT line. Bottom panel: memory (KiB) against k for both. | **APPX** (implementation detail, not an E1 result). Title claims "equal accuracy from k = 128" (true for drift/jump; check constant at the extended grid). |
| `ch6_budget_memory{,_all}_ds_vs_own.png`, `*_ownkll.png` | Same budget frontiers with the hand-written KLL. | Like `ch6_budget_memory`, with an extra MPL curve. | **APPX** at most; likely **DROP** (redundant with `rmse_vs_k`). |

## E2: fixed forgetting (§6.3)

| Figure | Intended message | What the plot shows | Verdict and fixes |
|---|---|---|---|
| `ch6_neff_optimum.png` | Effective memory trades variance against tracking bias. The measured optimum agrees with N\* = (2σ²/d²)^{1/3}, and the bank's weighted memory follows the local optimum. | (a) Xiao, (b) Hermite on Gaussian drift: squared bias (orange, rising), variance (blue, falling) and MSE (black, U-shaped) against N_eff (log–log, 20 memories), with model lines σ²/N and (dN/2)²; vertical lines at measured N\* = 393/430 and predicted 426/448. (c) MPL log bank: weighted memory Σw_k N_eff,k over time (green) against the local optimum N\*(t) (dashed), ρ_S drifting 0.19 → 0.79; the bank stays within a factor of about 1.3 after t ≈ 1000. | **MAIN.** Clear. **FIX the script** (`plot_neff_optimum.py`), which defaults to the untagged original-path files; the figure itself shows the matched numbers. The caption must state the scope (Xiao, Hermite, Gaussian drift). |

## E3: adaptive forgetting (§6.4)

| Figure | Intended message | What the plot shows | Verdict and fixes |
|---|---|---|---|
| `ch6_bank_forest.png` | The log bank beats every fixed and adaptive competitor after jumps, but loses to the hindsight-best fixed memory under drift, and is level with slow VFF there. | Two columns (drift, jump). Rows: fixed M = 199, best fixed M (hindsight), VFF slow, VFF fast, uniform weights, squared bank. Each row has 12 points (4 families × Xiao/Hermite/MPL; colour = estimator, marker = family), x = log bank minus competitor (RMSE ×10⁻³, symlog, paired ±2 SE), with n/12 counts on the right. Drift: best fixed 0/12 (all right of 0), VFF slow 0/12 (around 0), others left. Jump: all 12/12 left except squared 11/12. | **MAIN, after a FIX:** add a **constant** column (bank vs best fixed 0/12/0; vs VFF slow mostly level); add the **no-forgetting** row (MPL only, from Step 3) or note it in the caption; the symlog tick "−10¹" is hard to read; shorten the title. |
| `ch6_bank_recovery.png` | Where the jump advantage comes from: by window after the jump. | Three panels (Xiao, Hermite, MPL), grouped bars on a log y-axis for windows before / 0–100 / 100–500 / 500–1500, bars = fixed M 39/199/1999, uniform, VFF fast, squared bank, log bank; mean over 4 families × 60 seeds. | **FIX or APPX:** VFF *slow*, the key smooth-change competitor, is missing; log-scale bars mislead (use dots); averaging over families hides the per-family variation. A Step 3 window table may serve better. |
| `ch6_bank_compare.png` | Which estimator works best inside the log bank. | (a) whole-run RMSE (mean ± 2 SE) of the Xiao, Hermite and MPL log banks for 4 families × drift/jump; MPL lowest in drift, close in jump. (b) Jump windows: Xiao/Hermite minus MPL bank (paired); MPL better after the jump, slightly worse before it. | **APPX.** Not one of the E1–E5 questions (it mixes E1 and E3). |
| *(missing)* Step 3 density results | The bank's gains and losses also hold for excess log loss and Hellinger (MPL banks: log vs squared 11/0/1; vs best fixed 4/8/0; mixture vs single 10/0/2). | — | **Optional new figure:** a forest like `bank_forest` for MPL with excess and H² (12 cells). Otherwise one table in 6.4/6.5. |

## E4: scoring loss and reweighting (§6.5)

| Figure | Intended message | What the plot shows | Verdict and fixes |
|---|---|---|---|
| `ch6_hedge_vs_fixedshare.png` | Controlled comparison (only δ differs): fixed share restores weight to the shortest memory after a jump, then moves back to longer memories; plain Hedge cannot recover discarded arms and stays at short memory. | Rows Gaussian and Clayton jump (60 seeds). Columns: (1) Hedge, (2) fixed share — stacked seed-mean weights on N_eff 39…1999 over t = 2000–4000 (blue ramp, jump line at 2500); (3) weighted memory Σw N_eff (log), orange Hedge vs blue fixed share. Hedge jumps from 1999 to 99 and stays there; fixed share jumps to 39 (≈ 0.95–0.99), then back through 99/199/399 (memory ≈ 220–390 by t = 4000). | **MAIN.** Clear. The weights on N_eff 99 and 199 look similar in the blue ramp, which is acceptable. Pair it with the Step 4 table (accuracy by window, including the 100–500 qualification). |
| `ch6_hedge_vs_fixedshare_all.png` | As above, all four families. | 4 × 3 panels. | **APPX.** |
| `ch6_bank_weights.png` | Log vs squared bank weight dynamics after a jump. | 2 × 2 panels: squared-loss and log-loss bank, mean weights on M = 39…1999 (default matplotlib colours) and weighted memory; no family, estimator or δ stated. | **FIX or DROP:** the source file is untagged (it may be pre-matched); unlabeled; old style ("M" not N_eff). If kept, regenerate from matched data, label it, and use the Step 4 style. |
| `ch6_bank_weights_neff.png` | Hedge vs fixed share vs fixed share with a cumulative arm. | 3 × 3 panels; the title says **δ = 3·10⁻⁴** (the superseded δ) and the data are the earlier bank-weights runs. | **DROP:** superseded by the controlled matched `ch6_hedge_vs_fixedshare.png`. |

## E5: full-copula accuracy (§6.6)

| Figure | Intended message | What the plot shows | Verdict and fixes |
|---|---|---|---|
| `ch6_mixture_regime.png` | When the truth (a Clayton–Gumbel mixture) is outside the bank, the family bank still has the best density (except about 100 ticks after the change), but its upper tail is biased before the change because it projects onto Clayton. | (a) Excess log loss by window (before, 0–100, 100–500, 500–1500, after): family-bank mixture, single Gaussian, Gaussian mixture, best fixed arm (t₄, M 1999), Gaussian floor D (bar marks); ±2 SE. (b) Seed-mean shape weights over time (Clayton about 0.8 before; t₄ about 0.65 after). (c, d) χ_L and χ_U(0.05) paths: truth, family bank, corner counter M = 399. | **MAIN.** Clear. Caption: (c, d) show seed means (bias), not RMSE. |
| `ch6_family_summary.png` | The family bank against the forgetting banks: ρ_S, density, tails and switch delays. | 4 panels; (a, b) **19 cells including oscillation**, rows include **own-density banks (Hermite invalid)** and density comparisons against the Xiao/Hermite Gaussian-at-ρ̂ forecasts; (c) tails against counters and own-density rows; (d) weight-crossing delays (medians 174/68/94). | **FIX (essential) or DROP.** If kept, regenerate with the 15 cells (no oscillation), drop all own-density rows, compare densities only against the MPL single Gaussian and Gaussian mixture, and add H². Panel (d) is fine. |
| `ch6_tail_chi.png` | After a shape switch the family bank tracks χ_L quickly; corner counters adapt slowly; in the mixture the bank's tail is biased (projection). | 2 × 2 panels: χ_L(0.05) seed-mean paths over ticks 3000–9000 (**switch at 5000**: the separate tail-bank probe, not the 2500-switch family probe), for Gauss→Clayton, Clayton→Gumbel, Gauss→t₄ and the mixture; lines: truth, counters M = 4999 and 1999, **Hermite own-density bank**, family bank. | **FIX:** remove the Hermite own-density line; state in the caption that this is the tail-bank probe (T = 10 000, switch at 5000, original protocol); χ_U is not shown. APPX if the main text uses `mixture_regime` (c, d) for tails. |
| *(missing)* family bank vs Gaussian forecasts by window | The core E5 answer: under non-Gaussian truth the family bank removes the Gaussian floor (excess −23 to −178); it costs a little under Gaussian truth; it lags about 100 ticks after a family change; Hellinger agrees. | — | **Recommended new MAIN figure** (from existing Step 2 data): for the three switches and four constant base cells, excess log loss (FB minus single Gaussian, paired ±2 SE) by window, with the H² equivalents beside it and the floor D marked. |

## Summary for the main text

- **Main:** `ch6_budget_paired`, `ch6_budget_memory`, `ch6_neff_optimum`, `ch6_bank_forest` (with the constant
  column), `ch6_hedge_vs_fixedshare`, `ch6_mixture_regime`, plus one new E5 window figure.
- **Appendix:** `ch6_budget_memory_all`, `ch6_budget_runtime`, `ch6_budget_rmse_vs_k`, `ch6_bank_compare`,
  `ch6_hedge_vs_fixedshare_all`, `ch6_tail_chi` (fixed), `ch6_bank_recovery` (fixed).
- **Drop:** `ch6_bank_weights_neff` (δ = 3·10⁻⁴), `ch6_budget_recovery`, `ch6_bank_weights` (unless regenerated),
  the `*_ownkll` / `*_ds_vs_own` budget frontiers.
- **Regenerate before use:** `ch6_family_summary` (or replace it with the new E5 figure).
- **Script hygiene:** `plot_neff_optimum.py`, `plot_thesis_bank.py`, `plot_bank_compare.py` and
  `plot_family_summary.py` read untagged (superseded) files by default. Point them at the matched/final files.
