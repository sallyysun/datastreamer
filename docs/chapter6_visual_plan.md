# Chapter 6 visual plan (proposal, 2026-11-11)

**Principle:** one finding per figure, readable in 5–10 seconds, built from the paired comparison that carries the
evidence. Everything else goes to a table, the caption or the appendix. All data are existing validated results; no new
runs.

**Palette and conventions** (colour-blind-safe, the palette already used; validated):

| Role | Colour |
|---|---|
| Xiao | blue `#2a78d6` |
| Hermite | orange `#eb6834` |
| Recursive MPL | aqua `#1baf7a` |
| Method of interest when comparing *procedures* (log bank, fixed share, family bank) | violet `#6250d6` |
| Its baseline (comparator, plain Hedge, Gaussian forecast) | mid-grey `#8a8984` |

- **Paired differences:** always *method of interest minus comparator*, so **negative means the method of interest is
  better**. The zero line is drawn heavy black, with a shaded "better" side labelled in words.
- **Uncertainty:** paired ±2 SE.
- **Linear axes** for differences. Log axes only for memory (N_eff, a ratio scale).
- **Ordering:**
  - families Gaussian, Clayton, Gumbel, t₄;
  - paths constant, drift, jump;
  - windows before, 0–100, 100–500, after.
- **Labels:** memories as N_eff.
- **Size:** each figure sized for its printed width (≈ 15 cm), with text ≥ 8 pt at that size.

## Proposed main-text figures (6, about one per question)

### Figure 1 (E1): no estimator dominates. REDESIGN of `ch6_budget_paired`.

**Conclusion:** Xiao/Hermite are more accurate under constant dependence, MPL under drift, and after jumps it depends on
the copula family.

**Layout:**
- One panel, three stacked groups (constant / drift / jump), each with 4 family rows.
- Only one budget (64 KiB): the three budgets agree to within a few hundredths, so repeating them adds nothing.
- x = MPL minus the better of Xiao/Hermite, test RMSE ×10⁻² (paired mean ± 2 SE, 100 seeds).
- Heavy zero line; side labels "MPL more accurate" / "Xiao/Hermite more accurate".
- Each point coloured by the side it falls on (aqua = MPL better, blue = Xiao/Hermite better), grey if within 2 SE.
- The winning comparator (Xiao or Hermite) written as small text at the row end.

**Why not a table:** the sign pattern by regime is the finding, and a figure shows it at a glance. Table 6.x keeps the
numbers.

**Appendix:**
- the three-budget version;
- the memory-vs-RMSE frontiers (`ch6_budget_memory`, `_all`);
- runtime;
- RMSE vs k.

### Figure 2 (E2): the optimal memory balances variance and bias. SIMPLIFY `ch6_neff_optimum`.

**Conclusion:** under drift, error is U-shaped in N_eff, and the predicted optimum (426/448) lies close to the measured one
(393/430).

**Layout:**
- Two small panels (Xiao, Hermite), Gaussian drift.
- x = N_eff (log), y = MSE of ρ̂_S (×10⁻³, linear, so the U is not distorted).
- Solid black MSE curve; light orange squared bias and light blue variance, unlabelled except in a single shared legend.
- Two vertical markers: solid = measured optimum, dashed = predicted (2σ²/d²)^{1/3}, with values in the legend instead of
  on-plot text.
- Remove the model-line overlays and the panel (c) bank-memory trace.

**Why not a table:** the U shape is the explanation. Two numbers in a table cannot show *why* neither extreme is optimal.

**Appendix:** panel (c), the MPL bank's weighted memory against the local optimum N\*(t). That is an E3 property, not E2.

### Figure 3 (E3): adaptive banks help after jumps, not under stable or drifting dependence. REDESIGN `ch6_bank_forest`.

**Conclusion:** the log bank beats the hindsight-best fixed memory after every jump but loses to it in every constant and
drift cell; slow VFF matches it under smooth change and loses after jumps.

**Layout: 2 rows × 3 columns** (columns constant / drift / jump).
- **Row 1, comparator = hindsight-best fixed memory** (oracle, chosen per cell).
- **Row 2, comparator = slow VFF.**
- In each panel, 12 points: 4 family rows, each with 3 estimators (Xiao blue, Hermite orange, MPL aqua) slightly offset.
- x = log bank minus comparator, RMSE ×10⁻³ (linear, shared within a row), paired ±2 SE (60 seeds).
- Heavy zero line; shaded "bank better" side.
- Each panel's W/L/T count in its corner, e.g. "0/12/0".
- Negative results are as visible as positive ones: the constant and drift columns of row 1 sit visibly right of zero.
- **Dropped from the main figure** (no clutter): fixed M = 199, fast VFF, uniform weights and the squared bank. Fast VFF
  and uniform go to an appendix table; the squared bank belongs to E4.

**Why not a table:** the win/lose flip between constant/drift and jump is the finding, and it is immediate in a panel grid.

**Appendix:**
- one table of all competitors (fixed 199, VFF fast, uniform, squared, no forgetting for MPL) with W/L/T and mean
  differences;
- the window-by-window recovery (replacing the current bar chart `ch6_bank_recovery`).

### Figure 4 (E4): fixed share restores short memory after a jump, then returns to long memory. REDESIGN `ch6_hedge_vs_fixedshare`.

**Conclusion:** with only δ changed, fixed share moves the bank to its shortest-memory arm right after the jump and back to
long memory as dependence settles; plain Hedge cannot recover discarded arms and stays at short memory.

**Layout:** one column (Gaussian jump, 60 seeds), shared time axis t = 2000–4000, a vertical line at the jump.
- **(a) Thin context strip:** the true ρ_S path (0.19 → 0.79). The per-tick bank estimates were not saved in these runs,
  so there is no estimate overlay without a new run.
- **(b) Main panel:** weighted memory Σ_k w_k N_eff,k (log y, ticks at 39, 99, 199, 399, 1999). Fixed share violet,
  Hedge grey, each labelled directly at its line end.
- **(c) Supplementary panel:** two simple weight summaries per method instead of 5 stacked bands: the weight on the
  shortest arm (N_eff = 39, solid) and the weight on long arms (N_eff ≥ 399, dashed). Fixed share's N_eff = 39 weight
  spikes to ≈ 1 and decays; Hedge's stays ≈ 0.

**Accuracy:** the paired accuracy by window goes in the accompanying table (Step 4: 0–100 after, 100–500, post; with the
100–500 qualification).

**Why not a table:** the mechanism is temporal; only a time plot shows the jump-and-return.

**Appendix:** the stacked weights for all four families (current `_all`, kept as a supplement).

### Figure 5 (E5): accurate ρ_S does not give an accurate density; family flexibility does. NEW.

**Conclusion:** under non-Gaussian dependence the family bank removes almost all of the Gaussian forecasts' excess log loss;
it costs a little when the truth is Gaussian and lags for about 100 observations after a family change.

**Comparator:** the single Gaussian at the MPL log bank's combined ρ̂ (the Gaussian mixture is within 0.5; noted in the
caption).

**Layout: two panels with their own scales.**
- **(a) Stationary dependence, by generating copula** (constant path, whole run; 4 rows).
- **(b) Family change at constant ρ_S, by window** (rows Gauss → t₄, Clayton → Gumbel, Gauss → Clayton, Clayton–Gumbel
  mixture; four window markers per row: before, 0–100, 100–500, after).

**The scale problem:** in absolute terms the effects span +0.2 (Gaussian cost) to −76 (Clayton) ×10⁻³ nats. A linear axis
hides the Gaussian cost, and a log axis would exaggerate it. Proposal:
- **x = relative excess**, the family bank's excess log loss as a percentage of the single Gaussian's, minus 100%.
- So 0% = no difference, −100% = all excess removed, +15% = 15% worse.
- Paired uncertainty by the delta method over seeds.
- Shows: Gaussian +15% (cost, visible); Clayton, Gumbel, t₄ about −96 to −98%; and the post-switch +160% (Clayton →
  Gumbel, 0–100) as a clear positive excursion.
- Absolute values go at the row ends or in Table 6.y, so the magnitudes are not hidden.
- **Alternative,** if you prefer absolute units: the same layout with a linear axis and the Gaussian row annotated
  numerically.

**Hellinger:** not in the main figure; the appendix has the same layout with H² (agreement 138/144 stated in the caption).

**Why not a table:** the three-part message (big gain, small Gaussian cost, temporary penalty) is about signs and
magnitudes across regimes, which a figure makes immediate.

**Replaces** `ch6_family_summary` (dropped).

### Figure 6 (E5, optional second): the family bank's limit when the truth is outside it. SIMPLIFY `ch6_mixture_regime`.

**Conclusion:** for a Clayton–Gumbel mixture the family bank still predicts the density best, but it projects onto one
family, so its upper-tail coefficient is biased before the change.

**Layout: two panels.**
- **(a) Excess log loss by window:** family bank, single Gaussian, best fixed arm, Gaussian floor.
- **(b) χ_U(0.05) path:** truth, family bank, corner counter (N_eff = 399).
- Drop the weights panel and the χ_L panel to the appendix.

**Keep only if** the limitation is part of the 6.6 argument. Otherwise the appendix.

## Appendix figures (kept, possibly restyled)

| Figure | Content |
|---|---|
| A1 | E1 three-budget paired plot |
| A2 | `ch6_budget_memory_all` |
| A3 | `ch6_budget_runtime` |
| A4 | `ch6_budget_rmse_vs_k` |
| A5 | E2 bank memory against N\*(t) |
| A6 | E3 window recovery (dot plot) |
| A7 | E4 stacked weights, all families |
| A8 | E5 with H² |
| A9 | Mixture regime, full |
| A10 | `ch6_tail_chi`, with the Hermite own-density line removed, retitled as the tail-bank probe (switch at 5000, original protocol) |

## Drop

- `ch6_family_summary`: oscillation, invalid own-density rows, working-model densities. Replaced by Figure 5.
- `ch6_bank_weights_neff`: superseded δ = 3·10⁻⁴; replaced by Figure 4.
- `ch6_bank_weights`: unlabeled, unclear source; Figure 4 covers it.
- `ch6_budget_recovery`: lines overlap; inconsistent selection rule.
- `ch6_bank_recovery`: log-scale bars, no slow VFF; becomes A6.
- `ch6_bank_compare`: the log banks across estimators is not one of E1–E5. Drop, or keep as appendix context.
- `ch6_budget_memory_ownkll`, `_all_ownkll`, `ch6_budget_memory_ds_vs_own`, `_all_ds_vs_own`: redundant with A4.

## Tables accompanying the figures (main text)

| Table | Content |
|---|---|
| 6.1 | E1 cells at 64 KiB (already drafted) |
| 6.2 | E3 all competitors, W/L/T and mean paired differences |
| 6.3 | E4 log vs squared, and mixture vs single (excess, H², RMSE W/L/T) |
| 6.4 | E4 fixed share minus Hedge by window (RMSE, excess, H²) |
| 6.5 | E5 absolute excess and H² for the cells in Figure 5 |

## Data sources (all validated; no new runs)

| Figure | Source |
|---|---|
| F1 | `budget_sweep_final` / `budget_seeds_final` |
| F2 | `memory_sweep_matched` |
| F3 | `thesis_bank_matched500[_fam]` |
| F4 | `thesis_bank_weights_{hedge0,density4}_s*` |
| F5 | `family_bank_*_density3` (constant/drift) and `density3b` (jump, switches, mixture) |
| F6 | `density3b` claygum_mix + `family_bank_path_*_density2_claygum_mix` |
