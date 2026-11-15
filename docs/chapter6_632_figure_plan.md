# Section 6.3.2 figures and tables: plan and status

The figures and tables Sally asked for on 2026-11-12 for **Section 6.3.2: Comparison of Estimation Accuracy**. Ask for
"the 6.3.2 figure list" to pull this up. Status is updated as items are made.

## Parts of the subsection
1. Correlation estimation accuracy.
2. Predictive density accuracy.

**Goal:** compare fixed forgetting, VFF and MFF; show when adaptation improves estimation; keep correlation accuracy
separate from predictive-density accuracy. Figures and tables only for now: no thesis prose, no interpretation.

## Configuration (all items)
- The matched Chapter 6 runs of 12 November with cumulative DataSketches KLL margins (k = 256):
  `experiments/probe_strategies.cpp` and `results/strategies_*`.
- Estimators: RMPL, Counting Matrix (Xiao), Hermite Series.
- Families: Gaussian, Clayton, Gumbel, Student-t4.
- Scenarios: constant, gradual drift, abrupt jump.
- 60 paired seeds (42–101); prequential timing.
- Strategies: Fixed, VFF (α = 0.02 only) and MFF (log loss, fixed share; η = 1, δ = 1e-4).
- Fixed candidates: N_eff 39, 99, 199, 399, 1999.
- **Fixed benchmark, chosen per metric:** for each family × estimator × scenario, the N_eff minimising that metric's
  whole-run error averaged over the 60 seeds. The RMSE, excess-log-loss and Hellinger choices can differ. It is a
  hindsight reference, not a per-observation oracle, and an adaptive method doesn't have to beat it to be useful.
- **Colours** (one per strategy, the same in every figure): Fixed charcoal `#333333` solid; VFF purple `#8E6BB8` dashed;
  MFF crimson `#C2185B` solid; all 1.1 pt.

## Items

| Item | Content | Status | File |
|---|---|---|---|
| **Figure A** | 3×3 (columns RMPL / Counting Matrix / Hermite Series; rows constant / drift / jump); pointwise Spearman RMSE across the 60 seeds; Gaussian; Fixed = Gaussian RMSE-best N_eff, held constant; full period for constant and drift; jump window with a dashed line at the change; MSE smoothed over 25 observations within regimes, never across the jump | **Done**, checked (whole-run MSE from the per-tick data = per-seed MSE) | `images/ch6/ch6_strategies_rmse_time.png` (`plot_strategies.rmse_time`) |
| Figure A variant | Mean estimated ρ_S against the true ρ_S path, with and without a 10–90% band across seeds | **Done**; best used for the jump row only | `images/ch6/ch6_strategies_est_time.png`, `_noband.png` (`plot_strategies.est_time`) |
| **Table A** | Whole-run Spearman RMSE; rows scenario × estimator; columns Fixed / VFF / MFF; mean over 4 families, with SE; RMSE-best Fixed per family before averaging | **Done** | `experiments/results/ch6_table_a_rmse.tex` and `.txt` (`make_table_a.py`) |
| **Figure B** | Same 3×3; excess log loss over time (mean over seeds); Gaussian; Fixed = excess-best N_eff (not the RMSE choice); investigate the Hermite extremes, no silent clipping, omission or winsorising; flag any axis transformation | Candidate made (symmetric-log jump row, linear elsewhere, per-metric Fixed); needs final check | `images/ch6/ch6_strategies_excess_time.png`; diagnostic median version `_median.png` |
| **Figure C** | Same 3×3; squared Hellinger over time; Gaussian; Fixed = Hellinger-best N_eff; the same Gaussian plug-in forecasts as the other density metrics | Candidate made (checkpoints, no line across the jump, axis fitted to the data); needs final check | `images/ch6/ch6_strategies_h2_time.png` |
| **Table B** | Excess log loss and squared Hellinger (one table or two); rows scenario × estimator; columns Fixed / VFF / MFF; **Gaussian copula**; Fixed chosen per metric; with SE. Note: all density forecasts are Gaussian copulas from the predicted ρ_S, so for Counting Matrix and Hermite they are working-model plug-ins, not native densities | **To do** | — |
| **Appendix tables** | Full family-level results (Gaussian, Clayton, Gumbel, t4) by scenario, estimator and metric, for Fixed / VFF / MFF | Partly: `results/strategies_tables.tex` exists but still has VFF α = 0.1 columns; needs a 3-strategy version | `results/strategies_tables.tex` (`make_strategies_tex.py`) |
| **Figure D** | Grouped bars of whole-run RMSE (4-family mean, SE whiskers, not stacked): 3×3 (columns estimators, rows scenarios, 3 bars per panel) and a simpler 1×3; compare readability with Table A | Candidates made | `images/ch6/ch6_strategies_rmse_bars.png`, `_bars_1x3.png` |
| **VFF sensitivity** (appendix) | 7 rates (0.005–0.2); keep the existing results and figures; don't add several VFF rates to the main figures | **Done**; preserved | `images/ch6/ch6_vffsens_rmse.png`, `ch6_vffsens_h2.png`; `results/vffsens_2026-11-12.txt` |
| **Validation report** | Which values changed after the per-metric Fixed selection, plus the checks below | Partly: Table A unchanged; for Table B the per-metric choice differs only on the jump path (RMPL Gaussian excess and Hellinger N_eff 99; Hellinger 99 in 6 of 12 family–estimator cells) | — |

## Formatting (all figures)
- No titles inside figures (LaTeX captions); one shared legend.
- Consistent strategy colours and line styles; clear row and column labels; light gridlines.
- Axes shared within a scenario row where sensible; readability first in the 3×3 grids.
- Readable at page width; high-resolution PNG; no overlapping labels.
- Change points clearly marked.

## Validation checklist
1. Fixed chosen separately for RMSE, excess log loss and Hellinger.
2. Fixed N_eff constant within each stream.
3. VFF and MFF reproduce the 12 November runs (confirmed: the sensitivity rerun matches exactly).
4. Figures agree with the summary tables where comparable.
5. Predictions evaluated before the current observation is absorbed.
6. No EW-500 results mixed into the KLL runs.

Related: `docs/chapter6_structure.md` (how Chapter 6 is to be developed).

## Reorganised figure set (2026-11-12): use this one

PNG, in `images/ch6/`, all produced by `plot_strategies.reorganised()` in `experiments/plot_strategies.py`.

| Use | File | Content | Data |
|---|---|---|---|
| Main Fig. 1 | `ch6_strategy_rmse_bars.png` | 1×3 grouped bars of whole-run Spearman RMSE; columns constant / drift / jump; ± 2 SE | Mean over 4 families; Fixed = RMSE-best per family |
| Main Fig. 2 | `ch6_jump_adaptation.png` | 2×3, jump only: row 1 mean estimate vs true ρ_S, row 2 pointwise RMSE across seeds; dashed line at the change | Gaussian; Fixed = Gaussian RMSE-best (N_eff 199) |
| Main Fig. 3 | `ch6_gaussian_density_bars.png` | 2×3 bars: row 1 excess log loss (zero line), row 2 squared Hellinger; columns scenarios; ± 2 SE | Gaussian; Fixed chosen per metric (jump RMPL: N_eff 99 for both density metrics) |
| Appendix A | `ch6_appendix_est_trajectories.png` | 3×3 mean estimate vs true ρ_S (mean lines only, no spread band) | Gaussian |
| Appendix B | `ch6_appendix_rmse_trajectories.png` | 3×3 pointwise RMSE | Gaussian |
| Appendix (non-Gaussian density) | `results/strategies_tables.tex` | Family-level tables for all metrics (note: still carries VFF α = 0.1 columns) | All 4 families |

**Aggregation note:** Fig. 1 averages the four families, while Figs. 2–3 and Appendices A–B are Gaussian only (the time-series data were saved for the Gaussian copula, and the density figure is Gaussian by design).

## Section 6.3 figure set (latest brief, 2026-11-12): supersedes the set above

PNG, in `images/ch6/`; `plot_strategies.section_63()`.

| Fig. | Section | File | Status |
|---|---|---|---|
| 1 | 6.3.2 | `ch6_accuracy_bars.png`: 3×3 bars, rows RMSE / excess log loss / Hellinger, columns scenarios; Gaussian; Fixed chosen per metric; ± 2 SE; every panel on its own y scale | **Done** |
| 2 | 6.3.2 | `ch6_rmse_trajectories.png`: 3×3 pointwise RMSE (Gaussian; MSE smoothed over 25 observations within regimes) | **Done** (main text) |
| 3 | 6.3.2 | `ch6_jump_trajectories.png`: 1×3 jump, mean estimate vs true ρ_S | **Done** |
| 4 | 6.3.3 | Weighting loss: MFF uniform / squared / log, 2×3 bars (RMSE, excess log loss) | **Needs a KLL run.** Uniform and squared banks exist only on EW-500 (old E3) |
| 5 | 6.3.4 | Weights before and after the jump, and the fixed-share vs Hedge (δ = 0) ablation | **Needs a KLL run.** Weight paths and Hedge exist only on EW-500 (old E4) |
| App. | | `ch6_appendix_rmse_bars_4fam.png` (four-family RMSE bars); `ch6_appendix_est_trajectories_band.png` (3×3 with 10–90% bands) and `ch6_appendix_est_trajectories.png` (no bands); `ch6_vffsens_rmse.png` / `_h2.png`; family-level density tables `results/strategies_tables.tex` | Done (tables still have the α = 0.1 column) |
