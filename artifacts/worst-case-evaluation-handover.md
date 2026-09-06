# Handover: worst-case MSE versus worst-case regret

Prepared 28 September 2026.

## Objective

Evaluate whether the estimator banks remain attractive when judged by absolute worst-case estimation error, alongside the existing worst-case regret criterion. Determine which scenario drives each maximum and whether the two criteria select different memory strategies. Do not assume the bank will win.

This is initially an analysis of existing simulations, not a new estimator or a live-market experiment.

## Repository and evidence

The study currently lives on branch `adaptive-memory` in:

`/Users/sallysun/Documents/unsw/6thYear/datastreamer/.claude/worktrees/agent-a388b031c8ab7cf49`

Verify the current checkout and repository instructions before editing. Preserve unrelated work; do not change `main` or push as part of this handover.

Relevant paths, relative to that worktree:

- `experiments/results/adaptive_memory.csv`: aggregate results for fixed memory, VFF configurations and banks.
- `experiments/show_regret.py`: existing cross-scenario regret calculation and strategy identification.
- `experiments/probe_adaptive.cpp`: simulation, scoring and CSV generation.
- `notes/studies/results_2026_09_22.md`, §0: authoritative post-RNG-fix summary; older sections may contain superseded numbers.
- `notes/studies/adaptive_memory.md`: experimental context and comparator definitions.
- `notes/studies/rng_portability.md`: why pre-fix and post-fix results must not be mixed.

At inspection, the adaptive study used Gaussian-copula data, four scenarios (`constant`, `drift`, `jump`, `oscillate`), 5,000 ticks, 40 seeds starting at 42, and scoring indices `[500,5000)`. Confirm these against the checked-out code and record the revision and input checksum. Do not imply these results cover all copula families.

## Define the two questions precisely

Let j identify one fully specified strategy, s a scenario, r a simulation seed, and W_s the declared scoring window. Define

\[
E_{j,s,r}=\frac{1}{|W_s|}\sum_{t\in W_s}
(\widehat\rho_{j,s,r,t}-\rho_{s,t})^2,
\qquad
E_{j,s}=\frac1R\sum_{r=1}^R E_{j,s,r}.
\]

**Worst-case MSE:**

\[
W_{\mathrm{MSE}}(j)=\max_s E_{j,s}.
\]

This asks: how large is the strategy's average estimation error in its hardest tested scenario?

**Worst-case regret:** for estimator family a, let J_a contain all eligible tested strategies in that family, including fixed, VFF and bank configurations. Define

\[
E^*_{a,s}=\min_{k\in J_a}E_{k,s},\qquad
W_{\mathrm{regret}}(j)=\max_s\{E_{j,s}-E^*_{a,s}\}.
\]

This asks: how far behind the best tested strategy can j fall?

This second definition reproduces `show_regret.py`. Its comparator is not restricted to fixed-memory arms, so do not label it best-fixed-arm static regret. It is also different from the cumulative one-switch analysis.

The maximum is over scenarios AFTER averaging within each scenario. Do not substitute the worst seed, worst tick, or the average of per-seed maxima.

## Phase 1: use the existing aggregate CSV

1. Add an analysis script, suggested name `experiments/show_worst_case.py`, with an input-path argument and reproducible CSV output.
2. Read MSE as `rmse ** 2`. In the inspected probe, `rmse` is pooled RMSE across seeds and ticks. Do not square `rmse_mean_seed`: that produces a different statistic.
3. Identify fixed strategies by family and memory setting, and adaptive strategies by family, method tag and tuning configuration. Follow the current data schema; VFF tags include variants such as `vff_fast`. Never merge distinct bank variants under one label.
4. Require exactly one finite, nonnegative result per strategy/scenario. Report duplicates, absent scenarios and excluded strategies explicitly. Rank only strategies evaluated on the same complete scenario set.
5. Within each estimator family, compute both maxima and retain the scenario attaining each maximum. Preserve ties.
6. Produce the full rankings and a compact bank/fixed/VFF summary. Select the best fixed and best VFF configuration separately for each criterion. A memory selected under regret must not silently be called the best fixed memory under MSE.
7. Also score the regret-selected strategies under worst-case MSE and vice versa. This reveals the practical cost of changing the selection criterion.
8. Verify that the regret calculation reproduces the current `show_regret.py` output on the same CSV, to a tolerance consistent with CSV precision.

Suggested output columns:

`family, strategy_id, method, param, mem_ticks, mse_constant, mse_drift, mse_jump, mse_oscillate, worst_mse, worst_mse_scenario, worst_regret, worst_regret_scenario, rank_mse, rank_regret`

Keep CSV values in raw MSE units. If displaying values multiplied by 1,000, label the table as “MSE × 10³” or “values in units of 10⁻³ MSE” consistently.

Cross-family absolute MSE comparisons are possible on identical targets and streams. Family-relative regret values use different oracle baselines; do not treat a smaller value as proof of better absolute accuracy across estimator families.

## Phase 2: uncertainty, only if making a ranking claim

Aggregate CSVs support descriptive rankings but do not preserve enough information for paired uncertainty calculations. `se_rmse` and the existing paired RMSE summaries are not sufficient to recover uncertainty in a maximum of scenario mean MSEs.

If per-seed losses are unavailable, add a per-seed export to the probe and rerun the existing design without changing estimator settings. Suggested fields:

`family, strategy_id, scenario, seed, window_start, window_end, n_scored, mse`

Retain the same observations for all strategies within each scenario and seed. For paired bootstrap comparisons, resample complete seed vectors across strategies; preserve cross-scenario pairing if scenarios share seed-level randomness. In each replicate, recompute scenario means, maxima and oracle baselines. Report the paired difference in worst-case MSE and its bootstrap interval, together with the scenario means. Treat near-tied maxima cautiously.

Selecting the best configuration and reporting its score on the same seeds is an in-sample comparison. For a confirmatory claim, freeze configurations selected on the existing seeds and evaluate them on a fresh, predeclared seed set. Do not tune on the held-out results.

## Interpretation and focused follow-up

Answer these questions:

- Do worst-case MSE and worst-case regret select the same strategy?
- Which scenario drives each maximum? Does oscillation dominate absolute error for most methods?
- Does the bank improve both criteria, only relative regret, or neither?
- If the selected fixed memory changes, what error trade-off caused the change?
- Are rank differences substantial or small relative to seed variability?

An intrinsically difficult scenario can dominate worst-case MSE; this is part of what that criterion measures. Regret removes the scenario-specific best tested error, but depends on the comparator set. Neither criterion is universally superior.

Keep the first analysis on the existing whole-run windows. If jump recovery is later added as an additional stress case, publish it as a separate scenario-window analysis with explicit boundaries. Do not silently change the original four-scenario benchmark.

Do not call the selected method theoretically minimax-optimal. Use: “lowest measured worst-case MSE among the tested strategies across these four synthetic regimes.” This analysis alone does not establish performance on real data or under arbitrary dependence paths.

## Deliverables and acceptance checks

- Reproducible analysis script and machine-readable results.
- Short study note, suggested path `notes/studies/worst_case_mse.md`, with definitions, provenance, rankings, worst scenarios, uncertainty status and interpretation.
- One side-by-side visual comparing worst-case MSE and regret within each family, with separate labelled axes. Include all strategy results in a supporting table.
- Existing regret output reproduced; missing cells and duplicate identities detected; all reported winners correspond to a single fixed configuration across scenarios.
- Any numeric claim states the window, seed count, copula family, comparator set and whether tuning/evaluation used the same seeds.
- No invented confidence intervals from aggregate standard errors, and no estimator or data-generation changes hidden inside the analysis.

Complete Phase 1 first. Its results determine whether the extra uncertainty run or a more targeted experiment is useful. Do not update thesis claims before reviewing the actual outcome.
