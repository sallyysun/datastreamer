# Code readability review

**Status:** on `final-edits`, done: the replay and scoring value renames, `Mff::score_and_update_weights()`, the `StrategyResults` cleanup, the unreported mixcheck open, the live scorer's short-window mean, and `ForgettingStrategies`. The remaining renames are deferred.

`Simulation::run()` now gives the standard comparison a clear entry point. The replay loader and fixed-memory arms also have shared owners. The next gains are inside the existing files. The reader still has to decode which values are observations, which values train the estimator, and which values exist only for evaluation. Large routines get harder to read when the same short names also carry indexing and reporting details. [Simulation design](simulation.md) · [Standard simulation](../../simulation/simulation.cpp#L44)

This review covers the C++ producer, consumer, estimators and simulation runners, with examples from the Python analysis and plots. The changes below are proposals based on reading the current code.

## DependenceEstimator and ForgettingStrategies

- `DependenceEstimator` stays the common interface. RMPL, Counting Matrix, and Hermite implement it. Its `estimate()` method returns Spearman correlation. Scoring converts that correlation to the parameter of the chosen working copula.
- `ForgettingStrategies` replaces `EstimatorStrategies`. It owns the fixed-memory instances of one `DependenceEstimator` algorithm, the MFF combination of those instances, and the VFF instances that adapt their memory. A `Device` holds one `ForgettingStrategies` per algorithm and supplies their shared marginal inputs. The name says *strategies*, not *estimator*, because the class returns one forecast per strategy and does not implement `DependenceEstimator`. It also matches the `strategy` output column.

For example, the output row with estimator `rmpl` and strategy `fixed_199` is the forecast from RMPL with an effective memory of 199. The row with strategy `mff` weights several fixed-memory RMPL instances. Both come from the RMPL `ForgettingStrategies`. Keep the existing estimator and strategy labels for the reports. [Estimator interface](../../../src/estimators/dependence/estimator.hpp) · [Device](../../../src/estimators/device.hpp#L44)

## Start with the values that cross files

| Current name | Suggested name | Why |
|---|---|---|
| `CopulaDraw::target`, `StudySeries::target` | `target_spearman` | Requested correlation from the scenario. The generating family can clamp the parameter needed to achieve it. |
| `CopulaDraw::rho_s`, `StudySeries::rho`, `SeedResults::rho` | `true_spearman` | Correlation achieved by the generating parameter and used as the evaluation target. |
| `StudySeries::{u,v}` and `{pu,pv}` | `estimated_pit_{x,y}` and `true_pit_{x,y}` | The first pair comes from the prior KLL state. The second is used to evaluate the forecast against the generating distribution. |
| `StudySeries::{x,y}` | `observed_{x,y}` | These are generated normal scores in process and reconstructed returns over TCP. |
| `Stream::par`, `ScoringObservation::parameter` | `generating_parameter` | Distinguishes the truth used by the scorer from the working parameter computed from a forecast. |
| `ScoringObservation::{rho,u,v,log_density}` | `true_spearman`, `true_pit_x`, `true_pit_y`, `true_log_density` | The type should read consistently with the input series. |
| `ForecastScore::{excess,h2}`, `StrategyTrace::{excess,h2}` | `excess_log_loss`, `hellinger_squared` | Names the score and identifies the squared Hellinger convention. Use `_x1000` at the reporting boundary where the values are scaled. |

At the scoring call, consider a small per-tick view over the existing replay columns. Keep the replay storage and construct the view where a reader needs to see the input roles together. Give `Simulation` the observation pair and its estimated PIT for the update, and give `score_forecast` the held forecast and the generating truth. `MIXDENS` uses both coordinate systems for different outputs. Its diagnostic evaluates mixture log loss at the estimated PIT, while excess log loss uses the true PIT. [Replay fields](../../simulation/study_stream.hpp#L21) · [Generation](../../../src/streamgenerator/datasource/copula.hpp#L19) · [Scoring call](../../simulation/simulation.cpp#L64)

## Functions and constants worth renaming

| Current | Proposed | Where |
|---|---|---|
| `arm()`, `estimators()` | `make_fixed_memory_estimator()`, `estimator_names()` | [Estimator registry](../../../src/estimators/registry.hpp#L24) |
| `counting_matrix_config(lambda, shared)`, `hermite(lambda, shared)`, `rmpl(gain)` | `counting_matrix_config(retention, normal_score_inputs)`, `hermite_config(update_weight, normal_score_inputs)`, `rmpl_config(gain)` | [Estimator registry](../../../src/estimators/registry.hpp#L29) |
| `Device::sets()`, `DeviceState::dependence()` | `forgetting_estimators()`, `spearman_forecasts()` | [Device](../../../src/estimators/device.hpp#L76) · [Consumer state](../../../src/consumer/device_state.hpp#L59) |
| `Mff::score()`, `guard_mff_losses()` | `score_and_update_weights()`, `replace_nonfinite_losses()` | [MFF](../../../src/estimators/dependence/mff.hpp#L15) |
| `datasketches_kll::seed()`, `Device::seeded_` | `reseed_thread_compaction_rng()` and an explicit reseed during device setup | [KLL](../../../src/estimators/dependence/datasketches_kll.hpp#L30) · [Device construction](../../../src/estimators/device.hpp#L76) |
| `Win {name, a, b}`, `windows()`, `checkpoint()` | `EvaluationWindow {name, begin, end}`, `evaluation_windows()`, `is_hellinger_checkpoint()` | [Evaluation](../../../src/estimators/evaluation.hpp#L98) |
| `gauss_logc()`, `H2Table::mix_logc()`, `FamilyInverse` | `gaussian_log_density_from_spearman()`, `gaussian_mixture_log_density()`, `SpearmanToParameterTable` | [Density evaluation](../../../src/estimators/evaluation.hpp#L112) |
| `kN`, `kBurn`, `kChange`, `kMem`, `kK` | `kObservationCount`, `kBurnIn`, `kChangeTick`, `kFixedMemories`, `kFixedArmCount` | [Study config generator](../../common/write_study_config.py#L30) |
| `make_stream()`, `Stream` in the strategy runner | `load_scoring_series()`, `ScoringSeries` | [Loader](../../simulation/simulation.cpp#L15) |
| `Run::{seeds,cop,sfx,h2}` | `seed_count`, `generating_family_name`, `output_suffix`, `hellinger_table` | [Runner declarations](../../simulation/strategy_runs.hpp#L20) |
| `run_scoresfam()`, `run_famswitch()`, `run_regretfam()` | `run_family_score_comparison()`, `run_family_switching()`, `run_working_family_regret()` | [Mode declarations](../../simulation/strategy_runs.hpp#L70) |
| `regret_oracle()`, `prefix_loss` | `compute_regret_oracle()`, `best_prefix_loss_by_switch_budget` | [Regret](../../simulation/regret.cpp#L18) |
| `Serializer::symbol_from_json()`, `topic_matches(sub, pub)` | `tick_from_json()`, `topic_matches(subscription, published_topic)` | [Protocol](../../../src/protocol.hpp#L25) |

Keep established names such as `Mff`, `Vff`, `rmse`, `rho`, `nu`, `eta`, and `delta` where the mathematical context is local. The derivation inside `VffController::observe()` already names the derivative it computes. Its persistent fields would benefit from names such as `log_effective_memory_` and `mean_gradient_square_`. Likewise, `ax` is clear in a plotting function, while `S`, `W`, or `D` passed between analysis functions hide what the caller must supply. [VFF controller](../../../src/estimators/dependence/vff.hpp#L24)

## Code shape beyond naming

### Make the core loop readable at the call site

- `Simulation::run()` should still read as **get held forecasts, score the tick, absorb the observation**. Keep MFF weight updates visible in the API. `Mff::score()` changes weights using the estimated PIT, while `score_forecast()` evaluates a held forecast using truth. Rename the former first. Preserve the pre-update observation count for MFF, the count passed to VFF, and the current DataSketches compaction order. [Strategy update](../../../src/estimators/device.hpp#L59) · [Simulation order](../../simulation/simulation.cpp#L64)
- The scorer selects MFF mixture behavior via `strategy == kK` and interprets the first `kK` entries as fixed arms. Start with a named MFF index and a named span of fixed-arm forecasts. The live scorer and CLI also derive labels from separate positional lists. Build the VFF labels and rates together at CLI entry, preserving the literal rate text used in output labels. [Scorer](../../../src/estimators/evaluation.hpp#L181) · [Dispatch](../../simulation/experiment.cpp#L332)
- The MFF scoring mode is implicit at the call site. `score_forecast()` scores MFF as a weighted density mixture whenever it receives mixture weights, and as a point forecast otherwise. The live `Scorer` always passes the weights, while `Simulation` passes them only under `MIXDENS=1`. The difference is intentional, since `device_vs_probe.cpp` checks the device against the `MIXDENS=1` rows, but neither caller says which mode it uses. Replace the optional span with a named mode, or a separate mixture entry point, so each caller states its choice. [Scorer](../../../src/estimators/scorer.hpp#L34) · [Simulation](../../simulation/simulation.cpp#L56) · [Device comparison](../../simulation/test/device_vs_probe.cpp#L1)
- `make_stream()` reconstructs generating parameters through a single family lookup. Family switching computes the active family and parameter separately, using its own schedule mapping. A useful follow-up is to have the producer report the active family and parameter with each draw, then give scoring callers that truth directly. Check how TCP carries the same truth before adopting this change. In the meantime, leave a name on the existing lookup that makes its single-family assumption visible. [Strategy loader](../../simulation/simulation.cpp#L15) · [Switching scores](../../simulation/family.cpp#L228) · [Producer](../../../src/streamgenerator/datasource/copula.hpp#L124)

### Simplify the reporting code in place

- `StrategyResults` still makes the reader decode six `FILE*` members, three storage layouts, and several output modes. Start with `seed_scores_file_`, `jump_path_file_`, `tick_paths_file_`, `seed_dump_file_`, `extremes_file_`, and `mixture_check_file_`. Rename `jsum_` to `squared_error_sum_by_tick_` and `at()` to `path_offset()`. In `add()`, use `estimator_index`, `strategy_index`, and `seed_index` because those indices are easy to swap. Group per-tick path totals and per-seed samples by estimator and strategy in the accumulator. A small private record can give those arrays a home and eliminate repeated offset arithmetic. Keep the CSV rows in their current order. [Output members](../../simulation/experiment.cpp#L125) · [Aggregation](../../simulation/experiment.cpp#L203)
- Give output mode selection typed options at CLI entry. Today globals for strategy names and VFF rates are edited alongside environment parsing, and `DUMP_SEED` is parsed repeatedly in the seed loop. A parsed seed filter and one named output mode would make the loop read as scenario selection, seed selection, simulation and aggregation. In that same area, check the file-open policy. `MIXDENS` can request an output whose open failure is omitted from the constructor's `ok_` check. Treat error handling as a separate change from the renames. [Output setup](../../simulation/experiment.cpp#L163) · [Experiment entry point](../../simulation/experiment.cpp#L309)
- The live scorer and experiment summarizer each implement window means and the denser Hellinger sampling after a jump. Name the checkpoint inclusion rule, then check whether both callers can share it. Handle the live scorer's short-stream behavior explicitly before factoring the whole summary. It stops at the available observations and still divides by the full configured window length. [Live scorer](../../../src/estimators/scorer.hpp#L35) · [Experiment summary](../../simulation/simulation.cpp#L23)

### Keep study-specific work in the study files

- `regret.cpp` already has a coherent oracle and two callers. Name the per-tick loss matrix and scored tick indices (`L` and `tt`) and keep the distinction between exact-switch DP state and an at-most-switch result visible. The one-switch summary also identifies the arms and change tick, so preserve its tie-breaking when touching that code. Compare the two callers for duplicate preparation after their working-family and forecast differences are explicit. [Oracle](../../simulation/regret.cpp#L18) · [Working-family regime](../../simulation/regret.cpp#L89)
- `RollingBenchmarks` already keeps the rolling methods together. Use `rolling_spearman()` for `spearman_window()`, `solve_mpl_correlation()` for `mpl_root()`, and `append_forecast_scores()` for the method that writes the held benchmark forecasts and records timing. In the budget runner, rename `Cfg`, `Acc`, `build()` and `run_split()` to `BenchmarkConfig`, `ConfigResults`, `make_benchmark_estimator()` and `run_budget_split()`. Keep the benchmark-specific setup and timing local. [Rolling benchmarks](../../simulation/real_data_comparison.cpp#L107) · [Budget runner](../../simulation/benchmarks/budget_benchmark.cpp#L77)

### Why `family.cpp` is long

The 312-line file contains two experiment runners. Each parses options, runs estimators, scores forecasts, and writes CSVs. Both also repeat mixture calculations inside nested loops. Keep both runners in `family.cpp` and share the calculations they have in common.

| Mode | What it measures |
|---|---|
| `SCORESFAM` (§6.4) | How squared loss and log loss change the weights on the same fixed-memory estimators, across working copula families and learning rates. |
| `FAMSWITCH` (§6.5) | How quickly a mixture puts weight on the generating family, and how its forecasts compare with a Gaussian-only mixture when the generating family stays fixed or changes. |

1. **Move numerical checks to an explicit validation mode.** About 40 lines compare the parameter lookup and Hellinger approximations with direct calculations. `VERBOSE` gates the print statements, so the inverse-table comparison still runs in quiet mode and the Hellinger comparisons run whenever their metrics are enabled. Keep `scoresfam_diag` as part of the experiment because it records clamping and nonfinite values from the actual run. [Score checks](../../simulation/family.cpp#L44) · [Switching checks](../../simulation/family.cpp#L184) · [Diagnostic CSV](../../simulation/family.cpp#L149)
2. **Share mixture scoring.** Both modes convert held Spearman forecasts to working copula parameters, compute a weighted Spearman forecast and mixture log density, and apply the existing fixed-share update. Start by sharing the weighted forecast and mixture log-density calculation. Preserve each mode's handling of nonfinite densities and its Gaussian parameter clamp. Keep evaluation at the true PIT and weight updates at the estimated PIT explicit at the call sites. Replace the `In` record with `ScoreVariant` and a named score rule, name `snf` and `enf` for their separate nonfinite counts, and pair `WF` and `WN` in a working-family descriptor. [Family scoring](../../simulation/family.cpp#L96) · [Switching loop](../../simulation/family.cpp#L228)
3. **Return a per-seed switching result.** Have the tick loop return two named score traces, the family-weight path, and the guarded-loss count. Summarize the path afterward for time to 50%, time to 90%, and incorrect-majority events. Keep the CSV writing and across-seed path totals in the outer runner. A local result record is enough for this split. [Switching state](../../simulation/family.cpp#L220) · [Identification and output](../../simulation/family.cpp#L275)
4. **Use the shared replay loader directly in switching.** `make_stream()` adds a single-family `generating_parameter` column. Switching already derives the active family's parameter per tick, so it can use `load_study_series()` directly. [Switching input](../../simulation/family.cpp#L219) · [Loader](../../simulation/simulation.cpp#L15)

### Make Python tables and plots explain their inputs

| Current | Suggested | Where |
|---|---|---|
| `load()`, `get(W, sc, e, delta)`, `frame(ax, sc)` | `load_weight_paths()`, `select_weight_path(weight_paths, scenario, estimator, delta)`, `style_time_axis(ax, scenario)` | [MFF weights](../../6.3-adaptive-forgetting/plot_mff_weights.py#L20) |
| `load(section_dir, metric)`, `panel(ax, agg, sc, ym, metric)` | `load_strategy_summary()`, `plot_scenario_bars(ax, summary, scenario, y_limit, metric)` | [Shared strategy plot](../../common/strategy_allfixed.py#L21) |
| `Days`, `idx`, `classify(D, ...)`, `stat(Dm, M)` | `PairedTradingDays`, `calendar_dates`, `classify_large_changes(changes, ...)`, `summarize_masked_loss_difference(loss_difference, mask)` | [Hedge change analysis](../../6.6-real-data/analyze_hedge_change.py#L20) |

Name array axes where shape matters, especially arm × pair × day in the hedge analysis. Break the aligned change calculation into named steps and preserve the order of shifting by trading days before reindexing to calendar dates. Move reusable statistics into `summary_stats.py` only when callers use the same pairing, units and missing-data policy. Keep report strings near the output so the computation can be read without skipping formatted prose. [Day alignment](../../6.6-real-data/analyze_hedge_change.py#L27) · [Shared statistics](../../common/summary_stats.py)

## Suggested order

1. Rename replay fields, scoring fields, the stage-changing functions, and the generated study constants. Keep CLI flags, wire keys, CSV columns and file names stable.
2. Clean up `StrategyResults` and the standard CLI options in place. Then apply the same level of naming inside `family.cpp`, `regret.cpp`, and the Python helpers that are read across scripts.
3. Review behavior separately for generating truth, short live streams, diagnostics and output errors. For implementation verification, use the existing Linux checksum outputs and the path-specific comparisons documented in the simulation design. Preserve replay and update order throughout the naming pass. [Verification criteria](simulation.md#verification)
