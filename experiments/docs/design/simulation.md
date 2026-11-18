# Simulation design

**Status:** implemented on `final-edits`, not yet verified against the reference outputs. The file table below records the layout before the refactor. The grouped structure further down is the current layout.

`src/` already generates the stream. The experiment supplies a family, dependence scenario, seed, and observation count. The current `make_stream()` in the strategy runner calls the shared generator through the study adapter, then stores the observations. The fixed-ESS sweep keeps its own copy of that loader. In the proposed design, a new `load_study_series()` in the study adapter replaces both copies. The standard run, family modes, regret modes, and fixed-ESS sweep all replay the series it returns.

## How observations reach the experiments

- **In-process.** `make_scenario()` supplies a target Spearman correlation for each tick. `CopulaSource` uses that schedule and a seeded sampler to draw the pair. The experiment receives the draws and their ground truth from the code in `src/`. [Scenario factory](../../../src/streamgenerator/datasource/dependence.hpp#L81) · [Copula source](../../../src/streamgenerator/datasource/copula.hpp#L168)
- **Live.** The server uses the same copula draw path. `SyntheticSource` turns draws into price ticks and carries the truth alongside them. The experiment's TCP adapter reconstructs paired returns the same way the consumer does. A live experiment therefore receives reconstructed returns, while an in-process experiment receives the original draws. [Synthetic source](../../../src/streamgenerator/datasource/synthetic_source.hpp#L24) · [TCP adapter](../../simulation/run_live_stream.hpp#L46)
- **Preparation for scoring.** The study adapter computes shared KLL pseudo-observations using the preceding pairs. The strategy runner then stores those inputs alongside the true coordinates, Spearman correlation, and generating parameter used by the scorer. Buffering lets several estimators or forecast variants score exactly the same observations. [Study adapter](../../simulation/study_stream.hpp#L20) · [Strategy loader](../../simulation/simulation.cpp#L15)

The source should continue to own the dependence schedule, copula sampling, and random number generator. The experiment should own its choice of streams, replay policy, metrics, and outputs. The live device can continue consuming one observation at a time, without a stored study series.

The TCP adapter also stays a separate header. `real_data_comparison.cpp` includes it directly for file replay, so folding it into the study adapter would pull the study's KLL preparation into the real-data runner.

## Files before the refactor

The standard strategy run, family scoring, family switching, and regret modes link into one executable. The section scripts start the executable, and Python reads the resulting CSVs. [Build entry points](../../build.sh) · [Section workflows](../../README.md)

| File | Responsibility | Direct users or outputs |
|---|---|---|
| `strategy_experiments.cpp` | Dispatch modes and run the standard strategy comparison, weight paths, and validation | §6.3–6.5 section scripts, live jobs, and the checksum check |
| `family_scores.cpp` | Compare squared and log loss across working families and learning rates | `SCORESFAM`, §6.4 analysis and heatmaps |
| `family_switching.cpp` | Measure identification, adaptation, and predictive accuracy when the generating family stays fixed or switches | `FAMSWITCH`, §6.5 analysis and figures |
| `strategy_regret.cpp` | Compare predictive loss with the best fixed and switching sequences | `REGRET` and `REGRETFAM`, §6.4 regret analysis |
| `fixed_ess_sweep.cpp` | Sweep fixed memories and measure error, bias, variance, and jump recovery | §6.2 figures and tables, live jobs, and the checksum check |
| `real_data_comparison.cpp` | Score forecasts and rolling benchmarks on observed returns | §6.6 analysis and figures, and the checksum check |
| `budget_benchmark.cpp` | Measure accuracy, retained state, and timing across settings | §6.7 budget tables and figures, and the checksum check |
| `margin_ablation.cpp` | Measure how different marginal estimation methods affect accuracy | §6.7 marginal estimation report |
| `strategy_runs.hpp` | Declare modes and share replay data, fixed arms, inverse lookups, and window summaries | The four strategy source files |
| `study_stream.hpp` | Adapt in-process or live observations and provide shared KLL margins | Strategy runs and the fixed-ESS sweep |
| `run_live_stream.hpp` | Start the server and reconstruct paired returns from TCP ticks | The study adapter and the real-data runner |
| `family_hellinger.hpp` | Tabulate working densities and cache truth densities for Hellinger scoring | Family scoring and family switching |
| `regret_oracle.hpp` | Compute fixed and switching comparators from a loss matrix | Both regret modes |
| `test/device_vs_probe.cpp` | Score the device on synthetic draws or reconstructed returns | The in-process device comparison |
| `test/lab_vs_device.cpp` | Compare the browser lab's forecasts with the device | The browser-lab consistency check |

## Proposed design

For each seed, **run the producer, run the consumer logic, score the forecasts, and collate the results**. In process, “run” means calling the producer and consumer code in `src/` directly. With `LIVE=1`, the loader starts the server and reads the series over TCP.

1. **Producer.** Load one seeded series per scenario and seed with `load_study_series()`. `CopulaSource` generates the observations through `src/`. In live mode, the loader reads the same kind of observations from the TCP adapter after reconstructing returns from price ticks.
2. **Consumer.** Replay each pair into the estimator strategies in `src/`. The series carries one shared pair of KLL pseudo-observations per stream. The standard run uses the existing `ForgettingStrategies` on those prepared inputs. The family and regret modes keep the shared fixed-memory bank with their own weighting or oracle logic.
3. **Score.** Evaluate the forecast held before the pair was absorbed. The scoring functions in `src/` use the generated truth for accuracy and copula density comparisons. The estimator still returns Spearman correlation. The scorer converts it to a copula parameter for the chosen working family.
4. **Collate.** Keep scenario and seed selection, accumulated statistics, CSV output, and optional diagnostics in the experiments. Python reads the same CSVs.

Every strategy mode **replays a stored series**. The standard runner already generates each seed's observations once, and the TCP adapter already buffers its input, so a single-pass loop would mainly make the standard run look more like the live consumer. A stored series is a recording of the producer's observations. Replaying it does not call the generator a second time. Revisit streaming if incremental TCP processing becomes a goal.

A future single-pass loop has one constraint. DataSketches flips its compaction coins with a thread-local generator, so every consumer of that generator changes the draws of every other one. Today the study adapter reseeds it once and owns the only pair of DataSketches margins. The estimator strategies consume those pseudo-observations or raw values and flip no coins of their own. If `Device` owns a single-pass loop, it must also own margin preparation, with the study adapter's KLL removed from that path. [Device](../../../src/estimators/device.hpp#L51) · [Study adapter](../../simulation/study_stream.hpp#L42)

The budget and margin benchmarks run each estimator start to finish and still read estimates after updating. The fixed-ESS and real-data forecasting runners keep their existing order. KLL seeding, the use of prior pairs for pseudo-observations, and the distinction between in-process draws and reconstructed live returns carry through this design.

## Experiment and Simulation

The standard comparison has two lifetimes. An `Experiment` covers the full study. A `Simulation` covers one scenario and one seed. These classes cover the standard mode only. The family and regret modes keep their own entry functions, fixed-memory bank, and weighting or oracle logic.

| Class | Owns | Main method |
|---|---|---|
| `Experiment` | The run options for the standard mode, and a `StrategyResults` for the length of the study. | `run()` checks the options, opens the outputs, loops over scenarios and seeds, creates a fresh `Simulation` for each, and passes each result on. |
| `StrategyResults` | The CSV outputs and the per-scenario sums across seeds. | `add()` collates one seed's results and writes its per-seed rows. `end_scenario()` writes the summaries across seeds. |
| `Simulation` | The configuration and lifetime of one seeded run. The producer owns its random state, the loaded series owns the shared margins, and each estimator's strategies own their estimates and weights. | `run()` loads one series, replays it into each estimator's strategies, and scores each held forecast against truth before the update. It returns the results for that seed. |

`main()` constructs an `Experiment` for the standard mode. Each `Simulation` scores all configured estimators on the same replayed series. `main()` owns the `H2Table` and shares it through `Run`, because the family, regret, and validation modes use the same cache. Per-seed metrics start fresh. `SeedResults` is an ordinary result value that carries the metrics needed for the requested outputs. The simulation returns it to the experiment for aggregation and CSV output.

### UML class diagram

```mermaid
classDiagram
    class Experiment {
        -Run run
        +run() int
    }
    class Run
    class Simulation {
        -string scenario
        -unsigned seed
        +run() SeedResults
    }
    class SeedResults
    class StrategyResults
    class StudySeries
    class ForgettingStrategies
    class H2Table

    Experiment *-- StrategyResults : owns for the study
    Experiment --> Run : reads options
    Run --> H2Table : shared cache owned by main()
    Experiment ..> Simulation : creates one per scenario and seed
    Simulation ..> StudySeries : loads with load_study_series()
    Simulation ..> ForgettingStrategies : reads forecasts and updates
    Simulation ..> H2Table : uses during scoring
    Simulation ..> SeedResults : returns
    StrategyResults ..> SeedResults : collates
```

Scoring calls the shared functions in `src/estimators/` with the shared `H2Table`. The loader fills the series from `CopulaSource` in process, or from reconstructed TCP returns in live mode.

### Call order

```mermaid
flowchart LR
    main["main()"] --> experiment["Experiment::run()"]
    experiment --> simulation["Simulation::run() for each scenario and seed"]
    simulation --> tick["Load series → replay: forecast → score → update"]
    tick --> collate["Return seed results for the experiment to collate"]
```

## Proposed grouped file structure

Keep the sources for the strategy executable directly under `experiments/simulation/`. They already share an entry point and build target. `fixed_ess_sweep.cpp` also stays at the top level as the §6.2 runner. The `benchmarks/` folder groups the §6.7 cost programs, and `test/` groups the comparisons. `Experiment` stays private to `experiment.cpp`. `Simulation` and its per-seed result types have their own header. `StrategyResults` has its own pair of files and takes the strategy labels from `main()`. The replay series has one type, `StudySeries`; callers that score against the generating parameter compute it with `generating_parameters()`.

```text
src/
  streamgenerator/datasource/    Existing producer and scenario code
  estimators/
    device.hpp                 Shared fixed bank, strategy state, and Device
    evaluation.hpp             Shared scoring and Hellinger calculations
    scorer.hpp                 Scoring adapter for the live consumer

experiments/simulation/
  study_stream.hpp              load_study_series() for in-process draws and TCP input
  run_live_stream.hpp           Server start and return reconstruction over TCP
  experiment.cpp                main(), mode dispatch, and Experiment
  simulation.hpp/.cpp           Simulation, SeedResults, and the single-seed forecast loop
  strategy_results.hpp/.cpp     StrategyResults, window summaries, and the strategies_* CSVs
  strategy_runs.hpp             Run options, mode declarations, and small runner helpers
  validate.cpp                  VALIDATE=1 table and density checks
  working_families.hpp          Working families, rho_S tables, and family Hellinger
  family.cpp                    Family scoring and switching
  regret.cpp                    Regret modes and their oracle
  fixed_ess_sweep.cpp           §6.2 fixed-memory accuracy and recovery
  real_data_comparison.cpp      Real-data forecasts and rolling benchmarks
  benchmarks/
    budget_benchmark.cpp        Accuracy, memory, and timing
    margin_comparison.cpp       Accuracy across marginal estimation methods
  test/
    device_vs_probe.cpp         Device and experiment comparison
    lab_vs_device.cpp           Browser lab and device comparison
```

The proposed structure has 18 simulation files, up from the current 15. The split replaces `strategy_experiments.cpp` with `experiment.cpp`, `simulation.cpp`, and `strategy_results.cpp`, which link into the same `strategy_experiments` executable. `regret_oracle.hpp` folds into `regret.cpp`. The two Hellinger implementations merge in `src/estimators/evaluation.hpp`. That header already holds the Chapter 6 evaluation, so the family tables join code used only by the experiments, not code the live consumer needs. The section scripts keep their executable names, mode flags, CSV names, and output schemas.

Within the strategy executable, the file boundaries are for navigation and linking.

- **`experiment.cpp`** reads options and dispatches modes. For the standard comparison, `Experiment::run()` loops over scenarios and seeds, calls `Simulation::run()`, and writes the aggregated results.
- **`simulation.cpp`** runs one scenario and seed. It loads the series, replays it through the estimator strategies and scorer in `src/`, and returns `SeedResults`.
- **`strategy_results.cpp`** collates `SeedResults` across seeds and writes the `strategies_*` CSVs. Its window summary is shared with the family modes.
- **`strategy_runs.hpp`** holds the run options and the mode entry points. `Experiment` stays private to `experiment.cpp`.
- **`family.cpp`** keeps the two family comparisons together. One compares scoring rules across working copulas. The other measures adaptation when the generating family changes. Both modes need family-specific weighting and density scoring. Their entry functions stay separate, and both link into the same executable as the standard loop.
- **`regret.cpp`** owns both regret modes and their shared oracle. It compares forecast losses with the best fixed and switching sequences.

An *ablation* is a controlled comparison that removes or changes a component to see what it contributes. Here the benchmark compares margin treatments on the same generated data, including known margins, empirical ranks, and KLL approximations. `margin_comparison.cpp` is a clearer source name. The executable can still be called `margin_ablation` for the existing scripts.

## Refactoring order

1. Add `load_study_series()` to the study adapter. Move the strategy runner's and the fixed-ESS sweep's loaders onto it, keeping the single KLL reseed and shared pseudo-observations.
2. Share the fixed-memory bank and scoring in `src/`. Preserve warm-up, KLL seeding, forecast timing, and numerical methods. If the standard run moves onto `Device` later, make the `src/` change first: configurable VFF rates and access to the mixture diagnostic's inputs.
3. Group the experiment files and split the standard comparison into `experiment.cpp` and `simulation.cpp`. Check that every existing mode reaches the same output.

## Mode functions

The file layout is settled. Most of the remaining length sits in a few mode functions, each a single 60–145 line body that reads the environment, opens its CSVs, runs a table self-check, loops over scenarios and seeds, runs the per-tick loop, and prints. `Simulation` and `StrategyResults` already do this for the standard run.

| Function | Lines |
|---|---|
| `run_famswitch` (`family.cpp`) | 145 |
| `run_scoresfam` (`family.cpp`) | 138 |
| `main` (`fixed_ess_sweep.cpp`) | 93 |
| `run_regret` (`regret.cpp`) | 88 |
| `main` (`real_data_comparison.cpp`) | 83 |
| `run_regretfam` (`regret.cpp`) | 64 |
| `run_weights` (`experiment.cpp`) | 58 |

Most of that length restates MFF. `Mff` is fixed at the five study arms, so modes that needed a 20-arm family mixture or another learning rate rebuilt it from the weight-update helpers.

| Repeated piece | Copies |
|---|---|
| MFF mixture log density (log-sum-exp over weighted arm densities) | 5 |
| MFF point forecast (weighted mean, clamped) | 3, besides `Mff::estimate` |
| Per-arm working-copula loss | 4, besides `Mff::score_and_update_weights` |
| Readiness rule `t >= 30 && arms.ready()` | 4, besides `Mff::ready` |
| The four working families, their names and inverse tables | 3 |
| Comma-separated environment lists | 4 |
| Mean and standard deviation across seeds | 2 |
| Output open and the "run from the experiment folder" check | 7 |

`run_weights` mostly duplicates the standard run. Its δ = 1e-4 rows replay the same arms and the same `Mff` that `ForgettingStrategies` holds, in the same scoring order, so their weights, squared errors and mean forecasts are already computed there. Only the δ = 0 Hedge comparison is new. `SeedResults` can carry the MFF weight path, and the weight outputs can come from the standard run, with δ = 0 as an extra strategy enabled only under `WEIGHTS=1`.

### Order

Steps 1–5 are implemented on `final-edits`, not yet verified against the reference outputs. `VALIDATE=1` now prints the family table checks, so `validate_strategies.txt` changes by design.

1. Let `Mff` take any number of arms, a learning rate and a share rate. Give it `forecast()`, `mixture_log_density()`, `mixture_loss()` and `update()`, and name the 30-tick warm-up. The family, regret and real-data modes then hold `Mff` values instead of raw weight vectors. Keep each mode's existing log-sum-exp form and loss guard, so outputs stay identical.
2. Move the table self-checks in `run_scoresfam` and `run_famswitch` into `VALIDATE=1`.
3. Add a working-family table (family, name, inverse), one list parser, one results-file opener, and one across-seed mean and standard deviation.
4. Move `FamilyInverse` and `FamilyHellinger` from `src/estimators/evaluation.hpp` back to the experiments. Only `family.cpp` uses them, and the live consumer compiles that header.
5. Produce the weight paths from the standard run as above, and move `run_validate` out of `experiment.cpp`.

## Verification

Each path has its own pass condition.

| Path | Check | Pass condition |
|---|---|---|
| Linux, in process | [`check.sh`](../../check.sh) | Every output hash matches `check_reference.txt` exactly |
| Margin ablation | Before-and-after run with the same seeds | Identical outputs. `check.sh` does not run this benchmark |
| Live (`LIVE=1`) | Rerun with the same seeds | Matches the previous live run. The [documented differences](../../README.md#L45) between live and in-process runs are not by themselves grounds for a change |
| In-process draws versus returns | [`test/device_vs_probe.cpp`](../../simulation/test/device_vs_probe.cpp) | Reproduces the `MIXDENS=1` rows |
| Standalone TCP consumer | [`live/run_live.sh`](../../live/run_live.sh) | Matches its previous results |
