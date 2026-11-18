# Chapter 6 experiments

One folder per thesis section, with only scripts: its analysis and plotting code, `run.sh`, and `images/` containing exactly the figures in the thesis.
`results/` (raw and derived CSVs) is git-ignored.

| Folder | Thesis | Program | Figures / tables |
| --- | --- | --- | --- |
| `ch3-background/` | Ch. 3 | — | `ch3_parametric`, `ch3_empirical` |
| `6.2-fixed-forgetting/` | §6.2 | `probe_fixed_ess` | constant/drift/jump ESS RMSE, drift bias–variance, jump recovery; tab:optimal-ess-*, tab:jump-recovery |
| `6.3-adaptive-forgetting/` | §6.3 + app. | `scenarios` | strategy RMSE bars, trajectories, MFF weights / fixed share, VFF rate sensitivity |
| `6.4-copula-density/` | §6.4 + app. | `scenarios` (`SCORESFAM`, `REGRET`, `REGRETFAM`) | log- vs squared-loss heatmaps, density bars, MFF regret; tab:mff-regret-bounds, tab:app-mff-regret-families |
| `6.5-family-switching/` | §6.5 | `scenarios` (`FAMSWITCH`) | family-switch accuracy, initial identification, RMPL weights, adaptation |
| `6.6-real-data/` | §6.6 + app. | `probe_real_practical` | final regret, change days, weights/ESS, response, VIX, hedge, strength vs advantage |
| `live/` | 6.2–6.6 over TCP | the probes with `LIVE=1`; `../server` + `../consumer` | `run_all_live.sh` + `jobs.txt` (every 6.2–6.5 simulation), `compare_all_live.py`; `run_live.sh` (the consumer binary scoring §6.3–6.4) |
| `6.7-computational-cost/` | §6.7 + app. | `probe_budget`, `probe_margins` | storage, KLL comparison, processing time, memory curves, accuracy gain; tab:ch6-budget-configs, tab:budget-configurations-full |

```bash
./build.sh                          # EIGEN=/usr (Linux) or /opt/homebrew/opt/eigen (macOS)
6.2-fixed-forgetting/run.sh         # figures + tables from results/
SIM=1 6.2-fixed-forgetting/run.sh   # rerun the simulations first (60 seeds x 4 families)
```

The C++ lives in `src/`: `scenarios` is `src/streamgenerator/datasource/scenarios.cpp` (the fixed / VFF / MFF simulation
used by 6.3–6.5, one `run_*` function per mode); the other programs, `live_source.hpp` (a probe's streams over TCP from
seeded servers) and the consistency tests are in `src/simulation/`. `build.sh` puts each binary in its section's
`build/`. Shared scripts in `common/`: `config.py` (seeds, scenarios,
families, memories, labels and colours used by every script) and `plot_style.py` (the figure style). 6.6 shares
`real_common.py`, ch3 `ch3_style.py`. Every script is a short docstring, functions and a `main()`.

Conventions:

- Simulations: 60 seeds (42–101), T = 5,000, evaluation from observation 500, change at 2,500, scenarios from
  `make_scenario` in `src/streamgenerator/datasource/dependence.hpp` (constant 0.6, drift 0.2 → 0.8, jump
  0.2 → 0.8). 6.7 uses n = 15,000, validation seeds 1000–1039, test seeds 42–141.
- In-process streams come from `src/streamgenerator/datasource/copula_source.hpp`. The estimators come from
  `src/estimators/registry.hpp` (`thesis::arm`); MFF, VFF and the scoring from
  `src/estimators/dependence/mff.hpp` and `vff.hpp`, the scoring from `src/estimators/evaluation.hpp`, applied to shared KLL
  pseudo-observations. The live consumer runs the same code.
- §6.7 storage is `store_bytes()`: the object plus its heap. Since the 2026-11-17 cut of unused estimator options it
  counts only what the thesis configuration holds (Hermite 344 B, Counting Matrix 488 B, RMPL 456/648 B less than
  the version the thesis draft measured); accuracy is unchanged. The pre-cut inputs are in
  `archive/experiments/results-6.7-before-cut/`.
- `std::*_distribution` draws differ between libstdc++ and libc++, so the same seed gives different data on
  Linux and macOS. The thesis numbers come from Linux/libstdc++.
- Live mode: with `LIVE=1` a probe takes each stream from a seeded server over TCP (`src/simulation/live_source.hpp`: the
  server is started per stream on free ports, observations are rebuilt from prices exactly as the consumer does).
  6.6: `LIVE=1 build/probe_real_practical in.csv out.csv` has the server replay the file (`REPLAY=`).
  Full 6.2–6.5 run over TCP (2026-11-17, 44 jobs, 60 seeds): 31 of 106 result files byte-identical to the in-process
  runs, the rest within ~1e-3 relative (KLL near-ties on log returns); 17 of 26 figures byte-identical, the other 9
  visually identical (at most 0.017% of pixels; `famswitch_initial_id` only by its known colour change); every thesis
  table identical except the third decimal in 4 of 840 lines of the scores-family summary.
- `live/`: the same device and scoring as the probe (`src/estimators/thesis_device.hpp`, `evaluation.hpp`), fed
  over TCP. The consumer sees log returns ln(p_t/p_{t-1}) = 1e-4 + 0.01 z rather than the draws z; ranks are
  unchanged, but the KLL margins store floats, so a near-tie can resolve differently. Full run (60 seeds x 4 families x 3 scenarios, 720 streams):
  707 streams byte-identical to the probe; in the other 13, per-seed RMSE differs by at most 1.2e-5 and 60-seed means
  by at most 1e-5 relative. `src/simulation/test/device_vs_probe.cpp` reproduces both (draws and returns) in-process.
- 6.6 needs the real-data inputs, which are not in the repository (see `6.6-real-data/run.sh`).
