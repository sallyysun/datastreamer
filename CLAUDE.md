# datastreamer (final-thesis)

Honours thesis system: a **streaming** (single-pass, O(1)/tick, bounded-memory) estimator of time-varying copula
dependence, plus the TCP/WebSocket pipeline it runs live on. Central question: *how should an online estimator
trade statistical efficiency against adaptation to changing dependence?*

This branch holds only what the thesis uses. Everything else (superseded estimators, detectors, studies, notes,
raw results) is in the git-ignored `archive/` and on branch `bank-vff-study`.

## Layout

| Path | What |
| --- | --- |
| `src/streamgenerator/` | **server**: synthetic DGP (`datasource/`: copula families, `make_scenario`), producer, raw-TCP `Broker`. Publishes; does **not** estimate. |
| `src/consumer/` | **consumer**: TCP client, ring buffer, device state. Runs the thesis strategies. |
| `src/estimators/` | Ch. 4 estimators (`registry.hpp`: `thesis::arm`), Ch. 5 strategies MFF/VFF and Ch. 6 scoring (`evaluation.hpp`), the Ch. 6 device (`thesis_device.hpp`). |
| `src/protocol/` | `SymbolTick`, serializer, topic matching. |
| `client/` | React/Vite browser viewer: the MFF lab (`Lab.tsx`: per-viewer steerable stream + viewer-built MFF, run by a spawned `./consumer --live`) and the market view. **Strict viewer**: renders what it receives, computes nothing. |
| `experiments/` | One folder per Chapter 6 section (+ `ch3-background/`), each with code, `run.sh`, `images/`. See `experiments/README.md`. |
| `external/` | `uWebSockets`, `datasketches-cpp` (submodules). |

## Build & run

```bash
make CXX=g++ EIGEN=/usr JSON=/usr OPENSSL=/usr            # -> ./server (Linux paths)
make consumer CXX=g++ EIGEN=/usr JSON=/usr                # -> ./consumer
experiments/build.sh                                      # simulation programs -> experiments/*/build/
experiments/6.3-adaptive-forgetting/run.sh                # figures from results/;  SIM=1 reruns simulations
```

Server: `DEPENDENCE=constant|drift|jump` (default `jump`), `COPULA=...`, `SYMBOLS=X,Y`; ports 9002 (WS) / 9003 (TCP);
experiments: `SEED`, `N`, `RATE=0`, `WAIT_CONSUMERS=1` (lossless), `PORT`/`BROKER_PORT`. Consumer `--score` writes the
probe's rows; `experiments/live/run_live.sh` drives seeded server+consumer pairs.

## The estimator contract

`DependenceEstimator` (`src/estimators/dependence/estimator.hpp`): `update(x,y)`, `estimate()`, `reset()`,
`store_bytes()`, `name()`.

- **`estimate()` always returns Spearman's ρ**, so all estimators are comparable in one space.
- **`store_bytes()` is a thesis metric** (§6.7), not diagnostics. Keep it honest.
- MFF scores arms by the Gaussian log density at each arm's ρ_S, evaluated at the shared (U, V).

⚠️ **The forgetting knob is a mirror image across estimators.** All are set from one memory N_eff in
`registry.hpp`: Counting Matrix retention λ = (N−1)/(N+1) (larger = longer memory), Hermite weight 2/(N+1) and RMPL
gain 2/(N+1) (smaller = longer memory). Getting this backwards silently inverts an experiment.

⚠️ The probe (`experiments/common/probe_strategies.cpp`) and the live consumer share one device and one scorer
(`src/estimators/thesis_device.hpp`, `evaluation.hpp`, configured by `registry.hpp`). Change them there, never in a
probe or in the consumer; `common/test/device_vs_probe.cpp` must still reproduce the probe's rows.

## Conventions

- No number is a thesis result unless it comes from a 60-seed run (100 test seeds in §6.7) with error bars.
- `std::*_distribution` draws differ between libc++ (macOS) and libstdc++ (Linux); the thesis numbers are Linux.
- Do not commit `results/` CSVs; `images/` holds exactly the thesis figures for that section.
- Keep source nearly comment-free: a short header plus comments only for a non-obvious *why*.
- Before changing a probe, check old and new binaries give byte-identical CSVs on a short run.
