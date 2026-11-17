# datastreamer

Code for the honours thesis on **streaming estimation of time-varying copula dependence**: single-pass,
O(1)-per-observation, bounded-memory estimators of Spearman's ρ_S, the forgetting strategies that let them
adapt, and the live pipeline they run on.

| Thesis | Code |
| --- | --- |
| Ch. 2 data streams | `src/streamgenerator/` (server: DGP, producer, TCP broker), `src/consumer/`, `client/` |
| Ch. 4 estimators | `src/estimators/dependence/`: RMPL (`recursive_mpl.hpp`), Counting Matrix (`xiao_spearman.hpp`), Hermite Series (`hermiter.hpp`) |
| Ch. 5 forgetting | fixed N_eff (`registry.hpp`), VFF and MFF (`evaluation.hpp`, controller in `vff.hpp`) |
| Ch. 6 experiments | `experiments/<section>/`: one folder per section, with code, `run.sh` and `images/` |

## Live pipeline

```
server (DGP: scenario x copula family)  ──TCP 9003──▶  consumer (ring buffer + thesis VFF / MFF strategies)
                                         ──WS  9002──▶  client (browser viewer, computes nothing)
```

```bash
git submodule update --init --recursive        # uWebSockets, datasketches-cpp
make            # -> ./server    (uWebSockets, openssl, Eigen, nlohmann-json)
make consumer   # -> ./consumer  (Eigen, nlohmann-json, datasketches)
./run.sh dev    # server + vite client
```

On Linux pass the paths: `make CXX=g++ EIGEN=/usr JSON=/usr OPENSSL=/usr` (first build uSockets with
`make -C external/uWebSockets/uSockets WITH_OPENSSL=1`).

- Server: `DEPENDENCE=constant|drift|jump` (default `jump`) selects the thesis scenario (Table 6.2, over
  5,000 observations), `COPULA=gaussian|t|clayton|gumbel|gauss2t|gauss2clayton|clayton2gumbel` the family,
  `SYMBOLS=X,Y` the pair.
- Server experiment controls: `REPLAY=<file.csv>` (stream recorded returns instead of the DGP), `SEED` (DGP seed), `N` (stream N observations, then exit), `RATE` (observations per
  second, `0` = as fast as delivery allows; default 10), `WAIT_CONSUMERS` (hold the stream until that many consumers
  subscribe; delivery is then lossless), `PORT` / `BROKER_PORT`. Synthetic ticks carry the ground truth (true ρ_S,
  each leg's true PIT, the copula parameter and family) for scoring.
- Consumer: `./consumer [--host 127.0.0.1] [--port 9003] [--symbols AAPL,MSFT] [--print 20]` runs the Chapter 6
  device (`src/estimators/thesis_device.hpp`: shared KLL pseudo-observations; per estimator five fixed arms, MFF,
  VFF) and prints the true ρ_S with the MFF and VFF estimates. `--score <scenario> --seed <s> --out f.csv [--n N]`
  scores every strategy (ρ_S RMSE, excess log loss, H²) and appends the same rows the in-process probe writes.

## Experiments

See `experiments/README.md`. Every simulation program has a live mode (`LIVE=1`): its streams come from seeded servers
over TCP instead of in-process; `experiments/live/run_all_live.sh` runs all of 6.2–6.5 that way and compares with the
in-process results. In short: `experiments/build.sh` builds the simulation programs and each
`experiments/<section>/run.sh` rebuilds that section's figures and tables from its `results/` (`SIM=1` reruns
the simulations first).

Results CSVs are not committed (`experiments/*/results/` is ignored), and neither is `archive/`, which holds the
development history: superseded estimators, studies, notes and raw outputs (also on branch `bank-vff-study`).
