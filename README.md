# datastreamer

Code for the honours thesis on **streaming estimation of time-varying copula dependence**: single-pass,
O(1)-per-observation, bounded-memory estimators of Spearman's ρ_S, the forgetting strategies that let them
adapt, and the live pipeline they run on.

| Thesis | Code |
| --- | --- |
| Ch. 2 data streams | `src/streamgenerator/` (server: DGP, producer, TCP broker), `src/consumer/`, `client/` |
| Ch. 4 estimators | `src/estimators/dependence/`: RMPL (`rmpl.hpp`), Counting Matrix (`count_matrix.hpp`), Hermite Series (`hermiter.hpp`) |
| Ch. 5 forgetting | fixed N_eff (`registry.hpp`), MFF (`dependence/mff.hpp`), VFF (`dependence/vff.hpp`) |
| Ch. 6 experiments | simulation programs in `experiments/simulation/`; `experiments/<section>/`: the scripts, `run.sh` and `images/` |

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

The build uses Python 3 to generate C++ defaults from `study_config.json`. The experiment scripts read the same file.

- Server: `DEPENDENCE=constant|drift|jump` (default `jump`) selects the thesis scenario (Table 6.2, over
  5,000 observations), `COPULA=gaussian|t|clayton|gumbel|gauss2t|gauss2clayton|clayton2gumbel` the family,
  `SYMBOLS=X,Y` the pair.
- Server experiment controls: `REPLAY=<file.csv>` (stream recorded returns instead of the DGP), `SEED` (DGP seed), `N` (stream N observations, then exit), `RATE` (observations per
  second, `0` = as fast as delivery allows; default 10), `WAIT_CONSUMERS` (hold the stream until that many consumers
  subscribe; delivery is then lossless), `PORT` / `BROKER_PORT`. Synthetic ticks carry the ground truth (true ρ_S,
  each leg's true PIT, the copula parameter and family) for scoring.
- Consumer: `./consumer [--host 127.0.0.1] [--port 9003] [--symbols AAPL,MSFT] [--print 20]` runs the Chapter 6
  device (`src/estimators/device.hpp`: shared KLL pseudo-observations; per estimator five fixed arms, MFF,
  VFF) and prints the true ρ_S with the MFF and VFF estimates. `--score <scenario> --seed <s> --out f.csv [--n N]`
  scores every strategy (ρ_S RMSE, excess log loss, H²) and appends the same rows the in-process runner writes.

## MFF lab (browser)

`./run.sh dev` (or `make && make consumer && ./server`, then open http://localhost:9002) opens the **MFF lab**:
each viewer gets a private stream from a steerable generator and a consumer process running an MFF they build
(estimator, memories N_eff, working families, η, δ). Jump, drift, switch the copula family or turn on automatic
events, and watch ρ_S, the MFF and VFF estimates, the weights by memory and by family, and the effective memory
live. The server starts and stops each session's consumer (`./consumer --live`, `src/consumer/live_lab.hpp`);
the browser only draws the state the consumer publishes. With the Gaussian family and memories 39–1999 the lab's MFF
is the thesis MFF exactly (`experiments/simulation/test/lab_vs_device.cpp`). `MAX_SESSIONS` (default 8) caps sessions.
The Lab runs the first 31 observations in about five seconds, then uses the selected playback speed
(one observation every half-second by default).

### Market stream and saved replay

Market shows the shared simulated pair, its RMPL-based MFF and the weights over candidate memory lengths.
The server estimates dependence from every paired observation and publishes a chart update about once a second.

To offer recorded pairs, put the CSVs you want viewers to access in a dedicated directory, then start the app:

```bash
MARKET_REPLAY_DIR=/path/to/saved-pairs ./run.sh dev
```

Use the stock experiment format `date,x,y`, one pair of returns or standardised returns per row in chronological order.
A filename such as `AAPL_MSFT.csv` supplies the pair labels. Select it in **Market stream**, then press **Play**.
Replay advances at one observation every half-second and displays the original dates.

Set `MARKET_PRICES_DIR` to a directory of per-ticker adjusted-close CSVs in the thesis download format
(`TICKER.csv` with columns `unix,adj`). Choose **Saved stocks** and pick any two available tickers.
The server aligns their prices on shared dates and runs MFF on the log returns between those dates.

## Experiments

See `experiments/README.md`. Every simulation program has a live mode (`LIVE=1`): its streams come from seeded servers
over TCP instead of in-process; `experiments/live/run_all_live.sh` runs all of 6.2–6.5 that way and compares with the
in-process results. In short: `experiments/build.sh` builds the simulation programs and each
`experiments/<section>/run.sh` rebuilds that section's figures and tables from its `results/` (`SIM=1` reruns
the simulations first).

Results CSVs are not committed (`experiments/*/results/` is ignored), and neither is `archive/`, which holds the
development history: superseded estimators, studies, notes and raw outputs (also as git bundles in `archive/branches/`, e.g. `bank-vff-study.bundle`).
