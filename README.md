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

## Start the website

Run these commands from the repository root.

### On your computer

```bash
./run.sh dev
```

Open <http://localhost:5173>. This starts the C++ server and the Vite development site together. Press `Ctrl+C` in the terminal to stop them.

### Make it public with ngrok

Open two terminals in the repository root. In the first terminal, build and start the production site:

```bash
./run.sh prod
```

In the second terminal, forward the web server port:

```bash
ngrok http 9002
```

Open <https://thesaurus-process-jazz.ngrok-free.dev>. The public address must match `ws.prod` in `config.json`, and your ngrok account must route that address to port `9002`. Keep both terminals running while people use the site. Press `Ctrl+C` in both terminals to take it offline. The site is running from your computer, so it will also go offline if that computer sleeps or shuts down.

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
(estimator, memories N_eff, working families, η, δ). The guided demo uses a fixed seed, starts with a steady
period, then jumps and drifts at the observations set under `demo.lab.guided` in `config.json`. Compare MFF with the shortest and longest
fixed memories or switch the chart to VFF. Viewers can also apply their own events.
The charts show ρ_S, the estimates, the weights by memory and by family, and the effective memory
live. The server starts and stops each session's consumer (`./consumer --live`, `src/consumer/live_lab.hpp`);
the browser only draws the state the consumer publishes. The standard warm-up, Gaussian family and thesis memories
match the Chapter 6 device (`experiments/simulation/test/lab_vs_device.cpp`). `MAX_SESSIONS` (default 8) caps sessions.
Every demo number lives in the `demo` section of `config.json`: the warm-up (observations and seconds), the
playback speeds and default, η and δ, the Lab's form defaults and guided events, and the Market's event schedule.
The server rereads it when a page loads, when a Lab session starts and when a Market subscription starts, so
edit it and reload the page; no rebuild or restart is needed. The Lab form can override its own defaults.

### Shared Market broadcast

Market offers the server's simulation and configured replays in the **Source** picker.
The simulation generates five assets, A–E. Visitors pick any two and press **Subscribe** to begin tracking their prices and MFF
from the next observation. Each subscription starts with fresh weights. Lab sessions are private.

Start a simulation with `./run.sh dev` for local development or `./run.sh prod` for the Vite production build
served by C++. The production script reads the configured WebSocket URL from `config.json`. Point ngrok at
the server port to make the app available publicly.

The simulation alternates jumps and drifts automatically, on the schedule under `demo.market` in `config.json` (set `loop` to false to run the list once). `demo.startRho` and `demo.memories` are shared by the Lab and the Market; give `lab` or `market` its own `rho`/`startRho` or `memories` to override one tab.
Its MFF uses candidate memories of 11, 39, 99 and 399 to make adaptation visible on a short run.
The simulated broadcast starts estimating after 13 observations per subscription. Saved-price replay
uses the standard 31-observation start.
Set `MARKET_EVENTS` to a server-side CSV to use
your own schedule. Use one-based observation numbers and a duration of zero for a jump.
The running server picks up edits for future observations. A CSV containing only the header holds dependence fixed.

```csv
observation,target_rho,duration
40,0.80,0
100,0.25,30
```

To broadcast recorded prices, put the saved per-ticker `unix,adj` CSVs on the server and choose two to five
tickers by filename. The server streams their adjusted prices on shared dates and uses paired log returns
for MFF. The recording loops independently of the simulation. Each loop starts fresh charts and weights so
the last and first prices never form a return. Browse sources and pairs while your existing subscription
continues. Click **Subscribe** to switch to the new selection.

```bash
MARKET_PRICES_DIR=/path/to/saved-prices MARKET_TICKERS=AAPL,MSFT ./run.sh prod
```

To replay the exact Section 6.6 inputs, extract `data66/rerun/in/` from `origin/data-66-thesis`
(commit `b811b4e`) and set `MARKET_THESIS_DIR` to the extracted `in` directory. Choose **Thesis return replay**
and one of the recorded pairs. Its chart displays the committed volatility-standardised log returns,
and MFF receives the `x,y` values directly. The Market subscription begins at the current replay date.

Playback speed is chosen per viewer from `demo.playback`. Saved files and the event schedule are read by the server.

## Experiments

See `experiments/README.md`. Every simulation program has a live mode (`LIVE=1`): its streams come from seeded servers
over TCP instead of in-process; `experiments/live/run_all_live.sh` runs all of 6.2–6.5 that way and compares with the
in-process results. In short: `experiments/build.sh` builds the simulation programs and each
`experiments/<section>/run.sh` rebuilds that section's figures and tables from its `results/` (`SIM=1` reruns
the simulations first).

Results CSVs are not committed (`experiments/*/results/` is ignored), and neither is `archive/`, which holds the
development history: superseded estimators, studies, notes and raw outputs (also as git bundles in `archive/branches/`, e.g. `bank-vff-study.bundle`).
