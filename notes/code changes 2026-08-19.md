# Code changes — 2026-08-19

Branch: `feat/online-nonparametric-dependence`. Continues [code changes 2026-08-17](code%20changes%202026-08-17.md).
This session: restructure, config alignment, a `Schedule`→`Dependence` rename, dead-code cleanup,
and the **parametric branch** (online Gaussian-copula MLE / Fisher scoring).

---

## 1. Repo restructure

- **`experiments/` moved out of `src/`** to a top-level `experiments/` (harness `main.cpp`,
  `copula_source.hpp`, `analysis.ipynb`, `live_probe.py`, `plot.py`, `plot_live.py`, `results/`).
  Include paths, the Makefile target, and the CSV output path (`experiments/results/`) updated.
  (Briefly nested as `experiments/1/` then flattened back to `experiments/`.)
- **`run` → `run.sh`** (its own usage text already said `./run.sh`).
- **`client.html` → `archive/`** — unreferenced standalone prototype, superseded by the React client.
- **`architecture.md`, `roadmap.md` → `notes/`**; design discussion split out to
  `notes/application_2.md`.
- **`experiments/results/`** now holds all CSV + PNG outputs (offline scenarios + live captures).

## 2. Offline == online estimator configs

- **`registry.hpp` (offline) aligned to `dependence_monitor.hpp` (live)**: Xiao bins 20 /
  calibration 100, Hermite order 10 / burn-in 50 — so offline and live RMSE are directly
  comparable. Both files carry a keep-in-sync note.
- **Offline `oscillate` period 1000 → 400** to match the live server. This (not the configs) was the
  real driver of the earlier offline/live oscillate gap; after matching, offline ≈ live (~0.01).

## 3. Live regime selection + `Schedule` → `Dependence` rename

- **`DEPENDENCE` env var** picks the live regime at launch (constant | drift | jump | oscillate |
  stochastic) — same binary demos any regime without recompiling.
- **Renamed `Schedule` → `Dependence`** (it emits the true copula parameter rho_t each step;
  "Schedule" read like a cron/timetable):
  - base `Schedule` → `Dependence`; `Constant/Drift/Jump/Oscillating/StochasticSchedule` →
    `...Dependence`; file `schedule.hpp` → `dependence.hpp`.
  - `make_schedule`/`make_live_schedule` → `make_dependence`/`make_live_dependence`;
    member `schedule_` → `dependence_`; env var `SCHEDULE` → `DEPENDENCE`.
  - README, notebook, and experiment-1 summary updated (historical `code changes 2026-08-17.md`
    left as-is).

## 4. Dead-code cleanup

- **`two_pass_variance.hpp` deleted** — unused windowed-variance estimator, never referenced, didn't
  implement `DependenceEstimator`.
- **`objective.hpp`**: was an empty tracked file → first a documented stub, then implemented (§5).
- Reviewed all `.hpp`/`.cpp` splits: the four split pairs (`serializer`, `synthetic_source`,
  `file_source`, `producer`) are all defensible (`producer` borderline but consistent).

## 5. Parametric branch — online Gaussian-copula MLE (new)

**`src/estimators/objective.hpp`** (implemented)
- `GaussianCopulaObjective`: bivariate Gaussian-copula `log_likelihood`, `score` (dl/drho, unbiased:
  E[s]=0 at true rho), and closed-form `fisher` I(rho) = (1+rho^2)/(1-rho^2)^2. Estimator-agnostic
  pure math; will also back the forgetting-MLE and later Bayesian update.

**`src/estimators/dependence/gaussian_mle.hpp`** (new)
- `GaussianCopulaMLE : DependenceEstimator`. Recursive Fisher scoring in theta = atanh(rho)
  (unconstrained; Fisher info in theta is simply 1+rho^2). Update:
  `H_t = lambda*H_{t-1} + I_theta`, `theta_t = theta_{t-1} + s_theta / H_t`
  (lambda=0 => cumulative recursive MLE; lambda in (0,1) => exponential forgetting, ESS ~ 1/(1-lambda)).
- Marginals standardized online with **cumulative Welford** moments — the marginal scale is
  stationary, so it is NOT forgotten, only the dependence is. This decouples the timescales and
  avoids the `hermite_ew` cold-start. `estimate()` returns Spearman rho_S, same space as the others.
- Registered as `gaussian_mle` (cumulative) and `gaussian_mle_ew` (lambda=0.99); added to the harness.

**Validation (RMSE vs true rho_S, after burn-in 500):**

| estimator | constant | drift | jump | stochastic | oscillate |
|---|---|---|---|---|---|
| gaussian_mle    | 0.015 | 0.412 | 0.310 | 0.114 | 0.410 |
| gaussian_mle_ew | 0.039 | **0.052** | **0.076** | 0.057 | 0.358 |

Confirms the score/Fisher derivation: cumulative MLE is efficient on `constant` (~0.015, matches the
best nonparametric); the forgetting MLE is best/tied-best on `drift`/`jump`. Parametric and
nonparametric branches now compare apples-to-apples in the same harness.

## 6. Misc

- **`delay_strategies.hpp` moved** `streamgenerator/timing/` → `streamgenerator/runtime/` (next to
  `producer`, its only user); empty `timing/` removed; two includes updated.
- **README**: added a **Mermaid system-design diagram** (DGP → producer → dual delivery: monitor +
  EventBus/Consumer → WebSocket/React; offline harness reusing the same estimator code).
- **Plots**: added colours for `gaussian_mle` / `gaussian_mle_ew` (notebook + `plot_live.py`);
  regenerated all `experiments/results/*.png`.

---

## Build / run (unchanged interface, new regime knob)

```bash
make experiment && for s in constant drift jump stochastic oscillate; do ./experiment $s; done
DEPENDENCE=drift ./server        # live server, choose the regime
python experiments/live_probe.py --csv experiments/results/live.csv
```

`experiment` (the binary) and `server` are gitignored build artifacts — `make clean` removes both.

## 7. Mini-Kafka: producer -> broker -> per-device C++ consumers

Refactor to the thesis architecture (Ch. 2.3 / Fig 2.1): the server is a pure **broker** that
publishes **raw** events; **estimation moves into each consumer's device**; consumers are
independent **C++ processes**; the browser is a display client. No broker retention/offsets — each
consumer's `RingBuffer` is its only memory (thesis "buffer-type device").

- **`streamgenerator/runtime/broker.hpp`** (new) — minimal TCP pub/sub broker. Consumers connect,
  send `SUBSCRIBE <topic>\n`, and receive newline-delimited JSON for that topic (or `all`).
  Non-blocking sockets, best-effort delivery (slow consumer drops events), `poll()` accept/read loop
  on its own thread, mutex-guarded subscription table, `SIGPIPE` ignored.
- **`serializer.{hpp,cpp}`** — added `from_json` (JSON tick -> `MarketEvent`) so consumers can
  deserialize the raw stream.
- **`device/consumer.hpp`** — the device now owns a `DependenceMonitor` **inference layer**, fed in
  `process_available` (respecting the lag policy); `dependence()` exposes its estimates. Estimation
  is per-device, independent across consumers.
- **`src/consumer/main.cpp`** (new) — standalone C++ consumer binary: connects to the broker over
  TCP, subscribes, deserializes events, runs its own device (`RingBuffer` + `StateManager` +
  inference), prints per-estimator rho. Flags: `--host --port --topic --policy --print`.
- **`app/main.cpp`** — rewritten: server = producer + broker + display. Publishes **raw** events to
  the broker (topic from `EventType`: `trades`/`quotes`) and to the browser `ticks` topic. Removed
  the global `DependenceMonitor` and the in-process `EventBus`/`Consumer` wiring (estimation is no
  longer server-side). `config.json` may add `server.broker_port` (default 9003).
- **`Makefile`** — new `consumer` target (Eigen + nlohmann, no uWebSockets); `consumer` added to
  clean + `.gitignore`.
- **`EventBus` is now orphaned** — the network `Broker` supersedes the in-process bus. Left in the
  tree for now (only a comment + vestigial `sub_id_` reference remain); candidate for removal.

Verified end-to-end (loopback): `DEPENDENCE=jump ./server` + `./consumer --topic trades` — the
consumer receives raw events and its own inference warms up toward the true rho.

```bash
make server && make consumer
DEPENDENCE=jump ./server                 # producer + broker (tcp/9003) + display (9002)
./consumer --topic trades --policy all   # separate process; runs its own estimators
```

## Still missing / next
- Consolidate the two "brokers": the in-process `EventBus` is superseded by the network `Broker` —
  remove or repurpose it.
- Richer topics (per-symbol / per-pair) so consumers select more than event type; wire `FileSource`
  replay + live feeds as selectable sources.
- Broker retention/offsets deliberately omitted (minimal buffer-type devices) — revisit only if
  replay/at-least-once is needed.
- React client: display-only landing page where a consumer picks a topic then streams (deferred).
- Parametric: goodness-of-fit ("does the model fit?") to choose parametric vs nonparametric online;
  rank-PIT transform for non-Gaussian real marginals (currently standardization, exact for Gaussian).
- Change detection + restart (CUSUM / BOCPD); the `reset()` hook is in the interface.
- Multi-seed replication + error bars before any number is a thesis result.
