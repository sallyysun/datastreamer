# TODO

## Client vision — subscription-driven thin viewer

**Goal:** one web page anyone can open; each connection subscribes to the events/topics
it wants, and the display adapts to that subscription. The browser stays a **strict
viewer** — it renders what it receives and runs no estimators (those live in the C++
consumer devices).

**Why WebSocket (not raw TCP like the C++ consumers):** browsers cannot open raw TCP
sockets — only HTTP / WebSocket / WebRTC. So the browser must use WS. WS and the TCP
broker are just two transports carrying the *same* topic-based pub/sub, so the
subscribe-and-display model works identically on both.

### What to build
1. **Topic-aware WS on the server.** Today the WS side dumbly broadcasts everything to a
   single `ticks` topic. Give it the same per-connection subscription the TCP broker has:
   - client → server control message, e.g. `{"subscribe": ["trades", "quotes"]}`;
   - server parses it and calls `ws->subscribe(topic)` per topic (uWS pub/sub);
   - producer publishes each event to its topic (`app.publish(topic, msg)`), so a
     connection receives only the topics it asked for.
2. **React subscription picker + adaptive views.** A control to choose topics; the page
   renders different panels depending on what is subscribed (e.g. `trades` → price chart;
   a specific pair → that pair's view; later, a consumer's re-published estimates → a
   dependence panel). Encode the subscription in the URL so a link reopens the same view.
3. **Strict viewer.** Remove the client-side rolling-correlation calc added during the
   prettification — the browser should display only what it receives, no computation.

### Open questions
- **Topic taxonomy:** event type (`trades` / `quotes`) only, or also per-symbol / per-pair?
  And should consumers be able to **re-publish** their estimates to a topic the browser can
  subscribe to (so the display can show live dependence without the browser computing it)?
- Whether the WS and TCP broker should share one topic-routing implementation (currently
  the `Broker` is TCP-only; uWS has its own pub/sub).

---

## Backlog (consolidated from notes/code_changes 2026-08-19)

Status key: ✅ done · ⬜ open · ➖ deliberate non-goal.

### Broker / topics / sources
- ✅ **Consolidate the two "brokers".** `EventBus` removed from the tree; the network `Broker`
  is the only bus. (Was listed as "still missing" in the 2026-08-19 note; now complete.)
- ✅ **Richer topics — publish per-symbol, pair on the consumer.** Producers now publish one
  single-symbol `SymbolTick` per leg to `trades.equity.<SYMBOL>` (symbols set via `SYMBOLS=X,Y`,
  default `AAPL,MSFT`; equity-only for now). The consumer subscribes to two symbols and
  **FIFO-pairs** the legs into the bivariate sample its estimators consume — the device stack
  stays bivariate and untouched. Broker matching is dot-boundary prefix: `trades` /
  `trades.equity` get every symbol, `trades.equity.AAPL` gets one, `all` gets everything.
  - *Assumes synchronous producers* (one tick per symbol per step, same order) so arrival-order
    pairing is exact. **Async / multi-rate legs deferred** — will need timestamp/sequence
    alignment instead of FIFO, and `--rate` throttling can desync the legs today.
  - *Next: a multi-symbol DGP* — routing + pairing are in place, but the synthetic source still
    emits a single correlated pair (published as its two legs).
- ✅ **Throttled subscriptions.** `SUBSCRIBE <topic> [max_hz]` (consumer flag `--rate N`)
  rate-limits a connection to at most N messages/second broker-side; excess is dropped
  (consistent with the best-effort, buffer-type-device model).
- ⬜ **Wire `FileSource` replay + live feeds as selectable sources.** `file_source.{hpp,cpp}`
  exists but the consumer only reads the broker over TCP — no source selection yet.
- ➖ **Broker retention / offsets** — deliberately omitted (minimal buffer-type devices);
  revisit only if replay / at-least-once is needed.

### Client (see "Client vision" above)
- ⬜ **React topic-picker viewer** — `client/src` has no subscribe/subscription code yet;
  the WS side still broadcasts a single `ticks` topic.

### Estimators
- ⬜ **Validate C++ estimators against the reference submodules** (`external/hermiter` in R,
  `external/onlineNPCORR` in Python) on shared inputs.
- ✅ **Hermite scope vs `external/hermiter`:** we port only the *online bivariate Spearman's rho*
  path. Not ported (not needed for this thesis): univariate pdf/cdf/quantile, Kendall's tau,
  consistent merging of sub-estimators, batch mode. The old quadrature-vs-closed-form `W`
  difference is closed — `integral_matrix` is analytic (IBP symmetry + column recurrence).
- ⬜ **Parametric goodness-of-fit** (choose parametric vs nonparametric online) + **rank-PIT**
  transform for non-Gaussian real marginals (currently standardization, exact for Gaussian).
- ✅ **Change detection + restart, step 1: innovation CUSUM** (`dependence/kalman_cusum.hpp`,
  key `kalman_cusum`; build order step 3 in
  [change_detection.md](studies/change_detection.md)). Two one-sided CUSUMs on
  `KalmanLocalLevel`'s standardized innovations $e_t/\sqrt{F_t}$ (change_detection.md Part III
  Level 1); on alarm, calls the filter's `reset()` — the hook already in the interface
  (`estimator.hpp:16`). BOCPD is still open, tracked separately below.
  **Verified on the real pipeline, not synthetic $N(0,1)$ draws:** classical
  Page/Hawkins-Olwell defaults ($\delta=1,h=5$) produced **zero false alarms in ~80,000 simulated
  blocks** (8 seeds × 10000 blocks, `constant` scenario) — i.e. $\mathrm{ARL}_0>10{,}000$ blocks,
  a safe lower bound, *not* a calibrated number (that is the separate $\mathrm{ARL}_0$-calibration
  item below — this only confirms the defaults are sane enough to build on, not thesis-ready).
  On `jump` (0.2→0.8 at $t=2500$): 1 alarm, 0 false alarms before the jump, detection delay 59
  ticks (3 blocks). On `oscillate`: 21 alarms over 5000 ticks — plausible, since each half-period
  really is a large genuine shift, not noise.
  **Non-obvious finding — hard reset has a real RMSE cost, not just a detection-delay benefit:**
  `kalman_cusum`'s RMSE is *worse* than plain `kalman_fast` on both `jump` (0.118 vs 0.092) and
  `oscillate` (0.373 vs 0.316), despite (or rather because of) detecting the jump correctly. By
  the time the alarm fires, `kalman_fast` has already been adapting toward the new value for
  several blocks — the hard reset snaps the state back to $z=0$ and forces the filter to re-climb
  from a flat prior, adding transient error exactly where the smoothly-adapting filter was already
  closing in. This is the concrete instance of the Part V point that "VFF and detect-and-restart
  are not rivals, they are one dial at different decision hardness" — a soft posterior-weighted
  mix (Harrison-Stevens / IMM) would plausibly avoid this discontinuity; worth a direct comparison
  once BOCPD (softer than hard-reset CUSUM) exists.
  **Not done yet, and load-bearing before any number above is a thesis result:** $\mathrm{ARL}_0$
  calibration (next item) and the efficient-score / marginal-invariant CUSUM refinement
  (change_detection.md Part IV.2).
- ✅ **Local-level Kalman estimator** (`dependence/kalman_local_level.hpp`, keys `kalman`/
  `kalman_fast`). Emits $y_t=\operatorname{atanh}(\hat\rho^S_t)$ every `block` ticks from a
  windowed Spearman, with **known** observation variance $\sigma_v^2\approx 1.06/(\text{block}-3)$,
  so the filter is exact linear-Gaussian (no EKF) — the free knob is $\sigma_\omega^2$ (process
  noise), not a hand-picked $\lambda$.
  **Verified — Bridge 1 identity is exact, not approximate:** on a stationary stream the filter's
  steady-state Kalman gain matches the closed form $K=(\sqrt{q^2+4q}-q)/2$, $q=\sigma_\omega^2/\sigma_v^2$,
  to numerical precision across $q\in[4\times10^{-4}, 4.4]$ — confirms every fixed-$\lambda$
  estimator in this directory really is this filter's steady-state special case, not just an
  analogy.
  **Found and fixed a real bug via the harness, not guesswork:** the block observation model
  assumes $\rho$ is ~constant *within* a block; at the default `block=50` this broke down on
  `oscillate` (period 400 — a block spans 1/8 of a cycle, and near a zero-crossing $\rho$ moves by
  nearly its full range within one block), making `kalman`/`kalman_fast` **worse than every other
  estimator including non-adaptive ones** (RMSE 0.42-0.43 vs `windowed_rank`'s 0.36). Swept block
  size against all 5 scenarios and confirmed the mechanism (RMSE fell monotonically as block
  shrank on `oscillate`, rose slightly on `constant` from noisier per-block Fisher-z reads);
  `block=20` is the harness-wide sweet spot and is now the default.
  **Result, full harness (`experiments/results/experiment_*.csv`):** `kalman` (slow,
  $\sigma_\omega^2=10^{-4}$) is competitive with the cumulative estimators on `constant` (RMSE
  0.017 vs `gaussian_mle`'s 0.015); `kalman_fast` ($\sigma_\omega^2=10^{-2}$) is competitive with
  the best `_ew` estimators on `drift`/`jump`/`stochastic`, and **beats every other estimator in
  the suite, including all `_ew` variants, on `oscillate`** (RMSE 0.316 vs `windowed_rank`'s 0.361
  and `hermite_ew`'s 0.345) — the fastest-changing, hardest scenario, which is exactly where a
  genuinely adaptive filter should have the edge over a fixed window or fixed $\lambda$.
  Exposes `innovation()` / `innovation_variance()` / `standardized_innovation()` /
  `kalman_gain()` — the exact quantities an innovation-CUSUM (`change_detection.md` Part III
  Level 1) needs, and the next thing to build on top of this.

### Experiments / device
- ⬜ **Multi-seed replication + error bars** before any number is a thesis result (harness
  still uses a single fixed `seed = 42`).
- ⬜ **$\mathrm{ARL}_0$ calibration + detection-delay metrics** in the harness. Detectors are
  only comparable at matched in-control false-alarm rate, and rank/PIT feedback means
  $\mathrm{ARL}_0$ must be bootstrapped on the actual pseudo-observation stream rather than
  read off a formula. Blocks every detector result — see
  [change_detection.md](studies/change_detection.md) Part VI.
- ⬜ Decide whether to keep the `StateManager` moments (now inside `DeviceState`) exposed, or
  slim the device to buffer + copula inference only.
