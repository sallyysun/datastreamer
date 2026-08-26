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
- ⬜ **Change detection + restart** (CUSUM / BOCPD); `reset()` hook already in the interface
  (`estimator.hpp:16`). Design space + build order in
  [change_detection.md](studies/change_detection.md).
- ⬜ **Local-level Kalman estimator** (`kalman_local_level.hpp`) on Fisher-$z$'d blockwise
  Spearman. Strict generalization of the fixed-$\lambda$ estimators (steady-state Kalman gain
  $K=1-\lambda$), and the prerequisite for innovation-based detection.

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
