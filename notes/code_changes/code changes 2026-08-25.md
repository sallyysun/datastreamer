# Code changes — 2026-08-25

Branch: `feat/online-nonparametric-dependence`. Continues
[code changes 2026-08-19](code%20changes%202026-08-19.md).

Living doc — append new sections as the day's work continues. Themes today: a `std::numbers::pi`
cleanup, the broker gaining hierarchical topics + throttling, the switch to **per-symbol publish /
pair-on-consumer**, a branch-history repair, and the client design doc.

---

## 1. `std::numbers::pi` cleanup

- Replaced every hand-typed `kPi = 3.14159…` literal with **`std::numbers::pi`** (C++20 `<numbers>`)
  across `hermiter.hpp`, `gaussian_mle.hpp`, `synthetic_source.cpp`, `dependence.hpp`, and
  `experiments/copula_source.hpp`; removed the now-orphaned `kPi` constants.
- **`hermiter.hpp`**: `kQuarterRootPiInv = 1/π^¼` is now computed once
  (`inline static const … = 1.0/std::sqrt(std::sqrt(std::numbers::pi))`) instead of a magic literal,
  and `basis_integrals` reuses it (`z(0) = √2 / kQuarterRootPiInv`) — dropping a live `std::pow`.
  `std::sqrt` isn't `constexpr` until C++23, hence `static const` (computed at static-init), not
  `constexpr`.
- Fixed a **stale header comment**: it claimed the Spearman coupling matrix `W` is built by
  trapezoidal quadrature, but `integral_matrix` computes it in **closed form** (IBP symmetry + column
  recurrence). Rewrote the comment to describe the port's actual scope vs `external/hermiter`.
- Verified: `experiment` / `server` / `consumer` all build; RMSE unchanged (`std::numbers::pi` is the
  same `double`) — e.g. constant `gaussian_mle` 0.0146.

## 2. Broker — hierarchical topics + throttled subscriptions

**`streamgenerator/runtime/broker.hpp`**
- **Hierarchical dot-prefix matching** (`Sub::matches`): a subscription matches a published topic if
  it equals it, is the literal `all`, or is a dot-boundary prefix of it — so `trades` receives every
  `trades.<…>`, `trades.equity.AAPL` selects one, `all` gets everything. Backward compatible with the
  old flat `SUBSCRIBE trades`.
- **Throttled subscriptions**: `SUBSCRIBE <topic> [max_hz]` rate-limits a connection broker-side
  (per-connection `min_interval` + `last_sent`); excess messages are dropped — consistent with the
  existing best-effort, buffer-type-device model.
- Restructured `subs_` from `unordered_map<int, set<string>>` to `unordered_map<int, Sub>`.

Verified end-to-end: `--rate 1` delivered ~1 msg/s (server emits ~10/s); prefix/exact/wildcard/wrong
topic all route as expected.

## 3. Per-symbol publish → pair on consumer

The move from a single bivariate event to **per-symbol equity topics** (thesis Option A), keeping the
device/estimator stack bivariate and untouched.

- **New `src/protocol/symbol_tick.hpp`** — single-symbol `SymbolTick {symbol, price, volume, ret,
  timestamp, true_rho_s}`; `Serializer` gained `to_json(SymbolTick)` / `symbol_from_json`.
- **Producer (`streamgenerator/main.cpp`)** — each DGP step is split into two per-symbol ticks,
  published to `trades.equity.<SYMBOL>`. Symbols set via `SYMBOLS=X,Y` (default `AAPL,MSFT`; equity
  only). The browser `ticks` topic still carries the combined event.
- **Consumer (`consumer/main.cpp`)** — new `--symbols AAPL,MSFT`; subscribes to both symbol topics,
  routes ticks into one **FIFO queue per leg**, and pairs front-to-front into the bivariate
  `MarketEvent` the device consumes.
- **Assumption:** synchronous producers (one tick per symbol per step, ordered) → arrival-order
  pairing is exact. Async / multi-rate legs deferred (need timestamp/sequence alignment); `--rate`
  throttling can desync the legs, so leave it 0 for estimation. DGP still emits a single pair.

Verified: pairing exact (`true_rho_s=0.582` for constant ρ=0.6), estimators warm toward it;
unmatched symbol → 0 pairs; `SYMBOLS` override reroutes.

## 4. Branch history repair (squash-merge divergence)

PR #1 was **squash-merged** into `main` as one commit (`5f4c841`), but this branch still carried the
24 original commits on an old base — so a merge showed conflicts across ~50 files that were really
identical content. Confirmed `git diff origin/main 8f01b1b` was empty (main already contains the
branch's content up to its pre-existing tip), then rebuilt the branch as **`origin/main` + the one
genuinely-new commit** via `checkout -B origin/main` + `cherry-pick` (0 conflicts). Force-push left to
the user (history rewrite). Prevention notes captured: delete squash-merged branches / enable
auto-delete / rebase-after-merge.

## 5. Client design doc

- **`notes/design/client-vision.html`** (new) — design for the subscription-driven thin viewer:
  topic-aware WS, subscription picker + adaptive panels, strict viewer, and a Phase 3 "live estimates
  loop" (consumer republishes ρ̂ → broker → server WS bridge → browser dependence panel). Architecture
  diagram, wire protocol, UI mockup, phased build order, open decisions. Design only — no client code
  written yet.

## 6. Docs / housekeeping

- `notes/todo.md` — consolidated backlog (status keys), marked done items (EventBus removal, richer
  topics, throttling), recorded the per-symbol model + deferred async/multi-symbol-DGP work.
- Reorganised `notes/code_changes/`; added `notes/change_detection.md` (companion to roadmap).

## 7. Topic-aware WebSocket (shared dot-prefix matching)

- **New `src/protocol/topic.hpp`** — one `topic_matches(sub, pub)` predicate (equal / `all` /
  dot-boundary prefix) shared by **both transports**. The `Broker` now calls it instead of its own
  copy.
- **Server WS (`streamgenerator/main.cpp`)** is now topic-aware, symmetric with the TCP broker:
  on open it sends a `catalog` of available symbols; on a `{"action":"subscribe","topics":[…]}`
  message it expands each request (a prefix like `trades` included) against the catalog and calls
  `ws->subscribe()` per concrete topic. Per-symbol ticks are published to WS topics too (the old
  single `ticks` firehose is gone).

## 8. Client — subscription viewer (Phase 1–2)

- **`client/src/App.tsx`** rewritten: connects, reads the catalog, renders **symbol chips**; toggling
  one sends subscribe/unsubscribe and adds/removes that price line. Subscription is **encoded in the
  URL** (`?symbols=…`). Adaptive panels (per-symbol price lines merged by timestamp, tape), and a
  **display throttle** — buffered ticks flushed to the screen at most once per `intervalSec`
  (slider capped at 1s, dialable slower + number input). **Strict viewer**: `rollingCorr` removed;
  nothing is computed in the browser. Dependence panel is a labelled **Phase-3 placeholder**.
- **`client/src/App.css`** restyled into a polished dark "quant terminal".

## 9. Protocol refactor — SymbolTick is the atom, TickPair is a pair of them

- **`SymbolTick` is observables-only**: `{symbol, price, volume, timestamp, true_rho_s}` — **`return`
  dropped from the wire**. Each device derives `return = Δlog price` (price and return are the same
  information; deriving it keeps analysis on the device). `DeviceState` holds a prev-price per leg.
- **`MarketEvent` → `TickPair`**, defined as `{SymbolTick x, y}` **in `symbol_tick.hpp`**;
  **`market_event.hpp` deleted** and all ~15 references updated. A paired observation is now literally
  two ticks, with one definition of a leg's fields.
- **Dead code removed**: `EventType` (never read), the unused `MarketEvent` JSON
  (`to_json`/`from_json`), and `DeviceState::rolling_correlation`.
- Verified end-to-end: with returns derived from price, estimators still converge to the true ρ_S
  (constant: xiao ≈ 0.61, hermite ≈ 0.57 vs true 0.582 at n=200).

## 10. Design docs (for review before building)

- **`notes/design/client-vision.html`** — architecture/plan for the subscription viewer (topic-aware
  WS, picker, strict viewer, Phase-3 estimates loop).
- **`notes/design/frontend-mockup.html`** — high-fidelity visual mockup of the viewer (Phases 1–2).

---

## Build / run

```bash
make server && make consumer
cd client && npm run build && cd ..
SYMBOLS=AAPL,MSFT DEPENDENCE=constant ./server        # per-symbol producer + broker + WS + display
./consumer --symbols AAPL,MSFT --print 25             # subscribes to both, FIFO-pairs, estimates ρ
./consumer --symbols AAPL,MSFT --rate 1               # broker-side throttle to ~1 msg/s
# browser: http://localhost:9002  → pick symbols, set the display interval
```

## Still open / next
- Client: review the mockup, then align the live UI; decide Phase 3 (consumer republishes ρ̂ → WS).
- Multi-symbol DGP so the picker has real choices (routing + pairing already support it).
- Async / multi-rate leg alignment (replace FIFO pairing with timestamp/sequence matching).
- Optional: group `SyntheticSource`'s per-leg sim state (`price/volume/drift/vol`) into a `Leg` struct.
