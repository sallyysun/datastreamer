# Architecture

## Summary

**datastreamer** is a real-time market-data streaming system. A C++20 server generates synthetic (or file-based) market events, fans them out over WebSockets via [uWebSockets](https://github.com/uNetworking/uWebSockets), and a React/TypeScript client renders live price charts. The design follows a **producer → bus → per-client consumer** pipeline where each connected client gets its own ring buffer and lag policy, so a slow client never blocks others.

## Folder Tree

```
datastreamer/
├── src/
│   ├── app/
│   │   └── main.cpp                  # Entry point: config, HTTP, WS, producer thread
│   ├── streamgenerator/              # Data production layer
│   │   ├── datasource/
│   │   │   ├── datasource.hpp        # Abstract DataSource interface
│   │   │   ├── synthetic_source.*    # Correlated GBM via Eigen (Cholesky copula)
│   │   │   └── file_source.*        # CSV replay source
│   │   ├── runtime/
│   │   │   ├── producer.*           # Pulls from DataSource, applies delay, calls sink
│   │   │   └── event_bus.hpp        # Thread-safe pub/sub fan-out dispatcher
│   │   ├── protocol/
│   │   │   └── serializer.*         # MarketEvent → JSON
│   │   ├── event/
│   │   │   └── market_event.hpp     # Core event struct (prices, volumes, returns)
│   │   └── timing/
│   │       └── delay_strategies.hpp # Pluggable inter-event timing (fixed, Poisson, burst)
│   └── device/                       # Client-side processing layer
│       ├── consumer.hpp              # Per-client event processor with lag policies
│       ├── ring_buffer.hpp           # Lock-free-read circular buffer (atomic head)
│       ├── state_manager.hpp         # Incremental stats (Welford) + windowed correlation
│       └── running_stats.hpp         # O(1) online mean/variance (Welford's algorithm)
├── client/                           # React + Vite frontend
│   └── src/
│       ├── App.tsx                   # WS client, Recharts line chart + data table
│       └── main.tsx                  # React entry
├── R/                                # R streaming analytics client
│   ├── stream_client.R               # WS client, CSV logger, estimator harness
│   ├── ring_buffer.R                 # O(1) circular buffer (matrix-backed)
│   └── online_npcorr.R              # Port of onlineNPCORR (cutpoint-matrix Spearman/Kendall)
├── external/
│   └── uWebSockets/                  # Git submodule (HTTP + WS server)
├── config.json                       # Server port, WS URLs
├── Makefile                          # Builds C++ server binary
└── run                               # Launch script
```

## Key Functions

| Function / Class | File | Role |
|---|---|---|
| `main()` | `src/app/main.cpp:67` | Boots config, wires producer→bus→WS, starts HTTP server |
| `Producer::run(sink)` | `src/streamgenerator/runtime/producer.hpp:20` | Infinite loop: pull next event from source, apply delay, call sink |
| `EventBus::publish()` | `src/streamgenerator/runtime/event_bus.hpp:43` | Thread-safe fan-out to all matching subscribers |
| `EventBus::subscribe()` | `src/streamgenerator/runtime/event_bus.hpp:27` | Register a handler for an EventType (or `All`) |
| `SyntheticSource::next()` | `src/streamgenerator/datasource/synthetic_source.hpp:12` | Generates correlated price/volume ticks via Cholesky-decomposed multivariate normal |
| `FileSource::next()` | `src/streamgenerator/datasource/file_source.hpp:12` | Reads next CSV row as a MarketEvent |
| `Serializer::to_json()` | `src/streamgenerator/protocol/serializer.hpp:9` | Converts MarketEvent to JSON string for WS transmission |
| `fixed_delay()` / `poisson_delay()` / `burst_delay()` | `src/streamgenerator/timing/delay_strategies.hpp` | Pluggable inter-event sleep strategies |
| `Consumer::on_event()` | `src/device/consumer.hpp:41` | Ingests event into ring buffer, drains with lag policy |
| `Consumer::handle_policy()` | `src/device/consumer.hpp:90` | Applies ProcessAll / Snapshot / SkipToLatest cursor reconciliation |
| `RingBuffer::push()` / `get()` | `src/device/ring_buffer.hpp:16` | Mutex-guarded write, lock-free read via atomic head |
| `StateManager::update()` | `src/device/state_manager.hpp:33` | Folds event into incremental running stats |
| `StateManager::rolling_correlation()` | `src/device/state_manager.hpp:54` | Pearson correlation over a sliding window from the ring buffer |
| `RunningStats::add()` | `src/device/running_stats.hpp:11` | O(1) Welford online mean/variance update |
| `App()` (React) | `client/src/App.tsx:24` | WS client, maintains last 300 ticks, renders Recharts chart + table |

## Data Flow

```
┌──────────────┐
│  DataSource  │  SyntheticSource (Eigen/Cholesky)
│  (abstract)  │  FileSource (CSV replay)
└──────┬───────┘
       │ .next() → MarketEvent
       ▼
┌──────────────┐
│   Producer   │  Pulls events in a loop, applies DelayStrategy
└──────┬───────┘
       │ sink(event)
       ▼
┌──────────────────────────────────────────────────┐
│                    main()                        │
│                                                  │
│  1. bus->publish(event)     ──► EventBus fan-out │
│  2. app.publish("ticks")   ──► uWS broadcast     │
└──────┬──────────────────────────────┬────────────┘
       │                              │
       ▼                              ▼
┌──────────────┐              ┌──────────────┐
│   Consumer   │ (per-client) │  WebSocket   │
│  RingBuffer  │              │  (raw JSON)  │
│  StateManager│              └──────┬───────┘
│  LagPolicy   │                     │
└──────────────┘                     ▼
                              ┌──────────────┐
                              │ React Client │
                              │ (App.tsx)    │
                              │ Recharts     │
                              └──────────────┘
```

**Two parallel delivery paths exist:**

1. **EventBus path** — each Consumer subscribes a handler; on publish the bus pushes into that consumer's RingBuffer. The consumer drains with its lag policy (ProcessAll / Snapshot / SkipToLatest) and folds events into RunningStats via StateManager. This path supports server-side analytics.

2. **uWS broadcast path** — `app.publish("ticks", json)` pushes serialized JSON to all WS subscribers. The React client receives these, appends to a 300-tick sliding window, and renders a live line chart.

## Threading Model

| Thread | Responsibility |
|---|---|
| **Producer thread** (`std::thread`) | Calls `producer.run()`, publishes to EventBus + uWS |
| **uWS event loop** (main thread) | HTTP serving, WS connect/disconnect, subscribe/unsubscribe |

The EventBus mutex serializes cross-thread access. RingBuffer uses an atomic head for lock-free reads by consumers.

## Dependencies

- **uWebSockets** — HTTP + WebSocket server (git submodule)
- **Eigen** — Cholesky decomposition for correlated synthetic data
- **nlohmann/json** — JSON serialization
- **React + Vite + Recharts** — Frontend charting
