# datastreamer

Honours thesis system: a **streaming** (single-pass, O(1)/tick, bounded-memory) estimator of
time-varying copula dependence, plus the TCP/WebSocket pipeline it runs live on.
Central question: *how should an online estimator trade statistical efficiency against
adaptation to changing dependence?*

## Layout

| Path | What |
| --- | --- |
| `src/streamgenerator/` | **server** — synthetic DGP + producer + raw-TCP `Broker`. Publishes events; does **not** estimate. |
| `src/consumer/` | **consumer** — TCP client, ring buffer, device state. Runs the estimators. |
| `src/estimators/` | The estimators + `registry.hpp`. The thesis's actual subject. |
| `src/protocol/` | `SymbolTick`, serializer, topic matching. |
| `client/` | React/Vite browser viewer. **Strict viewer** — renders what it receives, computes nothing. |
| `experiments/` | Offline harness (own Makefile, deliberately not built by the root one) + `plot.py`, `live_probe.py`, `analysis.ipynb`, `results/`. |
| `notes/` | See below. |
| `external/` | Reference implementations: `hermiter` (R), `onlineNPCORR` (Python), `uWebSockets`. |

## Build & run

```bash
make            # -> ./server   (needs uWebSockets, openssl, eigen, nlohmann-json via brew)
make consumer   # -> ./consumer (eigen + nlohmann only)
./run.sh dev    # server + vite client together
```

Server listens on **9002** (HTTP/WebSocket, browser) and **9003** (raw TCP broker, C++ consumers).
Live DGP regime is chosen by env var: `DEPENDENCE=constant|drift|jump|stochastic|oscillate`
(default `oscillate`); symbol pair via `SYMBOLS=X,Y` (default `AAPL,MSFT`).

Consumer flags: `--host --port --symbols --policy all|snapshot|latest --print --rate`.
Leave `--rate 0` for estimation — throttling can desync the two legs of the pair.

Offline harness runs from **inside** `experiments/`:

```bash
cd experiments && make && ./experiment jump          # single seed, per-tick CSV
make multiseed && ./multiseed jump 20                # 20 seeds, mean±SE
./run_multiseed_grid.sh 20 42                        # every scenario × every family (~50 min)
```

Scenarios: `constant drift jump stochastic oscillate`. DGP copula family via
`COPULA=gaussian|clayton|gumbel|t` (default `gaussian`) — non-Gaussian families get a filename
suffix, so `results/experiment_<scenario>[_<family>].csv` and likewise `multiseed_…`.

**No number is a thesis result until it comes from `./multiseed`, not `./experiment`.** Both read
`all_estimator_keys()` from `registry.hpp`, so registering an estimator there is enough to get it
measured — a key that is registered but not in that list is silently never run, which is how
`xiao_win` went unmeasured until 2026-09.

## The estimator contract

`DependenceEstimator` (`src/estimators/dependence/estimator.hpp`) is five methods:
`update(x,y)`, `estimate()`, `reset()`, `store_bytes()`, `name()`.

- **`estimate()` always returns Spearman's ρ**, so parametric and nonparametric estimators are
  comparable in one space. Never return a native parameter.
- **`reset()` is the change-detection hook** — a detector wrapper calls it to restart.
- **`store_bytes()` is a thesis metric**, not diagnostics. Streaming-admissibility is a claim we
  measure, so keep it honest.
- New estimators are registered by string key in `src/estimators/registry.hpp` and added to
  `all_estimator_keys()` in the same file — both harnesses read that one list.

⚠️ **The forgetting parameter is a mirror-image knob across estimators** — for Xiao (and the
parametric copula MLEs, which follow Xiao's convention), larger λ = longer memory; for Hermite,
smaller λ = longer memory. Read the comment block at the top of `registry.hpp` before touching any
λ. Getting this backwards silently inverts an experiment.

⚠️ `registry.hpp` configs must stay **identical** to the live consumer's (`src/consumer/`), or
offline and online RMSE stop being comparable.

## Conventions

- Harness is **single-seed (42)**. No number is a thesis result until multi-seed with error bars —
  see `notes/todo.md`.
- Offline and live `oscillate` share period 400 deliberately, so their RMSE is directly comparable.
- Marginals are currently handled by standardization (exact only for Gaussian); rank-PIT is open.
- The browser must stay a strict viewer — if a calculation is creeping into `client/`, that's a bug.

## Where the thinking lives

Read the note, don't re-derive it.

- `notes/studies/roadmap.md` — thesis structure, the efficiency↔adaptation framing, the "ultimate
  streaming estimator" options, what makes a *copula* changepoint different (five distinctions,
  referenced by number elsewhere).
- `notes/studies/change_detection.md` — the detector design space (3 axes), the SSM view, build
  order. **Two identities worth knowing before touching an estimator:** the fixed-λ estimators are
  the steady-state limit of a Kalman filter (λ = 1−K), and CUSUM/SR/BOCPD/HMM are one recursion
  over a latent chain. Published as an artifact.
- `notes/studies/oracle_memory.md` — the thesis deliverable: an **online** evidence→memory rule for
  tail dependence. Known-changepoint soft restart ($\gamma^*=1/(1+Z^2)$) is the oracle it
  approximates, not the endpoint.
- `notes/studies/application_2.md` — covariance / hedging / vine application track.
- `notes/todo.md` — live backlog with ✅/⬜/➖ status. Check before proposing work.
- `notes/architecture.md`, `notes/code_changes/` — system design and dated change logs.

## Working style

- This is research code with a thesis attached: correctness of the *statistics* matters more than
  code elegance, and a claim in a note is not a result until the harness produces it.
- Keep source nearly comment-free: a short file header pointing at the note that explains it, plus
  comments only for a non-obvious *why* (guards, clamps, units, traps). Derivations and rationale go
  in `notes/`, not in the `.hpp`.
