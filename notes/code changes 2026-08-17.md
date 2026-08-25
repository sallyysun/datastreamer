# Code changes — 2026-08-17

Branch: `feat/online-nonparametric-dependence` (pushed to origin; PR not yet opened).

Goal for the session: get the **nonparametric** dependence estimators (Xiao empirical-copula
Spearman, Hermite series) working end to end, evaluate them offline against a known
time-varying ground truth, then wire them into the **live** streaming server so estimates
flow to the browser over the WebSocket (TCP).

---

## 1. Fixed / completed the nonparametric estimators

**`src/estimators/dependence/estimator.hpp`**
- Added `reset()` to the `DependenceEstimator` interface (for change-detection restart later).
- Documented that `estimate()` returns Spearman's rho for all estimators, so everything is
  compared in the same space.

**`src/estimators/dependence/xiao_spearman.hpp`** — rewritten (was truncated / didn't compile)
- Scale-invariant Spearman computed over a **double-weighted count grid**, so the same
  `estimate()` formula works for three memory regimes: cumulative, hard sliding window, and
  **exponential forgetting** (`counts *= lambda` each step).
- Fixed a cutpoint out-of-bounds bug; added `reset()`.

**`src/estimators/dependence/hermiter.hpp`** — fixed real bugs
- `estimate()` previously computed `12*u·Av - 12*u·Av` → **always ~0**. Corrected to
  `12(u - z/2)^T A (v - z/2)` (Spearman via centred marginal-CDF projections).
- Fixed Python-style `clamp(val:..., lo:...)` syntax.
- Fixed two `integral_matrix` bugs (`gaussian` undefined; `exp(2.0)` where `sqrt(2.0)` was meant).
- Added `reset()`.

**`src/estimators/dependence/windowed_rank.hpp`** — compile fixes
- `state_bytes` → `store_bytes` override, declared the tie counters, `std:size_t` typo, added `reset()`.
- Kept as a rank-correlation baseline.

**`src/estimators/registry.hpp`** — filled in (was empty)
- String → estimator factory (`make_estimator`). Documents the differing forgetting
  conventions: Xiao `lambda` = retention (larger = longer memory); Hermite `lambda` = EWMA
  smoothing (larger = faster adaptation; 0 = cumulative).

---

## 2. Dynamic DGP + offline evaluation harness

**`src/streamgenerator/datasource/schedule.hpp`** (moved here from `src/experiments/`)
- `Schedule` abstraction producing the true rho_t per step. Implementations:
  `ConstantSchedule`, `DriftSchedule`, `JumpSchedule`, `StochasticSchedule`
  (`z_t = z_{t-1}+omega`, `rho = tanh z`), and **`OscillatingSchedule`** (perpetual sine,
  added later for the live demo).
- Relocated because it is a data-generation primitive shared by the offline and live sources.

**`src/experiments/copula_source.hpp`**
- Bivariate Gaussian-copula source emitting `(x, y, true rho, true rho_S)`. Uses
  `rho_S = (6/pi)·asin(rho/2)` for the ground truth.

**`src/experiments/main.cpp`**
- Runs a chosen scenario against Xiao / Hermite / windowed-rank online, writes
  `experiment_<scenario>.csv` (estimate vs truth per step), and prints RMSE + memory per estimator.
- Scenarios: `constant | drift | jump | stochastic | oscillate`.

**`src/experiments/plot.py`** — optional quick single-scenario matplotlib plot.

**`Makefile`**
- Added `experiment` target (Eigen-only; independent of the uWS server build).

Validation (offline): cumulative estimators win under constant rho (efficiency); forgetting
variants win under drift/jump (adaptation); Xiao and Hermite agree closely (independent
implementations cross-checking).

---

## 3. Wired estimators into the live server (online, over TCP/WebSocket)

**`src/device/dependence_monitor.hpp`** (new)
- Server-side holder of the estimators, updated **once per event on the producer thread**
  (single-threaded, no locking). Live-tuned configs (short calibration / burn-in) so estimates
  appear in seconds. `snapshot()` returns name → current rho_S.

**`src/streamgenerator/protocol/serializer.{hpp,cpp}`**
- Added an overload attaching a `"dependence"` object (name → rho_S) to the tick JSON, and later
  a `"true_rho_s"` field. Refactored the base payload into a shared `base_json()` helper.

**`src/app/main.cpp`**
- Constructs a `DependenceMonitor`, updates it per event, and broadcasts estimates in the tick JSON.
- Raised tick rate `fixed_delay(0.5)` → `fixed_delay(10.0)` so estimators warm up in seconds,
  not ~10 minutes.

**`client/src/App.tsx`**
- Parses `dependence` (and `true_rho_s`), renders a live Spearman-rho chart + numeric readout,
  with the **true rho_S drawn as a bold reference line** so you can watch estimators track it.

---

## 4. Time-varying dependence in the LIVE stream (so the demo actually moves)

**`src/streamgenerator/datasource/synthetic_source.{hpp,cpp}`** — refactored
- Now **Schedule-driven**: the two return series are correlated at the schedule's rho_t
  (Gaussian copula) each tick, replacing the old fixed 4×4 Cholesky. Defaults to
  `ConstantSchedule(0.6)` to preserve prior behaviour.
- Emits the true rho_S for each event.

**`src/streamgenerator/event/market_event.hpp`**
- Added `double true_rho_s = 0.0;` as the **last** member (set by assignment, never positionally)
  so existing 7-field aggregate-inits in `synthetic_source.cpp` / `file_source.cpp` keep working.

**`src/app/main.cpp`**
- Drives the live source with `OscillatingSchedule(0.6, 400.0, 0.0)` — rho_S sweeps ±0.6 with a
  ~400-event (~40 s) period, so the estimators visibly chase a moving target.

---

## 5. Analysis notebook

**`src/experiments/analysis.ipynb`** (new)
- Loads `experiment_<scenario>.csv`, plots each estimator tracking the true rho_S (burn-in marked),
  and tabulates RMSE after burn-in as a heat-mapped table.
- Path resolution walks up from the kernel's cwd to find the repo root (folder with `Makefile` +
  `src/`), so it works regardless of where Jupyter launched; prints which CSVs it found.

---

## 6. Build hygiene

**`.gitignore`** + removed tracked `server` binary
- Ignore `/server`, `/experiment`, `/experiment_*.csv`; stopped tracking the committed binary so
  rebuilds don't produce noisy diffs.

---

## 7. Online testing — live accuracy probe

**`src/experiments/live_probe.py`** (new)
- Connects to the running server over its WebSocket (TCP), reads the `dependence` estimates and
  `true_rho_s` from each tick, and prints rolling per-estimator metrics in real time:
  instantaneous error, windowed **RMSE** and **bias**, and estimation **lag** (ticks each
  estimator trails the moving truth, via argmax cross-correlation).
- Tests the whole online path (server-side monitor + serialization + real-time timing), not just
  the math. Only dependency is `websocket-client` (metrics are pure Python; no numpy/pandas).
- Needs: `.venv/bin/pip install websocket-client` (already present, 1.9.0); the server running
  from this branch (so it emits `dependence` + `true_rho_s`); and a moving schedule
  (OscillatingSchedule) for `lag` to be meaningful. The server auto-subscribes each connection to
  the `"ticks"` topic, so the probe sends nothing — it just connects and reads.
- Run: `.venv/bin/python src/experiments/live_probe.py [--url ... --window 200 --interval 20 --csv live.csv]`.
  `--csv` logs `tick,true_rho_s,<estimators>` (same shape as the offline CSVs, loadable in the notebook).

**`src/experiments/plot_live.py`** (new) + `analysis.ipynb` "Live probe capture" cell
- Plot a `live.csv` capture (handles the `tick` index and reuses the estimator colours):
  `python3 src/experiments/plot_live.py live.csv` (writes `live.png`), or the notebook's
  `plot_live('live.csv')` cell. `estimator_cols()` now also ignores `tick` so it serves both
  offline and live CSVs.

### Live capture — first results (`live.csv`, ~1.6k ticks, OscillatingSchedule)
Note: the live `DependenceMonitor` runs four estimators (`xiao`, `xiao_ew`, `hermite`,
`hermite_ew`) — no `windowed_rank`. True rho_S swept +-0.58 (the sine). After warm-up:

| estimator | RMSE | corr w/ truth | own movement (std) | lag |
|---|---|---|---|---|
| `hermite` (cumulative)  | 0.42 | -0.09 | 0.04 | n/a |
| `xiao` (cumulative)     | 0.42 | -0.10 | 0.04 | n/a |
| `hermite_ew` (forgetting) | 0.35 | +0.57 | 0.26 | ~58 |
| `xiao_ew` (forgetting)    | 0.35 | +0.53 | 0.24 | ~60 |

- **Cumulative estimators don't track** a zero-mean oscillation: they sit ~flat near 0 (movement
  std ~0.04, corr ~0). Correct behaviour — they pool all history.
- **Forgetting estimators track but lag + attenuate**: corr ~0.55, but trail truth by ~60 ticks
  (~6 s) and under-shoot amplitude (0.25 vs 0.41). Cause: their effective window (ESS ~100-200)
  is a large fraction of the 400-tick oscillation period -> low-pass phase lag + amplitude loss.
  Bias is tiny (+-0.05), so error is timing, not offset.
- Xiao and Hermite agree tightly within each regime -> the two implementations are consistent.
- Live UI colours (for reference): price panel green `#00ff88` = `price_x`, blue = `price_y`;
  dependence panel white = true rho_S, amber = `xiao`, red = `xiao_ew`, cyan = `hermite`,
  violet = `hermite_ew`.
- To tighten live tracking: raise the schedule period (slower sine) or forget more aggressively
  (`xiao_ew` lambda 0.99->0.95, `hermite_ew` lambda 0.01->0.03), trading lag for variance.

---

## How to run (nothing here was auto-run)

```bash
# offline experiments -> CSVs (from repo root)
make experiment
for s in constant drift jump stochastic oscillate; do ./experiment $s; done

# analysis (use the project venv; it currently lacks pandas/matplotlib/numpy/jupyterlab)
.venv/bin/pip install pandas numpy matplotlib jupyterlab
.venv/bin/jupyter lab src/experiments/analysis.ipynb

# live demo
make server && ./server     # then open the client
```

Note: the offline `experiment` binary has **no inter-event delay** (unlike the live server's
`fixed_delay`), which is why the runs finish in milliseconds.

---

## Commits on the branch (oldest → newest)
1. `chore: ignore build artifacts and stop tracking server binary`
2. `docs: add nonparametric dependence-estimation roadmap`
3. `fix: make nonparametric dependence estimators compile and correct`
4. `feat: add dynamic copula DGP and offline evaluation harness`
5. `feat: stream live dependence estimates to clients over the socket`
6. `refactor: move Schedule into the datasource layer; add OscillatingSchedule`
7. `feat: time-varying dependence in the live server with truth overlay`
8. `docs: add experiment analysis Jupyter notebook`
9. `feat: add live online-test probe + notebook path fix + session notes`
10. `chore: ignore __pycache__`
11. `feat: add live-capture plotting + first live-result analysis`

## Still missing (next steps)
- Parametric online MLE / Fisher scoring branch; CUSUM change-detection + restart (the `reset()`
  hook is already in place); Bayesian / particle-filter branches.
- Multi-replication averaging + recovery-time / detection-delay metrics (current numbers are single-run).
- `requirements.txt` for the Python analysis env.
