# R Streaming Analytics

Online Spearman estimators compared against the grade correlation (the Pearson
product-moment correlation of the marginal CDFs, `Cor(F(X), G(Y))`).

## Do I need the C++ server?

**No** — for unit tests and simulation. Everything generates data in pure R:

```bash
Rscript R/test_npcorr.R      # validate the onlineNPCORR port
Rscript R/test_hermiter.R    # validate the hermiter estimator
Rscript R/simulate.R         # full simulation with live chart + CSV
```

**Yes** — only for `stream_client.R`, which connects to the C++ server's WebSocket.

## Install dependencies

```r
install.packages(c("websocket", "jsonlite", "hermiter", "later", "httpuv"))
```

## Files

| File | Purpose | Needs server? |
|---|---|---|
| `simulate.R` | Full simulation: terminal dashboard + live browser chart + CSV logging | No |
| `test_npcorr.R` | Unit tests for onlineNPCORR port (Pearson + Spearman, stationary + windowed) | No |
| `test_hermiter.R` | Unit tests for hermiter (Spearman, density, CDF, regime detection, merge) | No |
| `stream_client.R` | Live WS client against the C++ server | Yes |
| `online_npcorr.R` | Port of [wxiao0421/onlineNPCORR](https://github.com/wxiao0421/onlineNPCORR) to R | — |
| `ring_buffer.R` | Fixed-capacity circular buffer (matrix-backed, O(1) push) | — |

## Simulation (`simulate.R`)

Generates correlated bivariate normal data one observation at a time and feeds
to all estimators simultaneously. Three output channels run in parallel:

### 1. Terminal dashboard (ANSI overwrite)

```
── Observation 500 / 2000  (rho = 0.60) ─────────────
                              Spearman
                              ────────
  grade corr Cor(F(X),G(Y))   0.586951
  npcorr (stationary)         0.584510
  npcorr (window=100)         0.590120
  hermiter (stationary)       0.585600
  hermiter (exp-weighted)     0.589000
```

### 2. Live browser chart (`httpuv` + Chart.js)

Opens at `http://localhost:9090`. Streams every observation's Spearman estimates
over WebSocket to a line chart showing convergence against the grade correlation.

### 3. CSV log (`simulated-data/`)

Every observation is written to `simulated-data/sim_<mode>_<timestamp>.csv`:

```
obs,rho,grade_corr,npcorr_spearman,npcorr_spearman_win,hermiter_spearman,hermiter_spearman_ew
```

### Usage

```bash
# stationary (constant rho)
Rscript R/simulate.R                          # defaults: rho=0.6, N=2000
Rscript R/simulate.R --N 5000 --rho 0.8       # custom

# non-stationary (regime-switching: 0.7 → -0.3 → 0.9 → -0.6 → 0.2)
Rscript R/simulate.R --nonstationary

# options
#   --N <int>          number of observations (default 2000)
#   --rho <float>      correlation for stationary mode (default 0.6)
#   --window <int>     sliding window size (default 100)
#   --port <int>       chart server port (default 9090)
#   --delay <ms>       pause between observations (default 10ms)
#   --report <int>     terminal update frequency (default every 10 obs)
```

## Grade correlation

The theoretical target is the **grade correlation**: `Cor(F(X), G(Y))`, the
Pearson product-moment correlation of the two marginal CDFs.

For bivariate normal with Pearson correlation `rho`:

```
grade_corr = (6/π) × arcsin(ρ/2)
```

## What's in online_npcorr.R

Port of Xiao's Python package (Spearman + Pearson only):

| Python original | R port | Description |
|---|---|---|
| `online_corr` (pearsonr) | `new_online_pearson` / `pearson_update` / `pearson_value` | O(1) running Pearson via sufficient stats |
| `online_corr` (spearmanr) | `new_online_npcorr` / `npcorr_update` / `npcorr_value` | Cutpoint-matrix Spearman |
| `online_mv_corr` | `npcorr_remove` / `pearson_remove` | Sliding window by subtracting oldest obs |
| `get_index` | `get_bin_index` | Binary search bin assignment |
| `spearmanr_by_matrix` | `spearmanr_from_matrix` | Spearman from contingency matrix |
