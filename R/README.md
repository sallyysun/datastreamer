# R Streaming Analytics

Online Spearman estimators compared against the grade correlation (the Pearson
product-moment correlation of the marginal CDFs, `Cor(F(X), G(Y))`).

## Do I need the C++ server?

**No** — for unit tests and simulation. Everything generates data in pure R:

```bash
Rscript R/test_npcorr.R              # validate the onlineNPCORR port
Rscript R/test_hermiter.R            # validate the hermiter estimator
Rscript R/simulate.R                 # full simulation with live chart + CSV
Rscript R/mae_simulation_stationary.R  # Monte Carlo MAE study (Stephanou & Varughese 2021, §6.1)
```

**Yes** — only for `stream_client.R`, which connects to the C++ server's WebSocket.

## Install dependencies

```r
install.packages(c("websocket", "jsonlite", "hermiter", "later", "httpuv",
                    "tidyverse", "future", "furrr"))
```

`tidyverse` and `future`/`furrr` are only needed for `mae_simulation_stationary.R`.

## Files

| File | Purpose | Needs server? |
|---|---|---|
| `simulate.R` | Full simulation: terminal dashboard + live browser chart + CSV logging | No |
| `mae_simulation_stationary.R` | Monte Carlo MAE study replicating Stephanou & Varughese (2021) §6.1: Hermite vs. count-matrix, MAE against the batch Spearman statistic (+ grade correlation diagnostic) | No |
| `test_npcorr.R` | Unit tests for onlineNPCORR port (Pearson + Spearman, stationary + windowed) | No |
| `test_hermiter.R` | Unit tests for hermiter (Spearman, density, CDF, regime detection, merge) | No |
| `stream_client.R` | Live WS client against the C++ server | Yes |
| `online_npcorr.R` | Port of [wxiao0421/onlineNPCORR](https://github.com/wxiao0421/onlineNPCORR) to R | — |
| `ring_buffer.R` | Fixed-capacity circular buffer (matrix-backed, O(1) push) | — |
| `mae_results/` | Output CSVs from `mae_simulation_stationary.R`, one per replicated paper table | — |
| `katana/` | PBS job scripts + setup notes for running the MAE study on UNSW's Katana HPC cluster (see `R/katana/README.md`) | — |

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

## MAE study (`mae_simulation_stationary.R`)

Replicates the stationary simulation study in Stephanou & Varughese (2021), §6.1:
many independent Monte Carlo replications at fixed sample sizes, comparing the
Hermite series estimator against the count-matrix estimator via mean absolute
error (MAE). One function call runs the whole study:

```r
source("R/mae_simulation_stationary.R")
run_mae_study(n_values = 1000L, m = 1000L)
```

Or from the CLI with the same quick-draft defaults:

```bash
Rscript R/mae_simulation_stationary.R
Rscript R/mae_simulation_stationary.R --n 10000,50000,100000 --m 1000   # paper scale

# options
#   --n <int,...>              sample size(s), comma-separated (default 1000)
#   --m <int>                  Monte Carlo replications per (n, rho) (default 1000)
#   --rho <float,...>          correlation value(s) (default -0.75,-0.5,-0.25,0.25,0.5,0.75)
#   --hermiter_N <int>         Hermite series order (default 20)
#   --cutpoints <int,...>      count-matrix cut-point count(s) (default 20,30)
#   --cutpoint_method <str>    "theoretical" or "sample" (default theoretical)
#   --transform <str,...>      "identity", "exp", or both comma-separated (default identity,exp)
#   --seed <int>               RNG seed (default 42)
#   --parallel <TRUE|FALSE>    use future/furrr multisession (default TRUE)
```

Two error metrics are tracked per replicate:

- **`error2batch`** — absolute error against `cor(x, y, method = "spearman")`,
  the finite-sample batch Spearman statistic. This is what the paper itself
  calls the "exact Spearman correlation coefficient for the sample" (§6.1,
  steps 3–4) and is what its MAE tables are actually computed against.
- **`error2grade`** — absolute error against the population grade correlation
  (see below). Not part of the paper's own metric, but tracked as an extra
  diagnostic since it's arguably the more meaningful "true" target for large `n`.

Replications run in parallel via `future`/`furrr` (see the comments around
`future::plan()` in the script for what to change when moving from a laptop to
Katana — or see `R/katana/` for ready-made PBS job scripts and setup notes).
Only summary statistics (mean/SD) are written, one CSV per replicated paper
table, to `R/mae_results/`. Every filename carries a `run_tag` suffix (default:
derived from `--n`/`--m`) so separate runs — a local smoke test, a full Katana
run, or several split-by-`n` array tasks — never overwrite each other:

```
mae_table1_normal_c20_<run_tag>.csv      # identity transform, Hermite N=20 vs Matrix c=20
mae_table2_normal_c30_<run_tag>.csv      # identity transform, Hermite N=20 vs Matrix c=30
mae_table3_lognormal_c20_<run_tag>.csv   # exp transform,      Hermite N=20 vs Matrix c=20
mae_table4_lognormal_c30_<run_tag>.csv   # exp transform,      Hermite N=20 vs Matrix c=30
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
