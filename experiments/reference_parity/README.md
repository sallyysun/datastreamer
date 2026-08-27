# reference_parity

Validates the C++ online dependence estimators against the upstream reference
implementations they were ported from, on **byte-identical shared inputs**.

| C++ estimator (`src/estimators/dependence/`) | Reference (`external/`) | Language |
|---|---|---|
| `XiaoSpearmanEstimator` (empirical-copula Spearman) | `onlineNPCORR` — `spearmanr_by_matrix`, `online_corr`, `online_mv_corr` | Python |
| `HermiteEstimator` (Hermite-series Spearman) | `hermiter` — `update_sequential` + `spearmans` | R (installed pkg) |

The idea: feed the exact same `(x, y)` stream to the C++ estimator class and to
the reference, one observation at a time, and compare the estimated Spearman rho
at **every step**. If the ports are faithful, the two series agree to numerical
precision.

## Run

```bash
experiments/reference_parity/run.sh
```

Requires `clang++`, Eigen, `python3` (numpy), and `Rscript` with the `hermiter`
package installed. Initialize the reference submodules first if needed:

```bash
git submodule update --init external/onlineNPCORR external/hermiter
```

## What it does

- `gen_inputs.py` — writes shared datasets to `data/` at full float precision
  (Gaussian copula at rho ∈ {0.7, −0.4, 0.0}, plus a skew-marginal case).
- `harness.cpp` — drives the real C++ estimator classes and prints per-step
  estimates. For Xiao it also dumps the auto-calibrated cutpoints so the Python
  reference bins on the *identical* thresholds (isolating the estimator
  arithmetic from cutpoint choice). `#define private public` is a test-only hack
  to read those cutpoints; the shipped headers are untouched.
- `xiao_reference.py` — imports `spearmanr_by_matrix` **verbatim** from
  `onlineNPCORR` and runs the cumulative / sliding-window count-grid update.
- `hermite_reference.R` — runs the installed `hermiter` package.
- `compare.py` — reports per-step max / mean absolute difference.

Four shipped configs are covered: Xiao cumulative, Xiao window=500 (vs
`online_mv_corr`), Hermite cumulative, Hermite exp-weighted λ=0.01 (vs
`hermiter` `exp_weight_lambda`).

## Result (2026-08-26)

| estimator | config | reference | max abs diff |
|---|---|---|---|
| Xiao    | cumulative   | `online_corr`     | ~1e-16 (machine ε) |
| Xiao    | window = 500 | `online_mv_corr`  | ~1e-16 (machine ε) |
| Hermite | cumulative   | `hermiter`        | ~4e-11 |
| Hermite | ew λ = 0.01  | `hermiter` EW     | ~2e-10 |

**Xiao is bit-exact** to the reference formula (differences at the last bit of a
double). **Hermite agrees to ~1e-10**, the expected floating-point accumulation
gap between different linear-algebra backends (Eigen vs R/BLAS) and between the
closed-form coupling matrix `W` built in the C++ port and `hermiter`'s
precomputed serialized `W`. Both confirm algorithmic equivalence.

### Notes on the comparison alignment

- The C++ Xiao calibrates cutpoints from its first `calibration` (=100) samples,
  then bins all history with them; the Python reference is fed those same
  cutpoints and scored from step `calibration−1` on, where both grids hold the
  same points.
- For window=500 with calibration=100, the C++ window fills cumulatively to 500
  then slides; it becomes identical to the pure-500 reference from index 499 on,
  which is where scoring starts.
- The C++ Hermite is run with `burn_in=1` here so it emits from `n≥2`, matching
  `hermiter`'s live range; `spearmans()` is compared unclipped (the C++ clamp to
  [−1,1] is a no-op for the interior values seen here).
