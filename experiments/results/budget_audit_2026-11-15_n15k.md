# Section 6.2 (E1) memory-constraint audit, 2026-11-15 (extended RMPL grid k = 8-8192; the k <= 1024 version is budget_audit_2026-11-15_k1024.md)

Platform of the runs: Linux x86_64, libstdc++, g++ -O2 (budget_gainext_2026-11-11.txt header). No new simulation was run for this audit.

## 1. Design verification

- Test seeds per configuration: 100–100, range 42–141 (per-seed file holds test seeds only; validation enters only through the summary rows, split = val).
- Families: ['clayton', 'gaussian', 'gumbel', 't']; scenarios: ['constant', 'drift', 'jump']; estimators in the candidate set: ['hermite', 'mpl_dskll', 'mpl_exact', 'xiao'].
- Gain grid, constant: cum, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1
- Gain grid, drift: cum, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1
- Gain grid, jump: 0.005, 0.01, 0.02, 0.05, 0.1
- Approximation grid, Counting Matrix (bins B): [10, 20, 30, 45, 65, 90, 130, 180]
- Approximation grid, Hermite Series (order N): [5, 10, 15, 20, 30, 43, 60, 89, 128]
- Approximation grid, RMPL (KLL k): [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
- Validation seeds 1000–1039 (n_val = 40) and test seeds 42–141 (n_test = 100): from the probe_budget sweep invocation (`sweep 100 40`, kValSeed = 1000, kTestSeed = 42) recorded in the run header.

## 2. Selection uses validation only

- `pick()` (identical to analyze_budget_gainext.py and plot_budget.py) filters split = val rows and takes the minimum validation RMSE among candidates whose validation peak bytes ≤ cap. Test rows enter only after selection. Confirmed by reading the code; no test quantity feeds a configuration choice.
- All 108 selected configurations also fit their cap on the test seeds: max test peak / cap = 0.940; test peak = validation peak in 108/108 (RMPL's DataSketches compaction is randomised but its peak is deterministic here).
- Exception to 'validation only': the *comparator* in 'RMPL − better of CM/HS' is chosen on test means (draft §6.2 says so). Section 5 reports the validation-chosen comparator as well.

## 3. Test Spearman RMSE (×10⁻², mean ± SE over 100 test seeds) at the validated configuration

Cell entries: RMSE ± SE [approx, gain]. Approx = KLL k (RMPL), bins B (CM), order N (HS).

| Scenario | Family | Budget | RMPL | Counting Matrix | Hermite Series |
|---|---|---|---|---|---|
| constant | Gaussian | 16 | 1.13 ± 0.06 [512, cum] | 1.03 ± 0.05 [30, cum] | 1.07 ± 0.04 [20, cum] |
| constant | Gaussian | 64 | 1.13 ± 0.06 [2048, cum] | 1.03 ± 0.05 [30, cum] | 1.03 ± 0.04 [60, cum] |
| constant | Gaussian | 256 | 1.13 ± 0.06 [16384, cum] | 1.03 ± 0.05 [30, cum] | 1.03 ± 0.04 [60, cum] |
| constant | t4 | 16 | 1.26 ± 0.05 [256, cum] | 1.13 ± 0.05 [30, cum] | 1.21 ± 0.05 [10, cum] |
| constant | t4 | 64 | 1.26 ± 0.05 [256, cum] | 1.11 ± 0.05 [65, cum] | 1.21 ± 0.05 [10, cum] |
| constant | t4 | 256 | 1.26 ± 0.05 [256, cum] | 1.11 ± 0.05 [130, cum] | 1.21 ± 0.05 [10, cum] |
| constant | Clayton | 16 | 1.24 ± 0.06 [512, cum] | 1.05 ± 0.05 [30, cum] | 1.10 ± 0.05 [15, cum] |
| constant | Clayton | 64 | 1.24 ± 0.06 [2048, cum] | 1.05 ± 0.05 [30, cum] | 1.07 ± 0.05 [30, cum] |
| constant | Clayton | 256 | 1.24 ± 0.06 [16384, cum] | 1.05 ± 0.05 [30, cum] | 1.07 ± 0.05 [30, cum] |
| constant | Gumbel | 16 | 1.18 ± 0.05 [256, cum] | 1.01 ± 0.04 [30, cum] | 1.08 ± 0.04 [20, cum] |
| constant | Gumbel | 64 | 1.18 ± 0.05 [2048, cum] | 1.02 ± 0.04 [65, cum] | 1.02 ± 0.04 [60, cum] |
| constant | Gumbel | 256 | 1.18 ± 0.05 [16384, cum] | 1.02 ± 0.04 [90, cum] | 1.04 ± 0.04 [89, cum] |
| drift | Gaussian | 16 | 2.99 ± 0.05 [512, 0.002] | 3.29 ± 0.05 [30, 0.002] | 3.31 ± 0.05 [20, 0.002] |
| drift | Gaussian | 64 | 2.98 ± 0.05 [2048, 0.002] | 3.23 ± 0.05 [65, 0.002] | 3.23 ± 0.05 [60, 0.002] |
| drift | Gaussian | 256 | 2.98 ± 0.05 [16384, 0.002] | 3.22 ± 0.05 [130, 0.002] | 3.23 ± 0.05 [89, 0.002] |
| drift | t4 | 16 | 3.21 ± 0.05 [512, 0.002] | 3.53 ± 0.05 [30, 0.002] | 3.58 ± 0.05 [5, 0.002] |
| drift | t4 | 64 | 3.20 ± 0.05 [1024, 0.002] | 3.48 ± 0.05 [65, 0.002] | 3.48 ± 0.05 [60, 0.002] |
| drift | t4 | 256 | 3.20 ± 0.05 [1024, 0.002] | 3.47 ± 0.05 [130, 0.002] | 3.48 ± 0.05 [60, 0.002] |
| drift | Clayton | 16 | 3.24 ± 0.05 [512, 0.002] | 3.27 ± 0.05 [30, 0.002] | 3.27 ± 0.05 [20, 0.002] |
| drift | Clayton | 64 | 3.24 ± 0.05 [512, 0.002] | 3.23 ± 0.05 [65, 0.002] | 3.24 ± 0.05 [60, 0.002] |
| drift | Clayton | 256 | 3.24 ± 0.05 [512, 0.002] | 3.21 ± 0.05 [130, 0.002] | 3.24 ± 0.05 [60, 0.002] |
| drift | Gumbel | 16 | 2.70 ± 0.05 [512, 0.002] | 3.24 ± 0.05 [30, 0.002] | 3.22 ± 0.05 [20, 0.002] |
| drift | Gumbel | 64 | 2.68 ± 0.04 [2048, 0.002] | 3.18 ± 0.05 [65, 0.002] | 3.20 ± 0.05 [43, 0.002] |
| drift | Gumbel | 256 | 2.68 ± 0.04 [8192, 0.002] | 3.17 ± 0.05 [130, 0.002] | 3.20 ± 0.05 [43, 0.002] |
| jump | Gaussian | 16 | 5.95 ± 0.04 [512, 0.01] | 6.19 ± 0.04 [30, 0.005] | 6.29 ± 0.04 [20, 0.005] |
| jump | Gaussian | 64 | 5.95 ± 0.04 [2048, 0.01] | 6.17 ± 0.04 [65, 0.005] | 6.21 ± 0.04 [60, 0.005] |
| jump | Gaussian | 256 | 5.95 ± 0.04 [4096, 0.01] | 6.17 ± 0.04 [130, 0.005] | 6.20 ± 0.04 [89, 0.005] |
| jump | t4 | 16 | 6.48 ± 0.05 [512, 0.005] | 6.41 ± 0.04 [30, 0.005] | 6.51 ± 0.04 [20, 0.005] |
| jump | t4 | 64 | 6.48 ± 0.05 [1024, 0.005] | 6.40 ± 0.04 [65, 0.005] | 6.44 ± 0.04 [60, 0.005] |
| jump | t4 | 256 | 6.48 ± 0.05 [1024, 0.005] | 6.40 ± 0.04 [130, 0.005] | 6.44 ± 0.04 [60, 0.005] |
| jump | Clayton | 16 | 6.56 ± 0.04 [512, 0.01] | 6.25 ± 0.04 [30, 0.005] | 6.33 ± 0.04 [20, 0.005] |
| jump | Clayton | 64 | 6.56 ± 0.04 [512, 0.01] | 6.24 ± 0.04 [65, 0.005] | 6.28 ± 0.04 [60, 0.005] |
| jump | Clayton | 256 | 6.56 ± 0.04 [512, 0.01] | 6.24 ± 0.04 [130, 0.005] | 6.28 ± 0.04 [89, 0.005] |
| jump | Gumbel | 16 | 6.18 ± 0.05 [512, 0.005] | 6.28 ± 0.05 [30, 0.005] | 6.36 ± 0.04 [20, 0.005] |
| jump | Gumbel | 64 | 6.18 ± 0.05 [2048, 0.005] | 6.27 ± 0.05 [65, 0.005] | 6.30 ± 0.05 [60, 0.005] |
| jump | Gumbel | 256 | 6.18 ± 0.05 [8192, 0.005] | 6.27 ± 0.05 [130, 0.005] | 6.29 ± 0.05 [89, 0.005] |

## 4. Effect of storage (RMSE reduction ×10⁻², positive = more storage helps; paired over the 100 test seeds)

| Step | Estimator | cells with a different configuration | max reduction (cell) | median | mean | min (cell) | cells reduction > 2 SE | cells worse by > 2 SE |
|---|---|---|---|---|---|---|---|---|
| 16->64 | RMPL | 9/12 | +0.015 ± 0.004 (Gumbel drift) | +0.000 | +0.002 | -0.001 ± 0.001 (Gaussian jump) | 3/12 | 0/12 |
| 16->64 | Counting Matrix | 10/12 | +0.060 ± 0.004 (Gaussian drift) | +0.015 | +0.023 | -0.005 ± 0.007 (Gumbel constant) | 9/12 | 0/12 |
| 16->64 | Hermite Series | 11/12 | +0.102 ± 0.031 (t4 drift) | +0.054 | +0.049 | +0.000 ± 0.000 (t4 constant) | 7/12 | 0/12 |
| 64->256 | RMPL | 7/12 | +0.002 ± 0.001 (Clayton constant) | +0.000 | +0.000 | -0.000 ± 0.001 (Gumbel constant) | 0/12 | 0/12 |
| 64->256 | Counting Matrix | 10/12 | +0.014 ± 0.002 (Gaussian drift) | +0.001 | +0.004 | -0.001 ± 0.002 (t4 constant) | 4/12 | 0/12 |
| 64->256 | Hermite Series | 5/12 | +0.010 ± 0.008 (Gumbel jump) | +0.000 | +0.001 | -0.020 ± 0.015 (Gumbel constant) | 0/12 | 0/12 |
| 16->256 | RMPL | 9/12 | +0.016 ± 0.005 (Gumbel drift) | +0.001 | +0.003 | -0.001 ± 0.011 (Gumbel constant) | 3/12 | 0/12 |
| 16->256 | Counting Matrix | 10/12 | +0.074 ± 0.003 (Gaussian drift) | +0.016 | +0.028 | -0.005 ± 0.007 (Gumbel constant) | 9/12 | 0/12 |
| 16->256 | Hermite Series | 11/12 | +0.102 ± 0.031 (t4 drift) | +0.047 | +0.050 | +0.000 ± 0.000 (t4 constant) | 6/12 | 0/12 |

All estimators, 16->64: max +0.102, median +0.012, mean +0.025, median over cells where the configuration changed +0.020 (30/36 changed).

All estimators, 64->256: max +0.014, median +0.000, mean +0.002, median over cells where the configuration changed +0.001 (22/36 changed).

All estimators, 16->256: max +0.102, median +0.013, mean +0.027, median over cells where the configuration changed +0.020 (30/36 changed).

**Check of 'at most 0.12 ×10⁻²' (16 → 256 KiB):** 0 of 36 estimator-cells exceed 0.12 (unrounded):
- Restricted to constant and drift (the draft's scope): max +0.102 (Hermite Series, t4 drift).
- Largest single-step (16→64 or 64→256) reduction: +0.102 (Hermite Series, t4 drift, 16->64).
- More storage gave a *higher* test RMSE in 10 estimator-cell-steps (validation selection noise): RMPL Gaussian jump 16->64 -0.001 ± 0.001; RMPL Gaussian jump 16->256 -0.001 ± 0.001; Counting Matrix t4 constant 64->256 -0.001 ± 0.002; RMPL Clayton constant 16->64 -0.001 ± 0.007; RMPL Gumbel constant 16->64 -0.001 ± 0.011; RMPL Gumbel constant 64->256 -0.000 ± 0.001; RMPL Gumbel constant 16->256 -0.001 ± 0.011; Counting Matrix Gumbel constant 16->64 -0.005 ± 0.007; Counting Matrix Gumbel constant 16->256 -0.005 ± 0.007; Hermite Series Gumbel constant 64->256 -0.020 ± 0.015.

## 5. RMPL minus the better nonparametric estimator (×10⁻², paired mean ± SE; W/L at 2 SE)

- 16 KiB: test-chosen comparator W/L/T = 5/6/1; validation-chosen = 5/5/2; comparator differs in 3 cells.
- 64 KiB: test-chosen comparator W/L/T = 5/6/1; validation-chosen = 5/5/2; comparator differs in 4 cells.
- 256 KiB: test-chosen comparator W/L/T = 5/6/1; validation-chosen = 5/5/2; comparator differs in 3 cells.
- Cell-by-cell verdicts identical at 16, 64 and 256 KiB: True.

64 KiB table (draft Table tab:ch6-e1) recomputed:

| Scenario | Family | RMPL g | CM | HS | RMPL | RMPL − better (test-chosen) | RMPL − better (validation-chosen) |
|---|---|---|---|---|---|---|---|
| constant | Gaussian | cum | 1.03 | 1.03 | 1.13 | +0.10 ± 0.05 L (HS) | +0.10 ± 0.04 L (CM) |
| constant | Clayton | cum | 1.05 | 1.07 | 1.24 | +0.20 ± 0.05 L (CM) | +0.20 ± 0.05 L (CM) |
| constant | Gumbel | cum | 1.02 | 1.02 | 1.18 | +0.17 ± 0.04 L (HS) | +0.17 ± 0.04 L (CM) |
| constant | t4 | cum | 1.11 | 1.21 | 1.26 | +0.15 ± 0.05 L (CM) | +0.05 ± 0.05 T (HS) |
| drift | Gaussian | 0.002 | 3.23 | 3.23 | 2.98 | -0.25 ± 0.03 W (CM) | -0.25 ± 0.03 W (CM) |
| drift | Clayton | 0.002 | 3.23 | 3.24 | 3.24 | +0.01 ± 0.04 T (CM) | +0.01 ± 0.04 T (CM) |
| drift | Gumbel | 0.002 | 3.18 | 3.20 | 2.68 | -0.50 ± 0.03 W (CM) | -0.52 ± 0.03 W (HS) |
| drift | t4 | 0.002 | 3.48 | 3.48 | 3.20 | -0.27 ± 0.03 W (HS) | -0.27 ± 0.03 W (HS) |
| jump | Gaussian | 0.01 | 6.17 | 6.21 | 5.95 | -0.22 ± 0.03 W (CM) | -0.22 ± 0.03 W (CM) |
| jump | Clayton | 0.01 | 6.24 | 6.28 | 6.56 | +0.31 ± 0.03 L (CM) | +0.31 ± 0.03 L (CM) |
| jump | Gumbel | 0.005 | 6.27 | 6.30 | 6.18 | -0.09 ± 0.02 W (CM) | -0.09 ± 0.02 W (CM) |
| jump | t4 | 0.005 | 6.40 | 6.44 | 6.48 | +0.08 ± 0.02 L (CM) | +0.08 ± 0.02 L (CM) |

## 6. Retained state of the selected configurations (bytes; peak over every tick, max over seeds)

| Estimator | Budget | selected approx (cells) | peak bytes min–max | utilisation of cap min–median–max |
|---|---|---|---|---|
| RMPL | 16 KiB | 256 (2), 512 (10) | 6,792–11,848 | 41.5% – 72.3% – 72.3% |
| RMPL | 64 KiB | 256 (1), 512 (2), 1024 (2), 2048 (7) | 6,792–40,616 | 10.4% – 62.0% – 62.0% |
| RMPL | 256 KiB | 256 (1), 512 (2), 1024 (2), 4096 (1), 8192 (2), 16384 (4) | 6,792–132,200 | 2.6% – 34.5% – 50.4% |
| Counting Matrix | 16 KiB | 30 (12) | 13,192–13,192 | 80.5% – 80.5% – 80.5% |
| Counting Matrix | 64 KiB | 30 (2), 65 (10) | 13,192–40,352 | 20.1% – 61.6% – 61.6% |
| Counting Matrix | 256 KiB | 30 (2), 90 (1), 130 (9) | 13,192–142,792 | 5.0% – 54.5% – 54.5% |
| Hermite Series | 16 KiB | 5 (1), 10 (1), 15 (1), 20 (9) | 1,304–8,144 | 8.0% – 49.7% – 49.7% |
| Hermite Series | 64 KiB | 10 (1), 30 (1), 43 (1), 60 (9) | 2,784–61,584 | 4.2% – 94.0% – 94.0% |
| Hermite Series | 256 KiB | 10 (1), 30 (1), 43 (1), 60 (4), 89 (5) | 2,784–132,344 | 1.1% – 23.5% – 50.5% |

Component breakdown (diag_mem_breakdown.cpp, 5000 Gaussian draws, cumulative mode; results/budget_mem_breakdown.csv):

- **RMPL** (`RecursiveMPL::store_bytes`): object itself 712 B (copula parameter, Fisher-information / gain state, counters, config, and inline shells of the unused margin types) + both DataSketches KLL margins (counting allocator, actual heap) + 112 B of heap held by unused margin objects. k = 256: 712 + 2 × 2,848 + 112 = 6,520 B; k = 512: 712 + 2 × 5,108 + 112 = 11,040 B; k = 1024: 712 + 2 × 8,808 + 112 = 18,440 B. No shared or external state exists in the standalone runs. Identical for g = cum and g = 0.005.
- **Counting Matrix**: object 680 B + B² + 2B count doubles + 2(B − 1) cut-points + 32 B of rank-window shells; its peak also includes the start-up calibration buffer (4,336 B at B = 30), which is released after the cut-points freeze, so the steady state is smaller than the reported peak (B = 30: 8,856 vs 13,192 B).
- **Hermite Series**: object 520 B + A and W ((N + 1)² doubles each) + a_x, a_y, z + 64 B. W is a constant of the order and could be shared; counting it makes Hermite's figure conservative (×≈2).
- Excluded everywhere: transient working memory inside update()/estimate() and allocator headers (malloc check within 0–7%, budget_mpl.md §24).

## 7. Processing time per observation at the selected configurations (µs; update + estimate, Gaussian drift, g = 0.02, one thread)

| Estimator | Budget | mean tick time min–median–max | update share of tick (median) |
|---|---|---|---|
| RMPL | 16 KiB | 1.70 – 2.80 – 2.80 | 98% |
| RMPL | 64 KiB | 1.70 – 8.41 – 8.41 | 99% |
| RMPL | 256 KiB | 1.70 – 5.33 – 18.43 | 99% |
| Counting Matrix | 16 KiB | 1.32 – 1.32 – 1.32 | 33% |
| Counting Matrix | 64 KiB | 1.32 – 6.06 – 6.06 | 26% |
| Counting Matrix | 256 KiB | 1.32 – 34.69 – 34.69 | 19% |
| Hermite Series | 16 KiB | 0.41 – 1.26 – 1.26 | 56% |
| Hermite Series | 64 KiB | 0.59 – 6.62 – 6.62 | 58% |
| Hermite Series | 256 KiB | 0.59 – 6.62 – 14.14 | 58% |

Full timing grid (µs, mean / p95; update only in brackets):

| Estimator | approx | peak bytes | mean | p95 | update |
|---|---|---|---|---|---|
| Hermite Series | 5 | 1,304 | 0.41 | 0.43 | 0.18 |
| Hermite Series | 10 | 2,784 | 0.59 | 0.64 | 0.30 |
| Hermite Series | 15 | 5,064 | 0.91 | 0.99 | 0.53 |
| Hermite Series | 20 | 8,144 | 1.26 | 1.38 | 0.71 |
| Hermite Series | 30 | 16,704 | 2.08 | 2.18 | 1.23 |
| Hermite Series | 43 | 32,616 | 3.47 | 3.90 | 2.00 |
| Hermite Series | 60 | 61,584 | 6.62 | 8.47 | 3.84 |
| Hermite Series | 89 | 132,344 | 14.14 | 16.98 | 8.59 |
| Hermite Series | 128 | 269,936 | 28.38 | 36.72 | 16.23 |
| RMPL | 8 | 1,816 | 0.60 | 1.00 | 0.56 |
| RMPL | 16 | 1,904 | 0.62 | 1.02 | 0.57 |
| RMPL | 32 | 2,104 | 0.67 | 1.12 | 0.62 |
| RMPL | 64 | 2,640 | 0.79 | 1.25 | 0.74 |
| RMPL | 128 | 3,976 | 1.09 | 1.33 | 1.04 |
| RMPL | 256 | 6,520 | 1.70 | 2.34 | 1.64 |
| RMPL | 512 | 11,040 | 2.80 | 4.97 | 2.74 |
| RMPL | 1024 | 18,440 | 5.33 | 10.96 | 5.27 |
| RMPL | 2048 | 28,448 | 8.41 | 18.12 | 8.35 |
| RMPL | 4096 | 55,760 | 13.93 | 32.43 | 13.85 |
| RMPL | 8192 | 66,664 | 18.43 | 41.06 | 18.36 |
| Counting Matrix | 10 | 6,472 | 0.31 | 0.36 | 0.13 |
| Counting Matrix | 20 | 9,032 | 0.72 | 0.83 | 0.28 |
| Counting Matrix | 30 | 13,192 | 1.32 | 1.63 | 0.44 |
| Counting Matrix | 45 | 22,432 | 2.75 | 3.46 | 0.81 |
| Counting Matrix | 65 | 40,352 | 6.06 | 7.37 | 1.56 |
| Counting Matrix | 90 | 71,752 | 14.31 | 18.41 | 3.17 |
| Counting Matrix | 130 | 142,792 | 34.69 | 47.26 | 6.54 |
| Counting Matrix | 180 | 267,592 | 61.46 | 81.33 | 12.17 |

Run-to-run reproducibility: 5-run mean (2026-11-15) / single run budget_timing_final (2026-11-10): ratio median 1.20, range 0.83–2.38.

## 8. Hand-written KLL vs Apache DataSketches KLL inside RMPL, matched k

Bytes and times: timing grid above. RMSE: test RMSE at the gain validation picks for each implementation at that k (mean over the 12 family × scenario cells, and the largest absolute cell difference).

| k | own bytes | DS bytes | own / DS | own µs | DS µs | DS / own time | mean RMSE own | mean RMSE DS | max abs cell diff |
|---|---|---|---|---|---|---|---|---|---|

## 9. Family-averaged test RMSE for the figure (×10⁻²)

Per test seed, average the four families' RMSEs at their own validated configurations; mean ± SE over the 100 seeds (seed s of every family is one replication, so the SE does not treat 400 family–seed combinations as independent).

| Scenario | Budget | RMPL | Counting Matrix | Hermite Series |
|---|---|---|---|---|
| constant | 16 KiB | 1.204 ± 0.029 | 1.055 ± 0.024 | 1.113 ± 0.024 |
| constant | 64 KiB | 1.204 ± 0.029 | 1.050 ± 0.024 | 1.082 ± 0.025 |
| constant | 256 KiB | 1.204 ± 0.029 | 1.051 ± 0.024 | 1.087 ± 0.025 |
| drift | 16 KiB | 3.035 ± 0.023 | 3.334 ± 0.023 | 3.343 ± 0.024 |
| drift | 64 KiB | 3.028 ± 0.022 | 3.281 ± 0.022 | 3.290 ± 0.023 |
| drift | 256 KiB | 3.028 ± 0.022 | 3.269 ± 0.022 | 3.289 ± 0.023 |
| jump | 16 KiB | 6.292 ± 0.022 | 6.285 ± 0.021 | 6.374 ± 0.021 |
| jump | 64 KiB | 6.292 ± 0.022 | 6.272 ± 0.021 | 6.310 ± 0.022 |
| jump | 256 KiB | 6.292 ± 0.022 | 6.271 ± 0.021 | 6.304 ± 0.022 |
