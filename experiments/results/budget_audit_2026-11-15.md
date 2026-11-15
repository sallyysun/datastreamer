# Section 6.2 (E1) memory-constraint audit, 2026-11-15 (extended RMPL grid k = 8-8192; the k <= 1024 version is budget_audit_2026-11-15_k1024.md)

Platform of the runs: Linux x86_64, libstdc++, g++ -O2 (budget_gainext_2026-11-11.txt header). No new simulation was run for this audit.

## 1. Design verification

- Test seeds per configuration: 100–100, range 42–141 (per-seed file holds test seeds only; validation enters only through the summary rows, split = val).
- Families: ['clayton', 'gaussian', 'gumbel', 't']; scenarios: ['constant', 'drift', 'jump']; estimators in the candidate set: ['hermite', 'mpl_dskll', 'mpl_ew', 'mpl_exact', 'mpl_fit_kll', 'mpl_kll', 'xiao'].
- Gain grid, constant: cum, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1
- Gain grid, drift: cum, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1
- Gain grid, jump: 0.005, 0.01, 0.02, 0.05, 0.1
- Approximation grid, Counting Matrix (bins B): [10, 20, 30, 45, 65, 90, 130, 180]
- Approximation grid, Hermite Series (order N): [5, 10, 15, 20, 30, 43, 60, 89, 128]
- Approximation grid, RMPL (KLL k): [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
- Validation seeds 1000–1039 (n_val = 40) and test seeds 42–141 (n_test = 100): from the probe_budget sweep invocation (`sweep 100 40`, kValSeed = 1000, kTestSeed = 42) recorded in the run header.

## 2. Selection uses validation only

- `pick()` (identical to analyze_budget_gainext.py and plot_budget.py) filters split = val rows and takes the minimum validation RMSE among candidates whose validation peak bytes ≤ cap. Test rows enter only after selection. Confirmed by reading the code; no test quantity feeds a configuration choice.
- All 108 selected configurations also fit their cap on the test seeds: max test peak / cap = 0.940; test peak = validation peak in 108/108 (RMPL's DataSketches compaction is randomised but its peak is deterministic here).
- Exception to 'validation only': the *comparator* in 'RMPL − better of CM/HS' is chosen on test means (draft §6.2 says so). Section 5 reports the validation-chosen comparator as well.

## 3. Test Spearman RMSE (×10⁻², mean ± SE over 100 test seeds) at the validated configuration

Cell entries: RMSE ± SE [approx, gain]. Approx = KLL k (RMPL), bins B (CM), order N (HS).

| Scenario | Family | Budget | RMPL | Counting Matrix | Hermite Series |
|---|---|---|---|---|---|
| constant | Gaussian | 16 | 1.71 ± 0.11 [512, cum] | 1.53 ± 0.08 [30, cum] | 1.60 ± 0.07 [20, cum] |
| constant | Gaussian | 64 | 1.71 ± 0.11 [2048, cum] | 1.53 ± 0.08 [30, cum] | 1.58 ± 0.08 [30, cum] |
| constant | Gaussian | 256 | 1.71 ± 0.11 [2048, cum] | 1.53 ± 0.08 [30, cum] | 1.58 ± 0.08 [30, cum] |
| constant | t4 | 16 | 1.81 ± 0.08 [256, cum] | 1.66 ± 0.07 [30, cum] | 1.79 ± 0.09 [10, cum] |
| constant | t4 | 64 | 1.81 ± 0.08 [256, cum] | 1.64 ± 0.07 [65, cum] | 1.79 ± 0.09 [10, cum] |
| constant | t4 | 256 | 1.81 ± 0.08 [256, cum] | 1.64 ± 0.07 [90, cum] | 1.79 ± 0.09 [10, cum] |
| constant | Clayton | 16 | 1.81 ± 0.09 [512, cum] | 1.51 ± 0.07 [30, cum] | 1.60 ± 0.07 [15, cum] |
| constant | Clayton | 64 | 1.81 ± 0.09 [4096, cum] | 1.51 ± 0.07 [30, cum] | 1.53 ± 0.08 [30, cum] |
| constant | Clayton | 256 | 1.81 ± 0.09 [8192, cum] | 1.51 ± 0.07 [30, cum] | 1.53 ± 0.08 [30, cum] |
| constant | Gumbel | 16 | 1.67 ± 0.08 [512, cum] | 1.51 ± 0.08 [20, cum] | 1.58 ± 0.07 [20, cum] |
| constant | Gumbel | 64 | 1.67 ± 0.08 [4096, cum] | 1.51 ± 0.08 [20, cum] | 1.49 ± 0.07 [60, cum] |
| constant | Gumbel | 256 | 1.67 ± 0.08 [8192, cum] | 1.51 ± 0.08 [20, cum] | 1.52 ± 0.07 [89, cum] |
| drift | Gaussian | 16 | 4.16 ± 0.09 [512, 0.005] | 4.66 ± 0.08 [30, 0.005] | 4.71 ± 0.08 [20, 0.005] |
| drift | Gaussian | 64 | 4.15 ± 0.09 [4096, 0.005] | 4.62 ± 0.08 [65, 0.005] | 4.64 ± 0.08 [60, 0.005] |
| drift | Gaussian | 256 | 4.15 ± 0.09 [4096, 0.005] | 4.61 ± 0.08 [130, 0.005] | 4.65 ± 0.08 [89, 0.005] |
| drift | t4 | 16 | 4.67 ± 0.09 [512, 0.005] | 4.89 ± 0.08 [30, 0.005] | 4.95 ± 0.08 [20, 0.005] |
| drift | t4 | 64 | 4.65 ± 0.09 [1024, 0.005] | 4.86 ± 0.08 [65, 0.005] | 4.88 ± 0.08 [60, 0.005] |
| drift | t4 | 256 | 4.65 ± 0.09 [1024, 0.005] | 4.85 ± 0.08 [90, 0.005] | 4.84 ± 0.08 [89, 0.005] |
| drift | Clayton | 16 | 4.47 ± 0.09 [512, 0.005] | 4.68 ± 0.08 [30, 0.005] | 4.75 ± 0.08 [20, 0.005] |
| drift | Clayton | 64 | 4.47 ± 0.09 [512, 0.005] | 4.65 ± 0.08 [65, 0.005] | 4.69 ± 0.08 [43, 0.005] |
| drift | Clayton | 256 | 4.47 ± 0.09 [512, 0.005] | 4.64 ± 0.08 [130, 0.005] | 4.66 ± 0.09 [89, 0.005] |
| drift | Gumbel | 16 | 3.99 ± 0.08 [512, 0.005] | 4.66 ± 0.09 [30, 0.005] | 4.77 ± 0.09 [20, 0.005] |
| drift | Gumbel | 64 | 3.98 ± 0.08 [2048, 0.005] | 4.63 ± 0.09 [65, 0.005] | 4.66 ± 0.09 [43, 0.005] |
| drift | Gumbel | 256 | 3.98 ± 0.08 [2048, 0.005] | 4.62 ± 0.09 [130, 0.005] | 4.65 ± 0.09 [89, 0.005] |
| jump | Gaussian | 16 | 7.91 ± 0.07 [512, 0.01] | 8.22 ± 0.07 [30, 0.01] | 8.37 ± 0.07 [20, 0.01] |
| jump | Gaussian | 64 | 7.91 ± 0.07 [4096, 0.01] | 8.19 ± 0.07 [65, 0.01] | 8.25 ± 0.06 [43, 0.01] |
| jump | Gaussian | 256 | 7.91 ± 0.07 [4096, 0.01] | 8.19 ± 0.07 [65, 0.01] | 8.24 ± 0.07 [89, 0.01] |
| jump | t4 | 16 | 8.63 ± 0.08 [512, 0.01] | 8.52 ± 0.08 [30, 0.01] | 8.64 ± 0.08 [20, 0.01] |
| jump | t4 | 64 | 8.63 ± 0.08 [1024, 0.01] | 8.52 ± 0.08 [30, 0.01] | 8.54 ± 0.08 [60, 0.01] |
| jump | t4 | 256 | 8.63 ± 0.08 [1024, 0.01] | 8.51 ± 0.08 [90, 0.01] | 8.55 ± 0.08 [89, 0.01] |
| jump | Clayton | 16 | 8.49 ± 0.09 [512, 0.01] | 8.29 ± 0.09 [30, 0.01] | 8.41 ± 0.09 [20, 0.01] |
| jump | Clayton | 64 | 8.49 ± 0.09 [512, 0.01] | 8.29 ± 0.09 [30, 0.01] | 8.36 ± 0.09 [60, 0.01] |
| jump | Clayton | 256 | 8.49 ± 0.09 [512, 0.01] | 8.29 ± 0.09 [30, 0.01] | 8.36 ± 0.09 [60, 0.01] |
| jump | Gumbel | 16 | 8.12 ± 0.08 [512, 0.01] | 8.32 ± 0.08 [30, 0.01] | 8.46 ± 0.08 [20, 0.01] |
| jump | Gumbel | 64 | 8.12 ± 0.08 [2048, 0.01] | 8.31 ± 0.08 [65, 0.01] | 8.36 ± 0.08 [43, 0.01] |
| jump | Gumbel | 256 | 8.12 ± 0.08 [2048, 0.01] | 8.31 ± 0.08 [65, 0.01] | 8.36 ± 0.08 [89, 0.01] |

## 4. Effect of storage (RMSE reduction ×10⁻², positive = more storage helps; paired over the 100 test seeds)

| Step | Estimator | cells with a different configuration | max reduction (cell) | median | mean | min (cell) | cells reduction > 2 SE | cells worse by > 2 SE |
|---|---|---|---|---|---|---|---|---|
| 16->64 | RMPL | 9/12 | +0.014 ± 0.003 (t4 drift) | +0.001 | +0.004 | -0.005 ± 0.008 (Clayton constant) | 4/12 | 0/12 |
| 16->64 | Counting Matrix | 7/12 | +0.046 ± 0.006 (Gaussian drift) | +0.013 | +0.017 | +0.000 ± 0.000 (Gaussian constant) | 5/12 | 0/12 |
| 16->64 | Hermite Series | 11/12 | +0.126 ± 0.024 (Gaussian jump) | +0.075 | +0.073 | +0.000 ± 0.000 (t4 constant) | 8/12 | 0/12 |
| 64->256 | RMPL | 2/12 | +0.000 ± 0.000 (Clayton constant) | +0.000 | +0.000 | +0.000 ± 0.000 (Gaussian constant) | 0/12 | 0/12 |
| 64->256 | Counting Matrix | 6/12 | +0.011 ± 0.002 (Gumbel drift) | +0.000 | +0.003 | -0.003 ± 0.003 (t4 constant) | 3/12 | 0/12 |
| 64->256 | Hermite Series | 8/12 | +0.039 ± 0.017 (t4 drift) | +0.000 | +0.003 | -0.033 ± 0.024 (Gumbel constant) | 1/12 | 0/12 |
| 16->256 | RMPL | 9/12 | +0.014 ± 0.003 (t4 drift) | +0.001 | +0.004 | -0.005 ± 0.008 (Clayton constant) | 4/12 | 0/12 |
| 16->256 | Counting Matrix | 8/12 | +0.057 ± 0.005 (Gaussian drift) | +0.012 | +0.020 | +0.000 ± 0.000 (Gaussian constant) | 5/12 | 0/12 |
| 16->256 | Hermite Series | 11/12 | +0.134 ± 0.024 (Gaussian jump) | +0.077 | +0.077 | +0.000 ± 0.000 (t4 constant) | 7/12 | 0/12 |

All estimators, 16->64: max +0.126, median +0.013, mean +0.031, median over cells where the configuration changed +0.032 (27/36 changed).

All estimators, 64->256: max +0.039, median +0.000, mean +0.002, median over cells where the configuration changed +0.005 (16/36 changed).

All estimators, 16->256: max +0.134, median +0.013, mean +0.033, median over cells where the configuration changed +0.033 (28/36 changed).

**Check of 'at most 0.12 ×10⁻²' (16 → 256 KiB):** 2 of 36 estimator-cells exceed 0.12 (unrounded):
- Hermite Series, Gaussian jump: +0.134 ± 0.024 [20,0.01 → 89,0.01]
- Hermite Series, Gumbel drift: +0.125 ± 0.029 [20,0.005 → 89,0.005]
- Restricted to constant and drift (the draft's scope): max +0.125 (Hermite Series, Gumbel drift).
- Largest single-step (16→64 or 64→256) reduction: +0.126 (Hermite Series, Gaussian jump, 16->64).
- More storage gave a *higher* test RMSE in 8 estimator-cell-steps (validation selection noise): RMPL Gaussian constant 16->64 -0.001 ± 0.003; RMPL Gaussian constant 16->256 -0.001 ± 0.003; Hermite Series Gaussian drift 64->256 -0.014 ± 0.018; Counting Matrix t4 constant 64->256 -0.003 ± 0.003; Hermite Series t4 jump 64->256 -0.004 ± 0.017; RMPL Clayton constant 16->64 -0.005 ± 0.008; RMPL Clayton constant 16->256 -0.005 ± 0.008; Hermite Series Gumbel constant 64->256 -0.033 ± 0.024.

## 5. RMPL minus the better nonparametric estimator (×10⁻², paired mean ± SE; W/L at 2 SE)

- 16 KiB: test-chosen comparator W/L/T = 6/6/0; validation-chosen = 6/5/1; comparator differs in 1 cells.
- 64 KiB: test-chosen comparator W/L/T = 6/6/0; validation-chosen = 6/5/1; comparator differs in 2 cells.
- 256 KiB: test-chosen comparator W/L/T = 6/6/0; validation-chosen = 6/5/1; comparator differs in 2 cells.
- Cell-by-cell verdicts identical at 16, 64 and 256 KiB: True.

64 KiB table (draft Table tab:ch6-e1) recomputed:

| Scenario | Family | RMPL g | CM | HS | RMPL | RMPL − better (test-chosen) | RMPL − better (validation-chosen) |
|---|---|---|---|---|---|---|---|
| constant | Gaussian | cum | 1.53 | 1.58 | 1.71 | +0.18 ± 0.08 L (CM) | +0.18 ± 0.08 L (CM) |
| constant | Clayton | cum | 1.51 | 1.53 | 1.81 | +0.30 ± 0.09 L (CM) | +0.30 ± 0.09 L (CM) |
| constant | Gumbel | cum | 1.51 | 1.49 | 1.67 | +0.18 ± 0.06 L (HS) | +0.16 ± 0.07 L (CM) |
| constant | t4 | cum | 1.64 | 1.79 | 1.81 | +0.17 ± 0.07 L (CM) | +0.02 ± 0.08 T (HS) |
| drift | Gaussian | 0.005 | 4.62 | 4.64 | 4.15 | -0.47 ± 0.04 W (CM) | -0.47 ± 0.04 W (CM) |
| drift | Clayton | 0.005 | 4.65 | 4.69 | 4.47 | -0.18 ± 0.06 W (CM) | -0.18 ± 0.06 W (CM) |
| drift | Gumbel | 0.005 | 4.63 | 4.66 | 3.98 | -0.65 ± 0.05 W (CM) | -0.65 ± 0.05 W (CM) |
| drift | t4 | 0.005 | 4.86 | 4.88 | 4.65 | -0.20 ± 0.05 W (CM) | -0.20 ± 0.05 W (CM) |
| jump | Gaussian | 0.01 | 8.19 | 8.25 | 7.91 | -0.28 ± 0.04 W (CM) | -0.28 ± 0.04 W (CM) |
| jump | Clayton | 0.01 | 8.29 | 8.36 | 8.49 | +0.20 ± 0.04 L (CM) | +0.20 ± 0.04 L (CM) |
| jump | Gumbel | 0.01 | 8.31 | 8.36 | 8.12 | -0.19 ± 0.04 W (CM) | -0.19 ± 0.04 W (CM) |
| jump | t4 | 0.01 | 8.52 | 8.54 | 8.63 | +0.11 ± 0.04 L (CM) | +0.11 ± 0.04 L (CM) |

## 6. Retained state of the selected configurations (bytes; peak over every tick, max over seeds)

| Estimator | Budget | selected approx (cells) | peak bytes min–max | utilisation of cap min–median–max |
|---|---|---|---|---|
| RMPL | 16 KiB | 256 (1), 512 (11) | 6,520–11,040 | 39.8% – 67.4% – 67.4% |
| RMPL | 64 KiB | 256 (1), 512 (2), 1024 (2), 2048 (3), 4096 (4) | 6,520–55,760 | 9.9% – 43.4% – 85.1% |
| RMPL | 256 KiB | 256 (1), 512 (2), 1024 (2), 2048 (3), 4096 (2), 8192 (2) | 6,520–66,664 | 2.5% – 10.9% – 25.4% |
| Counting Matrix | 16 KiB | 20 (1), 30 (11) | 9,032–13,192 | 55.1% – 80.5% – 80.5% |
| Counting Matrix | 64 KiB | 20 (1), 30 (4), 65 (7) | 9,032–40,352 | 13.8% – 61.6% – 61.6% |
| Counting Matrix | 256 KiB | 20 (1), 30 (3), 65 (2), 90 (3), 130 (3) | 9,032–142,792 | 3.4% – 21.4% – 54.5% |
| Hermite Series | 16 KiB | 10 (1), 15 (1), 20 (10) | 2,784–8,144 | 17.0% – 49.7% – 49.7% |
| Hermite Series | 64 KiB | 10 (1), 30 (2), 43 (4), 60 (5) | 2,784–61,584 | 4.2% – 49.8% – 94.0% |
| Hermite Series | 256 KiB | 10 (1), 30 (2), 60 (1), 89 (8) | 2,784–132,344 | 1.1% – 50.5% – 50.5% |

Component breakdown (diag_mem_breakdown.cpp, 5000 Gaussian draws, cumulative mode; results/budget_mem_breakdown.csv):

- **RMPL** (`RecursiveMPL::store_bytes`): object itself 712 B (copula parameter, Fisher-information / gain state, counters, config, and inline shells of the unused margin types) + both DataSketches KLL margins (counting allocator, actual heap) + 112 B of heap held by unused margin objects. k = 256: 712 + 2 × 2,848 + 112 = 6,520 B; k = 512: 712 + 2 × 5,108 + 112 = 11,040 B; k = 1024: 712 + 2 × 8,808 + 112 = 18,440 B. No shared or external state exists in the standalone runs. Identical for g = cum and g = 0.005.
- **Counting Matrix**: object 680 B + B² + 2B count doubles + 2(B − 1) cut-points + 32 B of rank-window shells; its peak also includes the start-up calibration buffer (4,336 B at B = 30), which is released after the cut-points freeze, so the steady state is smaller than the reported peak (B = 30: 8,856 vs 13,192 B).
- **Hermite Series**: object 520 B + A and W ((N + 1)² doubles each) + a_x, a_y, z + 64 B. W is a constant of the order and could be shared; counting it makes Hermite's figure conservative (×≈2).
- Excluded everywhere: transient working memory inside update()/estimate() and allocator headers (malloc check within 0–7%, budget_mpl.md §24).

## 7. Processing time per observation at the selected configurations (µs; update + estimate, Gaussian drift, g = 0.02, one thread)

| Estimator | Budget | mean tick time min–median–max | update share of tick (median) |
|---|---|---|---|
| RMPL | 16 KiB | 1.70 – 2.80 – 2.80 | 98% |
| RMPL | 64 KiB | 1.70 – 8.41 – 13.93 | 99% |
| RMPL | 256 KiB | 1.70 – 8.41 – 18.43 | 99% |
| Counting Matrix | 16 KiB | 0.72 – 1.32 – 1.32 | 33% |
| Counting Matrix | 64 KiB | 0.72 – 6.06 – 6.06 | 26% |
| Counting Matrix | 256 KiB | 0.72 – 10.18 – 34.69 | 24% |
| Hermite Series | 16 KiB | 0.59 – 1.26 – 1.26 | 56% |
| Hermite Series | 64 KiB | 0.59 – 3.47 – 6.62 | 58% |
| Hermite Series | 256 KiB | 0.59 – 14.14 – 14.14 | 61% |

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
| 8 | 4,616 | 1,816 | 2.54 | 0.80 | 0.43 | 0.54 | 12.52 | 9.39 | 4.91 |
| 16 | 5,896 | 1,904 | 3.10 | 0.78 | 0.44 | 0.56 | 8.07 | 7.82 | 1.14 |
| 32 | 8,840 | 2,104 | 4.20 | 0.76 | 0.47 | 0.62 | 5.97 | 6.18 | 0.60 |
| 64 | 14,472 | 2,640 | 5.48 | 0.66 | 0.63 | 0.95 | 5.13 | 5.16 | 0.21 |
| 128 | 23,688 | 3,976 | 5.96 | 0.67 | 0.91 | 1.35 | 4.87 | 4.88 | 0.03 |
| 256 | 38,024 | 6,520 | 5.83 | 0.63 | 1.33 | 2.11 | 4.81 | 4.81 | 0.05 |
| 512 | 58,312 | 11,040 | 5.28 | 0.69 | 1.99 | 2.88 | 4.80 | 4.79 | 0.06 |
| 1024 | 82,888 | 18,440 | 4.50 | 0.88 | 2.88 | 3.26 | 4.79 | 4.78 | 0.03 |

## 9. Family-averaged test RMSE for the figure (×10⁻²)

Per test seed, average the four families' RMSEs at their own validated configurations; mean ± SE over the 100 seeds (seed s of every family is one replication, so the SE does not treat 400 family–seed combinations as independent).

| Scenario | Budget | RMPL | Counting Matrix | Hermite Series |
|---|---|---|---|---|
| constant | 16 KiB | 1.750 ± 0.046 | 1.554 ± 0.038 | 1.640 ± 0.039 |
| constant | 64 KiB | 1.751 ± 0.045 | 1.549 ± 0.038 | 1.597 ± 0.040 |
| constant | 256 KiB | 1.751 ± 0.045 | 1.550 ± 0.038 | 1.606 ± 0.040 |
| drift | 16 KiB | 4.322 ± 0.044 | 4.725 ± 0.044 | 4.797 ± 0.044 |
| drift | 64 KiB | 4.313 ± 0.044 | 4.688 ± 0.044 | 4.716 ± 0.044 |
| drift | 256 KiB | 4.313 ± 0.044 | 4.679 ± 0.043 | 4.699 ± 0.045 |
| jump | 16 KiB | 8.288 ± 0.038 | 8.335 ± 0.037 | 8.472 ± 0.036 |
| jump | 64 KiB | 8.285 ± 0.038 | 8.327 ± 0.037 | 8.376 ± 0.036 |
| jump | 256 KiB | 8.285 ± 0.038 | 8.325 ± 0.036 | 8.375 ± 0.036 |
