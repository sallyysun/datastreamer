# Storage, processing time and accuracy, extended RMPL grid (2026-11-15)

Reproduction of RMPL k = 1024 by the extension run: 9200 per-seed test rows, max |ΔRMSE| 0.0e+00; 184 summary rows, max |ΔRMSE| 0.0e+00, peak bytes equal True.

## 1. Timing table (gain 0.02; mean of 5 repeated runs ± SD across runs; RMSE family-averaged at the validation-best gain per size, ×10⁻²)

| Estimator | size | retained KiB | µs / obs (± SD, 5 runs) | p95 µs | update share | RMSE constant | RMSE drift | RMSE jump |
|---|---|---|---|---|---|---|---|---|
| RMPL | k = 8 | 1.77 | 0.60 ± 0.07 | 1.00 | 92% | 7.81 | 9.28 | 11.07 |
| RMPL | k = 16 | 1.86 | 0.62 ± 0.06 | 1.02 | 92% | 5.90 | 7.67 | 9.89 |
| RMPL | k = 32 | 2.05 | 0.67 ± 0.09 | 1.12 | 92% | 3.43 | 6.09 | 9.01 |
| RMPL | k = 64 | 2.58 | 0.79 ± 0.11 | 1.25 | 94% | 2.14 | 4.84 | 8.49 |
| RMPL | k = 128 | 3.88 | 1.09 ± 0.17 | 1.33 | 95% | 1.83 | 4.47 | 8.33 |
| RMPL | k = 256 | 6.37 | 1.70 ± 0.24 | 2.34 | 97% | 1.77 | 4.37 | 8.30 |
| RMPL | k = 512 | 10.78 | 2.80 ± 0.47 | 4.97 | 98% | 1.75 | 4.32 | 8.29 |
| RMPL | k = 1024 | 18.01 | 5.33 ± 0.69 | 10.96 | 99% | 1.75 | 4.32 | 8.29 |
| RMPL | k = 2048 | 27.78 | 8.41 ± 0.50 | 18.12 | 99% | 1.75 | 4.32 | 8.29 |
| RMPL | k = 4096 | 54.45 | 13.93 ± 1.37 | 32.43 | 99% | 1.75 | 4.31 | 8.29 |
| RMPL | k = 8192 | 65.10 | 18.43 ± 0.90 | 41.06 | 100% | 1.75 | 4.31 | 8.29 |
| Counting Matrix | bins = 10 | 6.32 | 0.31 ± 0.03 | 0.36 | 42% | 1.77 | 5.14 | 8.47 |
| Counting Matrix | bins = 20 | 8.82 | 0.72 ± 0.11 | 0.83 | 39% | 1.57 | 4.79 | 8.36 |
| Counting Matrix | bins = 30 | 12.88 | 1.32 ± 0.22 | 1.63 | 33% | 1.55 | 4.73 | 8.34 |
| Counting Matrix | bins = 45 | 21.91 | 2.75 ± 0.23 | 3.46 | 29% | 1.54 | 4.70 | 8.33 |
| Counting Matrix | bins = 65 | 39.41 | 6.06 ± 0.92 | 7.37 | 26% | 1.54 | 4.69 | 8.32 |
| Counting Matrix | bins = 90 | 70.07 | 14.31 ± 1.41 | 18.41 | 22% | 1.54 | 4.68 | 8.32 |
| Counting Matrix | bins = 130 | 139.45 | 34.69 ± 2.20 | 47.26 | 19% | 1.54 | 4.68 | 8.32 |
| Counting Matrix | bins = 180 | 261.32 | 61.46 ± 1.07 | 81.33 | 20% | 1.54 | 4.68 | 8.32 |
| Hermite Series | order = 5 | 1.27 | 0.41 ± 0.10 | 0.43 | 43% | 1.83 | 5.11 | 8.90 |
| Hermite Series | order = 10 | 2.72 | 0.59 ± 0.05 | 0.64 | 51% | 1.69 | 4.92 | 8.61 |
| Hermite Series | order = 15 | 4.95 | 0.91 ± 0.09 | 0.99 | 59% | 1.65 | 4.83 | 8.52 |
| Hermite Series | order = 20 | 7.95 | 1.26 ± 0.10 | 1.38 | 56% | 1.59 | 4.80 | 8.47 |
| Hermite Series | order = 30 | 16.31 | 2.08 ± 0.04 | 2.18 | 59% | 1.58 | 4.75 | 8.43 |
| Hermite Series | order = 43 | 31.85 | 3.47 ± 0.20 | 3.90 | 57% | 1.59 | 4.73 | 8.39 |
| Hermite Series | order = 60 | 60.14 | 6.62 ± 0.44 | 8.47 | 58% | 1.56 | 4.71 | 8.38 |
| Hermite Series | order = 89 | 129.24 | 14.14 ± 1.06 | 16.98 | 61% | 1.56 | 4.70 | 8.37 |
| Hermite Series | order = 128 | 263.61 | 28.38 ± 1.65 | 36.72 | 57% | 1.56 | 4.70 | 8.36 |

Timing stability: coefficient of variation across the 5 runs median 9.9%, max 23.4% (Hermite Series order = 5). Ratio to the earlier single run (budget_timing_final.csv, k ≤ 1024): median 1.20, range 0.83–2.38.

## 2. Effect of the gain on time (ratio to gain 0.02, same run conditions, one run each)

| Estimator | cumulative (g = 0) | g = 0.005 | g = 0.01 |
|---|---|---|---|
| RMPL | 0.92 (0.90–0.96) | 0.96 (0.87–1.14) | 0.96 (0.91–1.02) |
| Counting Matrix | 0.73 (0.63–0.82) | 0.89 (0.80–1.04) | 0.89 (0.82–1.06) |
| Hermite Series | 0.95 (0.88–0.99) | 0.96 (0.87–1.20) | 1.02 (0.88–1.37) |

## 3. Budget selections with the extended RMPL grid (validation only, as before)

| Estimator | budget | selected sizes (cells) | retained KiB max | µs at selected configs (g = 0.02) min–max | µs at selected gains min–max |
|---|---|---|---|---|---|
| RMPL | 16 KiB | 256 (1), 512 (11) | 10.78 | 1.70–2.80 | 1.52–2.55 |
| RMPL | 64 KiB | 256 (1), 512 (2), 1024 (2), 2048 (3), 4096 (4) | 54.45 | 1.70–13.93 | 1.52–13.36 |
| RMPL | 256 KiB | 256 (1), 512 (2), 1024 (2), 2048 (3), 4096 (2), 8192 (2) | 65.10 | 1.70–18.43 | 1.52–17.53 |
| Counting Matrix | 16 KiB | 20 (1), 30 (11) | 12.88 | 0.72–1.32 | 0.45–1.10 |
| Counting Matrix | 64 KiB | 20 (1), 30 (4), 65 (7) | 39.41 | 0.72–6.06 | 0.45–5.64 |
| Counting Matrix | 256 KiB | 20 (1), 30 (3), 65 (2), 90 (3), 130 (3) | 139.45 | 0.72–34.69 | 0.45–36.22 |
| Hermite Series | 16 KiB | 10 (1), 15 (1), 20 (10) | 7.95 | 0.59–1.26 | 0.57–1.52 |
| Hermite Series | 64 KiB | 10 (1), 30 (2), 43 (4), 60 (5) | 60.14 | 0.59–6.62 | 0.57–8.15 |
| Hermite Series | 256 KiB | 10 (1), 30 (2), 60 (1), 89 (8) | 129.24 | 0.59–14.14 | 0.57–13.56 |

RMSE reduction 16 → 256 KiB with the extended RMPL grid (×10⁻², paired):

- RMPL: median +0.001, max +0.014; 64 → 256 KiB max +0.000
- Counting Matrix: median +0.012, max +0.057; 64 → 256 KiB max +0.011
- Hermite Series: median +0.077, max +0.134; 64 → 256 KiB max +0.039
- All 36: median +0.013, max +0.134

## 4. Dominated configurations (family-averaged RMSE per scenario; a configuration is dominated if another, of any estimator, is no larger in retained KiB and µs and has RMSE lower by more than 2 SE of the difference treated as unpaired)

- **constant**: lowest RMSE 1.54 (Counting Matrix bins = 130, 139.4 KiB, 34.69 µs). Dominated: 11/28 — RMPL 8 (by Hermite Series 5); RMPL 16 (by Hermite Series 5); RMPL 32 (by Hermite Series 5); RMPL 64 (by Hermite Series 5); RMPL 128 (by Hermite Series 10); RMPL 256 (by Hermite Series 15); RMPL 512 (by Hermite Series 20); RMPL 1024 (by Hermite Series 20); RMPL 2048 (by Hermite Series 20); RMPL 4096 (by Hermite Series 20); RMPL 8192 (by Hermite Series 20)
  - within 1 µs: best Counting Matrix bins = 20 (RMSE 1.57, 8.8 KiB, 0.72 µs)
  - within 2 µs: best Counting Matrix bins = 30 (RMSE 1.55, 12.9 KiB, 1.32 µs)
  - within 5 µs: best Counting Matrix bins = 45 (RMSE 1.54, 21.9 KiB, 2.75 µs)
- **drift**: lowest RMSE 4.31 (RMPL k = 4096, 54.5 KiB, 13.93 µs). Dominated: 15/28 — Hermite Series 20 (by RMPL 128); Hermite Series 30 (by RMPL 128); Hermite Series 43 (by RMPL 128); Hermite Series 60 (by RMPL 128); Hermite Series 89 (by RMPL 128); Hermite Series 128 (by RMPL 128); RMPL 8 (by Hermite Series 5); RMPL 16 (by Hermite Series 5); RMPL 32 (by Hermite Series 5); Counting Matrix 30 (by RMPL 128); Counting Matrix 45 (by RMPL 128); Counting Matrix 65 (by RMPL 128); Counting Matrix 90 (by RMPL 128); Counting Matrix 130 (by RMPL 128); Counting Matrix 180 (by RMPL 128)
  - within 1 µs: best Counting Matrix bins = 20 (RMSE 4.79, 8.8 KiB, 0.72 µs)
  - within 2 µs: best RMPL k = 256 (RMSE 4.37, 6.4 KiB, 1.70 µs)
  - within 5 µs: best RMPL k = 512 (RMSE 4.32, 10.8 KiB, 2.80 µs)
- **jump**: lowest RMSE 8.29 (RMPL k = 1024, 18.0 KiB, 5.33 µs). Dominated: 4/28 — Hermite Series 20 (by RMPL 128); Hermite Series 30 (by RMPL 256); RMPL 8 (by Hermite Series 5); RMPL 16 (by Hermite Series 5)
  - within 1 µs: best Counting Matrix bins = 20 (RMSE 8.36, 8.8 KiB, 0.72 µs)
  - within 2 µs: best RMPL k = 256 (RMSE 8.30, 6.4 KiB, 1.70 µs)
  - within 5 µs: best RMPL k = 512 (RMSE 8.29, 10.8 KiB, 2.80 µs)
