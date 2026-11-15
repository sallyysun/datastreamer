# Section 6.5 predictive accuracy across estimators (family-augmented vs Gaussian-only MFF)

Full post-switch window (observations 2500–4999), 60 paired replications. Absolute values ×10⁻³ (nats/obs for excess log loss). Percentage reduction = 100 (L_G − L_F)/L_G with ± 2 SE from a paired-seed bootstrap of the ratio of means (95% percentile interval in brackets).

| Metric | Switch | Estimator | Gaussian-only | Family-augmented | Absolute reduction ± 2 SE | % reduction ± 2 SE [95% CI] |
|---|---|---|---|---|---|---|
| Excess log loss | Gaussian → t4 | RMPL | 33.79 | 3.32 | +30.47 ± 1.90 | 90.2 ± 0.6 [89.5, 90.8] |
| Excess log loss | Gaussian → t4 | Counting Matrix | 33.91 | 3.35 | +30.56 ± 1.85 | 90.1 ± 0.6 [89.5, 90.7] |
| Excess log loss | Gaussian → t4 | Hermite Series | 33.92 | 3.44 | +30.48 ± 1.88 | 89.8 ± 0.6 [89.3, 90.4] |
| Excess log loss | Gaussian → Clayton | RMPL | 78.18 | 3.90 | +74.29 ± 2.07 | 95.0 ± 0.3 [94.7, 95.3] |
| Excess log loss | Gaussian → Clayton | Counting Matrix | 78.15 | 3.91 | +74.24 ± 2.09 | 95.0 ± 0.3 [94.7, 95.3] |
| Excess log loss | Gaussian → Clayton | Hermite Series | 78.32 | 3.96 | +74.37 ± 2.14 | 94.9 ± 0.3 [94.6, 95.3] |
| Excess log loss | Clayton → Gumbel | RMPL | 28.86 | 4.24 | +24.62 ± 1.16 | 85.3 ± 0.8 [84.5, 86.1] |
| Excess log loss | Clayton → Gumbel | Counting Matrix | 28.86 | 4.26 | +24.60 ± 1.15 | 85.2 ± 0.8 [84.5, 86.0] |
| Excess log loss | Clayton → Gumbel | Hermite Series | 28.89 | 4.34 | +24.55 ± 1.15 | 85.0 ± 0.8 [84.2, 85.7] |
| Squared Hellinger | Gaussian → t4 | RMPL | 6.91 | 0.85 | +6.06 ± 0.14 | 87.6 ± 1.3 [86.3, 88.9] |
| Squared Hellinger | Gaussian → t4 | Counting Matrix | 7.06 | 0.85 | +6.20 ± 0.12 | 87.9 ± 1.1 [86.8, 88.9] |
| Squared Hellinger | Gaussian → t4 | Hermite Series | 6.96 | 0.86 | +6.10 ± 0.13 | 87.7 ± 1.2 [86.4, 88.8] |
| Squared Hellinger | Gaussian → Clayton | RMPL | 20.53 | 1.18 | +19.34 ± 0.11 | 94.2 ± 0.4 [93.8, 94.7] |
| Squared Hellinger | Gaussian → Clayton | Counting Matrix | 20.47 | 1.18 | +19.28 ± 0.11 | 94.2 ± 0.5 [93.8, 94.7] |
| Squared Hellinger | Gaussian → Clayton | Hermite Series | 20.42 | 1.22 | +19.20 ± 0.13 | 94.0 ± 0.5 [93.5, 94.5] |
| Squared Hellinger | Clayton → Gumbel | RMPL | 7.35 | 1.48 | +5.87 ± 0.13 | 79.9 ± 1.7 [78.1, 81.5] |
| Squared Hellinger | Clayton → Gumbel | Counting Matrix | 7.37 | 1.47 | +5.90 ± 0.11 | 80.1 ± 1.6 [78.4, 81.5] |
| Squared Hellinger | Clayton → Gumbel | Hermite Series | 7.34 | 1.47 | +5.87 ± 0.12 | 80.0 ± 1.7 [78.2, 81.6] |

Denominators: smallest per-seed Gaussian-only value across all cells = 16.31 ×10⁻³ (excess) / 6.32 ×10⁻³ (H²); all positive and far from zero.

## RMPL: early vs later recovery (absolute reduction ± 2 SE, ×10⁻³; positive favours family augmentation)

| Switch | Metric | first 100 (2500–2599) | remaining 2400 (2600–4999) | full post-switch |
|---|---|---|---|---|
| Gaussian → t4 | Excess log loss | +8.17 ± 6.15 | +31.40 ± 1.95 | +30.47 ± 1.90 |
| Gaussian → t4 | Squared Hellinger | +0.82 ± 0.30 | +6.29 ± 0.14 | +6.06 ± 0.14 |
| Gaussian → Clayton | Excess log loss | +24.31 ± 7.27 | +76.37 ± 2.16 | +74.29 ± 2.07 |
| Gaussian → Clayton | Squared Hellinger | +4.72 ± 0.96 | +20.06 ± 0.11 | +19.34 ± 0.11 |
| Clayton → Gumbel | Excess log loss | -42.58 ± 5.77 | +27.42 ± 1.18 | +24.62 ± 1.16 |
| Clayton → Gumbel | Squared Hellinger | -11.82 ± 1.75 | +6.88 ± 0.10 | +5.87 ± 0.13 |

H² uses checkpoints: every 10 observations in 2500–2599 (rec_0_100) and every 50 elsewhere; excess log loss uses every observation.

## Estimator agreement (full post-switch % reduction: max − min across the three estimators)

| Metric | Switch | spread (pp) | largest 2 SE (pp) |
|---|---|---|---|
| Excess log loss | Gaussian → t4 | 0.32 | 0.64 |
| Excess log loss | Gaussian → Clayton | 0.07 | 0.33 |
| Excess log loss | Clayton → Gumbel | 0.34 | 0.83 |
| Squared Hellinger | Gaussian → t4 | 0.26 | 1.28 |
| Squared Hellinger | Gaussian → Clayton | 0.21 | 0.49 |
| Squared Hellinger | Clayton → Gumbel | 0.17 | 1.71 |
