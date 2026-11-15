# Section 6.5 (RMPL / KLL configuration) — family-augmented vs Gaussian-only MFF

Configuration: RMPL arms N_eff ∈ {39, 99, 199, 399, 1999} on shared cumulative KLL (k = 256) pseudo-observations; family-augmented MFF = 20 models (Gaussian, t4, Clayton, Gumbel × 5 memories; each RMPL ρ_S mapped to the family parameter by inversion, as in the Section 6.4 scoring-function experiment); Gaussian-only MFF = the 5 Gaussian models; both log loss, η = 1, δ = 10⁻⁴, mixture density forecast, warm-up as Chapter 6. 60 replications (seeds 42–101), T = 5000, ρ_S ≈ 0.582 constant, switch at observation 2500, evaluation from observation 500.

Verification: on the Gaussian constant path, Gaussian-only MFF reproduces the existing Section 6.3 mixture-density MFF and Section 6.4 log-loss MFF (RMPL) exactly (ρ_S RMSE and excess log loss, all 60 seeds, max difference 0).

## Initial family identification (constant families, uniform start)

| Generating family | 50% weight, median [IQR] | 90% weight, median [IQR] | reached 50% / 90% | another family ≥ 50% after identification | share of later obs. with true family ≥ 50% (median, min) | mean true-family weight, obs. 500–4999 |
|---|---|---|---|---|---|---|
| Gaussian | 148.5 [58.25, 242.25] | 261 [129.25, 317] | 60/60 / 60/60 | 32/60 | 0.998 (0.928) | 0.987 |
| Student's t4 | 47.5 [33, 119] | 105 [34, 197.25] | 60/60 / 60/60 | 38/60 | 0.996 (0.928) | 0.980 |
| Clayton | 47 [39, 104] | 96 [57, 152.75] | 60/60 / 60/60 | 35/60 | 0.998 (0.938) | 0.995 |
| Gumbel | 94 [44, 213.25] | 233.5 [112.25, 390.75] | 60/60 / 60/60 | 42/60 | 0.993 (0.830) | 0.971 |

## Adaptation following a family switch

| Switch | new family 50%, median [IQR] (obs. after switch) | 90%, median [IQR] | reached 50% / 90% | new-family weight before switch | new-family weight, obs. 3000–4999 | another family ≥ 50% after identification |
|---|---|---|---|---|---|---|
| Gaussian → t4 | 178.5 [137, 285] | 268 [188, 389.75] | 60/60 / 60/60 | 0.0030 | 0.978 | 36/60 |
| Gaussian → Clayton | 82.5 [64.75, 112.25] | 114 [83.25, 136] | 60/60 / 60/60 | 0.0011 | 0.994 | 30/60 |
| Clayton → Gumbel | 63.5 [41.75, 126.25] | 159 [97.25, 217] | 60/60 / 60/60 | 0.0007 | 0.981 | 32/60 |

## Predictive density accuracy: ΔL = L(Gaussian-only MFF) − L(family-augmented MFF), ×10⁻³ nats/obs, mean ± 2 SE over 60 paired replications

| Scenario | first 100 after switch (2500–2599) | remaining 2400 (2600–4999) | full post-switch (2500–4999) | pre-switch (500–2499) | L Gaussian-only / family (post) |
|---|---|---|---|---|---|
| Gaussian → t4 | +8.2 ± 6.1 | +31.4 ± 2.0 | +30.5 ± 1.9 | -0.1 ± 0.1 | 33.79 / 3.32 |
| Gaussian → Clayton | +24.3 ± 7.3 | +76.4 ± 2.2 | +74.3 ± 2.1 | -0.1 ± 0.1 | 78.18 / 3.90 |
| Clayton → Gumbel | -42.6 ± 5.8 | +27.4 ± 1.2 | +24.6 ± 1.2 | +75.2 ± 2.4 | 28.86 / 4.24 |

Constant families (observations 500–4999): ΔL (negative = cost of family augmentation)

| Generating family | ΔL, mean ± 2 SE | L Gaussian-only / family |
|---|---|---|
| Gaussian | -0.07 ± 0.03 | 0.84 / 0.91 |
| Student's t4 | +33.85 ± 1.41 | 34.83 / 0.98 |
| Clayton | +75.88 ± 1.65 | 77.12 / 1.24 |
| Gumbel | +28.72 ± 1.02 | 29.95 / 1.23 |

Spearman ρ_S RMSE (×10⁻², whole run): Gaussian-only / family, and difference (positive favours family)

| Scenario | RMSE | difference |
|---|---|---|
| Gaussian | 2.27 / 2.25 | +0.02 ± 0.01 |
| Student's t4 | 3.08 / 2.70 | +0.38 ± 0.13 |
| Clayton | 2.37 / 2.29 | +0.08 ± 0.09 |
| Gumbel | 2.48 / 2.30 | +0.18 ± 0.09 |
| Gaussian → t4 | 2.69 / 2.73 | -0.04 ± 0.09 |
| Gaussian → Clayton | 2.44 / 2.53 | -0.09 ± 0.07 |
| Clayton → Gumbel | 2.42 / 2.70 | -0.29 ± 0.11 |

Non-finite model log densities replaced by Mff's guard: 0 in total.
