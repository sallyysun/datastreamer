# Initial copula-family identification (constant families, 60 replications, uniform start)

| Truth | t50 median [IQR] | t90 median [IQR] | t50 ≤ 100 / 250 / 500 / 1000 | t90 ≤ 100 / 250 / 500 / 1000 | sustained ≥50% from, median [IQR] | never below 50% after first crossing | share of later obs. ≥50% (median) | mean weight at n = 100 / 250 / 500 / 1000 |
|---|---|---|---|---|---|---|---|---|
| Gaussian | 166.5 [108, 264.25] | 299 [241.25, 420] | 13 / 43 / 56 / 60 (of 60) | 0 / 19 / 47 / 60 | 1233 [289.75, 3078.75] | 7/60 | 0.991 | 0.23 / 0.58 / 0.87 / 0.95 |
|  | never reached: 0 | never reached: 0 | | | | | min share 0.807 | |
| Student's t4 | 108 [82.5, 135.25] | 178 [132.5, 222] | 23 / 57 / 60 / 60 (of 60) | 1 / 48 / 59 / 60 | 189 [118.75, 2078] | 17/60 | 0.998 | 0.37 / 0.86 / 0.98 / 0.98 |
|  | never reached: 0 | never reached: 0 | | | | | min share 0.965 | |
| Clayton | 44 [39, 52.5] | 68 [55, 86.25] | 59 / 60 / 60 / 60 (of 60) | 51 / 60 / 60 / 60 | 81 [49.75, 2364.75] | 14/60 | 0.999 | 0.90 / 0.99 / 0.99 / 0.99 |
|  | never reached: 0 | never reached: 0 | | | | | min share 0.987 | |
| Gumbel | 43 [37.75, 53] | 92 [76, 141] | 58 / 59 / 60 / 60 (of 60) | 35 / 52 / 58 / 60 | 408 [53.75, 2920.75] | 11/60 | 0.998 | 0.78 / 0.85 / 0.96 / 0.99 |
|  | never reached: 0 | never reached: 0 | | | | | min share 0.903 | |

**Definitions.** n = observations processed (weights held before observation n). t50 / t90: first n at which the true
family's weight, summed over ESS 99/399/1999, reaches 50% / 90%. "Sustained from" = one more than the last n < 5000 with
weight below 50% (strict: any later dip resets it). "Share of later obs." = fraction of observations after t50 at which the
true family holds ≥ 50%.

**Reading.** Identification from a uniform start is fast and complete: every replication reaches both thresholds, median
t50 = 166.5 (Gaussian), 108 (t4), 44 (Clayton), 43 (Gumbel); asymmetric-tail families are recognised fastest and the
Gaussian slowest (it is nested-like: t4, Clayton and Gumbel fit Gaussian data almost as well early on). After first
crossing, the true family holds ≥ 50% for 99–99.9% of later observations in the median replication (minimum 81%,
Gaussian), but brief dips below 50% occur in most replications (only 7–17 of 60 never dip), so the strict
"sustained-from" time is long and highly variable and should not be read as slow identification. The dips correspond to
the occasional false switches on constant paths (e.g. 14/60 Gaussian replications in which another family reaches 50%).

**Provenance.** probe_family_bank (code: INITID output added; outputs only), `SCENARIOS=matched FB_SCENARIOS=constant
INITID=1 TAG=initid`, seeds 42–101, clang++ -O2; configuration unchanged (24 arms = 8 shapes × ESS 99/399/1999,
CopulaFit arms, EW margins N_eff,m = 500, η = 1, δ = 2.1 × 10⁻⁴, T = 5000, ρ_S ≈ 0.58). Reproduction check: all
constant-path false-switch rows and family-bank metric rows equal the existing density3 run exactly. Data:
`results/family_bank_initid_42_initid[_t|_clayton|_gumbel].csv`; table script `experiments/famswitch_initid.py`.
