# Section 6.2 (Estimation under memory constraints) — results summary

Context: honours thesis on streaming (single-pass, bounded-memory) estimation of time-varying copula dependence. Section 6.2 asks
how much retained-state storage three online estimators need, and whether more storage improves accuracy or only costs time.

## Design
- **Estimators** (each reports Spearman's ρ_S):
  - **RMPL**: recursive Gaussian-copula maximum pseudo-likelihood; marginals from two KLL quantile sketches (Apache DataSketches);
    approximation size = sketch capacity k (8 to 8192; to 16384 in the long-stream run).
  - **Counting Matrix**: Xiao's bivariate binned rank-count estimator; size = number of bins (10–180).
  - **Hermite Series**: bivariate Hermite-series estimator; size = truncation order (5–128).
  - Forgetting gain g ∈ {no forgetting, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1} (jump: 0.005–0.1).
- **Scenarios:** constant (ρ_S ≈ 0.58), gradual drift (linear over the stream), abrupt jump (ρ_S 0.19 → 0.79 at mid-stream).
  **Copula families:** Gaussian, Student t₄, Clayton, Gumbel.
- **Storage budgets:** 16, 64, 256 KiB of *measured* retained state (peak over every observation, including both RMPL margins and
  its parameter and gain state; checked against an exact allocation count to within 7%).
- **Protocol:** within each budget, each estimator's size and gain are chosen on validation seeds 1000–1039 only, then scored on
  test seeds 42–141 (100 replications). Stream length n = 5000 (main) and n = 15,000 (robustness check).
  RMSE is in ×10⁻²; uncertainties are paired over test seeds; family averages are taken per seed before averaging.

## Main findings (n = 5000)
1. **More storage barely improves accuracy.** RMSE reduction from 16 to 256 KiB across all 36 estimator–family–scenario cells:
   median 0.013, maximum 0.134 (×10⁻²); 94% of it is already reached at 64 KiB. By estimator: RMPL median +0.001 (max +0.014),
   Counting Matrix +0.012 (max +0.057), Hermite Series +0.077 (max +0.134; the only one with consistent gains).
2. **Accuracy plateaus at small sizes:** RMPL from k ≈ 512 (≈ 11 KiB), Counting Matrix from ≈ 45 bins (≈ 22 KiB), Hermite from
   order ≈ 60. Validation still selects larger configurations when they fit, on differences of 0.005 or less, which is noise.
3. **The estimator ranking depends on the scenario, not the budget** (family-averaged test RMSE at 64 KiB):
   constant RMPL 1.75 / Counting Matrix 1.55 / Hermite 1.60; drift 4.31 / 4.69 / 4.72; jump 8.29 / 8.33 / 8.38.
   RMPL against the better nonparametric estimator (chosen on validation): 6 wins, 5 losses, 1 tie in 12 cells, with identical
   verdicts at 16, 64 and 256 KiB. Nonparametric estimators win under constant dependence, RMPL under drift; after jumps it depends
   on the copula family (RMPL wins Gaussian and Gumbel, loses Clayton and t₄).
4. **Processing time** (update + estimate per observation, one thread, gain 0.02, mean of 5 runs): 0.6–2.8 µs at the 16 KiB
   selections. Counting Matrix and Hermite grow roughly quadratically in their size (61.5 µs at 180 bins, 28.4 µs at order 128);
   RMPL grows roughly linearly in k (1.7 µs at k = 256, 18.4 µs at k = 8192), an implementation effect (each marginal rank query
   scans the retained sketch items). Larger, slower configurations buy no accuracy. The selected gains change times by 10% or
   less, except Counting Matrix without forgetting (about 27% faster).
5. **KLL implementations:** a hand-written KLL needs 4.5–6.0× more storage than Apache DataSketches at k ≥ 128 for the same
   accuracy (RMSE difference ≤ 0.06), but DataSketches is 1.4–3.3× slower at those sizes.
6. **RMPL's remaining gap is the cost of rank-based margins, not of sketching.** At n = 5000, k = 8192 never compresses (it keeps
   every observation), yet RMPL's RMSE equals k = 512's. With the true margins RMPL reaches 1.23 against 1.75 (Gaussian, constant).

## Long-stream check (n = 15,000; same protocol; drift spread over the longer stream)
- Sketches now compress, so RMPL can use up to 129 KiB, but its accuracy still stops improving at k ≈ 512 (≈ 12 KiB).
- More storage: median gain 0.013, maximum 0.102 (×10⁻²); RMPL max +0.016, Counting Matrix max +0.074, Hermite median +0.047.
- Family-averaged RMSE at 64 KiB (RMPL / Counting Matrix / Hermite): constant 1.20 / 1.05 / 1.08; drift 3.03 / 3.28 / 3.29;
  jump 6.29 / 6.27 / 6.31. All errors fall with n; the rankings under constant and drift are unchanged.
- RMPL against the better nonparametric estimator: 5 wins, 5 losses, 2 ties (Clayton drift becomes a tie), again identical at
  every budget.
- RMPL with true margins: 1.02 (constant) against 1.21 with sketched margins; the rank-margin gap persists.
- Caveat: the drift is 3× slower at n = 15,000, so drift and jump numbers are not directly comparable with n = 5000.

## Conclusion for the thesis
Within the tested range, accuracy is limited by the forgetting gain and the sample size, not by summary size: roughly 10–15 KiB
per estimator suffices, and larger summaries mainly add processing time. This holds at both stream lengths. Which estimator is
most accurate depends on the type of dependence change, not on the memory budget.

## Caveats
- Selected storage reflects validation choices, not each estimator's minimum requirement.
- Timings come from a shared 4-core cloud machine; absolute values vary up to about 2× between sessions, so compare within a run.
- Gains are tuned per scenario offline (an oracle no online procedure has), so this benchmarks best achievable accuracy.
