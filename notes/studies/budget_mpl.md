# Matched memory budgets: Xiao vs Hermite vs recursive MPL on KLL sketches

**Date:** 2026-11-06. **Platform:** Linux x86_64 / libstdc++, branch `bank-vff-study`.

**Code:**
- `experiments/probe_budget.cpp` is the harness. `experiments/analyze_budget.py` makes the tables
  and `experiments/plot_budget.py` makes the figures.
- `src/estimators/dependence/recursive_mpl.hpp` is the estimator, and
  `src/estimators/dependence/kll_sketch.hpp` the sketch.

**Seeds:** 100 test seeds (42–141). Tuning used 40 separate validation seeds (1000–1039).

**Figures:** `notes/thesis/figures/ch6_budget_{memory,runtime,recovery}.png`.

## 1. Design

The question: under the same streaming memory budget, which method is most accurate, adapts
fastest, and costs least time per tick?

- **Same data for every method.** Nobody skips observations, since that would confound the method
  with the amount of data it sees. Every estimator in a seed sees the identical stream.
- **The estimators:**

  | Estimator | Approximation knob | Grid | Marginals |
  | --- | --- | --- | --- |
  | Xiao | bins B | 10, 20, 30, 45, 65, 90, 130, 180 | its own (cut-points frozen after 300 obs) |
  | Hermite | order N | 5, 10, 15, 20, 30, 43, 60, 89, 128 | its own (standardised) |
  | Recursive MPL | KLL capacity k, two sketches | 8, 16, …, 1024 | KLL ranks |

- **Recursive MPL** is the Gaussian copula with a constant gain:
  r ← clip(r + g·I(r)⁻¹·s(r), ±0.999).
  - Ranks are read from the sketches *before* the pair is inserted, then clipped to
    [10⁻³, 1 − 10⁻³].
  - r starts at 0, and the copula updates only after 30 marginal observations.
- **Gain grid:** g ∈ {0.005, 0.01, 0.02, 0.05, 0.10}, used as the per-observation update weight for
  all three. That means retention 1 − g for Xiao and update weight g for Hermite.
- **Memory** is each object's measured peak `store_bytes()` over the run. KLL state includes level
  vector capacity. Xiao's state includes its 300-pair calibration buffer.
- **Budgets:** 16, 64 and 256 KiB.
  - Within a budget, (knob, gain) is chosen per family × scenario by validation RMSE, and test RMSE
    is reported.
  - Jump recovery is tuned separately, on post-change RMSE.
- **Scenarios** (N = 5000, scored on [500, 5000) every 5th tick):
  - constant ρ = 0.6;
  - drift −0.5 → 0.8;
  - jump 0.2 → 0.8 at t = 2500.
- **Families:** Gaussian, Clayton, Gumbel and t₄. Margins are standard normal and stationary.
  - For the last three, the MPL is *misspecified*: it always fits the Gaussian.
- **SEs are unpaired**, so they overstate the uncertainty of differences between methods. All
  methods see the same seeds, so paired SEs would be tighter.

## 2. Accuracy at matched memory

Test RMSE ×10⁻² at **64 KiB**, validation-tuned:

| Family | Scenario | Xiao | Hermite | MPL + KLL |
| --- | --- | --- | --- | --- |
| Gaussian | constant | 3.49 ± .05 | 3.61 ± .06 | **3.01 ± .05** |
| | drift | 6.73 ± .11 | 6.79 ± .11 | **6.26 ± .07** |
| | jump | 8.19 ± .07 | 8.25 ± .06 | **7.91 ± .07** |
| Clayton | constant | 3.63 ± .05 | 3.70 ± .05 | **3.49 ± .06** |
| | drift | 5.91 ± .10 | 5.94 ± .09 | **5.67 ± .09** |
| | jump | **8.29 ± .09** | 8.36 ± .09 | 8.51 ± .08 |
| Gumbel | constant | 3.64 ± .06 | 3.71 ± .06 | **3.32 ± .05** |
| | drift | 6.03 ± .10 | 6.03 ± .10 | **5.47 ± .09** |
| | jump | 8.31 ± .08 | 8.36 ± .08 | **8.12 ± .08** |
| t₄ | constant | 3.90 ± .05 | 4.00 ± .06 | 3.94 ± .05 |
| | drift | 6.83 ± .11 | 6.92 ± .11 | 6.71 ± .11 |
| | jump | 8.52 ± .08 | 8.54 ± .08 | 8.63 ± .08 |

- **Correctly specified, the MPL wins.** On Gaussian data it beats Xiao by 4–14% in every scenario.
- **Misspecified, it still wins or ties on Clayton and Gumbel**, except Clayton `jump`, where it
  is 0.22 behind.
  - On t₄ the three are within about 1–2 SE of each other.
  - Its fixed Gaussian family costs bias, which the recovery figure shows as a lower settled level
    on Clayton, but this is offset by lower variance.
- **At 16 KiB the MPL is held to k = 64**, and the advantage shrinks or reverses:
  - Gaussian: 3.35 / 6.43 / 8.05 against Xiao's 3.49 / 6.75 / 8.22.
  - Clayton: 3.91 / 5.93 / 8.74 against 3.64 / 5.93 / 8.29.
- **Xiao and Hermite do not use the extra memory.** Their curves are flat beyond about 8–16 KiB
  (B ≈ 20–30, N ≈ 30–43), and 256 KiB moves no cell by more than 0.05.
- **Only the MPL turns memory into accuracy.**
  - Its KLL rank error halves each time k doubles, until it meets the exact-PIT oracle at about
    k = 256–512 (25–60 KiB).
  - Below k = 32 (about 8 KiB) it is the worst method by a wide margin.

**Caveat on stream length.** At N = 5000, storing every observation exactly costs 2 × 40 KB. KLL
at k = 1024 (82 KB) is already larger than that. KLL's advantage is that its state stays bounded
as the stream grows, which a 5000-tick test cannot show.

## 3. Recovery after the jump (64 KiB, tuned on post-change RMSE)

| Family | Xiao post-RMSE / delay | Hermite | MPL + KLL |
| --- | --- | --- | --- |
| Gaussian | 9.83 / 50 | 10.20 / 50 | **7.84 / 45** |
| Clayton | 10.23 / 50 | 10.61 / 45 | **9.86 / 45** |
| Gumbel | 10.49 / 45 | 10.78 / 45 | **9.06 / 45** |
| t₄ | 10.68 / 45 | 11.23 / 115 | **9.38 / 45** |

- **Post-RMSE** is over the 1000 ticks after the change (×10⁻²).
- **Delay** is the number of ticks until the mean path covers 90% of its own pre → settled change.
  Measuring against its own settled level means a biased estimator isn't scored as "never
  recovering".
- **All three pick gain 0.05 and recover in 45–50 ticks.** Hermite on t₄ is the exception: it
  picks 0.02 and takes 115 ticks.
- **The MPL's lower post-change error is variance, not speed.** At the same gain its path is less
  noisy.

## 4. Same gain, different memory

At equal g, the MPL's RMSE is 20–25% lower on `constant`: 3.01 against Xiao's 3.49 at g = 0.005,
and 13.0 against 17.2 at g = 0.10.

The same g does not mean the same statistical memory:
- the MPL averages an efficient score;
- Xiao and Hermite average noisier per-observation contributions.

So "common gain" comparisons favour the MPL. The validation-tuned table in §2 is the fair one.

## 5. Marginal ablation (MPL, val-best gain; test RMSE ×10⁻²)

| Family · scenario | exact PIT | rank window 500 | KLL 16 | KLL 64 | KLL 256 | KLL 1024 | registered `CopulaFit` on KLL 256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian constant | 2.85 | 3.07 | 6.98 | 3.35 | 3.02 | 3.01 | 2.94 |
| Gaussian drift | 6.22 | 6.30 | 7.91 | 6.43 | 6.26 | 6.25 | 6.29 |
| Gaussian jump | 7.87 | 7.92 | 9.67 | 8.05 | 7.91 | 7.91 | 7.77 |
| Clayton jump | 8.45 | 8.38 | 10.89 | 8.74 | 8.52 | 8.50 | 8.47 |
| t₄ drift | 6.68 | 6.74 | 8.27 | 6.89 | 6.72 | 6.70 | 6.82 |

All 12 rows are in `experiments/results/budget_tables.txt`.

- **The cost of approximating the margins:**
  - KLL with k ≥ 256 is within 1–2 SE of exact PIT everywhere except Gaussian `constant`
    (+0.17, about 3 SE). That is where variance is smallest, so a marginal error shows most.
  - k = 64 costs 0.15–0.5.
  - k = 16 roughly doubles the error.
- **KLL against the window:** the sliding rank window of 500 (16 KB, O(W)) performs about as well
  as KLL 256 (37 KB).
- **The two recursions are interchangeable.** The registered decaying-information recursion
  (`CopulaFit`, `expected_info = true`, retention 1 − g) and the spec's constant-gain recursion
  differ by ≤ 0.15. So "the registered MPL" and "the constant-gain recursive MPL" can be described
  as one algorithm.

## 6. Time per tick

`./probe_budget time`: Gaussian drift, one core, 5 seeds × 4500 ticks, update + estimate per tick.

| Estimator | Setting | Peak state | Mean | p95 |
| --- | --- | --- | --- | --- |
| Xiao | B = 20 / 30 / 65 / 180 | 8.3 / 12.5 / 39.6 / 267 KB | 0.45 / 0.86 / 4.2 / 49 µs | 0.47 / 1.0 / 4.7 / 65 µs |
| Hermite | N = 20 / 43 / 89 / 128 | 3.9 / 16.2 / 66 / 135 KB | 1.0 / 2.7 / 10.6 / 21 µs | 1.0 / 2.8 / 11.0 / 24 µs |
| MPL + KLL | k = 64 / 256 / 512 / 1024 | 13.9 / 37.5 / 57.8 / 82 KB | 0.61 / 0.98 / 1.35 / 1.9 µs | 1.0 / 1.0 / 1.6 / 2.5 µs |
| MPL, exact PIT | — | 72 B | 0.14 µs | 0.17 µs |
| MPL, rank window 500 | — | 16.2 KB | 0.41 µs | 0.46 µs |

- **Xiao is O(B²) per tick**: every update multiplies all B² counts by λ, and estimate() sums
  over them. Hermite is O(N²).
- **The MPL's cost is the KLL rank query**, which scans the retained items. It grows slowly with k.
  The p95 shows KLL compactions as occasional spikes of about 1.6× the mean.
- **At the configurations validation picked**, all three cost under 2 µs per tick. Time separates
  them only when Xiao or Hermite are pushed to large B or N, which buys them nothing.

## 7. What this supports, and what it does not

**Supported:**
- Under a fixed memory budget of about 25 KiB or more, the recursive Gaussian MPL on two KLL
  sketches is the most accurate of the three:
  - on Gaussian data;
  - on Clayton and Gumbel data, except Clayton `jump`;
  - including after a change.
- On t₄ it ties.
- It is the only one of the three whose accuracy improves with memory.

**Not supported, or not tested:**
- **Non-stationary margins.** KLL is cumulative and would need forgetting.
- **Tail functionals.** A Gaussian MPL cannot represent a change in tail shape.
- **Streams longer than 5000 ticks**, where KLL's bounded state would matter.
- **The bank-level comparison**, where K MPL arms share one pair of sketches: state is O(S + K),
  against K·B² for Xiao.

**Bug found along the way.** `copula::spearman_from_param` builds the Student-t table behind plain
statics, which is a data race under threads.
- In the first run it corrupted every t cell (RMSE ≈ 0.6).
- The harness now builds the table before spawning threads, and the t family was rerun.
- `tail_validate.cpp` (std::async over t streams) may be exposed. A fix task is queued.

## 8. Reproduction

```bash
cd experiments && mkdir -p results && make -B probe_budget EIGEN=/usr CXX=g++
./probe_budget sweep 100 40        # ~14 min on 4 cores -> results/budget_sweep.csv, budget_paths.csv
./probe_budget time                # ~1 min, run on an idle machine -> results/budget_timing.csv
cd .. && python3 experiments/analyze_budget.py experiments/results/budget_sweep.csv
python3 experiments/plot_budget.py
```

`./probe_budget sweep 100 40 t` reruns one family into `budget_sweep_t.csv`.
