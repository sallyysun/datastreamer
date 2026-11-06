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

## 9. Time complexity, derived from the code and checked against §6

**Notation:**
- D = N + 1 is the number of Hermite basis functions;
- B is Xiao's bins per margin;
- k is the KLL capacity, and S is the number of items a sketch retains;
- n is the stream length so far;
- K is the number of arms in a bank.

Costs are per tick, for update() followed by estimate(), since the online output is produced every
tick.

### 9.1 Per-tick cost

| Estimator | update() | estimate() | Per tick | Memory |
| --- | --- | --- | --- | --- |
| Xiao, forgetting λ | 2 binary searches O(log B); rescale every count by λ: **B² + 2B** multiplies | mid-ranks O(B); covariance double loop **B²** | **Θ(B²)** | Θ(B²), plus Θ(C) calibration pairs until t = C |
| Hermite, order N | basis recursion O(D); outer product hxhyᵀ **D²**; ‖outer − A‖ **D²**; A ← A + w(outer − A) **D²** | four D × D matrix–vector products (W·ax, W·ay, A·v, A·z) **4D²** | **Θ(N²)** | Θ(N²) |
| MPL + KLL, as implemented | 2 rank queries, each a scan of S items: **O(S)**; 2 inserts, amortised O(log k + log(n/k)), worst case O(k log k) at a compaction; copula step O(1) | O(1) (arcsin) | **O(S) = O(k + log(n/k))** | O(k + log(n/k)) |
| MPL + rank window W | sorted-vector insert and erase O(W), plus a front erase of the FIFO O(W) | O(1) | **O(W)** | Θ(W) |
| MPL, exact PIT (oracle) | O(1) | O(1) | **O(1)** | O(1) |

**KLL's retained size.** Level capacities are k(2/3)^d, with at least 2 per level, and there are
H = O(log(n/k)) levels. So
S ≤ k·Σ_d (2/3)^d + 2H ≤ 3k + O(log(n/k)).

That bound does not depend on n beyond the logarithm. This is the property a sliding window or an
exact store lacks: their state is Θ(W) or Θ(n).

**Where the time goes in each estimator:**
- **Xiao: forgetting, not counting.** Without forgetting, update() is O(log B). Multiplying the
  whole B × B table by λ on every tick is what makes it Θ(B²).
  - A global scale factor would bring update() down to O(log B): store counts divided by λᵗ, and
    renormalise when the factor underflows.
  - estimate() would still be Θ(B²), because inserting one point shifts every mid-rank.
  - An incremental estimate using 2-D Fenwick trees, as in online Kendall's τ, looks possible in
    O(log² B). It is not implemented or proven here.
- **Hermite: the stored matrix.** Θ(D²) is a lower bound for any update that stores A explicitly,
  since a rank-one update touches every entry.
  - estimate() is also Θ(D²): it is a quadratic form in a matrix that changes on every tick.
  - This matches Stephanou & Varughese's "constant in n" claim, which holds for fixed N.
- **MPL: the rank query, not the copula step.**
  - Keeping each level sorted and binary-searching it would make the query O(H log k) =
    O(log(n/k)·log k).
  - A merged, weighted, sorted view rebuilt at each compaction would make it O(log S).

**Total over a stream:**
- n·Θ(B²) for Xiao;
- n·Θ(N²) for Hermite;
- n·O(k + log(n/k)) for MPL + KLL.

All three are linear in n, which is what makes them streaming-admissible. They differ in the
constant multiplying n, and in what that constant scales with.

**Banks of K arms:**
- **K Xiao arms:** Θ(K·B²) time and memory.
- **K Hermite arms:** Θ(K·N²) time and memory.
- **K MPL arms sharing two sketches:**
  - **Time:** Θ(S + K) per tick. The rank query is done once per margin and its two
    pseudo-observations feed all K arms, each of which then takes an O(1) score step.
  - **Memory:** O(k + log(n/k) + K).

### 9.2 Checked against the measured times (§6)

Log-log slopes of time per tick, over the upper half of each grid:

| Estimator | Predicted exponent | Measured: update | Measured: update + estimate |
| --- | --- | --- | --- |
| Xiao, in B (B ≥ 30) | 2 | **1.99** | 2.35 |
| Hermite, in D (N ≥ 20) | 2 | 1.72 | 1.71 |
| MPL + KLL, in k (k ≥ 128) | 1 in S, below 1 in k at this n | 0.48 | 0.47 |

- **Xiao's update has the predicted slope exactly.** The total is steeper than 2 at large B because
  the table outgrows the cache: at B = 180 it is 259 KB, more than L2 holds. estimate()'s double
  loop then becomes memory-bound, giving 24 µs at B = 130 and 40 µs at B = 180.
- **Hermite is still approaching its asymptote.** At these sizes the O(D) basis recursion (an
  exp and two square roots per term) and a heap allocation of the D × D outer product every tick
  still matter. The local exponent is already 2.2 between N = 89 and N = 128.
- **The MPL is linear in S, not in k.** At n = 5000 a sketch retains S ≈ 1.4k at small k, falling
  to 1.4k/k ≈ 1.4 at k = 1024 (S = 1417), because n is too short to fill the sketch. There is also a
  fixed cost of about 500 ns: probit, clipping, the score step, and timer overhead.
  - The cost above the k = 8 baseline is 1305 ns for two scans of 1417 items, about 0.46 ns per
    item. That is a linear, vectorised scan.
- **Bounded growth in n**, measured directly at k = 256:

  | n | 10³ | 10⁴ | 10⁵ | 10⁶ |
  | --- | --- | --- | --- | --- |
  | S | 382 | 662 | 743 | 763 |
  | query | 301 ns | 484 ns | 575 ns | 598 ns |
  | insert, amortised | 79 ns | 131 ns | 170 ns | 235 ns |

  - S levels off at about 3k, as the bound says.
  - The amortised insert grows like log(n/k). That growth is the compaction cascade, plus
    `total_capacity()` being recomputed in O(H) with a `pow` call on every insert.

### 9.3 A memory-accounting gap this turned up

Hermite's `store_bytes()` counts (2D + D² + 6) doubles, but each instance actually allocates three
D × D matrices:
- A_;
- W_, the integral matrix, which is constant for a given order and could be shared;
- A_fast_, which is allocated even when `surprise_fast_rate` = 0, so it is pure waste in that case.

So Hermite's true footprint is about 3× what the memory figures show. N = 43 is about 48 KB, not
16 KB.

This barely changes the budget result, because Hermite's accuracy is flat beyond N ≈ 20–30.
- Under honest counting, its 16 KiB choice would drop from N = 43 to N ≈ 20–30.
- That costs at most about 0.1 in RMSE, by the frontier in the memory figure.

It should still be fixed before the figure goes in the thesis. Either count W_ and A_fast_ (or
mark W_ as a shared constant), or stop allocating A_fast_ when it is unused. Fixing either changes
the store_bytes() of every registered Hermite key, so it is a decision for the thesis audit, not
for this note.

## 8. Reproduction

```bash
cd experiments && mkdir -p results && make -B probe_budget EIGEN=/usr CXX=g++
./probe_budget sweep 100 40        # ~14 min on 4 cores -> results/budget_sweep.csv, budget_paths.csv
./probe_budget time                # ~1 min, run on an idle machine -> results/budget_timing.csv
cd .. && python3 experiments/analyze_budget.py experiments/results/budget_sweep.csv
python3 experiments/plot_budget.py
```

`./probe_budget sweep 100 40 t` reruns one family into `budget_sweep_t.csv`.
