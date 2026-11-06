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
- **The MPL's lower post-change error is not faster reaction**: its delay is the same as the
  others'. Whether the difference is variance or bias is untested. The harness stores only mean
  paths, so a bias/variance split, and paired seed-by-seed CIs, need a rerun that writes per-seed
  output.

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

## 10. The marginal sketch from Apache DataSketches (2026-11-06)

**What changed:**
- `external/datasketches-cpp` is now a submodule, pinned at `c2f89d7`.
- `src/estimators/dependence/ds_margin.hpp` wraps its `kll_sketch<float>` (and `req_sketch`) behind
  the same interface as `KllSketch`.
- `RecursiveMPL::Margin::DsKll` selects it, and `probe_budget` runs it as `mpl_dskll`.

**Three integration details:**
- **Reproducibility.** DataSketches flips its compaction coins with a `thread_local` generator
  seeded from the clock plus the thread id.
  - There's no data race, but runs are not reproducible.
  - `override_seed()` reseeds a different generator.
  - `ds_margin::seed()` reseeds `random_utils::random_bit` when each MPL is built. Two runs with
    the same seed give identical rank sequences (checked).
- **Rank query.** `get_rank()` rebuilds a full sorted view after every update. The wrapper instead
  sums the weights of retained items below x in one pass over the sketch's iterator: O(S), no
  allocation.
- **Memory.** `store_bytes()` uses a counting allocator, so it reports the heap the sketch actually
  holds, not its serialised size.

**Marginal sketch alone** (n = 5000, N(0,1) stream, 20 seeds; rank error against the exact
empirical CDF):

| Sketch | k | mean \|err\| | tail \|err\| (rank < .05 or > .95) | Peak bytes | ns per query + insert |
| --- | --- | --- | --- | --- | --- |
| own KLL | 64 / 256 | 0.0059 / 0.0016 | 0.0056 / 0.0015 | 6 920 / 18 696 | 337 / 493 |
| DataSketches KLL | 64 / 256 | 0.0060 / 0.0015 | 0.0053 / 0.0015 | **908 / 2 848** | 309 / 575 |
| DataSketches REQ (HRA, k/2) | 64 / 256 | 0.0011 / 0.0002 | 0.0009 / 0.0002 | 11 688 / 29 960 | 1 152 / 2 102 |

- **Same accuracy as my KLL, 6.5–7.6× less memory.** The difference is floats in one contiguous
  buffer, against doubles in growable vectors.
- **REQ is about 6× more accurate at the same k, but larger and 2–4× slower.** At matched memory,
  KLL with a larger k is as accurate at n = 5000.
  - REQ's high-rank-accuracy mode protects only the upper tail. Both tails would need a second
    sketch fed −x.

**Inside the MPL** (20 seeds, test RMSE; a quick check, not the budget run):

| Case | exact PIT | own k=64 | DataSketches k=64 | own k=256 | DataSketches k=256 |
| --- | --- | --- | --- | --- | --- |
| Gaussian constant, g = .005 | .0282 | .0335 | .0345 | .0303 | .0303 |
| Clayton jump, g = .01 | .0870 | .0900 | .0920 | .0878 | .0880 |
| Peak state (both margins) | 72 B | 13.9 KB | **1.9 KB** | 37.5 KB | **5.8 KB** |

**Time per tick** (`./probe_budget time`, kept in `results/budget_timing_ds.csv`):
- MPL on DataSketches KLL: 0.81 µs at k = 64, 1.65 µs at k = 256.
- MPL on my KLL in the same run: 0.73 µs and 1.11 µs.
- The machine was noisier in this run (Xiao B = 30 read 1.35 µs against 0.86 in §6), so compare
  within the run only.
- DataSketches is a little slower at large k. The likely cause is the iterator's pair proxy, but
  that is not profiled.

**What this changes in §7.**
- With DataSketches KLL at k = 256, the standalone MPL holds **5.8 KB**, less than Xiao at its
  chosen B = 20–30 (8–12 KB). Its ρ_S accuracy is unchanged. So "the MPL takes more memory" was a
  property of my sketch, not of the method.
- In a bank, the shared marginal state would fall from about 16 KB (two rank windows of 500) to
  about 6 KB.
- The matched-budget tables in §2 are from my KLL. With DataSketches KLL, the MPL would reach its
  plateau inside the 16 KiB budget, where my sketch held it to k = 64.
  - That needs the budget rerun to confirm: `mpl_dskll` is already in `probe_budget`'s grid.

### 10.1 Budget rerun with DataSketches KLL (100 test / 40 val seeds)

**Done, 2026-11-06.** Same seeds as §3.3 (100 test, 40 validation). The own-KLL, Xiao and Hermite
columns reproduce §3.3 exactly, so the harness is deterministic.
- **Tables:** `experiments/results/budget_tables.txt`.
- **Figures:** `ch6_budget_memory.png` (Gaussian + Clayton) and `ch6_budget_memory_all.png` (all
  families). DataSketches is the yellow series.

**Test RMSE ×10⁻² at 16 KiB** (DataSketches picks k = 256–512, using 5.8–10.3 KB):

| Family | Scenario | Xiao | Hermite | MPL + own KLL (k = 64) | MPL + DataSketches KLL |
| --- | --- | --- | --- | --- | --- |
| Gaussian | constant | 3.49 | 3.61 | 3.35 | **3.01** |
| | drift | 6.75 | 6.79 | 6.43 | **6.25** |
| | jump | 8.22 | 8.25 | 8.05 | **7.91** |
| Clayton | constant | 3.64 | 3.75 | 3.91 | **3.45** |
| | drift | 5.93 | 5.94 | 5.93 | **5.64** |
| | jump | **8.29** | 8.33 | 8.74 | 8.49 |
| Gumbel | constant | 3.64 | 3.74 | 3.53 | **3.34** |
| | drift | 6.07 | 6.03 | 5.83 | **5.49** |
| | jump | 8.32 | 8.36 | 8.27 | **8.12** |
| t₄ | constant | **3.91** | 4.06 | 4.04 | 3.94 |
| | drift | 6.84 | 6.92 | 6.89 | **6.72** |
| | jump | **8.52** | 8.64 | 8.75 | 8.63 |

- **With DataSketches, the MPL reaches its plateau inside 16 KiB.** Its 16 KiB numbers equal its
  64 and 256 KiB numbers. So the budget stops mattering for all three estimators.
- **What remains is the method difference:**
  - The MPL is best in 9 of 12 cells.
  - Xiao is best on Clayton jump (by 0.20, about 1.6 unpaired SE), t₄ constant (0.03) and t₄ jump
    (0.11).
  - The t₄ gaps are within noise.
- **Rank error isn't the binding constraint.** In Gaussian drift at g = .01, the RMSE against peak
  memory is:

  | Memory | 1.1 KB (k = 8) | 1.9 KB (k = 64) | 5.8 KB (k = 256) | 10.3 KB (k = 512) |
  | --- | --- | --- | --- | --- |
  | RMSE | .0875 | .0644 | .0626 | .0625 |

  The exact-PIT oracle is .0622, so beyond about 6 KB the remaining error isn't marginal
  approximation.
- **Recovery after the jump** (16 KiB; post-change RMSE ×10⁻², delay to 90%):

  | | Xiao | MPL + DataSketches |
  | --- | --- | --- |
  | Gaussian | 9.83 / 50 | **7.84 / 45** |
  | Clayton | 10.23 / 50 | **9.81 / 45** |
  | Gumbel | 10.51 / 45 | **8.97 / 45** |
  | t₄ | 10.71 / 45 | **9.39 / 45** |

  Hermite is 10.2–11.4.
- **Marginal ablation:** DataSketches KLL 256 matches own KLL 256 within 0.02 in all 12 cells.
  At k = 64 the two are equally coarse.
- **Time per tick, idle machine** (`results/budget_timing.csv`, re-timed after the run; supersedes
  §3.3's timing table):

  | | Peak state | Time per tick |
  | --- | --- | --- |
  | MPL + DataSketches, k = 64 / 256 / 1024 | 1.9 / 5.8 / 17.7 KB | 0.64 / 1.24 / 3.7 µs |
  | MPL + own KLL, k = 64 / 256 / 1024 | 13.9 / 37.5 / 82 KB | 0.66 / 1.0 / 1.96 µs |
  | Xiao, B = 20 / 30 / 180 | | 0.52 / 1.15 / 63 µs |
  | Hermite, N = 30 / 43 | | 1.9 / 3.1 µs |
  | exact PIT | | 0.14 µs |
  | window 500 | | 0.44 µs |

  DataSketches is somewhat slower per tick at large k (iterator overhead), but every configuration
  that validation picks costs under 2.5 µs.

**Bottom line.** Implemented well, the recursive MPL on rank sketches is both *smaller* than Xiao
(5.8 KB against 8–12 KB) and more accurate for ρ_S in most cells. The exceptions are Clayton jump
and t₄, where Xiao is as good or slightly better.

## 11. Spec vs code, and the MPL bank switched to the spec (2026-11-06)

### The agreed specification

Two online marginal summaries give paired pseudo-observations. These feed a recursive Gaussian
copula score update, which gives the fitted copula and the Spearman estimate.

1. **Pseudo-observations:** Û = T_δ(F̂_{X,n−1}(X_n)), V̂ likewise, with z = Φ⁻¹(Û, V̂).
2. **Score:** s(r) = r/(1−r²) + [(1+r²)z₁z₂ − r(z₁²+z₂²)]/(1−r²)².
3. **Fisher information:** I(r) = (1+r²)/(1−r²)².
4. **Update:** r̂_n = Π_{±0.999}[r̂_{n−1} + λ·I(r̂_{n−1})⁻¹·s_n(r̂_{n−1})].
5. **Outputs:** ρ̂ = (6/π) arcsin(r̂/2) and Ĉ = C^Gauss_{r̂}.

### Line-by-line check before the switch

**`RecursiveMPL`, constant-gain mode (the budget study): matches the spec.**
- Ranks are queried before insertion and are strict-less ranks over n. They are clipped to
  δ = 10⁻³, passed through the probit, and use the same score, information, update, clamp and
  output.
- **Differences:**
  - The copula updates only after 30 marginal observations.
  - No prequential score is computed inside the estimator.
  - Ĉ is mathematical only: no CDF method, so no ISE.
  - The `RankWindow` margin uses (below+1)/(n+2), not below/n.

**The old `mle_bank*` arms (`ParametricCopulaMLE` → `CopulaFit`): a different recursion.**
1. **Parameter:** φ = atanh r̂, with each step capped at |Δφ| ≤ 0.5.
2. **Discounted information:** Δφ = s_φ/H_t, with H_t = λ_ret·H_{t−1} + (1 + r̂²) and H₀ = 1.
   - In steady state Δr ≈ (1 − λ_ret)·I(r)⁻¹·s_r, which is the spec with λ = 1 − λ_ret.
   - At start-up the gain is about 1/t.
3. **A second marginal step:** the bank's window ranks (below+1)/(n+2) are passed as z = Φ⁻¹(u).
   Each arm then re-standardised z with its own cumulative mean and variance:
   u′ = Φ((z − m̂)/ŝ), clipped to [10⁻⁹, 1 − 10⁻⁹].
4. **Start-up:** the arms updated from tick 1, and reported 0 until 51 observations.

The §5 ablation (`CopulaFit` on KLL within 0.15 ×10⁻² of constant gain) tested points 1–2
numerically, not 3–4.

### The change (commit `40ebc21`)

- **The `mle_bank`, `_log`, `_own`, `_sq32` and `_uniform` arms are now `RecursiveMPL`.**
  - Constant gain λ ∈ {0.05, 0.02, 0.01, 0.005, 0.001}, i.e. the old 1 − λ_ret, so the memories
    are still 39 to 1999 ticks.
  - The margin is `ExactNormal`. The bank passes z = Φ⁻¹(u), and Φ(z) recovers the bank's u
    exactly, so there is no second standardisation.
- **`RecursiveMPL` implements `PredictiveDensity`** (the Gaussian copula at r̂), which
  `mle_bank_own` needs.
- **Builds without DataSketches still work.** The DataSketches margin is compiled only when its
  headers are on the include path (`__has_include`), so `experiment`, `multiseed` and the
  consumer build without it (checked).
- **Not changed:**
  - `mle_bank_block` (`GaussianCopulaMLE` arms).
  - The MPL variable-forgetting comparator in `probe_thesis_bank.cpp`, which still uses
    `ParametricCopulaMLE` with `vff_rate`.

**5-seed smoke test** (Gaussian jump, RMSE; new arms against old):

| | new | old |
| --- | --- | --- |
| squared bank | .0681 | .0660 |
| log bank | .0605 | .0570 |
| own-density bank | .0652 | .0647 |
| sq32 bank | .0786 | .0794 |
| uniform | .0868 | .0855 |
| memory | 16.9 KB | 18.8 KB |

### Consequences

- **The MPL-bank rows in `docs/chapter6_results.md` and the Chapter 6 draft are stale.**
  `probe_thesis_bank` builds its fixed arms from the same registry keys, so rerunning it
  regenerates them. The Xiao and Hermite rows do not move.
- **Start-up handicap, open decision.**
  - The spec's gain is constant from the first observation, starting at r̂ = 0. The gain-0.001 arm
    therefore needs about 2000 ticks to warm up. The old arms started like a running mean.
  - The proposed fix is λ_n = max(λ, 1/n). It is identical to the spec after about 1/λ
    observations, and is the same rule as Hermite's `init_first_obs`.
  - It is not implemented: waiting on Sally.
- **The η and δ tuned in `probe_bank_tune` were tuned with the old arms.** They are kept.

## 12. MPL bank: start-up rule and the old arms, paired (2026-11-06)

**Run:** `probe_thesis_bank` with `COMPONENTS=mle,mlea,mleold TAG=startup`, 4 families × 60 seeds
(42–101), the same streams as Chapter 6. Output: `results/thesis_bank_startup[_fam].csv`, tables in
`experiments/startup_table.py` → `results/startup_table.txt`. Single-threaded, so the t-table race
does not apply.

**The three variants:**
- **(b) `mle`** (the thesis version): constant-gain RecursiveMPL with start-up gain max(λ, 1/m).
- **(a) `mlea`:** constant gain from the first update.
- **`mleold`:** the previous `ParametricCopulaMLE` arms.

**Sanity:** `mleold` reproduces the Chapter 6 MPL numbers exactly (log bank, Gaussian: 67.0 drift
and 59.6 jump).

**Log bank, RMSE ×10⁻³:** paired difference, mean ± 2 SE; * = beyond 2 SE.

| Scenario, window | Family | (b) | (a) | old | (b) − old | (a) − old | (b) − (a) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| drift, whole | Gaussian | 67.3 | 67.5 | 67.0 | +0.3 ± 0.7 | +0.5 ± 1.0 | −0.2 ± 0.8 |
| | Clayton | 57.7 | 56.5 | 60.2 | −2.5 ± 0.4* | −3.7 ± 0.4* | +1.2 ± 0.4* |
| | Gumbel | 58.8 | 57.6 | 61.0 | −2.2 ± 0.5* | −3.4 ± 0.4* | +1.2 ± 0.3* |
| | t₄ | 77.8 | 78.2 | 77.0 | +0.8 ± 0.7* | +1.3 ± 1.1* | −0.4 ± 1.0 |
| jump, whole | Gaussian | 58.6 | 62.7 | 59.6 | −1.1 ± 0.5* | +3.1 ± 0.5* | −4.2 ± 0.5* |
| | Clayton | 65.8 | 69.9 | 68.4 | −2.6 ± 0.6* | +1.6 ± 0.6* | −4.1 ± 0.5* |
| | Gumbel | 62.7 | 66.7 | 63.6 | −0.9 ± 0.7* | +3.2 ± 0.7* | −4.1 ± 0.5* |
| | t₄ | 70.4 | 74.8 | 70.2 | +0.2 ± 0.6 | +4.6 ± 0.6* | −4.4 ± 0.5* |
| jump, early [500, 2000) | all four | 43.7–54.6 | 57.5–68.0 | 42.4–53.6 | +0.1 to +1.4 (n.s.) | +13.6 to +15.9* | −13.4 to −14.5* |
| jump, post-change | Gaussian / Clayton / Gumbel / t₄ | 68.4 / 79.0 / 73.9 / 81.5 | 67.8 / 78.6 / 73.2 / 81.5 | 70.0 / 83.4 / 76.2 / 81.6 | −1.6* / −4.4* / −2.3* / −0.1 | −2.2* / −4.9* / −2.9* / −0.2 | +0.6* / +0.5* / +0.6* / +0.1 |
| constant, whole | all four | 24.3–35.1 | 37.0–48.6 | 27.5–37.2 | −1.0 to −4.9 (3 of 4 *) | +9.4 to +11.4* | −11.4 to −14.3* |

**What it shows:**
1. **The five-seed difference was start-up, not the update rule.**
   - (a) loses to the old arms by 13–16 in the first 2000 ticks of `jump`, and by 9–11 over a whole
     `constant` run.
   - (b) removes all of that.
2. **(b) is as good as or better than the old arms in 11 of 12 whole-run cells:**
   - 8 better beyond 2 SE;
   - 3 ties (Gaussian drift, t₄ jump, Gumbel constant).
   - The exception is t₄ drift, worse by 0.8 ± 0.7.
   - After the change it is better in 3 of 4 families (−1.6 to −4.4).
3. **(a) is slightly better than (b) on Clayton and Gumbel `drift`** (−1.2). That's an artefact:
   those paths start at the independence edge (Clayton and Gumbel can't go negative, so the clipped
   drift begins at ρ_S ≈ 0), which is exactly where (a)'s r̂ = 0 starts.
4. **One genuine rule difference, in the slowest arm alone.**
   - On `jump`, the M = 1999 arm is 10–12 worse under (b) than under the old arms over the whole run,
     with no difference in the early window.
   - The likely cause: the old arm divides by *accumulated* information H = Σ λʲ I(r̂_{t−j}). After a
     jump upward, H still reflects the smaller information at the old level, so the old arm takes
     larger steps. (b) uses the current I(r̂)⁻¹, which is smaller at r̂ ≈ 0.8.
   - The bank barely weights that arm after a jump, so the bank is unaffected.

**Bank against fixed memory under (b)** (log bank, whole run, paired, ×10⁻³):

| | Gaussian | Clayton | Gumbel | t₄ |
| --- | --- | --- | --- | --- |
| drift: minus best fixed (M) | +4.0 ± 0.7 (199) | +3.1 ± 0.9 (399) | +4.1 ± 1.2 (399) | +7.4 ± 1.4 (399) |
| drift: minus M = 199 | +4.0 ± 0.7 | −4.6 ± 1.6 | −4.3 ± 1.6 | +3.3 ± 1.4 |
| jump: minus best fixed (M = 199) | −20.2 ± 1.2 | −17.8 ± 1.4 | −18.5 ± 1.3 | −15.4 ± 1.9 |

**The same pattern as the old arms in `chapter6_results.md`** (old drift +3.1 / +2.5 / +3.4 / +7.4;
old jump −18.5 / −16.0 / −16.9 / −15.5):
- it beats the hindsight-best fixed memory on every jump;
- it loses to it on every drift;
- against M = 199 on drift it is better on Clayton and Gumbel, and worse on Gaussian and t₄.

**The Chapter 6 bank conclusions are unchanged by the switch.** The `chapter6_results.md` MPL rows
are still to be regenerated from the (b) arms. That is waiting on Sally before any rewrite.

## 13. Mixed bank: Xiao + Hermite + MPL arms together (2026-11-06)

**Setup.**
- Key `mixed_bank*` (registry; not in `all_estimator_keys()`): the 5 Xiao, 5 Hermite and 5 spec-MPL
  arms (with the start-up rule) of the single-estimator banks, K = 15. Same η and δ as each loss
  variant, and the shared 500-tick rank window.
- Run: `probe_thesis_bank` with `COMPONENTS=mixed TAG=mixed`, 4 families × 60 seeds (42–101), the same
  streams as the regenerated Chapter 6 banks.
- Tables: `experiments/mixed_table.py` → `results/mixed_table.txt`. Weight shares are from a
  20-seed side run.

**Log bank, whole-run RMSE ×10⁻³.** Mixed, and mixed minus the best single bank (paired ± 2 SE;
* = beyond 2 SE):

| | Gaussian | Clayton | Gumbel | t₄ |
| --- | --- | --- | --- | --- |
| drift | 68.1, +0.7 ± 0.8 vs MPL | 60.2, **+2.5 ± 0.7*** vs MPL | 61.3, **+2.5 ± 0.7*** vs MPL | 75.6, +0.1 ± 1.0 vs Xiao (−2.2* vs MPL) |
| jump | 59.5, **+1.0 ± 0.6*** vs MPL | 65.4, −0.5 ± 0.9 vs MPL | 62.5, −0.2 ± 0.8 vs MPL | 69.4, **−1.0 ± 0.7*** vs MPL |
| constant | 24.7, +0.4 ± 0.8 | 25.8, +1.4 ± 0.8* | 28.8, +0.7 ± 1.0 vs Xiao | 35.3, +1.4 ± 1.3* vs Xiao |
| oscillate | 185.0, +0.3 ± 0.4 | 169.9, −1.1 ± 0.7* | 171.5, −0.2 ± 0.8 | 204.1, +0.8 ± 0.9 vs Xiao (−12.5* vs MPL) |

- **Against Xiao's bank:** better in 13 of 16 cells.
- **Against the best single bank:**
  - better in 2 cells (t₄ jump, Clayton oscillate);
  - worse in 5 (Clayton and Gumbel drift, Gaussian jump, Clayton constant, t₄ constant), by 1.0–2.5;
  - tied in the rest.
- **The squared bank** shows the same pattern: worse than the MPL bank by up to +3.2 on drift.

**Mean weight by estimator type, log bank:**

| | Gaussian | Clayton | Gumbel | t₄ |
| --- | --- | --- | --- | --- |
| drift: Xiao / Hermite / MPL | .15 / .40 / .44 | .38 / .31 / .31 | .36 / .33 / .31 | .27 / .45 / .29 |
| jump: Xiao / Hermite / MPL | .28 / .25 / .46 | .34 / .31 / .35 | .31 / .30 / .39 | .34 / .35 / .32 |

**What it shows.**
- **The bank cannot tell the estimator types apart.**
  - All three predict nearly the same ρ_S from the same pseudo-observations, so their per-tick
    losses barely differ.
  - The weight spreads about ⅓ each, even onto Hermite, which is never the best single bank.
  - The bank still separates *memories* (fast against slow arms) but not *estimators*.
- **So the mixed bank behaves like an average across estimator types.**
  - It loses a little where the MPL is clearly best: Clayton and Gumbel drift, +2.5.
  - It gains where the MPL is weak: t₄ drift (−2.2 vs MPL), and t₄ oscillate (−12.5 vs MPL).
- **It is a hedge, not an improvement.** It is never far from the best single bank (at most +2.5),
  and it removes the MPL bank's few bad cells.
- **Cost:** 15 arms, about 42 KB and about 8–9 µs per tick, against about 17 KB and about 2 µs for
  the MPL bank alone.

**For the thesis:**
- Combining estimator types adds robustness, not accuracy.
- Weighting by one-step loss separates memories, because a wrong memory is wrong for many ticks.
  It cannot separate estimators whose predictions differ by less than the per-tick noise.

## 14. Forgetting against no forgetting (2026-11-06)

**Setup.**
- Probe: `experiments/probe_noforget.cpp`; table: `experiments/noforget_table.py` →
  `results/noforget_table.txt`.
- 4 families × 60 seeds (42–101), on the `probe_thesis_bank` streams. The sanity check matches
  `chapter6_results.md`: Xiao log bank, Gaussian, 70.6 drift and 62.0 jump.
- For each estimator, three versions are compared:
  - **no forgetting:** cumulative, on the bank's rank-PIT inputs. That is the `xiao` and `hermite`
    registry keys, and RecursiveMPL with gain 1/m, a running mean.
  - **one fixed memory:** its M = 199 arm;
  - **its log bank.**
- Raw Xiao on its own frozen cut-points (`xiao_raw_cum`) is within 1 ×10⁻³ of the cumulative
  rank-fed Xiao everywhere.

**Whole-run RMSE ×10⁻³, Xiao** (Hermite and the MPL show the same pattern, within a few units):

| Scenario | no forgetting | M = 199 | log bank | bank − no forgetting |
| --- | --- | --- | --- | --- |
| constant (Gaussian / Clayton / Gumbel / t₄) | 18.1 / 16.2 / 16.1 / 17.8 | 49.5 / 52.5 / 52.7 / 55.2 | 26.1 / 26.9 / 28.0 / 33.9 | **+8.0 / +10.6 / +12.0 / +16.0** (worse) |
| drift | 384.5 / 273.9 / 274.4 / 382.9 | 68.2 / 67.8 / 68.7 / 71.8 | 70.6 / 62.9 / 64.9 / 75.4 | −314 / −211 / −210 / −308 |
| jump | 319.1 / 316.6 / 315.2 / 318.2 | 81.7 / 83.1 / 83.2 / 85.5 | 62.0 / 66.2 / 64.7 / 70.5 | −257 / −250 / −251 / −248 |
| jump, early [500, 2000) | 30.8 / 26.0 / 26.7 / 32.3 | 67.4 / 70.7 / 65.7 / 73.0 | 43.5 / 44.0 / 41.2 / 50.3 | +12.7 / +18.0 / +14.5 / +18.0 (worse) |
| oscillate | 411.1 / 225.3 / 225.0 / 410.9 | 352.5 / 204.3 / 205.3 / 350.8 | 191.6 / 172.1 / 174.6 / 203.3 | −220 / −53 / −50 / −208 |

All differences are beyond 2 SE (± 1.5–4).

**Answers.**
1. **Does forgetting beat no forgetting? Yes, by a factor of 4–6 whenever the dependence moves.**
   - Drift: about 0.07 against 0.27–0.39.
   - Jump: about 0.065 against 0.32.
   - After the jump: about 0.08 against 0.43.
   - A cumulative estimator converges to the *time average* of ρ_S, so its error grows with the size
     of the change.
   - **When nothing moves, no forgetting is best.** On constant, and in the first 2000 ticks before
     the jump, the cumulative estimator beats both M = 199 (by 31–45) and the bank (by 8–18). Pure
     variance; no bias to trade.
2. **Does the bank beat raw (no-forgetting) Xiao?**
   - **Yes, massively,** on drift, jump and oscillate: by 210–314, 248–257 and 50–220 ×10⁻³.
   - **No on constant:** +8 to +16 worse. Its longest arm is M = 1999, which is still shorter than
     "all data", and fixed share keeps some weight on short arms.
   - It is still far better than a single M = 199 on constant (26–34 against 50–55).
3. **The bank's cost when nothing changes** is about 8–16 ×10⁻³ against the ideal no-forgetting
   estimator. That is the insurance premium for 4–6× smaller error when something does change.
   - Adding a cumulative arm (M = ∞) to the bank would likely shrink that premium. Untested.

## 15. A cumulative (no-forgetting) arm in the bank (2026-11-06)

**Setup.**
- Keys `xiaocum_*`, `hermitecum_*`, `mlecum_*`: the usual five forgetting arms plus one cumulative arm
  (K = 6), with the same η and δ as the five-arm banks.
- Run in `probe_noforget` on the same streams (4 families × 60 seeds). Table:
  `experiments/cumarm_table.py` → `results/cumarm_table.txt`.

**Six-arm bank minus five-arm bank, RMSE ×10⁻³** (paired ± 2 SE; all three estimators and both losses
agree within about 1):

| Window | Gaussian | Clayton | Gumbel | t₄ |
| --- | --- | --- | --- | --- |
| constant, whole | −2.1 to −2.8 | −2.6 to −3.2 | −1.9 to −3.6 | −2.2 to −3.1 |
| jump, first 2000 ticks | −2.0 to −4.1 | −2.6 to −5.5 | −2.3 to −4.4 | −2.5 to −4.0 |
| jump, after the change | +0.9 to +1.3 | +1.2 to +3.9 | +1.2 to +2.4 | +1.6 to +8.4 |
| jump, whole | −0.5 to +0.7 | 0 to +1.6 | +0.2 to +0.6 | +0.5 to +4.5 |
| drift, whole | +0.9 to +1.4 | +0.2 to +0.7 | +0.1 to +0.8 | +1.5 to +2.9 |

**Weight on the cumulative arm** (Xiao log bank; the MPL is similar):

| | Weight |
| --- | --- |
| constant | 0.41–0.46 |
| jump, whole run | 0.17–0.18 |
| jump, after the change | 0.01–0.02 |
| drift | 0.04–0.18 |

**What it shows.**
- **It helps on stable stretches, but only partly.** It cuts the bank's stationary penalty from
  8–16 to 5–14 ×10⁻³, closing about 25–30% of the gap to the pure cumulative estimator.
  - The bank gives the cumulative arm only about 0.45 of its weight, even on `constant`. Over 4500
    ticks its predictions barely differ from the M = 1999 arm's, so one-step loss can't separate
    them. This is the same limit as the mixed bank (§13).
- **It costs after changes.** It is +1 to +8 worse after the jump (worst on t₄), and +0.1 to +2.9
  worse on drift.
  - Before the change, more of the weight sits on slow arms. So at the change, the bank has further
    to move to reach the fast arms.
  - The cumulative arm itself is dropped almost at once (weight 0.01–0.02).
- **Net: a trade, not a win.** Whole-run `jump` is −0.5 to +4.5, and drift is slightly worse.
- **Recommendation:** keep the five-arm bank. A cumulative arm is worth it only if long stable
  stretches dominate the stream.
- **A restart-on-detection version is untested.** It could keep the stationary gain without the
  post-change cost.

## 8. Reproduction

```bash
cd experiments && mkdir -p results && make -B probe_budget EIGEN=/usr CXX=g++
./probe_budget sweep 100 40        # ~14 min on 4 cores -> results/budget_sweep.csv, budget_paths.csv
./probe_budget time                # ~1 min, run on an idle machine -> results/budget_timing.csv
cd .. && python3 experiments/analyze_budget.py experiments/results/budget_sweep.csv
python3 experiments/plot_budget.py
```

`./probe_budget sweep 100 40 t` reruns one family into `budget_sweep_t.csv`.
