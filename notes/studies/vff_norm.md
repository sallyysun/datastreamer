# Reproduction, matched table, and the Xiao VFF normalisation ablation

Branch `claude/vff-norm-ablation`, off `adaptive-memory` @ `1edc2bb`, adding three
changes (below). Platform: Linux, clang++ (libstdc++), Eigen 3.4.0,
`-std=c++20 -O2`. Protocol unchanged: N = 5000, burn-in 500, scored on ticks 500–4999,
40 seeds from 42, Gaussian copula. All RMSE differences are **paired over seeds** (per-seed
RMSE), reported as mean with a 95% t-interval (t₃₉ = 2.023).

Changes on this branch (nothing merged into `adaptive-memory` or `main`):

- `experiments/probe_adaptive.cpp`: also writes `adaptive_memory_seeds.csv` (per-seed RMSE)
  and `adaptive_memory_cost.csv` (ns/tick and `store_bytes` after one stream).
- `src/estimators/dependence/xiao_spearman.hpp`: ablation hooks only — a setter for the
  VFF null variance and getters for the controller's per-block state. Default behaviour is
  unchanged (verified: fixed-1 RMSEs reproduce `probe_adaptive` exactly).
- `experiments/probe_vff_norm.cpp`: the ablation.

## 1. Matched comparison (bank vs each alternative), % RMSE change; negative = bank better

| Xiao | constant | drift | jump | oscillate |
|---|---|---|---|---|
| vs one_setting (in-sample) | −62.9 | −31.4 | −13.9 | −18.5 |
| vs best fixed, post hoc | +77.6 | +6.9 | −1.7 [CI includes 0] | +3.4 |
| vs VFF r = 0.05 | −29.6 | −2.1 [CI incl. 0] | −1.0 [CI incl. 0] | −35.1 |
| vs VFF r = 0.5 | +22.1 | −24.9 | −27.9 | −26.4 |

| Hermite (debiased) | constant | drift | jump | oscillate |
|---|---|---|---|---|
| vs one_setting | −62.5 | −33.1 | −16.6 | −17.0 |
| vs best fixed, post hoc | +56.1 | +7.9 | −2.0 | +3.1 |
| vs VFF r = 0.05 | −45.0 | −6.1 | −2.2 | −40.1 |
| vs VFF r = 0.5 | −45.7 | −7.1 | −3.2 | −39.7 |

| Kalman | constant | drift | jump | oscillate |
|---|---|---|---|---|
| vs one_setting | −69.0 | −38.7 | −35.5 | −9.8 |
| vs best fixed, post hoc | +82.6 | +5.1 | −18.1 | +5.3 |
| vs VFF slow | −42.1 | −1.3 [CI incl. 0] | −23.8 | −7.8 |
| vs VFF fast | +18.8 | −39.8 | −33.3 | −15.9 |

Absolute differences and intervals: `results/matched_table.csv` (`python3 matched_table.py`).

Cost after one 5000-tick stream (single timing run, indicative only):

| | fixed | VFF | bank |
|---|---|---|---|
| Xiao ns/tick · bytes | 566 · 3,824 | ~1,360 · 19,968 | 10,179 · 43,288 |
| Hermite ns/tick · bytes | ~600 · 1,192 | ~600–720 · 1,192 | ~8,600–9,000 · 24,864 |
| Kalman ns/tick · bytes | 66 · 512 | ~75 · 512 | 86 · 736 |

The note's Xiao byte counts (5,120 / 21,264 / 52,360) were taken from freshly constructed
estimators, which still hold the 100-point calibration buffer; after a stream they are
smaller. State which one is reported.

## 2. Ablation: fixed null variance 1 vs oracle Var(p_t)

Oracle Var(p) for the Gaussian copula by quadrature: 1.000 (d = 0), 0.988 (0.2), 0.900 (0.6),
0.841 (0.8); matches the Monte Carlo check.

RMSE, paired difference oracle − fixed1 [95% CI]:

| scenario | r = 0.05 | r = 0.5 |
|---|---|---|
| constant | 0.0423 → 0.0554, **+0.0131** [0.0124, 0.0138] | 0.0244 → 0.0552, **+0.0308** [0.0227, 0.0389] |
| drift | 0.0703 → 0.0731, +0.0028 [0.0025, 0.0030] | 0.0916 → 0.0869, −0.0048 [−0.0109, 0.0014] |
| jump | 0.0786 → 0.0793, +0.0007 [0.0005, 0.0009] | 0.1079 → 0.0940, **−0.0138** [−0.0181, −0.0096] |
| oscillate | 0.2916 → 0.2624, **−0.0291** [−0.0296, −0.0287] | 0.2571 → 0.2039, **−0.0531** [−0.0594, −0.0469] |

Controller state, seed-averaged:

| | ratio ē²/denominator, t ≥ 1000 | nominal memory | Kish ESS |
|---|---|---|---|
| constant, fixed1 (r = 0.5) | 0.900 | 946 | 911 |
| constant, oracle (r = 0.5) | 1.004 | 394 | 356 |
| jump post-change (t 3000–4990), fixed1 | 0.84 | → 995 by t = 4900 | |
| jump post-change, oracle | 1.00 | → 387 by t = 4900 | |
| jump r = 0.5, memory before → minimum after | fixed1 543 → 441 (1.2×) | oracle 400 → 174 (2.3×) | |
| jump r = 0.05, memory before → minimum after | fixed1 166 → 166 | oracle 156 → 148 | |

### What it shows

1. **The mechanism is confirmed.** With the fixed denominator the post-jump ratio sits at
   0.84, so the multiplicative controller lengthens memory through the new regime (to ~995
   ticks). With the oracle the ratio is 1.00 and memory stays put.
2. **Correcting the scale trades stationary accuracy for adaptation.** It improves
   `oscillate` (−10% / −21%) and `jump` at r = 0.5 (−13%), and worsens `constant`
   (+31% / +126%). The fixed-1 bias was lengthening memory in exactly the scenarios where long
   memory pays, so part of the shipped VFF's stationary performance came from the
   miscalibration.
3. **Under correct calibration this controller has almost no information about memory
   length in a stable regime.** With the oracle, the stationary ratio is 1.003–1.007 at
   memories of ~150–400 ticks, consistent with E[s²]/Var(p) ≈ 1 + 1/N_eff when the residual
   is uncorrelated with the estimate's error. Between memories of 150 and 1000 that is a
   difference of about 0.006 in a block-mean statistic whose noise is far larger. So only
   transients (peak ratio ~1.3) move it, and nothing pulls memory long when the regime is
   stable. *Interpretation, not yet tested:* the bank scores arms on the same tick, so the
   common Var(p) term cancels in its comparisons. That may be why the same residual
   discriminates between memories in a bank but not as an absolute ratio.
4. **At r = 0.05 the controller barely responds at all** (memory 166 → 166 through the jump,
   either scale). The rate, not the scale, is the binding constraint there.

### Limits

Gaussian copula only (the oracle is the Gaussian Var(p)); one jump size; rates 0.05 and 0.5
only; memory clamp 25–1000 unchanged. The oracle removes the Var(p) misspecification but
not rank-PIT estimation or estimator lag, which point 3 suggests are small at stationarity.
Platform: Linux/libstdc++, not bit-identical to the note's macOS runs.

Reproduce (from `experiments/`; on macOS drop `EIGEN=/usr`):

```bash
mkdir -p results
make probe_adaptive probe_vff_norm EIGEN=/usr
./probe_adaptive && ./probe_vff_norm
python3 matched_table.py
```
