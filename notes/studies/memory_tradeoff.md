# Memory trade-off: does a bank improve jump recovery without paying on drift or stationarity?

Probe: `experiments/probe_memory_tradeoff.cpp`. Analysis and figures:
`experiments/analyze_memory_tradeoff.py`. Raw outputs: [`memory_tradeoff/`](memory_tradeoff/).
2026-10-20. Builds on `adaptive_memory.md` (the banks) and §8.5 of `kalman_changepoint_results.md`
(the fixed nonparametric density).

**Question.** After fair tuning, does the discounted-loss bank improve recovery after a jump
without paying too much during drift or stationary periods? Compared against fixed share, a VFF
and fixed memory.

**Answer, in one paragraph.** Over the whole `jump` run, yes. The discounted bank has the lowest
excess log loss of the four, beating tuned fixed memory by 6.8 ×10⁻³ nats/tick, fixed share by
0.6 and the VFF by 3.6, all with paired CIs excluding zero. But it does **not** recover faster
immediately after the jump: in the first 100 ticks it is the worst of the four (+8.2 against
fixed memory). Its advantage comes 100–1500 ticks after the change, and before the change.
On `drift` it pays 1.0 ×10⁻³ against fixed memory (about 2% of the excess loss), the same as
fixed share. On `constant` it beats the jump/drift-tuned fixed memory by 10.2, and on the tuning
seeds it is within about 0.6 of the best fixed memory chosen for `constant` in hindsight. The
VFF fails for an identifiable reason, not because it was tuned too cautiously: **its signal
points the wrong way after this jump** (§4).

All of this is for one density estimator (an 8×8 forgetting histogram copula), Gaussian
margins, and this VFF design. It is not a statement about banks versus VFFs in general.

---

## 1. Design, fixed before any evaluation run

**One density estimator for everything.** An 8×8 histogram copula with exponential forgetting.
Counts are in observation units, there is a Laplace prior of 0.5 per cell, and it reports
Spearman's ρ as the weighted rank correlation on the grid. All methods see the same 500-tick
rank-PIT inputs.

**Memory range.** Ten fixed memories, log-spaced from 25 to 8000 ticks. Fixed-1000 was the best
fixed memory and sat at the top of the previous range, so the range now extends well past it. The
banks use exactly these ten as components; the VFF is clamped to the same [25, 8000].

**Equal tuning budgets: 12 configurations per adaptive method.**

| Method | Grid |
| --- | --- |
| Discounted bank: $L\leftarrow\beta L+(1-\beta)s$, $w\propto e^{-\eta L}$, $\eta=c/(1-\beta)$ | β ∈ {.98, .99, .995, .998} × c ∈ {1, 2, 4} |
| Fixed share (`EstimatorBank`'s rule, log score, 20-tick blocks) | α ∈ {.0005, .002, .008, .03} × η ∈ {0.5, 1, 2} |
| VFF (`XiaoSpearmanEstimator`'s rule, driven by the Rosenblatt residual, §4) | rate ∈ {.02, .05, .2, .5} × starting memory ∈ {250, 1000, 4000} |
| Fixed memory | the 10 memories |

A Spearman-scored fixed share (exactly `EstimatorBank`) is carried along as a reference for
Spearman RMSE only.

**Tuning objective.** The mean over `jump` and `drift` of the mean excess log loss over
[500, 5000), on tuning seeds 42–71 (30 seeds). One configuration per method; nothing is tuned
per scenario. `constant` is not in the objective, so it can serve as a check on the cost of
adapting.

**Evaluation.** 60 fresh seeds (5042–5101), identical across methods. Paired 95% CIs ($t_{59}$).

**Metric.** Excess log loss $\log c_{\text{true}}(U^*,V^*)-\log\hat c(U^*,V^*)$, evaluated at the
**true** PIT $U^*=\Phi(X)$ for every method. Its expectation is the per-tick KL divergence from
the true copula, so a change in the true copula's entropy at the jump does not register as a loss.
Spearman RMSE is reported separately, as a secondary metric.

**Windows, fixed in advance.**

| Scenario | Windows |
| --- | --- |
| `jump` (Pearson knob 0.2 → 0.8 at t = 2500) | pre [500, 2500), recovery 0–100, 100–500, 500–1500 ticks after the change, post [2500, 5000), whole [500, 5000) |
| `drift` (−0.5 → 0.8 over the run) | whole |
| `constant` (0.6), the stationary control | whole |
| Secondary: `oscillate`, Gaussian→t at ρ≈.58, Clayton→Gumbel at ρ≈.58 and ρ≈.79 | whole, plus 0–500 ticks after the switch |

## 2. Checks (`memory_tradeoff/checks.txt`)

- **Bank loop.** This probe's fixed-share loop, run with Xiao components and the ρ score,
  reproduces `EstimatorBank` to 4×10⁻¹⁶.
- **True densities.** All twelve family × dependence combinations integrate to 1 within Monte Carlo
  error (0.997–1.003).
- **Rosenblatt calibration.** See §4.

## 3. Results (`memory_tradeoff/summary.md` has every cell)

Chosen configurations: fixed memory **1170 ticks** (now inside the range); fixed share **α = .008,
η = 1**; discounted **β = .995, c = 2**; VFF **rate .05, starting memory 1000**.

Values are excess log loss in 10⁻³ nats per tick. Lower is better, so a negative difference
favours the method named first. `*` means the CI excludes 0.

### 3.1 `jump`, by window

| window | disc − fixed | disc − fs | disc − vff |
| --- | --- | --- | --- |
| pre [500, 2500) | −5.46 ± 0.52 * | +0.67 ± 0.21 * | −9.68 ± 0.66 * |
| recovery 0–100 | **+8.17 ± 2.87 *** | **+7.07 ± 3.43 *** | **+12.20 ± 4.10 *** |
| recovery 100–500 | **−37.18 ± 3.25 *** | +1.76 ± 1.25 * | **−16.65 ± 2.64 *** |
| recovery 500–1500 | −6.06 ± 0.75 * | **−4.41 ± 0.32 *** | +3.15 ± 0.53 * |
| post [2500, 5000) | −7.95 ± 0.60 * | −1.64 ± 0.23 * | +1.25 ± 0.47 * |
| **whole** | **−6.84 ± 0.39 *** | **−0.61 ± 0.17 *** | **−3.61 ± 0.37 *** |

Means over the whole run: fixed 93.7, fixed share 87.4, discounted 86.8, VFF 90.4.

### 3.2 `drift` and `constant`: the cost of adapting

| scenario | disc − fixed | fs − fixed | disc − fs | disc − vff |
| --- | --- | --- | --- | --- |
| `drift` | +1.01 ± 0.21 * | +0.94 ± 0.18 * | +0.07 ± 0.18 | −1.38 ± 0.33 * |
| `constant` | −10.20 ± 0.41 * | −10.40 ± 0.43 * | +0.19 ± 0.08 * | −10.58 ± 0.49 * |

⚠️ **Read `constant` against the right baseline.** The fixed control is tuned for jump and drift
(1170 ticks), not for `constant`, where the best fixed memory is the longest (8000). Against that
hindsight choice, on the tuning seeds, the banks lose only 0.4–0.6: 40.50 (fixed share) and
40.71 (discounted) against 40.11. So the stationary cost of adapting is small, and against any
*single* memory that must also handle jumps and drift, adapting is a clear gain.

### 3.3 Secondary checks

- **`oscillate`:** both banks beat fixed memory (−8.8, −10.3). Fixed share beats discounted
  (+1.55 ± 0.31).
- **Shape switches at fixed ρ:** over the whole run the banks beat fixed memory by 3–9. After
  Gaussian→t they stay better (−7.9 in the first 500 ticks). In the 500 ticks after a
  Clayton→Gumbel switch they are *worse* (+3.1 to +4.5): they hold about
  6000 ticks of memory beforehand, so they adapt to the new shape more slowly than the
  1170-tick control.
- **Spearman RMSE** (×10⁻³, whole `jump` run): ρ-scored fixed share 79.9, fixed share 104.8,
  discounted 121.3, VFF 137.6, fixed 157.9. As before, weights tuned for the copula are not
  tuned for ρ.

### 3.4 What the memory traces show (`memory_tradeoff/traces.png`)

- **Before the jump**, both banks run at about 6000–7000 ticks of memory, far longer than the
  fixed control. That is where the "pre" gain comes from, and it is also why they are worst in
  the first 100 ticks after the jump: they start from the longest memory.
- **After the jump**, both banks drop to about 350 ticks within about 300 ticks, then lengthen
  again. The discounted bank lengthens back sooner than fixed share, which is where it wins
  500–1500 ticks after the change.
- **On `drift`**, the banks' memory slides from about 3000 to 1200 ticks.

## 4. Why the VFF does not react: the signal has the wrong sign here

The VFF follows `XiaoSpearmanEstimator`'s rule: $C\leftarrow C+r(e^2-C)$, then
$\alpha\leftarrow\alpha\,C^r$, so $e^2>1$ shortens memory. Here $e^2$ is the mean of the two
squared Rosenblatt residuals of the predictive histogram, $z=\Phi^{-1}(H(V\mid U))$ and the same
for $U\mid V$. Under a correct predictive copula, $\mathbb E[z^2]=1$.

**Calibration.** The null level is not 1, and it depends on the dependence. Mean $e^2$ for
fixed-memory histograms under a stationary copula, ticks 1500–2500, 20 seeds:

| copula | mem 250 | mem 1000 | mem 4000 |
| --- | --- | --- | --- |
| independence, rank PIT | 1.061 | 1.014 | 1.002 |
| Gaussian ρ≈.58, rank PIT | 0.934 | 0.958 | 0.968 |
| Clayton ρ≈.58, rank PIT | 0.921 | 0.957 | 0.975 |

True-PIT values are within 0.02 of these, so this is not caused by the rank PIT.

**Trajectory (bottom row of the figure).** Over the jump from weak to strong dependence, $e^2$
**falls** from about 1.03 to about 0.75 and stays below 1 for over 1000 ticks. A forecast that is
too *diffuse* (the old, nearly independent histogram facing strongly dependent data) makes the
conditional PITs cluster near ½, which gives $z^2<1$. The statistic reads "too diffuse" as
"better than calibrated". So:

- the tuned VFF's memory *lengthens* after the jump (about 850 → 1700 ticks);
- the aggressive VFF (rate .5, start 250) shows the same wrong-way reaction more strongly. Before
  the jump, near independence, $e^2\approx1.05$ pins it at the 25-tick floor. After the jump it
  climbs to about 7000 ticks;
- after the Clayton→Gumbel switch, $e^2$ does rise above 1 (to about 1.1). The aggressive VFF
  shortens its memory (about 7000 → 2800 ticks), in the right direction. The tuned VFF's rate is
  too slow for this to do more than slow its lengthening.

**So the problem is neither a weak signal nor cautious tuning. The statistic has the wrong sign
for a change toward stronger dependence, and a dependence-dependent null.** Faster rates do not
rescue it. Across the VFF grid, the best `jump` loss (87.9, rate .2, start 1000) comes with a
`drift` loss of 77.3, against the banks' 52.6–52.8: a faster rate amplifies the dependence-
dependent null as much as the change signal. This is the copula-level counterpart of `adaptive_memory.md`
§5.1, where a ρ-level scale also pointed the wrong way just after a jump. A two-sided statistic,
such as a test of whether the conditional PITs are uniform, or the log score compared with its
own predicted expectation, would be needed. That is untested.

## 5. Caveats

- **One density estimator** (8×8 histogram, prior 0.5) and Gaussian margins. With fewer
  parameters, the best memories, and possibly the ranking, could change.
- **One VFF design.** §4 explains its failure. It is not evidence about VFFs in general.
- **`jump` is a single, large change toward stronger dependence.** A jump toward *weaker*
  dependence would make the forecast too *narrow*, $e^2>1$, and the VFF might react correctly.
  That is untested.
- **Windows were fixed in advance, but several were compared.** Treat individual window
  differences of a few ×10⁻³ with that in mind. The whole-run `jump`, `drift` and `constant`
  comparisons are the primary results.
- **Platform.** Numbers are from Linux/libstdc++. `std::normal_distribution` differs under
  libc++, so a macOS run will give different draws (see §8.5 of `kalman_changepoint_results.md`).

## 6. Reproduction

```bash
cd experiments
make probe_memory_tradeoff            # no Eigen needed; CXX=g++ on Linux
mkdir -p ../notes/studies/memory_tradeoff
./probe_memory_tradeoff               # ~1 min: checks, tuning, evaluation -> CSVs
python3 analyze_memory_tradeoff.py    # summary.md + traces.png
```

Outputs: `checks.txt` (sanity checks and calibration), `tuning.csv` (every configuration ×
scenario, tuning seeds), `eval_windows.csv` (per seed × window × method), `traces.csv` (per-tick
means over the 60 evaluation seeds, ticks 2000–4000), `summary.md` and `traces.png`.
