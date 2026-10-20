# Thesis audit — what is implemented, what was run, what is missing (2026-10-10, refreshed 2026-10-20)

**2026-10-20 refresh.** Since the original audit, merge #10 brought `adaptive-memory`,
`claude/tail-dependence` and `claude/async-memory` into `main`, and `bank-vff-study` added the
density-scored bank/VFF study (`memory_tradeoff.md`). This refresh narrowed the scope paragraph below,
changed §0, §1.1 (VFF, bank and MPL rows), §1.4 items 7–8, §1.5, the status of §3 and the claims
table in §4, and added §1.6 (`CopulaSelector`) and the glossary (§6). Everything else is the 2026-10-10 text. The `branch:path:line` references below are the
pre-merge branches, and their line numbers date from 2026-10-10; the files now live on `main`.

Scope of the thesis for this audit: **memory / forgetting** — fixed forgetting, VFF and a bank —
for **Xiao and Hermite**, the two estimator classes the current claims rest on (§4, claim 7). The
online pseudo-likelihood (MPL) is audited for what exists, as the planned extension (claim 7+,
§3 MUST-1 to MUST-3), not as part of the supported claims. Kalman, state-space, CUSUM/BOCPD and
method of moments are out of scope and appear here only where an in-scope result depends on them.
*(Scope paragraph narrowed 2026-10-20; the original audit listed the MPL as a third in-scope
class.)*

Every claim below is tagged **[code]** (read in source, file:line given), **[rerun]** (rebuilt and
re-run in this audit), **[note]** (stated in a repo note, not re-verified), or **[interp]** (my
reading, not a fact about the code). File paths are given as `branch:path:line`; `am` =
`origin/adaptive-memory`, `td` = `origin/claude/tail-dependence`, `as` = `origin/claude/async-memory`.

---

## 0. Where the work actually lives

State on 2026-10-20:

| Branch | Holds | Merged to `main`? |
| --- | --- | --- |
| `main` | Xiao, Hermite, Gaussian MLE, Clayton/Gumbel/t MLE, Kalman + detectors, `multiseed`, family grid (#8), and since #10 everything that was on `am`, `td` and `as` below | — |
| `adaptive-memory` (`am`) | memory sweep, ρ-scored bank/VFF study, VFF code, `EstimatorBank`, rank-PIT, Hermite `debias`, `copula_v1–v3.md`, `memory_sweep.md`, `adaptive_memory.md` | ✅ #10 |
| `claude/tail-dependence` (`td`) | χ studies, steps 1–3b (`tail_dependence.md`) | ✅ #10 |
| `claude/async-memory` (`as`) | async-arrival / thinning study, the f^(−1/3) check (`async_memory.md`) | ✅ #10 |
| `bank-vff-study` | `NonparamCopula` fix, `XiaoSpearmanEstimator::log_density_uv`, the density-scored bank/VFF study (`memory_tradeoff.md` + committed CSVs) | no (on top of #10) |
| `claude/hermite-ew-startup` | a second, different Hermite start-up fix, w = max(λ, 1/n) | no, kept separate deliberately |
| `fix-estimator-resets`, `massive-refactor` | the same `ParametricCopulaMLE::reset()` fix (+ contract checks / design notes) | no |
| `r-analytics` | June R prototype (MAE study, `online_npcorr.R`) | no, left as is |

The divergence the original audit described (no single commit reproducing all results) is resolved
by #10 for code. Two things still block one reproducible thesis build: the Hermite start-up fix is
not chosen (`debias` is opt-in on `main`, `max(λ, 1/n)` is on its own branch), and the numbers come
from two platforms (§0.1).

**Thesis PDF in the repo** (`Sally_Sun_Honours_Thesis.pdf`, 21 pp., 19 Aug 2026) has no
forgetting chapter and no MPL section; §3.3 still reads "describe method IN THIS DRAFT" for both
Xiao and Hermite; §4.1 describes the *live server's* GBM price/volume DGP (4-variate Gaussian
copula), which is **not** the offline harness DGP used for every result (§2.1 below). The MPL
background you describe is not in the repo; I audited the code against your prompt's equations.

**Committed results.** `experiments/results/` is gitignored, so for most studies the numbers exist
only in notes. Exception: `memory_tradeoff.md`'s per-seed CSVs are committed in
`notes/studies/memory_tradeoff/`. Committed figures: `am:notes/studies/figures/{memory_fig1_ucurve,
memory_fig2_families,adaptive_fig1_trajectory,adaptive_fig2_table}.png` (scripts
`plot_memory.py`, `plot_adaptive.py`) and `td:notes/studies/figures/{tail_mixture,tail_restart,
tail_soft}.png` (`plot_tail_*.py`).

### 0.1 ⚠️ Reproducibility finding [rerun]

I rebuilt and re-ran `probe_memory`, `probe_adaptive` (branch `am`) and `tail_soft` (`td`) in a
Linux container (gcc 13, **libstdc++**). Runs are deterministic, and `probe_memory` at the commit
that wrote `memory_sweep.md` (`d7c2db0`) gives byte-identical output to today's code — so the code
did not change. But the numbers do **not** match the memory/adaptive notes:

| cell (Xiao unless stated) | note | rerun here |
| --- | --- | --- |
| `constant` best | 0.0154 at M ≥ 30 000 | 0.0187 at M = 9 793 (curve flat to 1e-4 from M ≈ 4 600; SE ≈ 0.0013) |
| `drift` best | 0.0659 at M = 235 | 0.0649 at M = 235 |
| `jump` recovery / settled | 0.0934 @ 53 / 0.0199 @ 495 | 0.0912 @ 53 / 0.0196 @ 495 |
| `oscillate` best | 0.1827 at M = 36 | 0.1833 at M = 36 |
| Xiao bank vs per-scenario oracle, `jump` | −3.3σ | **−1.8σ** |
| Hermite (debiased) bank vs oracle, `jump` | −2.7σ | −2.2σ |
| bank vs one fixed setting | 12/12 wins | 12/12 wins |

The `tail_soft` rerun reproduces `tail_dependence.md` step 3b **exactly** (48/48 cells, same
ranges). Cause [interp]: `std::normal_distribution` / `uniform_real_distribution` are
implementation-defined, so libc++ (Homebrew/macOS, where the memory notes appear to have been
produced) and libstdc++ (cloud sessions, where the tail notes were produced) draw different data
from the same seed. **Every optimal memory reproduces; RMSEs move by ~1–3 SE; the "bank beats the
oracle on `jump`" claim for Xiao is not robust to it.** Before any number goes in the thesis:
generate all of them on one platform from one commit, and either commit the CSVs or replace the
`std::*_distribution` calls with a portable sampler.

**Platform labels (2026-10-20).** Until the numbers are regenerated, each note carries its
platform:

| Platform | Notes |
| --- | --- |
| macOS / libc++ | `memory_sweep.md`, `adaptive_memory.md` (labelled at the top of each, with the Linux rerun values from the table above) |
| Linux / libstdc++ | `tail_dependence.md`, `async_memory.md`, `vff_norm.md`, `memory_tradeoff.md`, `kalman_changepoint_results.md` §8.5 |

Every other note quoting these numbers inherits the label of its source.

---

## 1. Task 1 — the audit

### 1.1 Component table

| Component | Implemented? | File / class | Exact method (as coded) | Experiments using it | Notes / problems |
| --- | --- | --- | --- | --- | --- |
| **Xiao** | yes | `xiao_spearman.hpp`, `XiaoSpearmanEstimator` | B×B count grid (B = 20 registered); EW: every count ×λ then +(1−λ) on the hit cell (`am:…/xiao_spearman.hpp:240-255`); ρ̂ = rank correlation on grid mid-ranks (`:97-129`) | memory sweep, bank/VFF table, family grid, async | λ = **retention**. Cut-points = empirical quantiles of the **first 100 raw observations, frozen forever** (`:83-93`, registry `calibration = 100`); rank-PIT cut-points only in `xiao_pit`. Discretisation bias ≈ 0.006 in ρ [note]. |
| **Hermite** | yes | `hermiter.hpp`, `HermiteEstimator` | order K = 10 Hermite functions; A ← A + w(h_x h_yᵀ − A), w = λ (`am:…/hermiter.hpp:101-118`); ρ̂ = 12uᵀAv − 6uᵀAz − 6zᵀAv + 3zᵀAz (`:160-177`) | memory sweep, bank/VFF table, family grid, async | λ = **update weight** (= α). Marginals EW-standardised **with the same λ** (`:239-266`). **Start-up shrinkage toward 0** when λ small; two competing fixes, both unmerged: opt-in `debias` (`am`, divide by 1−(1−w)ⁿ) and always-on w = max(λ,1/n) (`claude/hermite-ew-startup`). |
| **Gaussian "MLE"** | yes | `gaussian_mle.hpp`, `GaussianCopulaMLE` | θ = atanh r; H_t = λH_{t−1} + (1+r̂²); θ_t = θ_{t−1} + s_θ/H_t, clipped at \|r\| ≤ 0.999 (`am:…/gaussian_mle.hpp:61-98`) | family grid (λ = 0.99); 5-point λ set (`gaussian_mle_l95…l999`, `am` only); async grid | **Not pseudo-likelihood**: input is (x − x̄ₙ)/sₙ with *cumulative* Welford moments (`:66-67, :138-151`), i.e. parametric Gaussian margins. **Never in the memory sweep or the bank/VFF table.** |
| **Parametric copula MLE** (Clayton/Gumbel/t/Gaussian) | yes | `parametric_copula.hpp`, `ParametricCopulaMLE` / `copula::CopulaFit` | φ unconstrained; H_t = λH_{t−1} + s_φ² (**outer product of scores**, H₀ = 1); step = clip(s_φ/H_t, ±0.5) (`am:…/parametric_copula.hpp:31-96`) | family grid (λ = 0.99), rank-PIT invariance test | Default input u = Φ(standardised x) (`:194-195, :219-222`). Rank-PIT only if `rank_pit_window > 0` (`am` only). Different information estimate from `GaussianCopulaMLE`. |
| **Online MPL (ranks)** | partly | `ParametricCopulaMLE` + `RankPit` (`am:…/marginal_rank_pit.hpp:25-43`) | u_t = (#{last ≤500 obs < x_t} + ½)/(n_t + 1), then the recursion above | §9 invariance test only (`gaussian_mle_pit`, 10 seeds, one memory) | **No memory experiment uses a rank-based MPL** (still true on `main` after #10, checked 2026-10-20). Ranks are sequential and windowed, not full-sample R/(n+1). |
| **Fixed forgetting** | yes, all | per estimator | M = (2−α)/α for all; knob differs (table §2.7) | all | Half-life check confirms M within ±8% for Xiao/Hermite [note]; **unchecked for the MLE**. |
| **VFF** | yes, 5 *different* rules | Xiao `:206-233`; Hermite `:141-147`; `CopulaFit` `:56-65`; Kalman (out of scope); the Rosenblatt-residual VFF inside `probe_memory_tradeoff.cpp` (probe only, not registered) | see §2.8 and `memory_tradeoff.md` §4 | bank/VFF table (Xiao, Hermite, Kalman); `gaussian_mle_vff` only in one `jump` table; `memory_tradeoff.md` | Not one method: each estimator uses a different surprise signal and clamp. `gaussian_mle_vff` is `ParametricCopulaMLE` (OPG information), so it is not matched to `gaussian_mle_ew`. Registry `xiao_vff` rate is 0.5 (`am:registry.hpp:411`); the study ran 0.05 and 0.5 (`adaptive_memory.md` §5.1, corrected 2026-10-20). |
| **Bank** | yes | `am:…/estimator_bank.hpp`, `EstimatorBank` | exponential weights on a squared rank residual, fixed share 0.002 per 20-tick block (`:52-91`) | bank/VFF table (Xiao, Hermite); χ step 2/3b (tail variant); `memory_tradeoff.md` uses two *log-scored* banks built in its probe (§6) | **MLE bank never measured** (still true after #10). Registered `xiao_bank/hermite_bank/mle_bank` use a λ grid (M ≈ 39–1999), not the 7-arm tick grid the study used. |
| **DGPs** | yes | `dependence.hpp`, `copula.hpp`, `experiments/copula_source.hpp` | knob d_t → target ρ_S = (6/π)asin(d/2) → family parameter by bisection | everything | Margins always exactly N(0,1), stationary. Clayton/Gumbel cannot follow the negative half of `drift`/`oscillate` (clipped). |
| **Evaluation** | yes | `multiseed.cpp`, `probe_memory.cpp`, `probe_adaptive.cpp`, `tail_*.cpp` | RMSE vs true ρ_S per tick after burn-in; bias²/var decomposition; paired σ; χ: MSE vs χ_t(q), recovery time | — | Three aggregations in use (mean of per-seed RMSE; pooled RMSE; MSE decomposed across seeds). **No regret anywhere** (only a "not relevant" mention in `am:notes/studies/reference_review.md:60`). |
| **Processing rate** | partly | `as:experiments/async_memory.cpp` (`sync` arm); `am:experiments/bench_rate.cpp` (throughput only) | Bernoulli(p) observation of each leg; `sync` arm = pure thinning | `async_memory.md` | The "processing-rate study" that *derived* f^(−1/3) is **not in the repo**: `async_memory.md:12,227` cites it as "not on `main`", and `bench_rate.cpp:1` points to `notes/studies/rate_budget.md`, which exists on no branch. |

### 1.2 Your 17 questions, answered from code

1. **Implemented estimators (in scope):** Xiao, Hermite, `GaussianCopulaMLE`,
   `ParametricCopulaMLE` (4 families), `WindowedRankEstimator`, `EstimatorBank`. Out of scope but
   present: Kalman family, CUSUM/GLR/BOCPD/EWMA/SR charts, particle filter, EKF/UKF, `CopulaSelector`.
2. **Update equations:** §2.4–2.6.
3. **Exponential forgetting:** all of Xiao, Hermite, both MLE classes (and Kalman by
   construction).
4. **Parameterisation:** three conventions. Xiao and both MLEs: λ = retention (α = 1−λ). Hermite:
   λ = α. Kalman: σ_ω². Probes and the async/tail harnesses specify memory in ticks and derive the
   knob; registry keys specify λ directly.
5. **Effective memory:** M = (2−α)/α, α = 2/(M+1), the weight-equivalent ESS (code:
   `xiao_spearman.hpp:166-169`; probes' `alpha_from_ess`). This is exactly your formula. For the
   MLE it is an [interp] linearisation in θ = atanh r, not in ρ_S, and was never measured.
6. **VFF implemented:** `xiao_vff`, `hermite_vff`, `gaussian_mle_vff` (+ `kalman_vff`).
7. **Surprise signals:** Xiao: rank residual s = 12(u−½)(v−½) − ρ̂ with fixed null variance 1.
   Hermite: ‖h_x h_yᵀ − A‖_F, fast/slow EWMA ratio (the code's own comment, `hermiter.hpp:57-61`,
   says this statistic has little power). MLE: score drift mean²/var. Details §2.8.
8. **Bank:** yes. Arms in the study: M ∈ {25, 46.2, 85.5, 158.1, 292.4, 540.7, 1000} (`am:
   experiments/probe_adaptive.cpp:63-67`). Weights: Bayes / exponential-weights on
   −s²/(2v̂) per 20-tick block, **fixed share** p = 0.002. Loss: squared error of each arm's
   ρ̂ against the one-tick unbiased proxy 12(u−½)(v−½). Combination: weighted mean of arm
   estimates. Your remembered grid is correct, but it is the probe's grid, not the registered
   keys'.
9. **DGPs simulated:** `constant`, `drift`, `jump`, `oscillate`, `stochastic` (random walk), `shape`
   (Gaussian → t₄ at constant ρ_S), `frechet`, `mixgauss`; families Gaussian, Clayton, Gumbel, t₄.
   Tail harness: matched-ρ_S family switches and a Clayton–Gumbel mixture path. Async: four extra
   directional drifts.
10. **What is "true":** the generator is driven by d (the Gaussian-copula correlation r). Every
    estimator is scored against **ρ_S** = ρ_S(θ_t), the *achieved* Spearman's rho of the sampled
    family (`am:src/streamgenerator/datasource/copula.hpp:27-46, 55-58, 78-90`). θ is solved for by
    bisection. Kendall's τ is never a target (`HermiteEstimator::kendall()` exists, unused). The
    tail harness scores against χ_t(q), not λ.
11. **Metrics:** RMSE vs ρ_S over ticks 500–4999 (multiseed: mean ± SE of per-seed RMSE,
    `multiseed.cpp:85-90`); MSE = bias² + var across seeds (`probe_memory`); paired-σ vs baselines
    (`probe_adaptive`); `jump` split into recovery/settled windows; χ: MSE pre/post/total and
    recovery time (seed-mean within 10% of the jump); `store_bytes`; throughput. **No regret,
    no integrated error as such** (RMSE over the run is an integrated squared error).
12. **Memory sweeps run:** `probe_memory`: 20 points log 25→30 000 ticks (Xiao, Hermite), Kalman
    10 points; 4 scenarios, 40 seeds, Gaussian. Registry: Gaussian MLE λ ∈ {0.95, 0.98, 0.99,
    0.995, 0.999} (multiseed). Async: M ∈ {10, 20, …, 2560} pairs (doubling), drift + constant,
    Gaussian, 20 seeds, with Xiao/Hermite/Gaussian MLE/windowed/Kalman. Tail: M = 100·2^(k/2),
    k = 0..18, plus cumulative; 40 seeds.
13. **Fixed vs adaptive comparisons:** `probe_adaptive` (Xiao, Hermite raw + debiased, Kalman ×
    {per-scenario oracle, one fixed setting, VFF at rates 0.05/0.5, bank}, 4 scenarios, 40 seeds,
    Gaussian); one MLE table (`am:kalman_changepoint_results.md` §14.10.3, `jump` only:
    `gaussian_mle_vff` 0.0708 vs best fixed λ 0.0745); tail steps 2/3b (fixed vs oracle
    hard/soft restart vs bank).
14. **Processing-rate experiments:** only the async study's `sync` arm (random thinning to pair
    rate f) — Gaussian, drift/constant, 20 seeds. M* follows f^(−1/3) to 0.93–1.12 [note], but
    *anchored on each estimator's own full-rate optimum*, so it tests the scaling, not the
    constant. Hermite is excluded (start-up bug). Not rerun in this audit.
15. **Results/figures:** §0.
16. **Already done, of what you were planning:** the fixed-memory sweep for Xiao + Hermite on all
    four scenarios (40 seeds); the bias²/variance decomposition; the scenario → M* ordering; the
    jump recovery-vs-settled split; VFF vs bank vs fixed for Xiao + Hermite; the misspecification
    grid at one memory; the χ jump/soft-restart oracle with γ* ≈ 1/(1+Z²) (`td:tail_dependence.md`
    step 3b, 5 Oct — my `oracle_memory.md` re-derived it without knowing that); the f^(−1/3)
    thinning check.
17. **Discrepancies:** §1.4.

### 1.3 Your remembered results, checked

| Recollection | Exists? | What it actually is |
| --- | --- | --- |
| bank arms {25, 46, 85, 158, 292, 541, 1000} | ✅ | `probe_adaptive.cpp:63-67`, Xiao + Hermite banks only. Not the registered keys. |
| constant → very long memory | ✅ | Monotone; optimum at or near the grid edge (≥ 4 600 ticks, flat beyond) with N = 5000, so it effectively means "cumulative". |
| drift → intermediate | ✅ | 235 ticks (Xiao), 235–341 (Hermite). |
| oscillation → short | ✅ | 36 ticks, both estimators (grid floor is 25). |
| jump recovery → short | ✅ | 53 ticks in recovery vs 495 settled (Xiao); each choice costs 2.3–3.1× in the other window [note]. |
| M* ∝ f^(−1/3) | ⚠️ partly | Scaling confirmed in the async `sync` arm, anchored; the original processing-rate study is not in the repo. |
| regret | ❌ | Never computed. |
| VFF and bank | ✅ | For Xiao, Hermite (and Kalman). **Not for the MLE** (one `jump` cell). |
| all of the above for the MLE | ❌ | MLE absent from the memory sweep and the bank/VFF table. |

### 1.4 Discrepancies (written theory ↔ belief ↔ code)

1. **"MPL with R/(n+1)" vs code.** Registered "MLE" estimators use Φ((x − x̄ₙ)/sₙ) with
   cumulative moments — parametric margins, not ranks. The only rank-based variant (`*_pit`) uses
   **sequential, windowed** ranks, (r + ½)/(n + 1) over the past ≤ 500 observations, which is
   itself a second (marginal) memory of 500.
2. **"Fisher-scaled recursion" vs code.** `GaussianCopulaMLE` matches your recursion, with
   discounting (H_t = λH_{t−1} + I), in θ = atanh r, with *expected* information. The
   Clayton/Gumbel/t estimator uses the *observed outer product of scores* (s_φ²), H₀ = 1, and a
   step clip of ±0.5. **No plain SGD with γ_t exists.**
3. **Bank grid.** The remembered 7-arm grid is the probe's; the registered banks use 5 λ arms
   reaching M ≈ 1999 (inside Hermite's warm-up trap). MLE bank: registered, never measured, yet
   `copula_v1.md:431` asserted "`mle_bank` beats `kalman_bank`". `copula_v2.md` flags the
   registered banks as unused.
4. **VFF is four methods sharing one name** (§2.8). A cross-estimator "VFF" row compares
   surprise signals, not estimators.
5. **Hermite marginal memory.** Hermite forgets its marginal moments at the *same* α as its
   dependence; Xiao freezes cut-points after 100 observations; the MLE uses cumulative
   moments. At short memory, Hermite therefore pays extra marginal noise the others do not, which
   confounds "Hermite has 11–63% more variance at matched memory" [interp].
6. **Thesis §4.1 DGP** (GBM prices + log-normal volumes, 4-variate copula) ≠ the harness DGP
   (bivariate normal scores, §2.1).
7. ~~**`CLAUDE.md` on `main`** still says the harness is single-seed 42; multi-seed exists.~~
   Fixed 2026-10-20.
8. ~~**`xiao_vff` rate**: registry 0.5, note 0.05 (probe uses both, so no number is wrong).~~
   Note corrected 2026-10-20 (`adaptive_memory.md` §5.1).
9. **Platform dependence of the numbers** (§0.1).
10. **`oracle_memory.md` (this branch)** said the χ oracle work was not in the repo; it is on
    `td`. Corrected in that note.

### 1.5 A–E summary

- **A. Complete.** Xiao and Hermite fixed-memory sweeps with bias²/var decomposition, 4 scenarios,
  40 seeds, Gaussian. ESS algebra verified by step response (Xiao, Hermite). Bank and VFF vs
  fixed for Xiao and Hermite. Fixed-memory family-misspecification grid (20 seeds, one memory). χ
  stationary validation, jump oracle hard/soft restart, mixture path. Thinning scaling check.
- **B. Partial.** Gaussian MLE memory: 5 λ points via multiseed and the async drift/constant grid,
  never decomposed or ESS-checked. MPL with ranks: built (`*_pit`) but only used in an invariance
  test. MLE VFF: one cell. Hermite start-up fix: two versions, unmerged.
- **C. Missing** (re-checked on `main` after #10, 2026-10-20: all still missing). MLE on the
  common memory axis (all 4 scenarios); MLE bank; any memory sweep
  under a misspecified family; any rank-based MPL in a memory experiment; a χ *drift* scenario;
  χ from Xiao/MPL (χ exists only in a separate ratio estimator inside the tail harness); one
  consolidated, reproducible build; regret (if you want it at all).
- **D. Redundant / out of scope.** All Kalman/detector/BOCPD/particle/EKF/UKF work;
  `copula_select`; shape detection; Fréchet/mixgauss; Polyak–Ruppert/damping ablations; resolution
  sweeps; `copula_v1–v3.md` (they consolidate the Kalman-centred story); the online γ̂ rule in
  `oracle_memory.md` §4.
- **E. Issues.** §1.4, items 1–5 and 9, are the ones that change what the thesis can claim.

### 1.6 `CopulaSelector`: context only (added 2026-10-20)

Out of scope for the memory thesis (§1.5 D): it selects a copula **family**, whereas the banks
select a **memory**. Kept here as a short, corrected description because the notes cite it.

What it does (`copula_selector.hpp:94-212` on `bank-vff-study`): standardise each margin with a
running Welford mean and variance, map to u = Φ(z) (clamped to [10⁻⁹, 1 − 10⁻⁹]), score Gaussian,
Clayton, Gumbel and t₄ `CopulaFit`s plus a 10×10 `NonparamCopula` by an exponentially discounted
cumulative log score, S_m ← λS_m + log ĉ_m(u_t, v_t) (non-finite log densities count as −50), and
report the Spearman ρ of the current argmax. λ = 0 means undiscounted.

Corrections to how it has been described:

1. **The score is only partly prequential.** Each candidate is scored *before its own update*, but
   `update()` updates the marginal mean and variance with (x_t, y_t) *before* computing
   (u_t, v_t) (`:113-118`). A strictly prequential score would transform with the previous
   marginal estimates and update them afterwards. ⚠️ This is a methodological issue in every
   selector number so far (`kalman_changepoint_results.md` §8.5, `copula_v1–v3.md`, the family
   grid); fixing it means rerunning those numbers, so they must not be quoted as strictly
   prequential, and a revised description must not silently carry them over.
2. **The inputs are Gaussian-standardised, not rank pseudo-observations.** All candidates see the
   same (u, v), so the marginal term is common, but an error in those inputs (non-Gaussian
   margins, estimated moments) can change candidates' scores by different amounts.
3. **It is a cumulative prequential log score, not a marginal likelihood.** Parameters are fitted
   online and plugged in, not integrated out under a prior (Dawid 1984, prequential approach).
4. **"Window" has two senses.** With λ = 0.99 the score's weight sum is 1/(1 − λ) = 100 ticks; its
   effective sample size is (1 + λ)/(1 − λ) ≈ 199 ticks (§6.1). Name which one is meant.
5. **Negative dependence.** Clayton and Gumbel cannot represent it, so they should score badly
   there, but a finite-sample discounted score does not guarantee which candidate wins.

---

## 2. Task 2 — Methodology, as implemented

Each equation is followed by its source. ⚠️ marks a place where code and written theory differ.

### 2.1 Simulation design

Bivariate streams of length **N = 5000**, scored on ticks **t = 500, …, 4999** (burn-in 500).
Each tick the generator draws (U_t, V_t) from a copula C_{θ_t} and passes the estimators the
normal scores x_t = Φ⁻¹(U_t), y_t = Φ⁻¹(V_t) (`am:experiments/copula_source.hpp:30-34`;
`copula.hpp:59-61`). Marginals are therefore exactly N(0,1) and stationary; no marginal change is
ever simulated in the memory experiments. Seeds: memory sweep and bank/VFF study 42–81 (40);
`multiseed` grid 42–61 (20); async 20 from 42; tail harness 40 per run (seed bases 6060, 8080).
Random numbers come from `std::mt19937` through `std::*_distribution`, so the draws depend on the
C++ standard library (§0.1).

### 2.2 Dependence paths

A scalar knob d_t (the Gaussian-copula correlation) follows one of
(`am:src/streamgenerator/datasource/dependence.hpp`; `probe_memory.cpp:121-124`):

| scenario | d_t | true ρ_S range |
| --- | --- | --- |
| `constant` | 0.6 | 0.582 |
| `drift` | −0.5 + 1.3·min(1, t/N) | −0.483 → 0.786 |
| `jump` | 0.2 for t < 2500, 0.8 after | 0.191 → 0.786 |
| `oscillate` | 0.6·sin(2πt/400) | ±0.582 |
| `stochastic` | z_t = z_{t−1} + 0.01ω_t, d = tanh z, z₀ = 0.6 | seed-specific; excluded from the memory sweep |

### 2.3 Copula families and the true target

Target Spearman: ρ*_S(d) = (6/π)arcsin(d/2) (`copula.hpp:55-58`). Gaussian: r = d. Clayton,
Gumbel, t₄: the family parameter θ solves ρ_S(θ) = ρ*_S by 60-step bisection on the family's
parameter range (`copula.hpp:27-46`). **The truth is the achieved ρ_S(θ_t)** (`:87-88`), so for
the positive-only Clayton/Gumbel the truth is clipped at ≈ 0 on the negative parts of `drift` and
`oscillate`. Every estimator reports ρ̂_S. Parametric fits map θ̂ → ρ_S(θ̂)
(`GaussianCopulaMLE::estimate`, `:100-106`; `spearman_from_param`). So **your proposed common
loss (ρ̂_S − ρ_S)² is what is used**, and the cross-estimator comparison is valid in that
space.

### 2.4 Xiao (`XiaoSpearmanEstimator`)

B = 20 bins per margin; cut-points c_1 < … < c_{B−1} are the empirical k/B-quantiles of the first
100 observations, then frozen (`:83-93, :186-204`); the estimate is 0 until then. Let
b(x) = #{k: c_k ≤ x}. With retention λ:

N_t(i,j) = λN_{t−1}(i,j) + (1−λ)·1{i = b(x_t), j = b(y_t)}, and likewise the row sums R_t(i),
column sums C_t(j) and total T_t (`:245-255`).

Mid-ranks r_i = Σ_{k<i}R(k) + R(i)/2 − T/2 (same for s_j), and

ρ̂_t = Σ_{i,j} N_t(i,j) r_i s_j / √(Σ_i R_t(i) r_i² · Σ_j C_t(j) s_j²), clipped to [−1, 1]
(`:97-129`).

The ratio is scale-invariant, so the EW total need not reach 1: no start-up bias. λ = 0 means
cumulative counts; `window > 0` gives a hard sliding window instead.

### 2.5 Hermite (`HermiteEstimator`)

Standardisation with the same λ (`:239-266`): μ_t = (1−λ)μ_{t−1} + λx_t,
σ²_t = (1−λ)σ²_{t−1} + λ(x_t − μ_t)² (cumulative Welford if λ = 0). With x̃ = (x − μ)/σ and
h_0..h_K the normalised Hermite functions (K = 10 registered), and w = λ (or 1/n if λ = 0)
(`:101-118`):

a_x ← a_x + w(h(x̃) − a_x), a_y likewise, A ← A + w(h(x̃)h(ỹ)ᵀ − A).

With u = W a_x, v = W a_y, W_{kj} = ∫h_k(x)∫_{−∞}^x h_j, and z_k = ∫h_k:

ρ̂ = 12uᵀAv − 6uᵀAz − 6zᵀAv + 3zᵀAz (`:160-177`); 0 for n ≤ 50.

⚠️ No start-up correction by default: the accumulators start at 0, so for small λ the estimate is
shrunk toward 0 until t ≳ 3M [note, `memory_sweep.md` §6.2(b)]. The optional `debias` divides by
1 − (1−w)ⁿ.

### 2.6 Semiparametric / parametric estimator (`GaussianCopulaMLE`)

Marginals by cumulative Welford: z_x = (x_t − x̄_t)/s_t (`:66-67, :138-151`). Gaussian copula
log-density in normal scores:

ℓ(r) = −½log(1−r²) − [r²(z_x² + z_y²) − 2r z_x z_y]/[2(1−r²)] (`objective.hpp:15-21`).

Re-parameterise θ = atanh r. Score s_θ = (∂ℓ/∂r)(1−r²); expected information I(θ) = 1 + r². The
recursion (`gaussian_mle.hpp:69-93`) is

H_t = λH_{t−1} + I(θ̂_{t−1}),  θ̂_t = clip(θ̂_{t−1} + s_θ(θ̂_{t−1}; z_t)/H_t, ±atanh 0.999),

with H₀ = 0, θ̂₀ = 0, λ = 1 for "cumulative", and an estimate of 0 for n ≤ 50. Report
ρ̂_S = (6/π)arcsin(tanh(θ̂)/2).

**Terminology.** This is *recursive one-step Fisher scoring with exponentially discounted expected
information* — a stochastic-approximation recursive MLE (with λ = 1 the gain is 1/(nI(θ)), the
efficient Robbins–Monro gain). Its input is **estimated parametric (Gaussian) margins**, so it is
an online two-stage estimator in the IFM sense, **not** maximum pseudo-likelihood. In this DGP the
margins are exactly Gaussian, so the distinction never shows in any result.

`ParametricCopulaMLE` (Clayton/Gumbel/t, `parametric_copula.hpp:31-96`), with φ the
unconstrained coordinate:

H_t = λH_{t−1} + s_φ²,  φ̂_t = φ̂_{t−1} + clip(s_φ/H_t, ±0.5), H₀ = 1,

with the outer product of scores (BHHH / empirical Fisher) in place of I. Input is u = Φ(z) by
default; with `rank_pit_window = W > 0` it is the sequential rank-PIT
u_t = (#{x_s < x_t, s in the last W} + ½)/(n_t + 1). Only that configuration is an *online
pseudo-likelihood* estimator. ⚠️ Its ranks are windowed (W = 500) and sequential, not
R/(n+1) on the full sample. No configuration does plain SGD.

[interp] Effective memory: at steady state H ≈ I/(1−λ), so θ̂_t ≈ λθ̂_{t−1} + (1−λ)(θ_t + ε_t/I).
That is an EWMA in θ = atanh r with α = 1−λ, i.e. M = (2−α)/α **in the Fisher-z scale**, as for
the Kalman, whose ρ-space half-life reads ~27% short on a large step (`memory_sweep.md` §3).
H_t ∝ 1 − λ^t early on, so there is no start-up shrinkage. Both points need the step-response
check that Xiao and Hermite got.

### 2.7 Exponential forgetting and effective memory

Generic form: S_t = (1−α)S_{t−1} + αz_t, weights α(1−α)^k, M = (2−α)/α, α = 2/(M+1).

| estimator | code knob | α | longer memory |
| --- | --- | --- | --- |
| Xiao | `lambda` (retention) | 1 − λ | larger λ |
| Hermite | `lambda` (update weight) | λ | smaller λ |
| `GaussianCopulaMLE`, `ParametricCopulaMLE` | `lambda` (retention on H) | 1 − λ (θ-space) | larger λ |
| windowed rank | window W | — | M = W |

The probes and the async/tail harnesses take M in ticks and derive the knob; registry keys hard-code
λ. Registered defaults: `xiao_ew` λ = 0.99, `hermite_ew` λ = 0.01, `*_mle_ew` λ = 0.99 — all M ≈ 199.

### 2.8 Variable forgetting factors (as coded)

| | Xiao `vff_step` (`:206-233`) | Hermite (`:141-147`) | `CopulaFit` (`:56-65`) |
| --- | --- | --- | --- |
| signal | s_t = 12(u−½)(v−½) − ρ̂_{t−1}, (u, v) from a separate rank-PIT window of 500 | e_t = ‖h_x h_yᵀ − A‖_F | s_φ,t |
| statistic | e² = block mean of s²/1 over 20 ticks; C ← C + r(e² − C) after 5 burn-in blocks | f ← f + 0.05(e − f), g ← g + 0.002(e − g) | m ← m + r(s − m), v ← v + r(s² − v), D = m²/v |
| rule | α ← α·C^r (multiplicative, per block) | λ = λ₀(f/g)^r | λ = λ₀(1 + D)^(−10r) |
| clamp | M ∈ [25, 1000] | λ ∈ [0.2λ₀, 20λ₀] | λ ∈ [0.90, 0.9999] |

Base memory in the study: 158 ticks; rates r ∈ {0.05, 0.5}. ⚠️ The Xiao rule integrates log α
(α keeps moving while C ≠ 1), whereas the Hermite and MLE rules set λ as a function of the
current statistic. The "best VFF" rows in `adaptive_memory.md` pick the better of two rates per
scenario after seeing the result.

### 2.9 Bank (`EstimatorBank`)

K arms of one estimator class at memories M_k. Each tick, before updating (`:52-91`): u, v from a
rank-PIT window of 500; p_t = 12(u−½)(v−½); s_{k,t} = p_t − ρ̂_{k,t−1};
v̂ ← v̂ + 0.001(s_mix² − v̂); L_k ← L_k − s_{k,t}²/(2v̂). Every 20 ticks
w_k ← w_k·exp(L_k − max_j L_j), normalise, reset L, then fixed share
w_k ← (1 − 0.002)w_k + 0.002/K. Output ρ̂ = Σ_k w_k ρ̂_k; reported memory M̄ = Σ_k w_k M_k; 0 for
the first 200 ticks. [interp] An exponentially weighted forecaster (Hedge, learning rate 1/(2v̂))
on squared error against an unbiased one-tick proxy of ρ_S, with Herbster–Warmuth fixed share.
Arms in the study: 7 log-spaced, 25 → 1000 ticks.

### 2.10 Evaluation

Per tick, e_t = ρ̂_{S,t} − ρ_{S,t}, over t = 500…4999.
- `multiseed`: per-seed RMSE, reported as mean ± SE over seeds.
- `probe_memory`: per tick, across S seeds, bias(t) = mean_s ρ̂ − ρ, var(t), mse(t) = bias² + var
  (asserted to 1e-15), averaged over the window; RMSE pooled. Windows: `all`, plus `jump` recovery
  (t ∈ [2500, 3500)) and settled (t ∈ [3500, 5000)).
- `probe_adaptive`: paired-over-seeds difference of per-seed RMSE against two baselines:
  `known_best` (best fixed memory for that scenario, chosen post hoc) and `one_setting` (best
  single memory averaged over the four scenarios).
- Tail harness: MSE against χ_t(q) pre / post / total, and recovery time.

### 2.11 Processing rate (if kept)

`as:experiments/async_memory.cpp`: T = 20 000 clock steps, burn-in 2 000, scored every 10th step;
each leg observed w.p. p; the `sync` arm observes both legs at refresh times (pure thinning to
pair rate f). M ∈ {10, …, 2560} pairs, M* by log-parabolic interpolation, 20 seeds, Gaussian,
drift/constant only. Prediction from MSE(M) ≈ σ²/(fM) + (dM/2)²: M* = (2σ²/d²)^(1/3) f^(−1/3).

### 2.12 Other choices that move results

Burn-ins (Xiao 100-observation calibration; MLE/Hermite 50; bank 200; rank-PIT needs 30) all end
before scoring starts. Estimates are clipped to [−1, 1]; r̂ is clipped at |r| ≤ 0.999. N = 5000
caps the identifiable optimum on `constant`. The bank's arm span (25–1000) caps it again.

---

## 3. Task 3 — Minimum additional experiments

**Status 2026-10-20.** None of MUST-1 to MUST-3 has been run. Claim 7 (§4) is now narrowed to
what Xiao and Hermite support, so MUST-1 to MUST-3 are the work needed to *extend* it to the MPL,
not a prerequisite for the narrowed thesis. MUST-4 is half done: #10 merged the code; the Hermite
start-up choice and the single-platform regeneration remain. MUST-1 to MUST-3 need Eigen.

### MUST

1. **Put the MLE on the common memory axis, correctly specified.** Add the Gaussian MLE to
   `probe_memory` (same 20-point grid, 4 scenarios, 40 seeds, Gaussian DGP). Run two variants:
   the current standardised `GaussianCopulaMLE`, and the rank-based MPL —
   `ParametricCopulaMLE{Gaussian, expected_info = true, rank_pit_window = 500}`, which needs **no
   new code**, only a config. Run the existing step-response (`./probe_memory ess`) for it, in
   the z-scale. This is the experiment the whole cross-estimator claim rests on.
2. **Put the MLE in the bank comparison.** A `bank_of("mle")` on the same 7-arm tick grid in
   `probe_adaptive` (config only). This completes fixed vs bank for all three classes.
3. **One misspecified memory sweep.** Repeat MUST-1 with `COPULA=t` (t₄): symmetric, so all four
   scenarios stay comparable (Clayton/Gumbel clip `drift`/`oscillate`). Run Xiao, Hermite and the
   Gaussian MPL. Question answered: *does M\*(scenario) move when the family is wrong?*
4. **Consolidate and regenerate.** Merge the branches into one thesis line, adopt one Hermite
   start-up fix by default, and regenerate every thesis number from one commit on one platform,
   committing the CSVs that the thesis cites. Without this, §0.1 shows the numbers in the notes are
   not reproducible.

### SHOULD

5. **One VFF rule for all three estimators.** Use the estimator-agnostic rank-residual signal
   (already used by `xiao_vff` and the bank) via a memory setter, so the VFF column compares
   estimators rather than surprise signals. The alternative is to report VFF for Xiao only and
   say so.
6. **Check the drift heuristic in absolute terms, with no new runs.** Fit
   MSE(M) ≈ σ²/M + (dM/2)² to the bias² and var columns `probe_memory` already writes; σ²
   from `constant`'s var × M, d from the true path. If it holds, M\* = (2σ²/d²)^(1/3) is a
   tested statement rather than a heuristic.
7. **One asymmetric misspecified family** (Clayton DGP → Gaussian MPL) on positive-only paths
   (e.g. drift 0.2 → 0.8, oscillate 0.5 ± 0.3). This needs a DGP parameter change only.

### OPTIONAL

8. Correct-family MPL (Clayton MPL on Clayton data) on the memory axis. Only worth doing if
   MUST-3 shows the Gaussian MPL's M\* moving.
9. Processing rate as a short section or appendix: the `sync`-arm result already covers Xiao, the
   MLE and windowed rank. Regenerate it on the consolidated build; don't extend it.
10. Per-arm predictive variance in `EstimatorBank` (the one structural difference from the Kalman
    bank).

### DO NOT DO

Kalman, BOCPD, CUSUM, GLR and every detector; particle/EKF/UKF; `copula_select`;
shape/Fréchet/mixgauss scenarios; the async pairing correction and its liquidity phase 2; the
online γ̂ rule (`oracle_memory.md` §4); MoM/Brahimi; resolution sweeps; Polyak–Ruppert/damping
ablations; regret machinery; bank grid or `switch_prob` tuning; re-designing the Xiao/Hermite
sweep (it exists — only regenerate it).

---

## 4. Task 4 — Narrative check

| # | Claim | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Online estimators must choose how much to retain | conceptual | — |
| 2 | Fixed EF gives a controllable memory scale | ✅ Xiao, Hermite; ❌ MLE | step-response ±8% (Xiao/Hermite); MLE unmeasured |
| 3 | Bias–variance / responsiveness–stability trade-off | ✅ Xiao, Hermite; ⚠️ MLE | exact decomposition, 40 seeds; MLE only via async drift grid (20 seeds, not decomposed) |
| 4 | Different dynamics prefer different memories | ✅ Xiao, Hermite (Gaussian); ⚠️ MLE | 36 → 53 → 235 → 495 → cumulative; reproduces on rerun |
| 5 | No single fixed memory is uniformly appropriate | ✅ | recovery vs settled 2.3×/3.1×; `one_setting` loses to per-scenario optima everywhere |
| 6 | VFF and bank respond to it (ρ-scored bank) | ✅ with an honest qualifier | banks beat one fixed setting 12/12 (rerun); vs the per-scenario oracle they win only on `jump`, and for Xiao that win is not robust (−1.8σ on rerun); VFF is erratic (loses to one fixed setting on `oscillate`) |
| 6b | A bank also helps when components are scored on the copula density (log score) | ✅ narrow | `memory_tradeoff.md`: 60 fresh seeds, paired CIs, whole-run `jump`, `drift`, `constant`. Holds for one 8×8 histogram (= Xiao in rank-PIT mode), Gaussian margins, and one VFF design; the ranking depends on the scoring window |
| 7 | Holds across two nonparametric estimator classes (Xiao's grid and Hermite's series) under Gaussian copula data | ✅ narrowed | claims 2–6 are measured for both (6b for Xiao only); the narrowed wording drops "semiparametric" |
| 7+ | *Extension:* holds for the online MPL, and under a misspecified family | ❌ not measured | MUST-1 (MPL on the memory axis + step response), MUST-2 (MPL bank), MUST-3 (t₄ sweep). Until they run, the thesis should not claim the trade-off for the MPL |

Wording notes for the claims:
- Claim 2: for the MLE, memory is defined on θ = atanh r (Fisher z), not on ρ_S (§6).
- Claim 4: the optimal memories reproduce on both platforms; the RMSE values do not (§0.1).
- Claim 6: say "these VFF designs", not "VFFs": five different rules share the name (§1.1). The
  "Xiao bank beats the per-scenario oracle on `jump`" result is −3.3σ on macOS and −1.8σ on Linux,
  so it should be qualified or dropped.

Claims 1–7 as narrowed are supportable now from Xiao and Hermite. Claim 7+ is exactly MUST-1 to
MUST-3.

---

## 5. Tail dependence

**What exists** (`td:notes/studies/tail_dependence.md`, all in copula space with *known* uniform
margins, using a ratio estimator χ̂_L(q) = J/K — the EW count of joint lower-corner events over
the count of marginal-corner events — which lives inside `experiments/tail_*.cpp`, not
`src/estimators/`):
- Step 1: finite-q bias table; Var χ̂ ≈ χ(1−χ)/(qM), verified to 0.89–1.08.
- Step 2: matched-ρ_S family switches, oracle hard restart vs fixed memory vs bank. Jump law
  M\* = √(4V/(qΔ²)) to 0.92–1.09.
- Step 3a/3b: Clayton–Gumbel mixture path at constant ρ_S; soft restart beats the best fixed
  memory in 48/48 cells (rerun, exact); γ\* ≈ 1/(1+Z²); a single γ = 0.25 wins every cell.
- Only **jumps** in χ. No χ drift, so the cube-root law for χ is untested. None of Xiao, Hermite
  or MPL produces χ.

**Why it fits the memory thesis rather than widening it** [interp]. Forgetting trades variance
against lag; it cannot remove either the finite-q threshold bias (nonparametric) or the
family-shape bias (parametric: a Gaussian MPL has λ_L = λ_U = 0 for every r, so its χ(q) is fixed
by ρ_S). That is one sentence connecting tails, memory and misspecification.

**Minimum tail extension (MUST if tails are in the thesis):** one experiment on the existing
Clayton–Gumbel mixture path, with π drifting linearly (ρ_S constant by construction), sweeping
memory for:
- (i) the nonparametric χ̂(q). [interp] Xiao's grid already holds it: with uniform (rank-PIT)
  cut-points and B = 20, `counts_[0]/row_counts_[0]` is exactly the ratio estimator at q = 0.05,
  under the same λ, so no new estimator is needed. Not yet implemented or checked.
- (ii) the Gaussian MPL's implied χ(q) = C_r̂(q,q)/q.
- (iii) optionally one tail-capable MPL (t₄).

Prediction: the ρ_S-optimal memory is "cumulative" (ρ_S does not move) while the χ-optimal memory
is finite. In other words, **the optimal memory depends on the functional being estimated, not
only on the data.** Hermite is not recommended for χ: a truncated smooth basis is weakest in the
corners. Its Fréchet failure (`copula_v3.md`) is the same mechanism.

---

## 6. Glossary (2026-10-20)

One definition per term; other notes should link here rather than re-derive.

### 6.1 Effective sample size is not memory in ticks

Write w_j for the normalised weight a scheme puts on the observation j ticks old (Σ_j w_j = 1).
Three different scales come out of the same weights:

- **Effective sample size (ESS)**, N_eff = 1/Σ_j w_j² (Kish). It sets the *variance*: a weighted
  mean of i.i.d. terms with variance σ² has variance σ²/N_eff. Any "noise ∝ N_eff^(−1/2)"
  statement means this quantity.
- **Mean age**, ā = Σ_j j·w_j. It sets the *lag*: on a linear drift of slope d the tracking bias
  is d·ā.
- **Memory in ticks**, M. The axis every probe and figure uses. It is a *nominal* length: the
  hard-window length whose ESS equals the scheme's steady-state ESS. It is a design coordinate,
  not a measured property, and for some schemes below it is neither the ESS nor twice the mean age.

| Scheme | Knob → α | Steady-state ESS | Memory in ticks as used | Mean age |
| --- | --- | --- | --- | --- |
| Hard window of W ticks (`windowed_rank`, `xiao_win`, the rank-PIT window) | — | W | M = W | (W − 1)/2 |
| Cumulative | — | n (grows) | "cumulative", no M | (n − 1)/2 |
| Exponential, retention λ (Xiao, both MLEs, the `memory_tradeoff` histogram) | α = 1 − λ | (2 − α)/α = (1 + λ)/(1 − λ) | M = ESS, so λ = 1 − 2/(M + 1) | (1 − α)/α ≈ M/2 |
| Exponential, update weight λ (Hermite) | α = λ | (2 − λ)/λ | M = ESS, so λ = 2/(M + 1) | (1 − λ)/λ ≈ M/2 |
| Exponential, *first t ticks* | as above | (1 + λ)(1 − λᵗ)/((1 − λ)(1 + λᵗ)), reaching M only for t ≫ M | M (the steady-state value) | < M/2 |
| Kalman local level, blocks of B ticks, gain K_t → K | α = K per block, **only once the gain has converged** | (2 − K)/K blocks of the filter's *state*, i.e. B(2 − K)/K ticks | M = B(2 − K)/K ticks — **approximate**: it describes the steady-state filter state, not the ρ̂ readout, whose step response reads ~27% short on a large step (`memory_sweep.md` §3) | ≈ B(1 − K)/K + (B − 1)/2 ≈ M/2, same caveats |
| Recursive MLE in θ = atanh r | α = 1 − λ on H_t | (2 − α)/α *in θ* [interp, linearisation] | M as for exponential; **not verified in ρ space** | unmeasured |
| VFF (any rule) | α_t varies, learned from the data | kernel ESS of the realised product weights w_j ∝ α_{t−j}∏_{s>t−j}(1 − α_s); it lags the current setting, and like the bank's it describes weights, not estimation variance | reported M_t = (2 − α_t)/α_t: the memory the *current* α_t would have if held fixed | lags too |
| Bank of K arms with memories M_k, weights π_k | per arm | **kernel-equivalent ESS under fixed bank weights**: 1/Σ_j(Σ_k π_k a_{k,j})² with a_{k,j} = α_k(1 − α_k)^j, the ESS of the mixed observation-weight kernel. It is at least the π-weighted harmonic mean 1/Σ_k(π_k/M_k), and can be far below the average arm memory. It is **not** in general the variance-equivalent ESS of the bank's estimate (see below) | **average arm memory** M̄ = Σ_k π_k M_k, what banks report and plot; a summary of the weights, not an ESS | Σ_k π_k ā_k for the mixed kernel |

The kernel-equivalent ESS describes the combined observation weights when the arms average
comparable weighted observations and the bank weights are held fixed. It is not, in general, the
variance-equivalent ESS of the bank's estimate: Xiao's ρ̂ (and the MLEs') is a nonlinear function
of the weighted data, and the bank weights π_k are learned from the same data.

Illustration of the kernel calculation only: arms at 25 and 8000 ticks with fixed π = ½ each have
an average arm memory of 4012 ticks, but a kernel-equivalent ESS of about 98 (harmonic-mean bound
50), because the short arm's heavy recent weights dominate Σ_j w_j². This does not mean such a
bank's estimate carries 98 observations' worth of information.

So: for a fixed exponential or hard-window scheme at steady state, M and ESS coincide by
construction and the mean age is about M/2. For a VFF, a bank, the start-up period, or a
parameter-space recursion they differ, and figures that plot "memory" for those methods are
plotting M_t or M̄, not an ESS. Xiao's and the MLEs' *estimates* are nonlinear in the weighted
counts, so the bank row's kernel view describes their weights, not their estimation variance.

Rank-PIT adds a **second, marginal memory**: a hard window of 500 ticks over which the margins are
ranked, independent of the dependence memory M.

### 6.2 Estimators

- **Xiao**: `XiaoSpearmanEstimator`, an exponentially forgotten B×B count grid; ρ̂ is the rank
  correlation of grid mid-ranks. In rank-PIT mode with equal-area cells it also gives a histogram
  density (`log_density_uv`).
- **Hermite**: `HermiteEstimator`, Hermite-series coefficients updated with weight λ.
- **MLE / two-stage (IFM)**: `GaussianCopulaMLE` and `ParametricCopulaMLE` with the default input
  u = Φ(standardised x). Parametric (Gaussian) margins; **not** pseudo-likelihood. All
  `gaussian_mle*` registry keys except `gaussian_mle_pit` are this.
- **MPL**: the same recursions fed sequential rank-PIT pseudo-observations
  (`rank_pit_window = 500`, e.g. `gaussian_mle_pit`). Only this is maximum pseudo-likelihood.

### 6.3 Banks

| Name in notes | Code | Arms | Score per arm | Weight rule |
| --- | --- | --- | --- | --- |
| ρ-scored bank (`xiao_bank`, `hermite_bank` in `adaptive_memory.md`) | `EstimatorBank` | probe: 7 memories 25–1000 ticks; registered keys: a λ grid, M ≈ 39–1999 | squared error of ρ̂_k against p_t = 12(u − ½)(v − ½) | exponential weights per 20-tick block + fixed share 0.002 |
| log-scored fixed share (`memory_tradeoff.md`) | in `probe_memory_tradeoff.cpp` | 10 memories 25–8000 ticks | predictive log density | as above, α = .008, η = 1 |
| discounted bank (`memory_tradeoff.md`) | in `probe_memory_tradeoff.cpp` | same 10 | predictive log density | L ← βL + (1 − β)s, π ∝ e^(−ηL), η = c/(1 − β); no fixed share |
| `kalman_bank` | `KalmanBank` | σ_ω² grid | Gaussian predictive density | fixed share; out of scope |

### 6.4 VFFs

Five rules share the name (signal → rule): Xiao (ρ-level rank residual, α ← α·C^r), Hermite
(coefficient-change ratio of fast/slow EWMAs), `CopulaFit` / `gaussian_mle_vff` (score drift
m²/v), Kalman (out of scope), and the Rosenblatt-residual VFF in `memory_tradeoff.md` §4 (Xiao's
rule driven by conditional-PIT residuals). Details: §2.8 and `memory_tradeoff.md` §4. A row
labelled "VFF" compares one of these, never VFFs in general.

### 6.5 Baselines, scores and uncertainty

- **Per-scenario best fixed memory (hindsight)**: `known_best` in `probe_adaptive`, "oracle" in
  `adaptive_memory.md`, "hindsight best" in `memory_tradeoff.md` §3.2. Chosen after seeing that
  scenario's result.
- **Single tuned fixed memory**: `one_setting` in `probe_adaptive`; the `fixed` control in
  `memory_tradeoff.md` (tuned on jump + drift, 1170 ticks).
- **RMSE**: against the true ρ_S per tick (§2.10).
- **ρ score**: squared error against the one-tick proxy p_t; with exact uniform margins its
  expectation is minimised by ρ̂ = ρ_S (with estimated rank-PIT margins p_t is biased by
  O(1/N_eff) of the rank window).
- **Log score**: predictive log density of the next observation. **Excess log loss**: log of the
  true copula density minus log of the predicted density, both at the *true* PIT; its expectation
  is the per-tick KL divergence. Reported ×10⁻³ nats/tick.
- **Paired σ** (`adaptive_memory.md`): mean paired difference over its standard error.
  **Paired 95% CI** (`memory_tradeoff.md`): mean ± t₀.₉₇₅·SE. |paired σ| > ~2 is the same
  statement as "CI excludes 0".

