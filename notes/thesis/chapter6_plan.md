# Chapter 6: what it should contain, and does the code run Algorithm 1? (2026-11-01)

Written against `Sally_Sun_Honours_Thesis_4.pdf` (59 pp.; Ch. 5 §5.3 and Algorithm 1). The regret
part of §5.3.4 is ignored, as asked. Code was read on `origin/bank-vff-study` (the newest results).
Shorthand: **EB** = `src/estimators/dependence/estimator_bank.hpp` (the registered
`xiao_bank` / `hermite_bank` / `mle_bank`, used in `adaptive_memory.md`). **MT** =
`experiments/probe_memory_tradeoff.cpp` (the log-scored banks in `memory_tradeoff.md`).

---

## Part A: Does the simulated bank match Algorithm 1?

**Short answer: not exactly.** The bank code follows the structure of Algorithm 1: score before
updating, multiplicative weights, a fixed-share floor, and K fixed-memory components. It differs
in six ways that Chapter 6 has to state, or that need a small config change and a rerun. The
discounted bank that gives the headline `jump` result is not in Chapter 5 at all.

| # | Algorithm 1 (thesis) | What the code does | Matters? | Fix |
| --- | --- | --- | --- | --- |
| 1 | **Squared loss** $\ell=(\hat p_t-\hat\rho^{(k)}_{t-1})^2$ with a **constant** η | EB uses $\ell/(2v_t)$, where $v_t$ is an EWMA (rate 0.001) of the *bank's own* residual variance. That makes the step a time-varying, data-dependent $\eta_t=1/(2v_t)$, i.e. a Gaussian predictive likelihood (Bayesian model averaging), not Hedge with fixed η. | Moderate. Ranking within a block is unchanged, but "η" in the thesis is not the η that was run. | Either describe $\eta_t$ in Ch. 5, or rerun with `var_rate = 0` (then $v=1$ and $\eta=\tfrac12$ is constant). |
| 2 | Weights updated **every tick** | EB and MT both update **once per 20-tick block**: losses are summed over the block, and the Hedge step and fixed share are applied at the block end. Hedge over a block equals the product of 20 per-tick steps, but weights are frozen inside the block, and fixed share (α per block) is applied 20× less often. Per tick that is roughly δ ≈ α/20, not identical. | Small, but it is a different algorithm. | Write Algorithm 1 with a block length B (B = 1 recovers the per-tick version), or rerun with B = 1. |
| 3 | "Update each estimator with $(\hat U_t,\hat V_t)$"; shared margins | **Scoring** margins are shared and prequential in both: `RankPit` (a 500-tick sliding window) ranks $x_t$ against the past *before* absorbing it ✓. But in EB the components are updated with **raw** $(x,y)$ and use **their own** margins: Xiao uses cut-points frozen from the first 100 observations, Hermite an EW standardisation with its own λ, `mle_bank` a cumulative Welford standardisation (parametric, so **not MPL**). Only MT (histogram, or Xiao in rank-PIT mode) feeds components the shared rank-PIT. | Your §5.3.1 bounds still hold, because they only need the *scoring* margins to be common, and they are. But line 12 of Algorithm 1 and the sentence "the bank uses the same marginal estimates across its estimators" are false for EB. | Reword line 12 ("update $E_k$ with $(X_t,Y_t)$ using its own marginal model"), or rerun EB with rank-PIT components. |
| 4 | Output $\hat\rho^{\text{bank}}=\sum_k w_k\hat\rho^{(k)}$ | EB ✓ (clamped to [−1, 1]). MT reports two outputs. ρ: the same weighted average ✓. **Density**: the mixture $\sum_k w_k\hat c^{(k)}$, which gives the excess-log-loss headline numbers. The mixture output is not in Ch. 5. | Yes, for every log-loss result. | Add the mixture construction to §5.3.3 (the sample draft's eq. 5.5–5.6 already has it, with the ρ-of-mixture identity). |
| 5 | Only Hedge and fixed share | MT's **discounted bank** ($L\leftarrow\beta L+(1-\beta)s$, $w\propto e^{-\eta L}$, $\eta=c/(1-\beta)$, no fixed share) is the best whole-run method on `jump`. | **Yes.** The main Ch. 6 result uses an algorithm Ch. 5 never defines. | Add discounted softmax to §5.3.2 (the sample draft's eq. 5.11–5.14), or drop it from Ch. 6. Adding it is better. |
| 6 | Starts at t = 1, $w_{k,0}=1/K$ | Scoring starts when the rank window holds 30 values. EB's `estimate()` returns 0 for the first 200 ticks, and all evaluation windows start at tick 500. Components also have burn-ins (Hermite and MLE 50, Xiao calibration 100) during which they may report 0 and still get scored. | Small. It mostly washes out before tick 500. | One sentence in Ch. 6 §6.1 on warm-up. |

**Other mismatches between Ch. 5 and the code**

- **λ convention.** The thesis (§5.1) uses λ = weight on the new observation. Xiao, the MLEs and
  the MT histogram use λ = **retention**; only Hermite matches the thesis. Every figure is in
  "memory in ticks" M, with λ = 1 − 2/(M+1) (retention) or 2/(M+1) (weight). Report M, or α = 1 − λ_code,
  in Ch. 6, never the raw code λ.
- **Which estimators are banked by which loss.** ρ-scored banks: Xiao, Hermite, and parametric
  Gaussian MLE (not MPL). Log-scored banks: **only** an 8×8 forgetting histogram. Xiao in rank-PIT
  mode reproduces it to every digit (`memory_tradeoff.md` §7). There is **no log-scored Hermite
  bank** (a truncated Hermite density can go negative) and **no log-scored MPL bank**, although an
  MPL Gaussian copula has a valid density. Ch. 5's log-loss section reads as if every estimator
  could be log-scored, so say which ones were.
- **The densities are not copula densities.** The histogram margins are only approximately
  uniform (the worst row or column deviates by about 13% at M = 1000). Your KL argument (§5.3.1) is
  fine as KL on the unit square, which is what the true-PIT excess log loss estimates. Just don't
  call it a copula density in Ch. 6.
- **The independence assumption** in your §5.3.1 proofs ($\mathbb E[p_t\mid\mathcal F_{t-1}]=\rho_{S,t}$)
  holds in the harness: draws are independent given the dependence path, and the margins are
  standard normal **in every scenario**, so $\varepsilon_t$ is pure rank-window sampling error
  (bias O(1/500)). Say this in Ch. 6. With real data the margins also drift, and the bound's
  $\varepsilon_t$ term would grow.
- **VFF.** §5.2 describes VFF generically. The VFFs actually run are specific rules: Xiao's
  ρ-residual rule in `adaptive_memory`, and Xiao's rule driven by Rosenblatt residuals in
  `memory_tradeoff`. Ch. 5 or Ch. 6 must name the rule. The negative VFF result applies to that rule only.
- **No uniform-average control.** Ch. 6 should compare the bank with uniform averaging of the
  same components. That control exists only for the Kalman bank (`kalman_bank_uniform`). Adding it
  to MT is one extra method (fixed share with α = 1).

**Verdict.** EB with `var_rate = 0` and `block = 1` would be Algorithm 1 exactly for the
squared loss. MT's fixed share would be Algorithm 1 for the log loss with B = 1. You have two
options. One is to change Ch. 5 to match the code: add a block length B, the time-varying $\eta_t$,
discounted softmax and the mixture output. The other is three config changes and one rerun of each
probe. Changing Ch. 5 is cheaper and more honest about what was run. A B = 1 rerun is worth doing
anyway as a robustness row.

---

## Part B: What Chapter 6 should contain

The current §6.1 describes the **live server's** GBM price and volume DGP (a 4-variate Gaussian
copula). **No offline result uses it.** Move that text to §2.3 (the implementation) and replace it
with the harness DGP below. I suggest renaming the chapter "Simulation Study", since it holds
results as well as methodology.

### 6.1 Experimental design

- **DGP.** `CopulaSource`: two standard-normal scores per tick, independent across ticks given a
  dependence path $d_t$. The copula family maps $d_t$ to a common target $\rho_S=(6/\pi)\arcsin(d_t/2)$,
  so the families are compared at matched Spearman's ρ. Families: Gaussian, Clayton, Gumbel and t (ν = 4).
- **Scenarios** (N = 5000, burn-in 500): `constant` d = 0.6; `drift` −0.5 → 0.8 linearly;
  `jump` 0.2 → 0.8 at t = 2500; `oscillate` amplitude 0.6, period 400; shape switches at fixed ρ
  (Gaussian→t, Clayton→Gumbel at ρ ≈ .58 and .79). Give a table with the true $\rho_{S,t}$ path of each.
- **Estimators and configurations.** Xiao (B = 20, or 8 in the log-bank study), Hermite (order 10),
  MPL / parametric MLE. Memory grid in ticks with the λ mapping (Part A). Rank-PIT window 500.
- **Banks and adaptive methods**, exactly as run: the block, η or $\eta_t$, α, β, c and arm grid of each.
- **Metrics.** (i) RMSE against the true $\rho_{S,t}$. (ii) Excess log loss at the true PIT,
  whose expectation is per-tick KL. (iii) Recovery after the jump (windows 0–100, 100–500 and
  500–1500 ticks). (iv) `store_bytes` and ns/tick (`adaptive_memory_cost.csv`, `bench_rate`).
- **Protocol.** Tuning seeds kept separate from evaluation seeds (MT: 30 tuning, 60 evaluation).
  One configuration per method, never tuned per scenario. Paired 95% CIs. One platform
  (Linux/libstdc++); `thesis_audit.md` §0.1 explains why that matters.

### 6.2 The effect of a fixed forgetting factor (source: `memory_sweep.md`)

- RMSE against memory, a U-curve per scenario, with the best memory and the bias/variance split.
  This supports §5.1.2 empirically.
- Xiao against Hermite at matched memory ("memory matters more than the algorithm").
- Status: **run, but needs regenerating on one platform.** The macOS and Linux numbers differ by 1–3 SE.

### 6.3 Adaptive memory: VFF against bank against fixed (sources: `adaptive_memory.md`, `memory_tradeoff.md`)

- **ρ-scored bank** (EB) against the per-scenario hindsight-best memory and against one tuned
  fixed memory, for Xiao and Hermite.
  - It beats the single fixed setting in 12/12 cells.
  - Whether it beats the hindsight best on `jump` is **not robust** (−3.3σ on macOS, −1.8σ on Linux). Don't claim it.
- **Log-scored banks** (fixed share and discounted) against VFF against fixed. This is the main table (`memory_tradeoff.md` §3.1–3.2):
  - Whole `jump`: discounted beats fixed by 6.8 and fixed share by 0.6 (×10⁻³ nats/tick).
  - The discounted bank is the **worst** of the four in the first 100 ticks after the jump.
  - `drift` costs about 1.0. `constant` is about 0.5 off the hindsight-best memory.
- **Figure:** the true ρ, the component estimates and the bank estimate, with the weights and the
  average arm memory over time (`traces.png`). Ch. 5's sample draft asks for exactly this.
- **Why the VFF fails** (Rosenblatt e² falls on a jump toward stronger dependence), stated as specific to that rule.

### 6.4 Which loss to score with

- Loss and metric cross table, on the whole `jump` run:
  - Spearman RMSE (×10⁻³): ρ-scored fixed share 79.9, log-scored fixed share 104.8, discounted 121.3.
  - The log-scored banks win on KL.
- This tests your §5.3.1 claim directly: the two losses agree to second order inside one family
  and diverge otherwise. **A bank tuned for the copula is not tuned for ρ.** Run the ρ-scored
  bank's KL to complete the 2×2. MT computes it but does not report it.

### 6.5 Changes in shape at constant ρ

- Gaussian→t and Clayton→Gumbel (`memory_tradeoff.md` §3.3). Squared loss is blind to these by
  construction (ρ does not change), so this is where the log loss earns its place.
- The banks are slower than the fixed control just after Clayton→Gumbel, because they carry
  about 6000 ticks of memory beforehand.

### 6.6 Computational cost

- Storage $\sum_k M_k+O(K)$ and time per tick, measured: `adaptive_memory_cost.csv` and
  `bench_rate.cpp`. One small table against stream length shows bounded state.

### 6.7 Summary of findings

- What generalises, and what holds only for one estimator, one change direction or one VFF rule.

### Remove from the current §6.5 draft, or run first

- **Integrated squared error of $\hat C$:** nothing computes it. Drop it (excess log loss already
  covers the full copula), or add it to MT.
- **"Offline batch estimators" as a comparator:** never run. Drop it, or add a hard-window batch
  MPL as a reference row.
- **"Adaptive forgetting methods" in the plural:** only one VFF rule per study was run.

---

## Part C: Minimum work before Ch. 6 is writable

1. **Choose:** update Ch. 5 to the code as run (block B, $\eta_t$, discounted softmax, mixture
   output). Recommended. The alternative is the three config changes in Part A plus a rerun.
2. **Regenerate** `memory_sweep` and `adaptive_memory` on Linux from one commit, and commit the CSVs.
3. **Add to MT** a uniform-average control (α = 1), the ρ-scored bank's KL, and optionally a B = 1 row.
4. **Optional:** a jump toward *weaker* dependence (0.8 → 0.2). This is the predicted case where
   the VFF should react correctly (`memory_tradeoff.md` §4).
