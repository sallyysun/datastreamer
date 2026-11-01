# Code changes 2026-11-01: make the code match thesis Chapters 3–5

Thesis version: `Sally_Sun_Honours_Thesis_4.pdf`, read without the regret part of §5.3.4. The audit
behind these changes is `notes/thesis/chapter6_plan.md`. Branch `claude/busy-goldberg-6qrua3`
includes `bank-vff-study`.

## 1. `ForgettingBank`: thesis Algorithm 1, exactly (`src/estimators/dependence/forgetting_bank.hpp`)

| Algorithm 1 line | Implementation |
| --- | --- |
| 1, 13 margins | One shared `RankPit` per margin (500-tick sliding window). `transform` ranks $x_t$ against the past, then absorbs it. |
| 2 | $w_{k,0}=1/K$ |
| 5 | $(\hat U_t,\hat V_t)$ from the margins held through $t-1$ |
| 7, squared | $(\hat p_t-\hat\rho^{(k)}_{t-1})^2$, $\hat p_t=12(\hat U_t-\tfrac12)(\hat V_t-\tfrac12)$ |
| 7, log | $-\log c_{g^{-1}(\hat\rho^{(k)}_{t-1})}(\hat U_t,\hat V_t)$: the method-of-moments copula of a chosen family (Gaussian by default), §4.3.1. This lets every component be log-scored, Hermite included. |
| 9 | Hedge every tick with a **constant** η, in log space |
| 11 | Fixed share every tick, $w=(1-\delta)\tilde w+\delta/K$ |
| 12 | Every component is updated with the shared pseudo-observations, passed as normal scores $\Phi^{-1}(\hat U_t)$ so that Hermite and the MLE can take them |
| 15 | $\hat\rho^{\text{bank}}=\sum_k w_k\hat\rho^{(k)}$ |

Choices Algorithm 1 leaves open. Each needs one sentence in the thesis:
- **Warm-up.** Scoring starts once the margin window holds 30 observations.
- **Non-finite losses.** A non-finite loss is replaced by the worst finite loss of that tick.
- **Bank density.** For the log metric the bank's own density is $c_{g^{-1}(\hat\rho^{\text{bank}})}$ (`log_density`).

**Registry**
- `xiao_bank`, `hermite_bank`, `mle_bank`: the squared-loss bank, η = 0.5, δ = 1e-4.
- `*_bank_log`: the log-loss bank, η = 1, δ = 4e-4.
- `*_bank_uniform`: η = δ = 0, i.e. frozen uniform weights (the averaging control).
- `*_bank_block`: the earlier `EstimatorBank` (20-tick blocks, $\eta_t=1/(2v_t)$), kept so that
  `adaptive_memory.md` stays reproducible.

The η and δ values are the per-tick equivalents of the tuned block rules. They have **not** been
retuned for Algorithm 1. Arms: M = 39, 99, 199, 399, 1999 ticks. `mle_bank` components are now
online MPL with expected Fisher scoring (`ParametricCopulaMLE`, Gaussian), not the IFM
`GaussianCopulaMLE`.

## 2. Expected Fisher information for every family (§3.5.4)

`copula::expected_info_phi`: $I(\phi)=\mathbb E[(\partial_\phi\log c)^2]$ in `CopulaFit`'s
coordinate. It is closed form for the Gaussian ($1+\rho^2$) and tabulated once by Gauss–Legendre
for Clayton, Gumbel and t. Monte Carlo agrees to within 1% at ρ_S ≈ .29, .58 and .84
(`check_thesis_bank`). `ParametricCopulaMLE::Config::expected_info` now applies to all families;
before, it covered only the Gaussian, and the others used the outer product of scores.

## 3. Online MPL keys (§4.3.2)

`mpl_gaussian`, `mpl_clayton`, `mpl_gumbel`, `mpl_t`: rank-PIT pseudo-observations (window 500)
feeding expected-Fisher scoring with retention 0.99. All four are in `all_estimator_keys()`.

## 4. Pseudo-observations are R/(n+1) (§3.5.1)

`RankPit` returned $(r+\tfrac12)/(m+1)$. It now returns $R/(m+2)$, with $R$ the rank of $x_t$
among the $m$ stored values plus itself: the mean-based pseudo-observation. Every rank-PIT number
moves by O(1/500).

## 5. Hermite start-up (§4.2.3)

The EW coefficient update now uses $w_n=\max(\lambda,1/n)$ by default (`init_first_obs`). That
gives $\hat A^{(1)}=h(X_1)h(Y_1)^\top$, as in the thesis, then the stationary running mean until
$n=1/\lambda$, then the exponential update.

The thesis recursion taken literally ($(1-\lambda)$ from $n=2$) was run first. It leaves the first
observation with weight $(1-\lambda)^n$, and M = 1999 Hermite on `constant` scored RMSE 0.077,
against 0.021 with this rule. The thesis equation should change to match (see `chapter6_plan.md`
Part D). The cumulative path is unchanged, and the reference harness sets
`init_first_obs = false` to keep parity with `hermiter`.

## 6. Other

- `dgp::param_from_spearman` moved to `copula::param_from_spearman`, so estimators can invert
  ρ_S without including the generator. `dgp::` keeps a `using` alias.
- `copula::log_density(Family, …)` dispatcher.
- Two `HermiteEstimator::Config` designated initialisers in `registry.hpp` were out of
  declaration order. clang accepts this; g++ rejects it. Reordered; no behaviour change.

## 7. New harness programs (`experiments/`)

- `check_thesis_bank` checks four things: Algorithm 1 against a hand-written loop (both losses, to
  1e-15); uniform weights when η = δ = 0; the Fisher tables against Monte Carlo; registry
  construction.
- `probe_thesis_bank [seeds]` (with `COPULA=` to change family) runs, for each of Xiao, Hermite
  and MPL: the five fixed arms, the squared bank, the log bank and the uniform control. Output
  goes to `results/thesis_bank[_fam].csv`, with weight traces in `results/thesis_bank_weights*.csv`.

First run (Gaussian, 20 seeds, Linux/libstdc++, η and δ untuned). The table shows RMSE ×10⁻³
over [500, 5000), as the difference from the per-cell hindsight-best fixed memory, with paired
95% CIs.

| scenario | best fixed | `bank` | `bank_log` | `uniform` |
| --- | --- | --- | --- | --- |
| constant (xiao / hermite / mle) | 20.5 / 21.2 / 22.0 (M = 1999) | +7.2 / +7.0 / +6.1 | +6.0 / +6.3 / +6.4 | +32 / +34 / +24 |
| drift | 66.1 / 68.7 / 61.9 (M = 199) | +3.6 / +2.2 / +3.9 | +3.9 / +2.7 / +4.5 | +23 / +24 / +23 |
| jump | 80.3 / 82.5 / 76.5 (M = 199) | −4.1 / −3.8 / −6.2 | **−20.1 / −19.5 / −19.2** | +12 / +12 / +11 |
| oscillate | 186.9 / 192.8 / 175.6 (M = 39) | +0.4 / +0.4 / +0.5 | +4.8 / +6.4 / +2.4 | +108 / +103 / +119 |

CI half-widths are 0.1–3.7. These are **not thesis results** yet: tune η and δ on separate seeds
first, then rerun with 60 evaluation seeds (`chapter6_plan.md` Part C).

Note: `experiments/Makefile` targets have no header dependencies. After editing a header,
rebuild with `make -B <target>`.
