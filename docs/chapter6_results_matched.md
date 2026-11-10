# Chapter 6 results inventory: matched dependence paths (2026-11-10). PRIMARY PROTOCOL

This supersedes `chapter6_results_final.md` for every cross-family comparison. That file remains the
record of the original paths, including the signed-drift results for Gaussian and Student-$t_4$.

**Configuration:** the frozen configuration plus three changes.
1. `SCENARIOS=matched`, defined in `experiments/scenarios.hpp`:
   - drift $d_n = 0.2 + 0.6(n-1)/(N-1)$ (linear in $d$; $\rho_S$ from 0.191 to 0.786);
   - oscillation $d_n = 0.5 + 0.3\sin(2\pi(n-1)/400)$ ($\rho_S$ from 0.191 to 0.786);
   - constant (0.6) and jump (0.2 → 0.8 at 2500) unchanged.
2. Log-bank fixed share $\delta = 10^{-4}$, re-selected on dev seeds 1–30 under the matched paths
   (it was $3\cdot10^{-4}$).
3. Everything else re-selected on dev seeds and **unchanged**:
   - squared bank $(\eta,\delta) = (1, 10^{-4})$;
   - log bank $\eta = 1$;
   - VFF rates 0.02 and 0.1, by the fixed rule in `select_vff_rates.py`.

**Code fingerprint:** `fe4dc392d4f5e518`.

**Generator check:** every family's achieved $\rho_S$ equals the target to machine precision for
$d \in [0.2, 0.8]$. Sample Spearman from 4·10⁵ draws per point is within 0.0032 of it (about 2 MC
SE), $t_4$ included. The true dependence paths are therefore **identical across the four families
in every scenario**, and cross-family differences reflect copula structure.

**Seeds:**
- evaluation: 42–101 (held out from this audit's decisions; used in earlier studies);
- budget: test 42–141, validation 1000–1039.

**Conventions:** as in `chapter6_results_final.md` (W/L/T at 2 paired SE per cell; parenthesised
values are the mean paired difference across cells, ×10⁻³, negative favouring the first-named method;
best comparators chosen once per cell). **The budget study now saves per-seed results**, so its
comparisons are paired too.

**Full tables:** `experiments/results/matched_eval_2026-11-10.txt`.
**Figures:** `notes/thesis/figures/matched/`.

## Main banks

| A vs B (RMSE of $\rho_S$) | $N_{\mathrm{eff},m} = 500$ | $N_{\mathrm{eff},m} = 1999$ |
|---|---|---|
| Log bank vs hindsight-best fixed memory, drift (12) | 0/12/0 (+4.6) | 0/12/0 (+4.3) |
| Log bank vs hindsight-best fixed memory, jump (12) | 12/0/0 (−16.6) | 12/0/0 (−16.7) |
| Log bank vs squared bank (24) | 23/0/1 (−3.8) | 23/0/1 (−4.0) |
| Squared bank vs uniform weights (24) | 24/0/0 (−16.3) | 24/0/0 (−16.5) |

## VFF (common forgetting schedule)

A = VFF, B = log bank.

| | constant | drift | jump | oscillation |
|---|---|---|---|---|
| Slow (0.02) | 0/7/5 (+2.6) | 1/0/11 (−0.2) | 0/12/0 (+27.6) | 8/3/1 (−3.4) |
| Fast (0.1) | 0/12/0 (+15.8) | 0/12/0 (+6.5) | 0/12/0 (+10.5) | 0/12/0 (+18.6) |

- **Slow VFF:**
  - no detectable difference from the log bank in 11 of 12 drift cells;
  - better in 8 of 12 oscillation cells;
  - worse in 7 of 12 constant cells and in every jump cell.
- **Fast VFF:** worse in all 48 cells.

## Regret (1440 runs; family bank 480; each run checked at 8 switch budgets)

- **Covered procedures:** 0 runs exceed their bound. These are the density-mixture log bank (fixed
  share $\delta = 10^{-4}$ and Hedge), squared loss at $\eta = 1/32$, and the family-bank
  mixture.
- **Uncovered procedures** against the reference expression:
  - squared bank ($\eta = 1$): 732/1440 runs (50.8%);
  - point-density log bank: 615/1440 (42.7%). The smaller $\delta$ tightens the reference
    expression.

## Optimal memory on drift (Gaussian; Fig. `matched/ch6_neff_optimum.png`)

- **Measured N\***: 393 (Xiao) and 430 (Hermite), against $(2\sigma^2/d^2)^{1/3}$ = 426 and 448
  with $d = 1.19\cdot10^{-4}$ per tick.
  - This shows the bias–variance model describes the measured optimum on this path. It is not an
    independent proof of the theory.
- **The MPL log bank's weighted memory** stays within a median factor of 1.28 of the local optimum
  $N^*(t)$ after tick 1000.

## Family bank (base families rerun on matched paths; shape switches unchanged)

- **$\rho_S$ RMSE** (19 cells):
  - against the Xiao, Hermite and MPL log banks: 9/5/5, 12/2/5, 8/7/4;
  - **against its own best single arm: 5/14/0**.
- **Excess log loss** against the log banks: 16/2/1, 18/1/0, 15/4/0.
- **$\chi_L(0.05)$ and $\chi_U(0.05)$** against the EW ratio estimators: 19/0/0 for every memory.
- **Shape-switch weight-crossing delays** (unchanged scenario): medians 174, 68, 94 ticks.
- **Tail bank:** unchanged. Its regimes are constant-$\rho_S$ shape switches; see
  `chapter6_results_final.md`.

## Full-copula density prediction at constant ρ_S (shape switches; targeted run, seeds 42–101)

Three metrics on the same streams:
- **Spearman RMSE:** dependence strength.
- **Full-copula excess log loss** at the true PIT, (1/T)·Σ[log c_n − log ĉ_{n−1}]: the whole density.
- **χ(0.05) RMSE:** joint extremes.

A = family bank (24 parametric arms). B = the **same machinery restricted to the Gaussian shape**
(`gauss_bank`). Paired, ×10⁻³ (log loss in 10⁻³ nats per observation). All densities compared are
valid and normalised.

| Regime | Window | Spearman RMSE | Excess log loss | χ_L | χ_U |
|---|---|---|---|---|---|
| Gaussian truth (before any switch) | pre | +1.4 | +0.5 | +2.3 | +5.5 |
| Gauss → t₄ | post | +3.3 | **−30.7** | −56.6 | −57.8 |
| Gauss → t₃ (arms assume t₄) | post | +8.2 | **−49.5** | −78.1 | −79.9 |
| Gauss → t₈ (arms assume t₄) | post | +5.2 | **−3.9** | −4.3 | tie |
| Clayton → Gumbel | pre (Clayton) | tie | **−75.3** | −271.9 | −197.5 |
| Clayton → Gumbel | 0–100 after | +18.2 | **+46.4** | +164.8 | +78.4 |
| Clayton → Gumbel | post | +9.9 | **−23.8** | −37.7 | −132.4 |
| Gauss → Clayton | post | +12.9 | **−72.2** | −228.6 | −165.1 |

**Findings:**
- **Modelling the full copula greatly improves density and tail prediction after a change of structure
  at constant ρ_S,** while Spearman RMSE gets slightly worse.
- **Flexibility costs a little when the truth is Gaussian.**
- **There is a temporary lag right after Clayton → Gumbel,** while the weights move.
- **The MPL log bank's Gaussian mixture density** is slightly better than its point density (−0.2 to
  −1.3), so the conclusion holds for either.
- **Full tables:** `experiments/results/density_objectives_2026-11-10.txt`. Method details:
  `ew_marginals.md` §32.

## Density follow-up: Gaussian comparators, best fixed arm, Gaussian floor, mixture truth (seeds 42–101)

Run `TAG=density2`: matched paths, log-bank δ = 10⁻⁴, outputs only. Fingerprint `54185f0f6c977037`. It
reproduces every earlier row exactly (switch log-bank rows shift slightly with the matched δ). Method,
checks and full commentary: `ew_marginals.md` §33. Tables: `experiments/results/density_followup_2026-11-10.txt`.
`hermite_bank_own` is excluded from all density tables (its score integrates to ≈1.07). The own-weight
Gaussian mixture uses the old (η, δ) and is provisional.

| Family-bank mixture minus | Excess log loss W/L/T (19 cells, whole) | ρ_S RMSE W/L/T |
|---|---|---|
| single Gaussian at the combined ρ̂ (MPL log bank) | 15/4/0 (losses: the four Gaussian-truth cells) | 8/8/3 |
| Gaussian mixture, own-bank weights | 17/2/0 | 15/4/0 |

- **Gaussian floor D(t) = KL(c_true ‖ c_G(r*))**, r* = E[Z₁Z₂], numerical (quadrature, checked by Monte
  Carlo). For non-Gaussian truth, almost all of every Gaussian forecast's excess is this floor: Clayton
  70–78, Gumbel 25–29, t₄ 32.6–33.9 (×10⁻³ nats per tick).
  - Estimation adds +1 to +3.5 in stationary windows.
  - The log-weight Gaussian mixture is 0–4.5 better than the single Gaussian. No Gaussian forecaster
    can remove the floor.
- **Best fixed family-and-memory arm** (hindsight, per cell):
  - the family bank is within +0.3 to +0.9 of it on stationary paths, where that arm is the true family
    at long memory;
  - it beats it over the whole window of every family switch (−7.7, −41.0, −24.9) and every ρ_S jump
    (−0.2 to −1.2).
  - So the gain over Gaussian forecasts is mostly **structural** (the right family is in the bank).
    Adaptive weighting adds the ability to change family.
- **Clayton–Gumbel mixture truth** (π 0.75 → 0.25, ρ_S constant; not in the bank):
  - **density:** the family-bank mixture beats all Gaussian forecasts before and after the change
    (−23.6 pre, −4.6 post vs the single Gaussian), but loses for the first 100 ticks (+19.5);
  - **against the best fixed arm** (t₄, M 1999): −7.3 over the whole window, +0.6 post;
  - **tails:** before the change its χ_U is worse than the M = 399 and 1999 counters (+13, +53 ×10⁻³) and
    χ_L worse than M = 1999 (+14). It has projected onto Clayton. After the change it beats every
    counter in both tails (t₄ projection);
  - **so** better density prediction coincided with worse upper-tail estimation before the change, but
    not after.

## Budget study (paired per-seed comparisons; test RMSE ×10⁻²)

**DataSketches-MPL against the better of Xiao and Hermite**, paired over 100 test seeds:

| Budget | W/L/T | Lowest observed RMSE |
|---|---|---|
| 16 KiB | **9/2/1** | 10/12 cells |
| 64 KiB | **9/2/1** | 9/12 |
| 256 KiB | **9/2/1** | 9/12 |

- **The two resolved losses** are Clayton jump (+0.20 ± 0.09) and $t_4$ jump (+0.12 ± 0.09). The
  tie is $t_4$ constant.
- **Drift is now attainable by every family**, and DataSketches-MPL leads in all four drift cells.
  For example, Gaussian drift 4.16 against 4.66, and Gumbel drift 3.99 against 4.66.
- **Constant and jump cells** are identical to the original run, as they should be.

## Stale or provisional (excluded from conclusions)

- **Own-density rows:** the Hermite score is not a normalised density, and the banks were not retuned.
- **Not rerun:** single-seed path figures (`ch6_bank_path*`, `ch6_family_path`), ISE figures,
  tail-CI.
- **Real-data (AAPL/GOOG) study:** not rerun.
