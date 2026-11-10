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
