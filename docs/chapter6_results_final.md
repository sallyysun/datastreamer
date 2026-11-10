# Chapter 6 results inventory, frozen final configuration (2026-11-09)

**Platform:** Linux / libstdc++.

**Configuration:** code fingerprint `96f6727b67e98cf8` (see `audit_archive_2026-11-09/MANIFEST.md`).
Algorithms and settings are in `notes/studies/ew_marginals.md` Part III §21.

**Seeds:**
- Development decisions used seeds 1–30.
- Evaluation used seeds 42–101. They were held out from this audit's design decisions, but **used in
  earlier studies** (the pre-audit Chapter 6 results and tuning checks), so they are not a pristine
  holdout.
- Budget study: test seeds 42–141, validation seeds 1000–1039.

**Full tables:** `experiments/results/final_eval_2026-11-09.txt`.

## Conventions (apply to every table below)

- **Cells:**
  - main banks: one family × estimator, so 12 per scenario and 24 over drift + jump;
  - family bank: one copula × scenario, 19 cells;
  - tail bank: one regime × q × tail, 20 cells.
- **W/L/T** compares the first-named method A with comparator B, cell by cell:
  - W: A has lower RMSE (or loss) than B by more than 2 SE of the **paired per-seed difference**
    (60 seeds);
  - L: higher by more than 2 SE;
  - T: no difference detected.
  - The criterion is per cell, with no multiplicity correction.
- **Parenthesised values:** the unweighted mean, across the cells of the row, of each cell's mean paired
  RMSE difference A − B, in units of 10⁻³. Negative values favour A.
- **Hindsight-best fixed memory and best single arm:** chosen **once per cell**, as the arm with the
  lowest mean RMSE across the 60 seeds. The per-seed differences are then taken against that one arm.
  It is not a per-seed oracle.
- **Paired statistics apply to the bank, VFF, family and tail results.** The budget study saves only
  per-configuration means and unpaired SEs (no per-seed output), so its comparisons are rankings plus a
  conservative unpaired check.
- **Excluded from all headline counts:**
  - own-density rows (provisional, §Stale);
  - stale ISE and real-data figures.

## Main banks (N_eff,m = 500; sensitivity N_eff,m = 1999)

| A vs B (RMSE of ρ_S) | N_eff,m = 500 | N_eff,m = 1999 | Pre-audit |
|---|---|---|---|
| Log bank vs hindsight-best fixed memory, drift (12) | 0/12/0 (+4.1) | 0/12/0 (+3.8) | 0/12/0 (+3.8) |
| Log bank vs hindsight-best fixed memory, jump (12) | 12/0/0 (−17.9) | 12/0/0 (−17.8) | 12/0/0 (−17.9) |
| Log bank vs squared bank (24) | 21/0/3 (−3.7) | 22/0/2 (−4.0) | 21/0/3 (−4.0) |
| Squared bank vs uniform weights (24) | 24/0/0 (−18.5) | 24/0/0 (−18.7) | 24/0/0 (−19.3) |

- **The bank beats every fixed memory after jumps,** including the hindsight-best one chosen per cell.
  Under gradual drift it pays an adaptation cost against that comparator.
- **Log weighting** lowers rank-correlation RMSE relative to squared weighting in most cells.
  - The log bank reported here outputs Σ_k w_k ρ̂_k and is scored as a point estimate.
  - It is distinct from the theorem-covered *density-mixture* forecaster that uses the same weights
    (see Regret).

## VFF comparators (common forgetting schedule)

A = VFF, B = log bank (12 cells each).

| | constant | drift | jump | oscillate |
|---|---|---|---|---|
| Slow (rate 0.02) | 3/3/6 (+0.2) | 5/0/7 (−0.8) | 0/12/0 (+28.8) | 9/2/1 (−3.3) |
| Fast (rate 0.1) | 0/12/0 (+13.4) | 0/12/0 (+9.4) | 0/12/0 (+11.8) | 0/11/1 (+12.4) |

- **Slow VFF:**
  - constant: mixed;
  - drift: better in 5 of 12 cells and never worse;
  - oscillation: better in 9 of 12;
  - jumps: worse in every cell.
- **Fast VFF:** worse in 47 of 48 cells, with no detected difference in the remaining (oscillation)
  cell.
- **Reading:**
  - The VFF imposes one forgetting schedule on every estimator. Under gradual or periodic change that
    schedule often performs better than the bank.
  - The bank's advantage is clearest after abrupt changes.

## Regret

- **Covered procedures** (the theorem's assumptions hold):
  - mixture log-loss forecaster (η = 1, fixed share δ = 3·10⁻⁴);
  - its Hedge version (δ = 0);
  - squared loss at η = 1/32 (δ = 10⁻⁴ and δ*);
  - the family-bank density mixture (η = 1, δ*).
- **The bound at each switch budget m:**
  - **with δ > 0** it is the fixed-share switching bound
    B_{T,m}(δ) = log K + (T−1−m)·log[1/(1−δ+δ/K)] + m·log(K/δ). At **m = 0** this is
    log K + (T−1)·log[1/(1−δ+δ/K)], the switching bound at zero switches, *not* the Hedge bound log K;
  - **with δ = 0** (Hedge) it reduces to log K at m = 0 and is infinite for m ≥ 1, so only m = 0 is a
    meaningful check;
  - for squared loss the bound is divided by η, i.e. multiplied by 32.
- **Results:**
  - covered procedures: 0 exceedances (0/1440 runs; family 0/480);
  - squared bank, η = 1: 730/1440 runs (50.7%) exceed the reference expression at some m, 725 of
    them at m = 0;
  - point-density log bank: 333/1440 (23.1%).
- **What the zero is evidence of:** zero exceedances is evidence that the covered procedures are
  implemented correctly. The guarantee itself comes from the assumptions being met.
- **Uncovered procedures:** the counts are exceedances of a reference expression, not violations of
  an applicable theorem.
- **Counting:** a run is one seed × family × scenario (drift, jump) × estimator, giving 1440 (480 for
  the family bank). Each run is checked at 8 switch budgets m ∈ {0, 1, 2, 3, 5, 10, 20, 50}, so
  per-check totals (11,520; 3,840) are dependent repetitions, not independent runs.

## Family bank and tail dependence

**The family bank is useful for representing changes in dependence structure beyond a single
rank-correlation coefficient. It does not uniformly improve rank-correlation estimation.**

**ρ_S RMSE** (A = family bank, 19 cells):
- against the log banks: 8/6/5 (Xiao), 11/6/2 (Hermite), 9/8/2 (MPL);
- against its own best single arm: **5/14/0**, so worse in 14 of 19 cells.

**Excess log loss of the predictive density:**
- against the log banks' Gaussian densities: 15/3/1, 16/1/2, 15/3/1. Both sides are valid normalised
  densities.

**Shape changes at constant ρ_S:** the delay until the **total weight on the new family** (summed over
its three memory arms) first reaches 50%.
- Medians over 60 seeds: 174 ticks (Gauss→t), 68 (Clayton→Gumbel), 94 (Gauss→Clayton). It was
  reached in 60/60 seeds.
- This is a **weight-crossing delay, not a calibrated change-detection delay**. On the no-switch
  Gaussian path, some wrong family's total weight reached 50% in 14 of 60 seeds.

**Tail dependence** (A = family bank, χ(q) RMSE, 20 cells):
- **Against the EW corner counters** M = 399 / 1999 / 4999: 20/0/0, 16/4/0, 16/2/2.
  - The four losses against M = 1999 are the Clayton–Gumbel mixture (misspecification).
- **Against EW ratio estimators** M = 99 / 399 / 1999, χ_L(0.05): 19/0/0 each (19 copula × scenario
  cells, family-bank probe).
- **Halfway crossing after a switch:** family bank 51–180 ticks; counters 102–187 (M = 399) and
  580–806 (M = 1999).

## Budget study (rankings; conservative significance)

**DataSketches-MPL has the lowest observed test RMSE** (validation-tuned) of {Xiao, Hermite, MPL} in
10/12 cells at 16 KiB and 9/12 at 64 and 256 KiB. That is a ranking.

**Conservative check:** its lead over the better of Xiao and Hermite exceeds 2 SE, with the unpaired
SEs combined in quadrature, in:
- **7/12 cells at 16 KiB;**
- **6/12 at 64 and 256 KiB.**

The two cells where it ranks lower (Clayton jump, t jump) are not resolved either.

Because all methods share the same streams, the unpaired SE overstates the uncertainty of the
differences. A paired test needs per-seed output, which the harness does not save. So: DataSketches-MPL
is frequently the best budget-constrained estimator by observed RMSE, and the differences are small
enough to need these uncertainty qualifications.

**Exact-PIT ablation:** a diagnostic, not an upper bound. EW margins (N_eff 500) beat it on
misspecified drift and jump.
- The explanation proposed (EW ranks bound the scores: Û ≥ ≈1/252, so |z| ≤ ≈2.65) is a
  **hypothesis**. No existing run isolates it.

## Stale or provisional (excluded from conclusions about the final configuration)

- **Own-density banks** (`*_bank_own`):
  - not retuned, and not rerun as a main result;
  - the Hermite own score is not a normalised density (it integrates to about 1.07);
  - all own-density rows, including those in `ch6_family_summary.png` and the tail tables, are
    provisional and excluded from the counts above.
- **Not regenerated:** `ch6_bank_path*.png`, `ch6_bank_weights_neff*.png`, `ch6_neff_optimum.png`,
  `ch6_family_path.png`, `ch6_ise_*.png`, `ch6_batch_ise.png`, `ch6_tail_ci.png`.
- **Real-data (AAPL/GOOG) studies:** not rerun.
- **`docs/chapter6_results.md`:** pre-audit, marked superseded.
