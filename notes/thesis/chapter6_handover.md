# Chapter 6 handover (2026-11-01)

Branch `bank-vff-study`, head `3072821`. **14 commits are local and not pushed** (push only on Sally's
say-so). Platform for every number: Linux / libstdc++. Evaluation seeds 42–101 (60); tuning seeds 1–30.
N = 5000 ticks, scored over [500, 5000), jump at t = 2500.

Where the numbers live:

- Every Chapter 6 bank number: `docs/chapter6_results.md`, with tables from
  `experiments/ch6_results_md.py`.
- The LaTeX draft: `notes/thesis/chapter6_draft.tex`.
- Which file and probe backs each section: `notes/thesis/chapter6_sources.md`.

## 1. What was built

| Thing | Where |
| --- | --- |
| Algorithm 1 (Hedge + fixed share over M = 39/99/199/399/1999, shared rank PIT, window 500) | `src/estimators/dependence/forgetting_bank.hpp` |
| Squared bank (η = 1, δ = 1e-4) | `*_bank` |
| MoM log bank (η = 1, δ = 3e-4): Gaussian method-of-moments density | `*_bank_log` |
| Own-density bank (η = 0.25, δ = 3e-3): each arm scored by its own predictive density | `*_bank_own`, `predictive_density.hpp` |
| Squared bank at the theory's η = 1/32 (δ = 1e-4) | `*_bank_sq32` |
| Uniform control (η = δ = 0) | `*_bank_uniform` |
| Old block bank | `*_bank_block` |

`*` is `xiao`, `hermite` or `mle`; all are registered in `registry.hpp`.

## 2. Results in one page

**Adaptive memory (§6.3).** Pooled over drift and jump, both η = 1 banks beat every fixed memory, both
VFF rates and the uniform average, in 12/12 family × estimator pairs.

- **Log bank against M = 199:** −5.7 to −12.5 ×10⁻³ RMSE pooled. M = 199 is the best single memory
  chosen in advance.
- **Squared bank against M = 199:** −3.2 to −7.0.
- **Jump:** the log bank beats the hindsight-best M in 12/12 cells, by −14.9 to −20.7.
- **Drift:** it loses to the hindsight-best M (+2.2 to +7.4). Against M = 199 it is better in 6 cells
  and worse in 6. Where it loses on drift, its jump gain is 4–17 times larger.
- **Log bank against the squared bank:** better in 21/24 cells, tied in 3 (t₄ drift).
- **Gaussian jump, one seed:** effective memory Σ w·M goes 1750 → 47 → 195 → 263 at ticks
  2499 → 2600 → 3000 → 4000.

**Tuning.**

- The optimum is η = 1 for both losses, with a flat region η ∈ [0.5, 2] × δ ∈ [1e-5, 1e-3].
- η ≥ 4 is unstable.

**Regret bounds (§6.4.1, §5.3.4).** Results are over 1440 runs (60 seeds × 4 families × 3 estimators
× drift/jump).

- **Log bank, as the mixture density:**
  - No violation at any m ∈ {0, 1, 2, 3, 5, 10, 20, 50}.
  - Drift static regret: max 2.61 against a bound of 2.80. It is close to tight.
  - Jump static regret: mean −17.2.
- **Hedge with δ = 0:** static max 1.61 = log K, but up to 43.8 against one-switch sequences, where it
  has no bound. With fixed share it stays below 8.2.
- **Log loss of the Gaussian density at the bank's ρ (not the mixture):** breaks the bound in 233 runs,
  mostly t₄.
- **Squared bank at η = 1:** outside the theory. It exceeds B/1 in 709 runs, but is within 32·B in
  all of them.
- **Squared bank at η = 1/32:**
  - Regret: within 32·B everywhere, by a factor of 4 or more. Its mean regret is still higher than
    the η = 1 bank's (drift 7.8 against 4.8; jump 3.9 against −7.6).
  - RMSE: loses 0/24 to both η = 1 banks (drift +3 to +5, jump +15 to +18 ×10⁻³) and beats uniform
    24/0. It is 67–88 behind the η = 1 bank in the first 100 ticks after the jump.
  - δ* ≈ 2.5e-4 instead of 1e-4 moves the mean regret by less than 1.

**Own-density bank (§6.4.2).**

- **Density:** it beats every fixed M and the MoM bank in excess log loss (held out, 20–24/24 wins).
  Against the MoM bank: Hermite −18 to −56, Xiao −6 to −14 ×10⁻³ nats/tick.
- **ρ_S:** it is worse than the MoM bank in RMSE (0/24). It keeps a longer memory: 1070–1710 ticks
  against 615–815.
- **Lesson:** what the bank is scored on is what it tracks.

**Shape switches at constant ρ_S (§6.5).** Gaussian → t₄ and Clayton → Gumbel, at tick 2500.

- **Hermite own bank, against M = 1999:**
  - −9.6 ± 0.7 and −11.9 ± 0.9 ×10⁻³ nats/tick over the whole run;
  - −13.6 and −27.5 in the first 100 ticks after the switch.
- **Xiao (5 × 5 histogram):** 2–4 behind M = 1999.
- **No bank shortens its memory after a shape change.** The gain comes from mixing arms, not switching.

**Recommendation as it stands.** Use the MoM log bank for ρ_S, framed as a Gaussian working model, and
the own-density bank for the copula. Keep the η = 1/32 squared bank as the labelled bound-covered
comparator. Whether η = 1 squared stays a headline is Sally's call.

## 3. Figures

| Figure | Status | Script → file |
| --- | --- | --- |
| Mean bank weights around the jump (squared and log bank, 60 seeds) | **in the draft** (Fig. 6.1, §6.3) | `plot_thesis_bank.py` → `figures/ch6_bank_weights.png` |
| Single-seed bank path: ρ vs truth and arms, stacked weights, effective memory Σ w·M (drift + jump) | **made, but not in the draft** | `probe_bank_path` + `plot_bank_path.py` → `figures/ch6_bank_path.png`, `ch6_bank_path_clayton.png` |

The single-seed bank path is "the bank plot" made earlier. It sits in `notes/thesis/figures/` but has
no `\includegraphics` in `chapter6_draft.tex` yet. Suggested home: §6.3, next to Fig. 6.1, as the
single-run view of how the bank switches.

## 4. Plots worth making

Data source given for each; ★ = data already on disk, only a script is needed.

1. ★ **Regret against the bound, by m.** Max (and mean) regret ÷ bound against m, for log_mix,
   log_mix_hedge, sq (η = 1, against B/1), sq32 (against 32B), split by drift and jump. It shows in one
   panel which banks are covered and how tight each bound is. Data: `results/bank_regret_42[_fam].csv`.
2. ★ **Paired-difference forest plot.** Bank_log − M = 199 and bank − M = 199 (mean ± 2 SE), one row per
   family × estimator, faceted by drift and jump. It is the headline table as a picture.
   Data: `results/thesis_bank[_fam].csv`.
3. ★ **η, δ tuning heatmap.** Mean regret against the hindsight-best M over the (η, δ) grid, for squared,
   log and own loss. It shows the flat optimum and the η ≥ 4 instability. Data: `results/bank_tune_*.csv`.
4. ★ **Recovery bars after the jump.** RMSE in the windows pre / 0–100 / 100–500 / 500–1500 for fixed
   arms, banks (including sq32 and uniform) and VFF. Data: `thesis_bank*.csv` (window column).
5. **Per-tick RMSE curve around the jump** (t = 2000–4000, mean over seeds). It is the smooth version of
   item 4. It needs `probe_thesis_bank` to also write the per-tick squared error averaged over seeds;
   that is cheap to add.
6. **Effective memory Σ w·M around the jump, one line per bank** (sq, log, own, sq32), mean over seeds.
   It shows that sq32 barely moves and that own keeps a long memory. It needs the weights of own and
   sq32 written in `probe_thesis_bank` (only sq and log are written now).
7. **Shape switch, excess log loss over time** for the own bank, the MoM bank and M = 1999, with the
   bank's Σ w·M underneath. It makes "no bank shortens its memory" visible. It needs a per-tick dump
   from `probe_bank_tune` shape mode.
8. ★ **Own vs MoM bank scatter.** For each cell, the density gain against the RMSE loss. It is the
   "scored on is what it tracks" trade-off in one figure. Data: `results/bank_tune_*` (excess_own,
   rmse columns).

## 5. Open items

- **Push** the 14 local commits (awaiting Sally).
- **Decisions for Sally:**
  - headline squared bank: η = 1 or 1/32;
  - η = 1 for the own bank, so that its log-loss bound holds exactly (currently 0.25);
  - whether Xiao should report its own checkerboard ρ_S.
- **§6.2:** the Linux rerun of the estimator-comparison numbers.
- **The §6.5 metrics in the PDF:** ISE and recovery time are not computed.
- **§6.5 tail-dependence results:** still from the earlier stationary probes (`tail_dependence.md`);
  marked `\pending`.
- **Built after this handover:** the family bank (`family_bank`, `notes/studies/thesis_bank_family.md`) and
  finite-q tails with the `tail_chi` corner counter (`notes/studies/thesis_bank_tails.md`).
- **probe_bank_regret:** not yet run for the own bank.
