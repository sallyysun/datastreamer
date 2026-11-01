# Scoring each arm by its own predictive density (2026-11-01)

Code: `src/estimators/dependence/predictive_density.hpp` (interface),
`log_predictive` in `xiao_spearman.hpp`, `hermiter.hpp` and `parametric_copula.hpp`, and
`ForgettingBank::Loss::Own`. Registry keys: `*_bank_own`.
Probe: `experiments/probe_bank_tune.cpp` (replay; now also emits `excess_own_x1000` and `mem`).
Tables: `experiments/tune_own_table.py`. Platform: Linux / libstdc++.

## Why

The `*_bank_log` bank scores every arm with the Gaussian copula at the arm's ρ̂ (method of moments).
Every arm is then one member of a single-parameter family, so the log score sees nothing beyond ρ:

- it cannot use the tail information that thesis §5.3.1 credits log loss with;
- under misspecification its target is the Gaussian KL projection rather than ρ_S. Measured
  offsets are at most 0.008 in ρ_S units, except Clayton at ρ_S = 0.79, where it is 0.025.

## What each arm's density is

Every density is expressed as a density of the bank's pseudo-observation (u, v), which the arm
receives as (probit u, probit v), and includes the Jacobian of the arm's own marginal step. The
densities are therefore comparable across arms.

| Component | Density |
| --- | --- |
| Xiao | Histogram on its calibrated cells, pooled to 5 × 5. Prior: 0.5 pseudo-observations per pooled cell, spread by cell area, so the prior is the independence copula. Counts are rescaled from weight (1 − λ) to observation units. |
| Hermite | Series density h(s_x)ᵀ A h(s_y) of its standardised inputs, divided by sd_x sd_y φ(z_x) φ(z_y). Clipped at 0 and mixed with independence at weight 0.01. |
| MPL (Gaussian) | Its fitted Gaussian copula at its own pseudo-observations, times dp/du. Still a Gaussian, but at its own fitted r, which targets the KL projection. |

Checks:

- At ρ = 0.6 after 2000 ticks, the densities integrate to 1.000 for Xiao and MPL, and 1.01–1.18
  for Hermite (the clipping; the excess is largest for the shortest arm).
- On Clayton data, Hermite's long arm scores 0.254 nats/tick against 0.200 for the Gaussian MPL.
  It sees the tail.
- The replay matches `xiao_bank_own`, `hermite_bank_own` and `mle_bank_own` from the registry
  exactly (whole-run RMSE, seed 42, Gaussian jump: 0.086739 / 0.100722 / 0.065394).

## Tuning (seeds 1–30, drift + jump, 4 families × 3 estimators)

**Objective:** the excess log loss of the bank's predictive density (the mixture of its arms' own
densities, evaluated at the true PIT) against the hindsight-best fixed memory's own density. This
is the quantity Loss::Own optimises. The RMSE objective is reported alongside.

**Grid:** η ∈ {0.05, 0.1, 0.25, 0.5, 1, 2}; δ ∈ {0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2}.

**Best: η = 0.25, δ = 3e-3, regret −9.5 ×10⁻³ nats/tick.** The optimum is interior.

- Within about 1.5 of the best: η ∈ [0.1, 0.5] with δ ∈ [1e-3, 3e-3].
- Per component: Hermite prefers η = 0.05, δ = 3e-3 (−27.3); Xiao prefers η = 1, δ = 0 (−6.0).

**On RMSE, no own-bank setting comes close to the MoM log bank.** The best own setting has a regret
of +20.8 (η = 0.25, δ = 1e-2), against −6.8 for the MoM log bank.

## Held out (seeds 42–101), own bank at η = 0.25, δ = 3e-3

W/L/T over the 24 drift + jump cells, paired ±2 SE. W means the own bank is better.

| against | density (excess own log loss) | RMSE |
| --- | --- | --- |
| M = 39 | 24/0/0 | 22/2/0 |
| M = 99 | 24/0/0 | 8/13/3 |
| M = 199 | 21/3/0 | 4/20/0 |
| M = 399 | 21/3/0 | 9/15/0 |
| M = 1999 | 24/0/0 | 24/0/0 |
| MoM log bank (`*_bank_log`) | 20/4/0 | **0/24/0** |

**Density.** The own bank is the best density forecaster. It beats every fixed memory except in
the 3 MPL drift cells, where every density is Gaussian. Against the MoM log bank:

- Hermite: −17.8 to −55.6 ×10⁻³ nats/tick;
- Xiao: −5.7 to −14.1;
- MPL: −2.5 to +1.0.

**RMSE in ρ.** It is worse than the MoM log bank everywhere:

- Hermite: +24 to +114 ×10⁻³;
- Xiao: +21 to +51;
- MPL: +6 to +10. MPL's density is the same Gaussian under both banks, so this gap comes only from
  the density-tuned δ.

**Why.** A nonparametric density has many degrees of freedom (25 cells; an order-10 series), so
its log score falls with effective sample size much more steeply than the error in ρ does. The
memory that predicts the density best is therefore far longer than the one that tracks ρ best.

- The own bank's mean weight-averaged memory is 1070–1710 ticks for Xiao and Hermite.
- The MoM log bank's is 615–815.
- The best fixed memory for ρ is 199–399.

**What you score is what the bank tracks.**

## Shape switches (ρ_S fixed, family switches at tick 2500; seeds 42–101)

`gauss2t` (Gaussian → t₄) and the new `clayton2gumbel` (lower tail → upper tail). Values are excess
own log loss ×10⁻³ nats/tick, own bank minus comparator, ±2 SE.

| | window | vs best fixed (M = 1999) | vs MoM log bank |
| --- | --- | --- | --- |
| Hermite, Gaussian → t | whole | −9.6 ± 0.7 | −6.9 ± 2.3 |
| | first 100 after | −13.6 ± 5.6 | −2.7 ± 3.3 |
| Hermite, Clayton → Gumbel | whole | −11.9 ± 0.9 | −4.3 ± 1.0 |
| | first 100 after | −27.5 ± 8.8 | −5.2 ± 3.8 |
| Xiao, Gaussian → t | whole | +3.6 ± 0.2 | −0.6 ± 0.8 |
| Xiao, Clayton → Gumbel | whole | +2.4 ± 0.2 | +0.4 ± 0.4 |
| MPL (Gaussian) | whole | +1.4 / +1.7 | +0.8 / +1.0 |

- **No bank shortens its memory after the switch.** The own bank stays at 1300–1840 ticks. M = 1999
  is the best fixed arm in every window, including the first 100 ticks after the switch.
- At ρ_S ≈ 0.58, a shape change is too small, relative to density-estimation noise at memories of
  400 or less, to be worth forgetting for.
- With the order-10 Hermite density, the own bank still predicts the copula better than any fixed
  memory and better than the MoM bank, by mixing arms rather than switching between them.
- The 5 × 5 Xiao histogram is too coarse to see the change.
- The earlier block-bank result (`memory_tradeoff.md` §3.3: log-scored banks worse than a fixed
  1170-tick control in the first 500 ticks after Clayton → Gumbel) does not carry over. With
  Algorithm 1 the Hermite own bank is ahead of every fixed memory in the first 100 and the first
  500 ticks.

## Reading

- **Two banks, two targets.**
  - For ρ_S, which the thesis evaluates by RMSE, keep the MoM log bank. Describe it as a
    Gaussian working model for ρ, and drop the §5.3.1 claim that it sees tails.
  - For the copula itself, use the own-density bank. It is the best density forecaster in 20–24
    of 24 cells and in the shape switches.
- **Neither bank detects a matched-ρ_S shape change by shortening memory at these sample sizes.**
  That needs a sharper density (a family bank, or a tail-specific score) or a larger change.
- **Regret bounds.** Log loss is 1-exp-concave, so with η = 0.25 the §5.3.4 bound becomes
  B_{T,m}(δ)/η = 4 B_{T,m}(δ) for the mixture of own densities. It was not run separately here.
  The mixture identity holds for any densities, so `probe_bank_regret` would confirm it by
  construction.
