# Empirical check of the §5.3.4 regret bounds (2026-11-01)

Probe: `experiments/probe_bank_regret.cpp`. Output: `results/bank_regret_42[_fam].csv`.
Platform: Linux / libstdc++. Seeds 42–101, 4 families × {drift, jump} × {Xiao, Hermite, MPL}.
That is 1440 runs, N = 5000, with T = 4970 scored ticks (scoring starts once the rank window
holds 30 values).

## What is compared

For each run, the bank's cumulative loss over the scored ticks is compared with:

- the best single arm in hindsight (m = 0);
- the best arm sequence with at most m switches, for m ∈ {0, 1, 2, 3, 5, 10, 20, 50}. This is an
  exact dynamic programme over (tick, arm, switches used).

The bound is the thesis v5 §5.3.4 one: R ≤ B_{T,m}(δ)/η, where

B_{T,m}(δ) = log K + (T − 1 − m) log 1/(1 − δ + δ/K) + m log(K/δ),

with K = 5 arms (M = 39, 99, 199, 399, 1999).

| Bank | Loss | η, δ | Covered by §5.3.4? |
| --- | --- | --- | --- |
| `log_mix` | −log Σ_k w_k c_k(Û, V̂), the mixture of the arms' Gaussian MoM copula densities | 1, 3e-4 (tuned) | **yes**: the bound holds with equality per tick at η = 1 |
| `log_mix_hedge` | the same mixture | 1, 0 | yes, static bound log K |
| `log_point` | −log c_{g⁻¹(Σ w_k ρ_k)}(Û, V̂), the Gaussian MoM density at the bank's own ρ | same weights as `log_mix` | no: this density is not the mixture |
| `sq` | (p̂ − Σ w_k ρ_k)² | 1, 1e-4 (tuned) | no: exp-concavity needs η ≤ 1/32 |

The weights are replayed with the same recursion and the same non-finite guard as
`ForgettingBank`. `probe_bank_tune` showed this replay matches the class exactly.

## Results

Regret is in nats (log) or squared-loss units (sq), cumulative over T = 4970 ticks. A negative
regret means the bank beat the comparator. `viol` counts runs out of 720 per scenario where the
regret exceeded the bound.

**`log_mix` (the forecaster the theorem covers): no violation in any of the 1440 × 8 checks.**

| scenario | m | mean | 95% quantile | max | bound | viol |
| --- | --- | --- | --- | --- | --- | --- |
| drift | 0 | 1.09 | 2.31 | 2.61 | 2.80 | 0 |
| drift | 1 | 4.85 | 6.20 | 8.15 | 12.52 | 0 |
| drift | 2 | 8.29 | 11.24 | 15.45 | 22.24 | 0 |
| drift | 5 | 16.31 | 22.36 | 29.15 | 51.41 | 0 |
| jump | 0 | −17.21 | −7.06 | 0.21 | 2.80 | 0 |
| jump | 1 | −3.59 | 2.36 | 6.00 | 12.52 | 0 |
| jump | 2 | 12.38 | 15.13 | 16.49 | 22.24 | 0 |
| jump | 5 | 27.79 | 34.68 | 40.70 | 51.41 | 0 |

Over all runs, at m = 10 / 20 / 50: max 66 / 104 / 177 against bounds 100 / 197 / 489.

**`log_mix_hedge` (δ = 0):** static regret mean 1.47, max 1.61, against log K = 1.61. Against
sequences with one switch, where δ = 0 has no bound, its regret reaches 43.8 (mean 10.2). The
fixed-share bank's maximum there is 8.2.

**`log_point`:** violates the static bound in 233 runs (200 on drift) and the one-switch bound in
62. Most violations are on the t family (54–63 per estimator, against 0–19 elsewhere). The worst
cases are t Hermite on jump, up to 213 nats (seed 93).

**`sq` at η = 1:** compared with B/η at the bank's own η = 1, it violates the static bound in 709
runs (688 of the 720 drift runs, where mean 4.84 > 2.01). It violates the one-switch bound in 42.
Every run is within 32·B_{T,m}, the bound a bank run at η = 1/32 would have: the static maximum is
11.2 against 64.

## Reading

- **The theorem is confirmed for the forecaster it covers, and it is close to tight where it
  matters.**
  - On drift, the static regret reaches 2.61 against a bound of 2.80. Most of the bound there,
    about 1.2, is the cost of fixed share, (T − 1)·δ(1 − 1/K).
  - Hedge without fixed share reaches log K almost exactly, because it locks onto the best arm and
    pays only for the uniform start.
- **On jump, the bank beats the best single memory** (static regret −17 on average). Against the
  best one-switch sequence it also comes out ahead on average (−3.6). That comparator is the oracle
  that knows the change time, but it must switch to one fixed arm. The bound becomes binding only
  from m = 2 onwards.
- **Fixed share is what buys the switching guarantee.** With δ = 0, regret against a one-switch
  comparator reaches 44 nats. With δ = 3e-4 it stays below 8, at a static cost of about 1.2.
- **The guarantee is about the mixture density, not the Gaussian copula at the bank's ρ.**
  - Algorithm 1's ρ output is the Spearman's ρ of the mixture, because ρ_S is linear in C. So the
    ρ-space results are unaffected.
  - But scoring the bank with the Gaussian density at that ρ breaks the guarantee, mostly on t
    (heavy tails: a mixture of Gaussian copulas fits a t sample better than any one Gaussian).
  - When the bank's log loss is reported, it should be the mixture's.
- **Squared loss at the tuned η = 1 runs outside the theory, and on drift the "η = 1 bound" fails
  in almost every run.** It stays far inside the guarantee of an η = 1/32 bank. That is a weaker
  statement, because the η = 1/32 bank does not adapt (`thesis_bank_tuning.md`).
