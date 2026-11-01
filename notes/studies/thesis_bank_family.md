# Family bank: arms over copula shapes × memories (2026-11-01)

**What it is.** Thesis Algorithm 1 with arms that differ in copula *family* as well as memory. Each arm
is an online MPL on the shared rank PIT, scored by its own copula log density.

**Why it matters.** The bank's predictive density is then a mixture of proper copula densities on one
scale, so the §5.3.4 log-loss bound holds exactly at η = 1. Its ρ_S, λ and χ(q) are exact for that
mixture. The finite-q tails are in `thesis_bank_tails.md`.

| | |
| --- | --- |
| Code | `src/estimators/dependence/family_bank.hpp` (`FamilyBank`), keys `family_bank`, `family_bank_uniform` in `registry.hpp` |
| Probes | `experiments/probe_family_bank.cpp`, `experiments/probe_bank_regret.cpp … family`; tables `experiments/family_bank_table.py` |
| Commit | `4fbe9d6` (code); results in the commit that adds this note |
| Platform | Linux / libstdc++ (g++), `-O2` |
| Seeds | 42–101 (60) |
| N | 5000; scored over [500, 5000); jump and shape switch at t = 2500 |

Run:

```bash
cd experiments && make probe_family_bank probe_bank_regret EIGEN=/usr CXX=g++
for c in gaussian clayton gumbel t gauss2t clayton2gumbel gauss2clayton; do COPULA=$c ./probe_family_bank 42 60; done
for c in gaussian clayton gumbel t; do COPULA=$c ./probe_bank_regret 42 60 family; done
python3 family_bank_table.py
```

The full run takes about 70 min on 4 cores. `COPULA=gauss2clayton` is new in `copula.hpp`.

## 1. Design

**Shapes (8).**

- Gaussian, $t_4$ (ν fixed at 4), Clayton and Gumbel.
- Clayton and Gumbel rotated by 90°, $c(1-u,v)$, and by 270°, $c(u,1-v)$. These give negative
  dependence.
- The 90° and 270° versions are each other's **transpose**, not the same copula. They would coincide
  only under radial symmetry, which neither family has.
  - Clayton-90 concentrates at $(u\to1,v\to0)$ and Clayton-270 at $(u\to0,v\to1)$.
  - Checked: on 3000 draws from each rotation, the bank puts ≥ 0.98 of its weight on the correct one.
- An earlier draft had only the 90° arms. That was wrong.

**Memories.** M = 99, 399, 1999, with retention $1-2/(M+1)$. So K = 24 arms.

**Arm fit.** `copula::CopulaFit` with expected Fisher scoring, the same online MPL as `mpl_*`, fed the
shared 500-tick rank PIT. A rotated arm is fed its base copula's coordinates, and its $\rho_S$ is
negated. Arms start updating once the PIT holds 30 values; that is also when scoring starts.

**Loss.** $-\log c_k(\hat u,\hat v)$ under the fit held before the tick. This is on the copula scale,
so no Jacobian is needed. The non-finite guard is as in ForgettingBank.

**Hedge and fixed share.**

- η = 1.
- δ = 2.1e-4. This is $\delta^*=Km/((K-1)(T-1))$ for m = 1, K = 24, T = 4970, rounded.
- Not tuned.

**Outputs.**

- $\hat\rho^{\rm bank}=\sum_k w_k\rho_S(\hat\theta_k)$, which is exactly the $\rho_S$ of the mixture
  copula.
- `log_density(u,v)`: the log of the mixture density $\sum_k w_k c_k$.
- `tails(q)`: $\sum_k w_k\{\chi_{L,k}(q),\chi_{U,k}(q),\lambda_{L,k},\lambda_{U,k}\}$, via the
  `TailEstimator` interface.
- Per-shape weights summed over memories, and the mean memory $\sum w_kM_k$.

**Comparators**, on the same streams: `xiao/hermite/mle_bank_log` and `xiao/hermite/mle_bank_own`, plus
the family bank's own 24 fixed arms (best in hindsight).

**Metrics.**

- $\rho_S$ RMSE against the true $\rho_S$.
- Excess log loss $\log c_{\rm true}-\log c_{\rm bank}$ at the true PIT, using each bank's own predictive
  density. The log banks use the MoM Gaussian density at the bank's $\rho$; the own banks use their
  mixture of own densities; the family bank uses its mixture.
- $\chi_L(0.05)$ and $\chi_U(0.05)$ RMSE against the **exact finite-q $\chi$ of the true copula at that
  tick**: closed form for Clayton and Gumbel, 64-point quadrature for Gaussian and $t$, matching a
  4M-draw Monte Carlo to within 0.003. The comparator is an EW ratio estimator J/K at M = 99, 399, 1999
  on the same rank PIT.

## 2. Results

All numbers are $\times10^{-3}$, mean ± 2 SE over 60 seeds. W/L/T counts the cells where the family bank
is lower / higher / within 2 SE.

### 2.1 ρ_S RMSE, whole run

Columns after the first are the family bank minus each comparator.

| copula | scenario | family bank | − xiao_log | − mle_log | − hermite_own | − mle_own | − best arm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian | constant | 23.8 | −2.3 ± 1.2 | −3.6 ± 1.4 | +1.6 ± 1.7 | −16.7 ± 1.2 | +6.1 ± 1.3 (gaussian_1999) |
| gaussian | drift | 69.8 | −0.9 ± 1.2 | **+2.7 ± 0.8** | −115.9 ± 2.5 | −6.1 ± 1.0 | +3.6 ± 0.9 (gaussian_399) |
| gaussian | jump | 63.5 | +1.4 ± 1.0 | **+3.8 ± 0.9** | −47.4 ± 3.2 | −3.6 ± 1.0 | −15.3 ± 1.5 (gaussian_99) |
| gaussian | oscillate | 265.5 | **+73.9 ± 1.6** | **+87.1 ± 1.6** | −130.7 ± 2.1 | +68.8 ± 1.4 | +6.4 ± 1.0 (t_99) |
| clayton | constant | 23.3 | −3.6 ± 2.1 | −6.0 ± 2.0 | +1.8 ± 2.0 | −21.1 ± 1.9 | +6.7 ± 1.3 |
| clayton | drift | 56.2 | −6.7 ± 1.7 | −4.1 ± 1.5 | −68.7 ± 2.7 | −13.8 ± 1.6 | +5.5 ± 1.8 |
| clayton | jump | 59.0 | −7.3 ± 1.7 | −9.4 ± 1.6 | −34.9 ± 2.4 | −18.1 ± 1.6 | −7.9 ± 1.2 |
| clayton | oscillate | 182.2 | +10.1 ± 1.8 | +16.4 ± 1.9 | −37.3 ± 1.7 | +16.8 ± 1.9 | +13.5 ± 1.6 |
| gumbel | constant | 25.5 | −2.6 ± 1.6 | −4.9 ± 1.6 | +3.3 ± 1.9 | −19.4 ± 1.7 | +6.2 ± 1.4 |
| gumbel | drift | 61.3 | −3.6 ± 1.3 | +0.3 ± 1.3 | −72.7 ± 2.5 | −9.0 ± 1.8 | +6.1 ± 1.6 |
| gumbel | jump | 61.8 | −2.9 ± 1.3 | −1.8 ± 1.3 | −46.0 ± 3.0 | −9.3 ± 1.3 | −8.5 ± 1.5 |
| gumbel | oscillate | 184.1 | +9.4 ± 2.0 | +18.0 ± 1.8 | −35.9 ± 2.3 | +18.3 ± 1.7 | +15.2 ± 1.4 |
| t | constant | 26.2 | −7.6 ± 1.9 | −11.0 ± 2.4 | +2.8 ± 1.8 | −23.8 ± 1.9 | +6.9 ± 1.4 |
| t | drift | 71.6 | −3.8 ± 1.4 | −5.3 ± 1.5 | −115.5 ± 2.5 | −14.5 ± 1.5 | +4.1 ± 0.9 |
| t | jump | 68.9 | −1.7 ± 1.4 | −1.4 ± 1.7 | −48.6 ± 3.7 | −7.2 ± 1.4 | −3.7 ± 1.6 |
| t | oscillate | 273.4 | +70.0 ± 2.1 | +67.5 ± 2.3 | −122.0 ± 2.7 | +58.0 ± 1.9 | −3.8 ± 0.4 |
| gauss2t | shape | 31.0 | +1.6 ± 1.3 | −1.3 ± 2.0 | +8.4 ± 1.8 | −15.2 ± 1.6 | +11.3 ± 2.0 |
| clayton2gumbel | shape | 33.1 | +5.2 ± 1.6 | +2.3 ± 1.7 | +10.2 ± 1.9 | −12.2 ± 1.4 | +14.8 ± 2.0 |
| gauss2clayton | shape | 32.6 | +6.2 ± 1.5 | +4.3 ± 1.8 | +10.3 ± 1.5 | −10.0 ± 1.4 | +15.1 ± 1.4 |

The table omits the hermite_log and xiao_own columns, which `family_bank_table.py` prints. W/L/T over
all 19 cells:

| comparator | W/L/T |
| --- | --- |
| xiao_log | 10/8/1 |
| hermite_log | 11/6/2 |
| mle_log | 8/8/3 |
| xiao_own | 13/3/3 |
| hermite_own | 12/5/2 |
| mle_own | 15/4/0 |
| best fixed arm | 5/14/0 |

- **Non-Gaussian families, constant/drift/jump: the family bank wins.**
  - Against xiao_log it wins 9/9, by −1.7 to −7.6.
  - Against mle_log it wins 7/9, by −1.8 to −11.0, and ties in the other 2 (Gumbel drift, t jump).
  - The log banks score every arm with a Gaussian density, which is misspecified here.
  - The family bank's arms include the true family.
- **Gaussian DGP: the MoM log bank is the right model there, and it wins on drift and jump.**
  - mle_log is ahead by 2.7–3.8. Its arms span M = 39–1999; the family bank's span only 99–1999.
- **Oscillate (period 400): the family bank loses badly**, by +9 to +87.
  - Its shortest memory is M = 99, while the Chapter 6 banks have M = 39.
  - Under a sinusoid even M = 99 lags, and on Gaussian and $t$ data the weight drifts to $t$, because a
    lagging Gaussian fit looks heavy-tailed: in the pre-switch window it is $t_4$ 0.50–0.61 on
    Gaussian oscillate.
  - Adding an M = 39 memory is the obvious fix (K = 32). Not run.
- **Shape switches (ρ_S constant): slightly worse than the log banks.** The gap is +1.6 to +6.2 whole run,
  and +9 to +24 in the 100–500 ticks after the switch.
  - When the weight moves to the new family, it lands on arms that spent 2500 ticks fitting the wrong
    family by MPL. Their $\rho_S$ is biased, and it re-converges at the arm's own memory.
  - That is a hypothesis; a per-arm $\rho_S$ trace would confirm it.
  - The log banks see no change at all, because $\rho_S$ does not move.
- **Against the best fixed arm (hindsight): the bank loses 14/19 by +3.6 to +15.** It wins only on the
  jumps.

### 2.2 Density: excess log loss at the true PIT, whole run (×10⁻³ nats/tick)

| copula | scenario | family bank | − mle_log | − hermite_own | − mle_own |
| --- | --- | --- | --- | --- | --- |
| gaussian | constant | 1.1 | −0.1 ± 0.1 | −22.9 ± 1.3 | −2.2 ± 0.2 |
| gaussian | jump | 7.8 | +1.7 ± 0.2 | −49.5 ± 1.4 | +0.2 ± 0.3 |
| gaussian | oscillate | 51.9 | +26.4 ± 0.8 | −65.1 ± 1.6 | +22.5 ± 0.7 |
| clayton | constant | 1.3 | −76.4 ± 1.7 | −21.5 ± 1.4 | −78.1 ± 1.8 |
| clayton | jump | 7.1 | −101.5 ± 1.7 | −52.5 ± 1.5 | −100.3 ± 1.7 |
| gumbel | jump | 6.1 | −31.7 ± 1.0 | −48.3 ± 1.2 | −31.7 ± 1.0 |
| t | jump | 6.9 | −36.9 ± 1.6 | −44.2 ± 1.6 | −32.6 ± 1.2 |
| gauss2t | shape | 2.6 | −17.6 ± 1.1 | −24.6 ± 1.3 | −18.6 ± 1.0 |
| clayton2gumbel | shape | 3.7 | −47.2 ± 1.2 | −34.5 ± 1.4 | −48.9 ± 1.3 |
| gauss2clayton | shape | 4.2 | −40.2 ± 1.2 | −26.5 ± 1.3 | −42.0 ± 1.2 |

W/L/T over 19 cells:

| comparator | W/L/T |
| --- | --- |
| xiao_log | 15/2/2 |
| hermite_log | 17/1/1 |
| mle_log | 16/3/0 |
| xiao_own | 19/0/0 |
| hermite_own | 19/0/0 |
| mle_own | 17/1/1 |

- **Wherever the copula is not Gaussian, the family bank's density beats every other bank beyond 2 SE**,
  by 2.2–145 ×10⁻³ nats/tick, oscillate included. Its own excess is 0.9–7.8 on every non-oscillating
  cell.
- **Losses**: Gaussian drift and jump (+0.7 to +1.7 against mle_log), and Gaussian oscillate (+17 to +26
  against the log banks).
- Against the own-density banks (`thesis_bank_own_density.md`), it wins 19/19 cells against
  xiao_own and hermite_own, and 17 against mle_own.
  - Hermite and Xiao own densities are nonparametric. The family bank's parametric arms are as good as a
    density can be when the family is in the bank.

### 2.3 Shape switches: how fast the weight moves

The table gives the ticks after the switch until the new shape's summed weight reaches 50% and 90%.

| switch | weight on new shape at t = 2499 | t50 median [IQR] | t90 median [IQR] | reached 50% / 90% |
| --- | --- | --- | --- | --- |
| Gauss → $t_4$ | 0.004 | 180 [138, 259] | 270 [177, 376] | 60/60, 60/60 |
| Clayton → Gumbel | 0.001 | 78 [46, 129] | 157 [120, 230] | 60/60, 60/60 |
| Gauss → Clayton | 0.002 | 98 [79, 123] | 136 [105, 166] | 60/60, 60/60 |

The single-seed figure quoted earlier (76 ticks for Gauss → t on seed 900) was lucky. The 60-seed median
is 180.

**False switches (null).** On the constant path (no switch), this counts the seeds in which a shape other
than the DGP's reaches 50% of the weight in the same 2500-tick window after tick 2500, out of 60:

| DGP | shapes that reach 50% | reach 90% |
| --- | --- | --- |
| Gaussian | t 2, Clayton 3, Gumbel 6 | t 1, Gumbel 3 |
| Clayton | Gaussian 4, t 1 | none |
| Gumbel | Gaussian 2, t 7, Clayton 1 | none |
| t | Gaussian 5, Clayton 3, Gumbel 7 | Gaussian 1 |

No rotated shape ever takes the weight on a positive-dependence path. When a false switch happens, it
takes a median of 370–1820 ticks to reach 50%, against 78–180 for a real switch.

**Why this is fast, and the earlier "≈ 5000 ticks for Gauss → t" is not a contradiction.**

- Log-loss Hedge at η = 1 accumulates the log-likelihood ratio of the arms.
- At ρ_S = 0.58, the KL from $t_4$ data to the best-fitting Gaussian copula is **0.034 nats/tick**, and
  from Gaussian data to the best $t_4$ it is 0.022. These are 200k-draw Monte Carlo estimates.
- Starting from a $t$ weight of 0.004, a log ratio of about −5.5 has to be made up, plus the arm's re-fit.
  $5.5/0.034\approx160$ ticks, close to the observed median of 180.
- The 5000-tick figure (`tail_dependence.md`, step 2) was for a corner counter of $\chi(q)$. That counter
  sees only about $q$ = 5% of ticks, and the change it must resolve is $\Delta\chi$ = 0.07–0.17 against
  a variance of $\chi(1-\chi)/(qM)$.
- The full likelihood uses the whole density, including the body, where Gaussian and $t_4$ also differ.
  So both findings are right; they measure different statistics.

### 2.4 Weights by scenario

Seed means over ticks; pre = [500, 2500), post = [2500, 5000).

- **Constant and jump**: the DGP's shape takes 0.90–0.99 of the weight in every family.
- **Mean memory**: 1270–1750 on constant paths. On the jump, it falls to 209–285 after the change.
- **Clayton/Gumbel drift, pre-change**: the weight is spread over all shapes (Clayton 0.19, the rotations
  0.09–0.15 each, Gaussian 0.24).
  - The DGP clamps a negative target ρ_S to independence for these families, so the first part of the
    drift is independence. Every shape fits it at θ near independence.
- **Oscillate**: weight is split, $t$ takes 0.50–0.61 on Gaussian data (§2.1), and the mean memory is
  140–510.

### 2.5 χ(q = 0.05) against the EW ratio estimator, whole run

- The family bank has the lower RMSE in **19/19 cells against every memory** of the ratio estimator
  (M = 99, 399, 1999), for both tails.
- The margin against M = 1999 is −12 to −140.
- In the first 100 ticks after a jump or switch, it ties in 1–3 cells per comparator and never loses.
- The tail study in `thesis_bank_tails.md` adds q = 0.10, a mixture DGP, and the limits of this result.

### 2.6 Regret against B_{T,m}(δ)

The bank's mixture log loss, compared with the best arm sequence with at most m switches. Bound at
η = 1 is $B_{T,m}(\delta)$; K = 24 gives $\log K$ = 3.18. Results cover 240 runs per scenario (60 seeds
× 4 families).

| bank | scenario | m | mean | max | bound | max/bound | violations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| log_mix (δ = 2.1e-4) | drift | 0 | 2.87 | 4.15 | 4.2 | 0.99 | 0/240 |
| | drift | 1 | 7.95 | 11.59 | 15.8 | 0.73 | 0/240 |
| | drift | 5 | 22.5 | 31.4 | 62.4 | 0.50 | 0/240 |
| | jump | 0 | −2.91 | 3.78 | 4.2 | 0.90 | 0/240 |
| | jump | 1 | 7.67 | 11.05 | 15.8 | 0.70 | 0/240 |
| | jump | 5 | 29.7 | 36.9 | 62.4 | 0.59 | 0/240 |
| log_mix_hedge (δ = 0) | drift / jump | 0 | 3.16 / 3.13 | 3.18 | 3.18 = log K | 1.00 | 0/240 |
| | jump | 1 | 13.7 | 19.8 | none | — | — |

- **The bound holds in every run at every m** in {0, 1, 2, 3, 5, 10, 20, 50}.
- **Static regret is nearly tight**: 0.99 of the bound on drift.
- **Pure Hedge** (δ = 0) sits at log K for m = 0. One switch costs it up to 19.8, where it has no bound;
  with fixed share the most is 11.6.
- This is the cleanest regret result in Chapter 6. The family bank's mixture density is exactly the
  forecaster the §5.3.4 log-loss bound covers, with no Gaussian working model in between.

## 3. Summary

- **Model selection over copula families, online, works.**
  - The bank finds the true family on every constant and jump path (0.90–0.99 of the weight).
  - It moves to the new family 78–180 ticks after a shape switch at constant ρ_S (60/60 seeds).
  - It rarely switches falsely: 0–7 seeds in 60 reach 50% in 2500 ticks, and almost never 90%.
- **Density and tails are where it pays.**
  - Its density is the best of all the banks wherever the copula is not Gaussian.
  - Its χ(0.05) beats the ratio estimator 19/19.
  - Its regret is covered exactly by the bound, with no violations.
- **For ρ_S it is a mixed result.**
  - It beats the MoM log banks on non-Gaussian constant, drift and jump paths.
  - It loses where the shortest memory matters (oscillate, and Gaussian drift/jump against mle_log),
    and on shape switches, where ρ_S does not move but its arms re-fit.
  - It is not a replacement for `*_bank_log` as the ρ_S headline.

## 4. Caveats and next steps

- **Untuned.** η = 1 and δ = 2.1e-4 are the theory's values. Its memory grid lacks M = 39, and adding it is
  the first thing to try for oscillate.
- **ν = 4 is fixed and equals the DGP's ν.** A ν mismatch, or a ν grid as arms, is not tested.
- **Clayton/Gumbel drift is partly independence.** For those families the negative phase of the drift is
  independence by the DGP's clamp, so the rotated arms are exercised only by Gaussian/t drift and the
  rotation unit test.
- **Cost.** 24 MPL fits per tick, and the $t$ arms need $t$ quantiles. Each 5000-tick run takes about
  0.6 s, after a one-off build of about 26 s for the $t$ expected-information table.
- **Never a tail-dependence detector.** The bank is not a detector. The weights are a soft, continuous
  shape classifier, and a detector built on them would need its ARL₀ calibrated.
