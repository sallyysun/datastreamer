# Copula ISE and offline batch comparators (2026-11-04)

**What this adds.** Two metrics asked for in the Chapter 6 methodology text: the integrated squared
error of the copula CDF, and a comparison with offline batch estimators. Both use the same streams as
`probe_family_bank`, so results pair seed by seed.

| | |
| --- | --- |
| Code | `experiments/probe_batch_ise.cpp`, tables `batch_ise_table.py`, figure `plot_batch_ise.py` → `notes/thesis/figures/ch6_batch_ise.png` |
| Platform | Linux / g++ `-O2` |
| Seeds | 42–101 (60) |
| N | 5000, scored over [500, 5000); jump and shape switch at 2500 |
| Run | `for c in gaussian clayton gumbel t gauss2t clayton2gumbel gauss2clayton; do COPULA=$c ./probe_batch_ise 42 60; done`, then `python3 batch_ise_table.py` (about 35 min on 4 cores) |

## Metrics

### Copula ISE

$e_t^C=\int_{[0,1]^2}(\hat C_t-C_t)^2$, by 12 × 12 Gauss–Legendre quadrature, every 50 ticks, and
reported ×10⁻⁴. How each estimator's $\hat C_t$ is obtained:

- **Family bank.** Its mixture CDF $\sum_kw_kC_k$, exact per arm:
  - Clayton and Gumbel, and their 90°/270° rotations, in closed form;
  - Gaussian and $t_4$ by the conditional-CDF quadrature, tabulated in $\rho$.

  Each of the 24 fixed arms is scored on its own, so the best one in hindsight can be reported.
- **Log banks.** The Gaussian copula at the bank's $\rho$, i.e. its method-of-moments working model.
- **Own-density banks.** The predictive density, integrated to a CDF on a 32 × 32 midpoint grid and
  normalised to mass 1. This density-to-CDF step adds a small discretisation error of its own.
- **Truth.** $C_t$ is the DGP family at the parameter of tick t.

### ρ_S RMSE

Scored at every tick. Online estimators are scored on the estimate held before the tick.

### Offline batch estimators (references, not contestants)

These see the whole sample, future included, so no online method can be expected to match them. They are
reported as reference points, and kept out of the online comparison and its tallies. The margins are global ranks, i.e. the full-sample
pseudo-observations.

| Estimator | What it is |
| --- | --- |
| `batch_full` | sample Spearman and empirical copula of all T ticks |
| `batch_segment` | the same, over the true regime segment containing t. It is an oracle: it knows the change point. On constant, drift and oscillate it equals `batch_full`. |
| `batch_kernel_h` | two-sided Epanechnikov-weighted Spearman and empirical copula centred on t, with h ∈ {50, …, 1600}. The table reports the **best h in hindsight** for each cell, so this is the offline gold standard for a smoothly varying target. |

## Results

Means over 60 seeds. W/L/T counts are family bank lower / higher / within 2 SE, paired over seeds.
The full tables, including windows 0–100 and 100–500 ticks after the change, are printed by
`batch_ise_table.py`.

### Copula ISE, whole run: family bank against each comparator (19 copula × scenario cells)

| Comparator | W/L/T | Detail |
| --- | --- | --- |
| Gaussian log banks (Xiao / MPL) | 13/4/2, 13/6/0 | It loses on oscillate in every family, where its shortest memory, 99, lags, and against MPL also on Gaussian drift and jump, where the Gaussian working model is right. On Clayton constant: 0.07 against 1.43. |
| Own-density banks (Xiao / Hermite) | 19/0/0, 19/0/0 | Their nonparametric densities, integrated, are 1.2–16 against 0.07–7. |
| Its own best fixed arm (hindsight) | 7/12/0 | It wins on jump, on two of the shape switches and on $t$ oscillate. It loses on constant, drift, three oscillate cells and Gauss → $t$, by 0.02–0.44. |
| *Reference:* offline, full sample | 14/5/0 | It loses on the constant paths, where the full-sample empirical copula is the best possible estimator (0.04 against 0.07–0.09), and on Gauss → t. |
| *Reference:* offline, oracle segment | 11/8/0 | It loses on constant and jump, where the oracle knows the change point (jump: 0.31–0.39 against 0.60–0.69). |
| *Reference:* offline, two-sided kernel (best h) | **19/0/0** | The weighted empirical copula is nonparametric and noisier than the bank's parametric mixture, even with the future and a tuned bandwidth. |

### ρ_S RMSE, whole run (19 cells)

| Comparator | W/L/T | Detail |
| --- | --- | --- |
| Log banks | 10/8/1 (Xiao), 8/8/3 (MPL) | as in `thesis_bank_family.md` |
| *Reference:* offline, two-sided kernel (best h) | **0/18/1** | drift 28–32 against 56–72; oscillate 98–107 against 182–273; constant 13–15 against 23–26; jump 57–61 against 59–69. The tie is Clayton jump (58.7 against 59.0). |
| *Reference:* offline, oracle segment | 8/11/0 | It wins on drift and oscillate, where a single segment is the whole run. It loses on constant, jump and shape (8–14). |

**The ρ_S gap to the offline kernel is the price of being causal.** On drift the online banks are about
twice the error of a two-sided smoother that uses the future. On jump they come within 0–8 ×10⁻³ of it,
because a symmetric kernel also smears an abrupt change.

### After the change

**First 100 ticks.**

- Every online method has an ISE of about 20 ×10⁻⁴ on jump.
- The oracle segment (0.3) is the floor.
- The best kernel is 8.5–9.3, because it straddles the change.

**Ticks 100–500.**

- The family bank (0.18–0.35) beats both log banks (6/0/1 against Xiao, 5/0/2 against MPL) and the
  kernel (7/0/0).
- It matches the oracle segment (3/0/4).

## What it says

1. **For the copula, the family bank is the best online estimator tested,** wherever its families contain
   the truth. Among the references:
   - It beats the offline kernel empirical copula in all 19 cells, even though the kernel has the
     future and a bandwidth tuned in hindsight.
   - Only the full-sample or oracle-segment empirical copula beats it, and only on stationary stretches
     or with knowledge of the change point.
2. **For ρ_S, the offline reference shows how much causality costs.** The online-vs-offline gap is about 2× on drift and
   oscillate, 1.7× on constant, and small on jump. That gap is the cost of causality, and no online
   method closes it.
3. **The ISE agrees with the excess log loss (KL).**
   - It ranks the estimators the same way.
   - It is gentler on tail differences. On Gauss → $t$, the family bank's ISE advantage over the log
     banks is only 0.04–0.06 ×10⁻⁴, while its excess-log-loss advantage is about 15 ×10⁻³ nats/tick.
   - So the KL, which works on densities, remains the more sensitive copula metric, and the ISE is a
     secondary check.

## Caveats

- **Grid resolution.** The ISE uses a 12 × 12 quadrature, and the own-density CDFs come from a 32 × 32
  grid. Their ISE includes some discretisation error, though not enough to change the 19/0/0.
- **Scoring frequency.** The batch kernel is scored every tick for ρ_S but every 50 ticks for the ISE.
  The 0–100-tick window therefore has only 2 ISE points per seed.
- **Copula space only.** The streams are in copula space with Gaussian marginals, as in all Chapter 6
  probes.
