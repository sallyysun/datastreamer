# Finite-q tail dependence from the family bank (2026-11-01)

The question: can tail dependence at a finite threshold q be estimated as a streaming quantity? Two
estimators are compared:

- the **family bank's mixture tails**, model-based;
- a **model-free corner counter**.

Both are scored against the exact $\chi(q)$ of the true regime at q = 0.05 and 0.10, never against the
limit $\lambda$.

| | |
| --- | --- |
| Code | `src/estimators/dependence/{tail_estimator,tail_chi,family_bank}.hpp`, `experiments/probe_tail_bank.cpp`, tables `experiments/tail_bank_table.py` |
| Commit | `4fbe9d6` (code); results in the commit that adds this note |
| Platform | Linux / libstdc++ (g++), `-O2` |
| Seeds | 42–101 (60). The post-switch sampler is seeded at seed + 100000 |
| Run | `make probe_tail_bank EIGEN=/usr CXX=g++ && for r in gauss2clayton clayton2gumbel gauss2t mix gaussian; do ./probe_tail_bank $r 42 60; done && python3 tail_bank_table.py` (about 15 min on 2 cores) |

## 1. What was built

### `TailEstimator` interface

This is a side interface: `TailEstimate tails(double q)`, returning $\chi_L(q)$, $\chi_U(q)$,
$\lambda_L$ and $\lambda_U$.

- `estimate()` stays $\rho_S$, per the estimator contract.
- Harnesses `dynamic_cast` to `TailEstimator` and log tails only when it is present.

### Family bank tails (`FamilyBank::tails(q)`)

Each arm's tail values come from its fitted copula; the bank reports their weighted sums.

**Per-arm $\chi$.** $\chi_L(q)=C(q,q)/q$ and $\chi_U(q)=(2q-1+C(1-q,1-q))/q$.

- Clayton and Gumbel use the existing closed-form `cdf`.
- Gaussian and $t_4$ use $C(q,q)=\int_0^q P(V\le q\mid U=w)\,dw$: a 64-point Gauss–Legendre integral of
  the closed-form conditional CDF. For the $t$ copula, $T_2\mid T_1=x$ is $t_{\nu+1}$ with location
  $\rho x$ and scale $\sqrt{(\nu+x^2)(1-\rho^2)/(\nu+1)}$.
  - Both are radially symmetric, so $\chi_U=\chi_L$.
  - The integral is tabulated once over 401 values of $\rho$ per $(q,\nu)$, the same way
    `spearman_from_param` builds its table.

**Rotated arms (90° and 270°).** They put the base copula's tail mass in an off-diagonal corner.

- Their $\lambda_L=\lambda_U=0$.
- Their diagonal $\chi(q)$ is computed exactly from $C_{90}(a,b)=b-C(1-a,b)$ and
  $C_{270}(a,b)=a-C(a,1-b)$, and is about 0 at $\rho_S=-0.58$.
- No 180° arm is included. Survival Clayton is Gumbel-like in its upper tail, but it is not in the bank.

**Per-arm $\lambda$.** These are the formulas in `tail_common.hpp::exact_lambda`.

**Bank output.** $\chi=\sum_k w_k\chi_k(q)$ and $\lambda=\sum_k w_k\lambda_k$. Both are exact for the
mixture copula $\sum_k w_k C_k$, because each is linear in $C$.

**Checks.**

- The arm formulas agree with a 4M-draw Monte Carlo (`tail::exact_corners`) to within 0.002 (Gaussian,
  $t$) and 0.0025 (Clayton, Gumbel), at q = 0.05 and 0.10.
- On the pure Gaussian path, the Gaussian arms report a mean $\chi_L(0.05)$ of 0.311–0.313 and a mean
  $\chi_L(0.10)$ of 0.390–0.393. The exact values are 0.3105 and 0.3902. **So finite-q $\chi$ is far
  from 0 even though $\lambda=0$.**

### `tail_chi_{399,1999,4999}` (new)

This is a model-free EW corner counter on a 500-tick rank PIT.

- **Update.** $a\leftarrow(1-\alpha)a+\alpha\,1\{\hat U\le q\}$ and
  $b\leftarrow(1-\alpha)b+\alpha\,1\{\hat U\le q,\hat V\le q\}$, with $\alpha=2/(M+1)$. Upper corners are
  tracked the same way. Both q = 0.05 and q = 0.10 are tracked.
- **Tail output.** `tails(q)` is $b/a$, reported as 0 until $a>10^{-12}$. $\lambda$ is NaN, because the
  counter has no model for the limit.
- **$\rho_S$.** `estimate()` is $12\,\mathrm{EW}(\hat U\hat V)-3$, which is a valid $\rho_S$, so it is
  registered and in `all_estimator_keys()`.
- **Counter logic.** It is `TailCusumDetector`'s, without the chart.

### Probe design

**Data.** `tail_common` samplers draw $(u,v)$ in copula space at $d=0.6$ ($\rho_S=0.5819$, matched on
both sides of the switch). The run is T = 10,000 ticks, the switch is at $\tau$ = 5,000, and scoring is
over [1000, T).

**Regimes.**

- Gauss → Clayton;
- Clayton → Gumbel;
- Gauss → $t_4$;
- `mix`: the Clayton/Gumbel mixture of `tail_soft`, with both components at $\rho_S=0.5819$ and
  $\pi=0.75\to0.25$;
- `gaussian`: no switch.

**Targets.** The exact finite-q $\chi$ of the regime in force at each tick. It is closed form for
Clayton, Gumbel and the mixture, and comes from the quadrature above for Gaussian and $t$. The targets
are computed once and checked against `exact_corners` (4M draws) at start-up, without touching
CopulaDraw or the live DGP.

**Margins.** Every estimator sees the same 500-tick rank PIT (fed $\Phi^{-1}(u)$, so the ranks are the
ranks of $u$).

- This differs from tail-study step 1, which used known uniform margins.
- `tail_chi_1999_known` is the same counter fed the true $(u,v)$, as the check.

## 2. Results

All numbers below are $\times10^{-3}$, mean ± 2 SE over 60 seeds.

### Whole-run $\chi(q)$ RMSE, family bank minus each counter

W = the bank is lower by more than 2 SE.

| regime | q | tail | − tail_chi_399 | − tail_chi_1999 | − tail_chi_4999 | − best fixed arm (in hindsight) |
| --- | --- | --- | --- | --- | --- | --- |
| gauss2clayton | 0.05 | lower | −65 ± 3 | −42 ± 3 | −71 ± 3 | −78 ± 2 (clayton_399) |
| gauss2clayton | 0.05 | upper | −61 ± 2 | −34 ± 3 | −49 ± 3 | −120 ± 2 (clayton_399) |
| gauss2clayton | 0.10 | lower | −44 ± 2 | −28 ± 3 | −50 ± 3 | −45 ± 2 (clayton_399) |
| gauss2clayton | 0.10 | upper | −48 ± 2 | −27 ± 2 | −41 ± 2 | −108 ± 1 (gaussian_1999) |
| clayton2gumbel | 0.05 | lower | −71 ± 3 | −65 ± 3 | −103 ± 4 | −139 ± 3 (clayton_99) |
| clayton2gumbel | 0.05 | upper | −66 ± 3 | −67 ± 3 | −108 ± 4 | −165 ± 2 (gaussian_1999) |
| clayton2gumbel | 0.10 | lower | −50 ± 2 | −47 ± 3 | −77 ± 3 | −91 ± 2 (clayton_399) |
| clayton2gumbel | 0.10 | upper | −47 ± 2 | −50 ± 2 | −85 ± 2 | −130 ± 2 (gaussian_1999) |
| gauss2t | 0.05 | lower | −80 ± 3 | −27 ± 3 | −25 ± 4 | −20 ± 2 (t_1999) |
| gauss2t | 0.05 | upper | −79 ± 3 | −25 ± 2 | −23 ± 3 | −20 ± 2 (t_1999) |
| gauss2t | 0.10 | lower | −55 ± 2 | −16 ± 2 | −14 ± 3 | −3 ± 1 (t_1999) |
| gauss2t | 0.10 | upper | −55 ± 2 | −15 ± 2 | −12 ± 2 | −2 ± 1 (t_1999) |
| **mix** | 0.05 | lower | −33 ± 3 | **+11 ± 3** | +0 ± 3 | −26 ± 3 (t_1999) |
| **mix** | 0.05 | upper | −17 ± 3 | **+20 ± 4** | +6 ± 4 | −18 ± 2 (gaussian_1999) |
| **mix** | 0.10 | lower | −21 ± 2 | **+9 ± 3** | −0 ± 3 | −24 ± 2 (t_1999) |
| **mix** | 0.10 | upper | −4 ± 2 | **+20 ± 2** | +7 ± 3 | −11 ± 2 (gaussian_1999) |
| gaussian | 0.05 | lower | −85 ± 3 | −30 ± 3 | −19 ± 3 | +4 ± 1 (gaussian_1999) |
| gaussian | 0.05 | upper | −82 ± 3 | −27 ± 3 | −17 ± 3 | +6 ± 2 (gaussian_1999) |
| gaussian | 0.10 | lower | −61 ± 2 | −19 ± 2 | −11 ± 3 | +4 ± 1 (gaussian_1999) |
| gaussian | 0.10 | upper | −60 ± 2 | −18 ± 2 | −10 ± 2 | +5 ± 1 (gaussian_1999) |

W/L/T over the 20 cells (W = family bank lower):

| comparator | W/L/T |
| --- | --- |
| tail_chi_399 | 20/0/0 |
| tail_chi_1999 | 16/4/0 |
| tail_chi_4999 | 16/2/2 |
| best fixed arm in hindsight | 16/4/0 |

- Every loss to tail_chi_1999 and tail_chi_4999 is on `mix`.
- Every loss to the best arm is the Gaussian arm on the pure Gaussian path. That arm is the oracle there,
  and the bank's cost for not knowing the family is 4–6.

### Nonparametric competitors: Hermite and Xiao read through their own densities

Added after the first run. Each competitor's $\chi(q)$ is P(both in the corner) / P(U in the tail),
computed from its own predictive density by a midpoint rule every 20 ticks, on the same rank PIT.

**The corner counter is Xiao's corner cell.** Xiao's 20 × 20 histogram has bins 0.05 wide, so at
q = 0.05 the corner counter *is* Xiao's corner cell read directly. The only differences are the shared
rank PIT, which replaces Xiao's frozen cut-points, and the free choice of q.

The density-based Xiao estimator here is a 20 × 20 histogram, with prior mass 12.5 pseudo-observations,
the same total as the default 5 × 5 pooling. With the default 0.5 per cell on a 20 × 20 grid, the prior
(200 pseudo-observations of independence) pulled $\chi$ to about q.

Whole-run RMSE, family bank minus each competitor, W/L/T over the 20 cells:

| competitor | W/L/T | where it loses |
| --- | --- | --- |
| Hermite own-density bank | 16/4/0 | mix, +13 to +20 |
| Hermite N = 10, M = 399 | 19/0/1 | — |
| Hermite N = 10, M = 1999 | 16/4/0 | mix, +13 to +20 |
| Xiao 20 × 20, M = 399 | 20/0/0 | — |
| Xiao 20 × 20, M = 1999 | 18/2/0 | mix q = 0.10, +8 and +13 |

- **Hermite's $\chi$ is biased low before the switch.** The order-10 series smooths the corner: on
  Clayton → Gumbel it reads 0.56 against a target of 0.62.
- **Hermite follows a switch faster than the counter at M = 1999, but much more slowly than the family
  bank.** The figure `ch6_tail_chi.png` shows this.
- **The pattern is the corner counter's**: the family bank wins everywhere its families contain the
  truth, and loses only on the mixture.

The full table is printed by `tail_bank_table.py`.

### Selected windows, q = 0.05, lower tail: RMSE and bias

| regime | method | RMSE pre | RMSE post 0–500 | RMSE post 2000–5000 | bias pre | bias post 0–500 | bias post 2000–5000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gauss2t | family_bank | 16 ± 1 | 68 ± 5 | 17 ± 2 | +1 ± 2 | −52 ± 7 | −0 ± 2 |
| gauss2t | tail_chi_1999 | 47 ± 4 | 84 ± 10 | 45 ± 4 | −5 ± 7 | −79 ± 11 | −7 ± 9 |
| gauss2t | tail_chi_4999 | 39 ± 4 | 93 ± 7 | 37 ± 4 | −6 ± 8 | −92 ± 7 | −26 ± 7 |
| gauss2t | arm t_1999 | 65 ± 2 | 27 ± 3 | 11 ± 1 | +64 ± 2 | −27 ± 3 | +1 ± 2 |
| clayton2gumbel | family_bank | 23 ± 2 | 121 ± 8 | 14 ± 1 | −2 ± 3 | +56 ± 9 | +3 ± 2 |
| clayton2gumbel | tail_chi_1999 | 49 ± 4 | 311 ± 10 | 39 ± 4 | −10 ± 9 | +307 ± 11 | +13 ± 7 |
| mix | family_bank | 74 ± 4 | 90 ± 8 | 70 ± 3 | +20 ± 5 | +32 ± 13 | +6 ± 10 |
| mix | tail_chi_1999 | 46 ± 3 | 153 ± 9 | 41 ± 3 | −7 ± 7 | +150 ± 9 | +3 ± 7 |
| mix | tail_chi_4999 | 38 ± 4 | 170 ± 7 | 47 ± 5 | −7 ± 7 | +170 ± 7 | +40 ± 5 |

The full tables, with both tails, both q values and all windows, are printed by `tail_bank_table.py`.

### Ticks after the switch for the seed-mean $\chi$ to cross halfway to the new target

| regime | q, tail | old → new | family_bank | tail_chi_399 | tail_chi_1999 | tail_chi_4999 | tail_chi_1999 (known margins) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gauss2clayton | .05 L | 0.31 → 0.62 | 115 | 131 | 703 | 1614 | 692 |
| gauss2clayton | .05 U | 0.31 → 0.11 | 110 | 135 | 662 | 1532 | 692 |
| clayton2gumbel | .05 L | 0.62 → 0.22 | 49 | 145 | 670 | 1445 | 701 |
| clayton2gumbel | .05 U | 0.11 → 0.52 | 57 | 121 | 685 | 1593 | 666 |
| gauss2t | .05 L | 0.31 → 0.41 | 195 | 173 | 638 | 1774 | 626 |
| gauss2t | .10 L | 0.39 → 0.46 | 195 | 182 | 699 | 1861 | 781 |
| mix | .05 L | 0.52 → 0.32 | 52 | 151 | 618 | 1399 | 642 |
| mix | .05 U | 0.21 → 0.42 | 63 | 118 | 701 | 1673 | 660 |

The counter's halfway time is about $\ln 2\cdot(M+1)/2$, as expected for an EW mean: 139, 693 and 1733
ticks. That is independent of the regime.

### Bank shape weights, seed means

| regime | pre ([1000, 5000)) | post ([7000, 10000)) |
| --- | --- | --- |
| gauss2clayton | Gaussian 0.98 | Clayton 0.99 |
| clayton2gumbel | Clayton 0.99 | Gumbel 0.98 |
| gauss2t | Gaussian 0.98 | t 0.98 |
| mix | Clayton 0.82, t 0.12, Gaussian 0.06 | t 0.54, Gumbel 0.32, Gaussian 0.14 |
| gaussian | Gaussian 0.98 | Gaussian 0.98 |

## 3. What this says

The expected story was: the bank has lower variance but is model-biased, especially on Gauss → t, where
it needs about 5000 ticks; tail_chi is unbiased but noisy. **That story is half right.**

- **Variance: confirmed.**
  - The bank's $\chi$ RMSE is 14–23 on every well-specified regime, pre-switch and late post-switch.
  - The counter's is 25–50 at M = 1999–4999 and about 100 at M = 399.
  - The bank turns all of the data into a few copula parameters, while the counter uses only the
    corner events: about $qM$ of them, with variance $\chi(1-\chi)/(qM)$.
- **Gauss → t: refuted.**
  - The bank crosses halfway in about 190 ticks. The only counter that is as fast is M = 399, and it is
    5× noisier.
  - After 2000 ticks the bank's bias is 0.
  - The 5000-tick figure belonged to tail-count detection. The likelihood sees the whole density, and
    the Gaussian–$t_4$ KL is 0.022–0.034 nats/tick (§6 of `thesis_bank_family.md`).
- **Model bias: confirmed, where the truth is outside the bank.**
  - On `mix`, a Clayton/Gumbel mixture with $\pi=0.75$, the bank puts its weight on one arm (Clayton
    0.82) instead of on the mixture. Its pre-switch bias is +20 and its RMSE is 74, against 46 for
    tail_chi_1999. It loses the whole run to tail_chi_1999 in all four mix cells, by 9–20.
  - The reason is that log-loss Hedge with $\eta=1$ is Bayes model averaging over the arms. Its
    posterior concentrates on the single best arm, whose $\chi$ is biased; it does not learn mixture
    weights. A mixture DGP needs mixture arms, or weights fitted by EM.
- **Switches: the bank wins across the board.**
  - Within the first 500 ticks after a switch, the bank has the lowest mean RMSE of the five main
    methods in all 16 cells (4 switches × 2 q × 2 tails).
  - This includes `mix`, where the bank's post-switch error (90) is far below the counters' (150–170).

So the trade-off is **model bias against variance**:

- The bank wins wherever its arms contain the truth, and on every switch.
- The model-free counter wins only when the copula is outside the bank's families, and only at a long
  memory.

**The bank loses on `mix`, and that is the case to report.**

**Known margins against the rank PIT.** At M = 1999 the counter's whole-run RMSE with known margins is
−0.4 to +4.7 higher, not lower, than with the rank PIT. That is beyond 2 SE in 11 of 20 cells.

- The rank PIT makes each marginal-corner frequency exactly $q$ over the window, which removes the
  denominator's sampling noise.
- So the shared rank-PIT footing is not what costs the counter.

## 4. Caveats

- The data are in copula space. The rank PIT is the only marginal step, so the comparison is about the
  dependence structure only.
- Every regime here is at one $\rho_S$, 0.58, with $t$ at $\nu=4$. The bank's $t$ arm has the DGP's $\nu$
  exactly; a $\nu$ mismatch would add model bias to the $t$ regimes.
- `mix` uses one mixing level ($\pi$ 0.75 → 0.25). Other levels would move the bank's bias.
- The bank's $\eta$ and $\delta$ are not tuned for tails: $\eta=1$, and $\delta=\delta^*$ for $m=1$.
