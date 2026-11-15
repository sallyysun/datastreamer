# Section 6.5 — family-switching results (verified from saved outputs)

## Family identification (density3b; 60 replications)

| Switch | t50 median [IQR] | reached 50% | t90 median [IQR] | reached 90% | original-family weight, pre (500–2499) | new-family weight, post (2500–4999) | new-family weight, late (3000–4999) |
|---|---|---|---|---|---|---|---|
| Gaussian → t4 | 174.5 [130.25, 246.75] | 60/60 | 261.5 [178, 342] | 60/60 | 0.958 | 0.895 | 0.980 |
| Gaussian → Clayton | 94 [80, 124] | 60/60 | 128 [99, 162] | 60/60 | 0.958 | 0.945 | 0.991 |
| Clayton → Gumbel | 68.5 [44.75, 142.75] | 60/60 | 156 [109, 213] | 60/60 | 0.992 | 0.922 | 0.976 |

## Stability under constant families (density3; no change; 60 replications)

False switch = any other shape's summed weight reaches 50% (90%) at some tick in 2500–4999; correct-family weight = mean over ticks 500–4999.

| Truth | correct-family weight | another shape reaches 50% | another shape reaches 90% | shapes involved (50%) |
|---|---|---|---|---|
| gaussian | 0.961 | 14/60 | 7/60 | {'clayton': 4, 'gumbel': 7, 't': 3} |
| t | 0.983 | 6/60 | 1/60 | {'clayton': 2, 'gaussian': 2, 'gumbel': 3} |
| clayton | 0.992 | 2/60 | 0/60 | {'gaussian': 1, 't': 1} |
| gumbel | 0.981 | 9/60 | 2/60 | {'clayton': 2, 't': 7} |

## Predictive accuracy: ΔL = L(Gaussian-only MFF) − L(family-augmented MFF), ×10⁻³ nats/obs, mean ± 2 SE (60 paired replications; positive favours family augmentation)

| Switch | first 100 after (2500–2599) | remaining 2400 (2600–4999) | full post-switch (2500–4999) | L Gaussian-only (post) | L family (post) |
|---|---|---|---|---|---|
| Gaussian → t4 | +11.6 ± 7.8 | +31.5 ± 2.0 | +30.7 ± 1.9 | 33.97 | 3.25 |
| Gaussian → Clayton | +17.2 ± 6.1 | +74.5 ± 2.2 | +72.2 ± 2.1 | 78.58 | 6.39 |
| Clayton → Gumbel | -46.4 ± 7.4 | +26.7 ± 1.2 | +23.8 ± 1.2 | 29.11 | 5.35 |

Cost of family augmentation under constant Gaussian dependence (ticks 500–4999): L(family) − L(Gaussian-only) = **+0.29 ± 0.10** ×10⁻³ nats/obs (L family 1.49, Gaussian-only 1.19).

Interpretation of the windows: in Clayton → Gumbel the family-augmented MFF is worse for the first 100 observations
(adaptation lag: weight is still on Clayton, with a transient share on t4) but better over observations 2600–4999 and
over the full post-switch period.

## Experimental provenance

| Item | Family-switch experiments (Section 6.5) |
|---|---|
| Candidates | 24 arms: 8 copula shapes (Gaussian, t4 with ν = 4, Clayton, Gumbel, and Clayton/Gumbel rotated by 90° and 270°) × ESS 99, 399, 1999 |
| Arm estimator | parametric `CopulaFit` online pseudo-likelihood in each shape (expected-Fisher scoring, forgetting 1 − 2/(N+1)); not the RMPL estimators of Sections 6.2–6.4 |
| Weighting | Hedge with fixed share, η = 1, δ = 2.1 × 10⁻⁴ (δ* for K = 24, m = 1, T = 4970; not tuned) |
| Scoring / forecast | each arm scored by its own copula log density before the observation; forecast = weighted mixture of the 24 arm densities |
| Margins | exponentially weighted empirical CDFs, N_eff,m = 500 (not the KLL sketch used in Sections 6.2–6.4) |
| Paths | T = 5000; switch in copula family at observation 2500 with ρ_S ≈ 0.58 held constant (d = 0.6); evaluation from observation 500; scoring and arm updates start once 30 observations are held |
| Replications | 60 (seeds 42–101) |
| Gaussian-only comparator | `gauss_bank`: identical machinery (same raw observations, same EW margins N_eff,m = 500, same Gaussian `CopulaFit` arms 99/399/1999, own-density log-loss scoring, η = 1, same true-PIT evaluation); only the candidate set is restricted to the Gaussian shape, and δ is the corresponding δ* for K = 3 (3/(2·4969) ≈ 3.0 × 10⁻⁴) |
| Metric | excess log loss log c_true − log ĉ at the true PIT, per-observation mean over each window, ×10⁻³ nats |
| Runs | `TAG=density3b` (switch scenarios, latest) and `density3` (constant paths), probe_family_bank, EW margins, matched paths. Switch times and weight paths are identical to the earlier EW-margin runs (audit "final", density, density2). The pre-audit medians 180/98/78 (rank-window PIT) are superseded and their outputs were not preserved. |
| Not compatible with | the Chapter 6 KLL configuration (KLL margins, RMPL arms, ESS 39–1999, δ = 10⁻⁴) |

Sources: `experiments/famswitch_final.py` (this summary, `results/famswitch_final.csv`), `experiments/plot_famswitch.py`
(figures), `experiments/probe_family_bank.cpp`, `src/estimators/dependence/family_bank.hpp`;
data `experiments/results/family_bank{,_switch,_path}_{42,57,72,87}_density3b_<regime>.csv` and `..._density3[_fam].csv`.
