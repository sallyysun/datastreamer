# Source rationale

Derivations, measured numbers and design reasons moved out of source comments on 2026-10-20 (comment trim on `massive-refactor`), by file. Material already in another note was not copied; the source headers point there instead.

## src/estimators/dependence/xiao_spearman.hpp

C++ port of Xiao (2019), "Novel Online Algorithms for Nonparametric Correlations with Application
to Analyze Sensor Data" (IEEE Big Data), reference implementation: external/onlineNPCORR (corr.py
online_corr / utility.py spearmanr_by_matrix, get_index). This estimate() reproduces
spearmanr_by_matrix exactly; we add two extensions beyond the paper: auto-calibrated cutpoints and
exponential forgetting (lambda).

Marginals are discretised into `bins` equiprobable cells whose cutpoints are fixed from an initial
calibration window. A bins x bins count grid then approximates the joint distribution, and
Spearman's rho is read off as a rank correlation over the grid (mid-rank of each marginal cell,
centred on the grand mean total_/2). Three memory regimes share the same estimate() formula:
lambda > 0 exponential forgetting (counts *= lambda each step); window > 0 hard sliding window
(oldest cell decremented); otherwise cumulative. rank_pit_window > 0 makes the cutpoints exactly
k/bins, so no calibration window is needed.

VFF: driven by the §14.14 rank residual s = 12 (u - 1/2)(v - 1/2) - rho_hat over its KNOWN null
variance. C_ then tracks E[e~^2], which is 1 when the memory is right -- the same statistic and the
same multiplicative rule as KalmanLocalLevel's vff_rate. Fixed null scale, not an estimated one: an
EWMA of s^2 chases the very lag the VFF exists to remove (14.11's trap); E[p^2] - rho_hat^2 biases
the other way, since rho_hat is the lagging quantity. Section 14.14 measured Var(s) at 0.92-1.00
across the whole rho range, so 1 is a KNOWN scale to within 8% and nothing has to be estimated.
This is the nonparametric stand-in for the Kalman's F.

RankPit members are declared after config_ because of the parametric_copula.hpp bug of 2026-09.

## src/estimators/dependence/hermiter.hpp

Port of Stephanou & Varughese (2021), "Sequential estimation of Spearman rank correlation using
Hermite series estimators"; reference implementation (R/Rcpp) external/hermiter. The orthonormal
Hermite-function, integral, full-domain-integral, and online-standardization recurrences match that
package exactly. Scope: only the online bivariate Spearman's-rho path. The reference package
additionally does univariate pdf/cdf/quantile estimation, Kendall's tau, consistent merging of
sub-estimators, and batch mode -- none ported.

estimate(): rho = 12 E[(F_X - 1/2)(F_Y - 1/2)] in the Hermite basis, with u = int F_X h dx,
v = int F_Y h dy, z = int h dx; the four terms are degree 3, 2, 2 and 1 in the accumulators.

Surprise statistics (§14.11): `surprise_fast_rate = 0` uses ||h_x h_y' - A||, whose norm is dominated
by ||h_x|| ||h_y|| (where in the margins the single point landed), so the dependence signal is a
small part of a large number. `> 0` compares two smoothed copies of the same matrix, so
per-observation noise cancels and only a persistent drift survives. The coefficient-matrix surprise
is the Hermite analogue of a Kalman innovation and carries shape change that the scalar rho throws
away. `surprise_baseline_rate` must not be the coefficient weight: the baseline would chase a change
at exactly the speed the coefficients absorb it (the §14.11 bug); 0 freezes after calibration
(Phase-I chart). `reset_boost > 0` is the Hermite analogue of covariance inflation: forget faster
for a short burst instead of discarding A_.

W closed form: integration by parts gives W(k,j) + W(j,k) = z_k z_j; column recurrence
W(k,j) = -sqrt(2/j)[k==j-1] + sqrt((j-1)/j) W(k,j-2). Bases: W(0,0)=z0^2/2, W(0,1)=-sqrt2,
W(k,1)=-sqrt2 [k==0], W(k,0)=z_k z_0 - W(0,k).

## src/estimators/registry.hpp

- `hermite_ew` differs from `hermite_pit` in BOTH order and marginal, so it cannot isolate rank-PIT;
  `hermite_ew20` is the control.
- `kalman_q*` sweep: if a plain filter at some other q beats the detectors, the detector comparison
  was being run at the wrong operating point.
- `kalman_periodic_inf` isolates "the alarm timing was informative" from "P_ was held higher".
- `gaussian_mle_cusum`: the MLE's score is already a standardized innovation, so it is the one
  detector that needs no statistic invented for it.
- `kalman_bank_uniform` isolates "carrying several memory lengths" from "picking between them".
- `hermite_o20/o40`, `xiao_b8/b40`: if Xiao beats Hermite on a singular copula by truncation of a
  smooth basis, raising Hermite's order must close the gap and changing Xiao's bins must not.
- `gaussian_mle_param` isolates whether a fault is in the marginals or in ParametricCopulaMLE's
  Gaussian path.
- check_registry: a listed key without a branch has segfaulted three times; the reverse direction
  (branch with no list entry) is checked by `make check-registry`.

## src/estimators/dependence/tail_cusum.hpp

Targets the change §14.13 found nothing detects: a copula SHAPE change at constant Spearman ρ.
A raw tail count is not a shape statistic (raising ρ raises tail co-occurrence too), so it is
divided by a middle co-occurrence:

    a = P(u <= q,   v <= q)      the deep tail
    b = P(u <= 1/2, v <= 1/2)    the middle, which rho alone determines
    r = a / b

⚠️ Measured, that ratio does NOT cancel ρ: for a Gaussian copula r runs 0.155 -> 0.228 as ρ_S goes
0.39 -> 0.79, while the whole Gaussian-to-t shape difference at fixed ρ is 0.19 -> 0.21. The ρ
sensitivity is five times the shape sensitivity. So the monitored statistic is the excess over a
Gaussian at the estimator's current ρ: `s = r_observed - r_gaussian(rho_hat)`, with r_gaussian
exact via Φ2(z_q, z_q; ρ) / Φ2(0, 0; ρ) (§A.8). Under any Gaussian copula s = 0 at every ρ; under a
tail-dependent one it is positive.

⚠️ Expect it to be SLOW: a is estimated from ~5% of the data at q = 0.15, so its standard error
falls ~4.5x more slowly than ρ's (distinction 4 in its most extreme form).

Why a Shewhart chart, not a CUSUM: a CUSUM's ARL0 grew LINEARLY in h (h=200 -> 667 blocks,
h=400 -> 1333) where an i.i.d. CUSUM's grows exponentially — the signature of a statistic whose
zero-floor never engages. tail_ is an EWMA with ESS ~1000 ticks, so consecutive blocks are
near-copies and the CUSUM just sums 50 of the same number. Decision rule: alarm when the smoothed
statistic sits L Phase-I SDs from its frozen baseline.

## src/estimators/dependence/gaussian_mle.hpp

Marginals use cumulative Welford moments: marginal scale is stationary, so it is NOT forgotten —
only the dependence is. This decouples the two timescales and avoids the hermite_ew cold-start
(where a forgotten variance took ~800 ticks to reach the right scale). For Gaussian marginals,
standardization gives exactly the normal scores.

Update (natural-gradient step, learning rate from accumulated information H):

    H_t     = lambda * H_{t-1} + I_theta,   I_theta = 1 + rho^2
    theta_t = theta_{t-1} + s_theta / H_t

λ ∈ (0,1) gives ESS ~ 1/(1-λ); λ = 0 (sentinel) means cumulative (step ~ 1/n). Deflating info_ on
an alarm is the exact analogue of Kalman covariance inflation (raises the gain, keeps theta_).
The standardized score s_theta/sqrt(I_theta) has mean 0, variance 1 at the true θ, so it is a
ready-made innovation for a detector. The clamp is needed because score(.,.,ρ) carries
1/(1-ρ²)² and dρ/dθ cancels only one power of it.

## src/estimators/dependence/kalman_bank.hpp

§14.8b: σ_ω² moves RMSE ~4x more than any detector, and its best value differs per scenario across
two and a half orders of magnitude; the value every detector wraps (1e-2) is right for two of five
scenarios. Hence model averaging over memory length — structurally the same device as BOCPD
(averaging over run length). Without switch_prob the weights converge once and never move again
(the cumulative-estimator trap). Cost: O(K) scalar updates per block on one shared block Spearman,
~500 bytes, no extra rank sorting.

## src/estimators/dependence/estimator_bank.hpp

KalmanBank only works for the local-level filter because it is the one estimator with a predictive
density. Spearman's ρ is an expectation, so s_k is a mean-zero residual with variance ~0.92–1.00
(§14.14), which is all a Gaussian predictive score needs — a bank over λ for ANY estimator. Cost is
K copies of the inner estimator; for the MLE that is 144 bytes each.

## src/estimators/objective.hpp

`rho` is the copula correlation, working in normal scores z_x = Phi^{-1}(u), z_y = Phi^{-1}(v)
(for Gaussian marginals these are just the standardized observations). Derivatives are w.r.t. rho:

- log_likelihood: l = -1/2 log(1-rho^2) - ( rho^2 (z_x^2 + z_y^2) - 2 rho z_x z_y ) / (2(1-rho^2))
- score: s = dl/drho = [ rho(1-rho^2) - rho(z_x^2+z_y^2) + z_x z_y (1+rho^2) ] / (1-rho^2)^2
  (unbiased: E[s] = 0 at the true rho)
- fisher: I(rho) = -E[d2l/drho2] = (1 + rho^2) / (1 - rho^2)^2 (closed form)

## src/estimators/dependence/mle_score_cusum.hpp

Every other detector wraps the Kalman filter, the one estimator that hands out a ready-made
standardized innovation. The MLE hands out one too: at the true theta the score has mean zero and
variance I_theta, so s / sqrt(I_theta) is already the quantity a CUSUM wants — nothing has to be
invented or re-estimated. Each term has unit variance and the MLE step makes successive terms only
weakly dependent, so the block sum is scaled by sqrt(block).

## src/estimators/dependence/marginal_rank_pit.hpp

Exact rank-PIT needs every past observation. This keeps a bounded sorted window instead, which is
what makes it streaming-admissible: O(log W) to rank, O(W) to maintain the sorted order. The probit
puts the output back on the real line where the Hermite basis expects it (van der Waerden normal
scores).

## src/estimators/dependence/kalman_var_chart.hpp

`cusum_k > 0`: the EWMA saturates at the new level; the CUSUM integrates it, which is the whole
difference on a small sustained shift. The null level of e~^2 is not exactly 1 because q over-models
a constant truth.

## src/estimators/dependence/hermite_cusum.hpp

Per tick the statistic is a rank-1 outer-product norm, dominated by the single observation's
magnitude; a block mean is what has usable SNR, and block = 20 makes the alarm rate comparable to
every other detector's. One-sided: only a *rise* in surprise indicates the coefficients have moved
away from what the estimator has absorbed.

## src/consumer/device_state.hpp

`DeviceState` is the recursively-updated state Z of a consumer device (thesis D = (Z, Psi, Z_0)).
`update()` is the one-pass update map Psi applied per event. It maintains, in O(1) per update:
fixed-dimensional moments (Welford mean/variance of prices and returns) and the inference layer
(a bank of online copula-dependence estimators). It folds the former `running_stats` /
`state_manager` / `dependence_monitor` into one.

## src/protocol/symbol_tick.hpp

Returns are not on the wire because price and return are the same information and estimation
belongs on the device. A device consumes one symbol's stream for univariate analysis (e.g.
variance), or two for bivariate dependence (`TickPair`).
