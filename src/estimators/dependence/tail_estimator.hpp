#pragma once

// Optional side interface for an estimator that also reports finite-q tail dependence:
// chi_L(q) = P(V <= q | U <= q), chi_U(q) = P(V > 1-q | U > 1-q), and the limits lambda_L, lambda_U
// (NaN where the estimator has no model for the limit, or q is not one it tracks). estimate() stays
// Spearman's rho, per the estimator contract. notes/studies/thesis_bank_tails.md.
struct TailEstimate { double chi_L, chi_U, lambda_L, lambda_U; };

class TailEstimator {
public:
    virtual ~TailEstimator() = default;
    virtual TailEstimate tails(double q) const = 0;
};
