#pragma once

// Optional second interface for a dependence estimator that can hand out its own predictive
// density, so a bank can score it by log loss without borrowing a parametric family.
// log_predictive(u, v) is the log density, on the unit square, of the pseudo-observation (u, v)
// that the caller is about to feed it as (probit(u), probit(v)), under the state held before that
// update. Each implementation maps (u, v) through its own marginal step and includes the
// Jacobian, so densities from different estimators are in the same coordinates and comparable.
// NaN means "no density in this configuration". archive/notes/studies/thesis_bank_own_density.md.
class PredictiveDensity {
public:
    virtual ~PredictiveDensity() = default;
    virtual double log_predictive(double u, double v) const = 0;
};
