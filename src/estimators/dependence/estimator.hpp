#pragma once

#include <cstddef>

// Common interface for all online dependence estimators.
//
// estimate() returns Spearman's rho (rho_S), so parametric and nonparametric
// estimators can be compared in the same space. reset() clears all state and
// is what a change-detection wrapper calls to "restart" the estimator.
class DependenceEstimator {
public:
    virtual ~DependenceEstimator() = default;

    virtual void update(double x, double y) = 0;
    virtual double estimate() const = 0;
    virtual void reset() = 0;
    virtual std::size_t store_bytes() const = 0;
    virtual const char* name() const = 0;
    // False while estimate() is a placeholder (no observation absorbed yet, a calibration sample still
    // filling, an output hold). Banks do not score an arm until it is ready.
    virtual bool ready() const { return true; }
};