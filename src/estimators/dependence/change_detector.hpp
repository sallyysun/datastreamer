#pragma once

// Minimal common interface for the change-detection wrappers around
// KalmanLocalLevel (kalman_cusum.hpp, kalman_glr.hpp, ...), so the harness
// can log per-block alarms and route on "did this detector fire" without a
// dynamic_cast per detector type. Kept separate from DependenceEstimator
// (estimator.hpp) because not every estimator detects changepoints, and not
// every changepoint detector need be a DependenceEstimator -- multiple
// inheritance keeps the two orthogonal.
class ChangeDetector {
public:
    virtual ~ChangeDetector() = default;
    virtual bool alarmed_this_block() const = 0;
};
