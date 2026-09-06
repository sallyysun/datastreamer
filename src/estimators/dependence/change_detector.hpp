#pragma once

// Common interface for the change-detection wrappers, kept separate from
// DependenceEstimator since not every estimator detects changepoints.
class ChangeDetector {
public:
    virtual ~ChangeDetector() = default;
    virtual bool alarmed_this_block() const = 0;
};
