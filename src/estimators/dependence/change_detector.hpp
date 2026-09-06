#pragma once

// Common interface for the change-detection wrappers, so callers can log
// alarms without a dynamic_cast per detector type. Kept separate from
// DependenceEstimator since not every estimator detects changepoints.
class ChangeDetector {
public:
    virtual ~ChangeDetector() = default;
    virtual bool alarmed_this_block() const = 0;
};
