#pragma once

#include <limits>

// Common interface for the change-detection wrappers, kept separate from
// DependenceEstimator since not every estimator detects changepoints.
class ChangeDetector {
public:
    virtual ~ChangeDetector() = default;
    virtual bool alarmed_this_block() const = 0;

    // The quantity compared against the threshold, oriented so larger = more
    // evidence of change. Threshold-free, so a probe can ask whether the signal
    // is present separately from whether h lets it through.
    // NaN, not 0, so an unimplemented detector is skipped rather than scored 0.5.
    virtual double monitor_statistic() const {
        return std::numeric_limits<double>::quiet_NaN();
    }
};
