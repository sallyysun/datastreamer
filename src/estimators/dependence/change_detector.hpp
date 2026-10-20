#pragma once

#include <limits>

// Interface for change-detection wrappers, separate from DependenceEstimator.
class ChangeDetector {
public:
    virtual ~ChangeDetector() = default;
    virtual bool alarmed_this_block() const = 0;

    // Larger = more evidence of change, threshold-free. NaN (not 0) so an unimplemented
    // detector is skipped rather than scored 0.5.
    virtual double monitor_statistic() const {
        return std::numeric_limits<double>::quiet_NaN();
    }
};
