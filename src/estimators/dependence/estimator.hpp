#pragma once

#include <cstddef>

// Common interface for all online dependence estimators. estimate() always returns Spearman's rho,
// never a native parameter; reset() is the change-detection restart hook; store_bytes() is a thesis
// metric (streaming admissibility), not diagnostics.
class DependenceEstimator {
public:
    virtual ~DependenceEstimator() = default;

    virtual void update(double x, double y) = 0;
    virtual double estimate() const = 0;
    virtual void reset() = 0;
    virtual std::size_t store_bytes() const = 0;
    virtual const char* name() const = 0;
};