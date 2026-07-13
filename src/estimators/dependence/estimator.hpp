#pragma once

#include <cstddef>

// Thesis Device
class DependenceEstimator {
public:
    virtual ~DependenceEstimator() = default;

    virtual void update(double x, double y) = 0;
    virtual double estimate() const = 0;
    virtual std::size_t store_bytes() const = 0;
    virtual const char* name() const = 0;
};