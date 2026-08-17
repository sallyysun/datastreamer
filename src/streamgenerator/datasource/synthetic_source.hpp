#pragma once

#include "datasource.hpp"
#include "schedule.hpp"

#include <memory>
#include <random>

// Synthetic bivariate market source. The dependence between the two return
// series is driven by a Schedule, so it can be constant or time-varying
// (drift / jump / oscillating / stochastic). Each event carries the true
// Spearman's rho for that step, giving online estimators a ground truth to
// track live.
class SyntheticSource : public DataSource {
public:
    // Defaults to a constant correlation of 0.6 (the previous fixed behaviour).
    explicit SyntheticSource(std::unique_ptr<Schedule> schedule =
                                 std::make_unique<ConstantSchedule>(0.6));

    MarketEvent next() override;

private:
    double price_x_;
    double price_y_;

    double volume_x_;
    double volume_y_;

    double drift_x_;
    double drift_y_;

    double vol_x_;
    double vol_y_;

    std::unique_ptr<Schedule> schedule_;

    std::mt19937 rng_;
    std::normal_distribution<double> normal_;
};
