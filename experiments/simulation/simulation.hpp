#pragma once

#include <string>
#include <vector>

#include "strategy_runs.hpp"

namespace strategy_experiments {

struct StrategyTrace {
    std::vector<double> forecast, squared_error, excess, h2;
};

struct EstimatorTrace {
    std::vector<StrategyTrace> strategies;
    double mixture_log_loss = 0;
};

struct SeedResults {
    unsigned seed;
    std::vector<double> true_spearman;
    std::vector<EstimatorTrace> estimators;
};

// One scenario and seed: replay the series into each estimator's strategies, scoring each held forecast first.
class Simulation {
public:
    Simulation(const Run& run, std::string scenario, unsigned seed, const std::vector<double>& rates, bool mixture);
    SeedResults run() const;

private:
    const Run& run_;
    std::string scenario_;
    unsigned seed_;
    const std::vector<double>& rates_;
    bool mixture_;
};

}  // namespace strategy_experiments
