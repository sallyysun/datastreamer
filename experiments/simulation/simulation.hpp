#pragma once

#include <string>
#include <vector>

namespace thesis {
struct H2Table;
}

namespace strategy_experiments {

struct StrategyTrace {
    std::vector<double> forecast, squared_error, excess, h2;
};

// MFF weights held before each tick, with that MFF's forecast and squared error.
struct WeightTrace {
    double delta;
    std::vector<double> weights;   // [t * kK + k]
    std::vector<double> forecast, squared_error;
};

struct EstimatorTrace {
    std::vector<StrategyTrace> strategies;
    double mixture_log_loss = 0;
    std::vector<WeightTrace> weight_paths;   // the study MFF, then plain Hedge (delta 0); WEIGHTS only
};

struct SeedResults {
    unsigned seed;
    std::vector<double> true_spearman;
    std::vector<EstimatorTrace> estimators;
};

// What a run records. scores: each strategy's forecast, squared error, excess log loss and H2 (MIXDENS scores MFF
// with the weighted density mixture); weight_paths: the MFF and plain-Hedge weights (WEIGHTS).
struct SimulationOptions {
    bool mixture_density = false;
    bool scores = true;
    bool weight_paths = false;
};

// One scenario and seed: replay the series into each estimator's strategies, scoring each held forecast first.
class Simulation {
public:
    Simulation(std::string cop, thesis::H2Table& h2, std::string scenario, unsigned seed, std::vector<double> vff_rates,
               SimulationOptions options);
    SeedResults run() const;

private:
    std::string cop_;
    thesis::H2Table& h2_;
    std::string scenario_;
    unsigned seed_;
    std::vector<double> vff_rates_;
    SimulationOptions options_;
};

}  // namespace strategy_experiments
