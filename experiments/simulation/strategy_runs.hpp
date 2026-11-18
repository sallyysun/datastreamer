#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "../../src/copula_family.hpp"
#include "../../src/estimators/device.hpp"
#include "study_stream.hpp"

namespace thesis {
struct H2Table;
struct Win;
}

namespace strategy_experiments {

struct Run {
    int seeds;
    std::string cop, sfx;
    const char* tag;
    thesis::H2Table& h2;
};

struct Stream : StudySeries { std::vector<double> generating_parameter; };

inline constexpr const char* kEst[] = {"xiao", "hermite", "rmpl"};
Stream make_stream(const std::string& family, const std::string& scenario, unsigned seed);

using thesis::FixedMemoryArms;

struct WindowScores {
    double rmse, excess_x1000, h2_x1000;
};

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

WindowScores summarize_window(const thesis::Win& window, const std::vector<double>& squared_error,
                              const std::vector<double>& excess, const std::vector<double>& hellinger);

int run_famswitch(const Run& run);
int run_scoresfam(const Run& run);
int run_regretfam(const Run& run);
int run_regret(const Run& run);

using thesis::FamilyHellinger;
using thesis::FamilyInverse;

}  // namespace strategy_experiments
