#pragma once

#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "../../src/estimators/evaluation.hpp"
#include "simulation.hpp"
#include "strategy_runs.hpp"

namespace strategy_experiments {

struct WindowScores {
    double rmse, excess_x1000, h2_x1000;
};

WindowScores summarize_window(const thesis::Win& window, const std::vector<double>& squared_error,
                              const std::vector<double>& excess, const std::vector<double>& hellinger);

struct StrategyOptions {
    bool mixdens, paths;
    const char* dump_seed;
    std::string dump_scenario;
};

// The strategies_* outputs and the per-scenario sums across seeds behind them.
class StrategyResults {
public:
    StrategyResults(const Run& run, const StrategyOptions& options, const std::vector<std::string>& strategy_labels);
    ~StrategyResults();
    StrategyResults(const StrategyResults&) = delete;
    StrategyResults& operator=(const StrategyResults&) = delete;

    bool ok() const { return ok_; }
    void begin_scenario(const std::string& scenario);
    void add(int seed_index, const SeedResults& result);
    void end_scenario();

private:
    // One estimator and strategy across the seeds of a scenario.
    struct StrategyTotals {
        std::vector<double> squared_error_sum;              // [t]
        std::vector<double> h2_sum_x1000;                   // [t], PATHS only
        std::vector<int> extreme_seeds_by_tick;             // [t], PATHS only
        std::vector<float> excess_x1000, forecast;          // [seed * kN + t], PATHS only
        std::vector<float> excess_after_burn_in_x1000;      // all seeds x ticks, extremes only
        int extreme_seeds = 0, extreme_seed_ticks = 0;      // extremes only
    };

    StrategyTotals& totals(std::size_t estimator, std::size_t strategy) { return totals_[estimator * strategy_count_ + strategy]; }
    void write_extremes();
    void write_paths();
    void write_jump_path();

    const std::string& cop_;
    const int seeds_;
    const bool paths_;
    const std::vector<std::string> strategy_labels_;
    const std::size_t strategy_count_;
    std::FILE* mixture_check_file_ = nullptr;
    std::FILE* seed_scores_file_ = nullptr;
    std::FILE* jump_path_file_ = nullptr;
    std::FILE* tick_paths_file_ = nullptr;
    std::FILE* seed_dump_file_ = nullptr;
    std::FILE* extremes_file_ = nullptr;
    bool ok_ = false;

    std::string scenario_;
    std::vector<thesis::Win> windows_;
    std::vector<StrategyTotals> totals_;   // [estimator * strategy_count_ + strategy]
    std::vector<double> true_spearman_;
};

}  // namespace strategy_experiments
