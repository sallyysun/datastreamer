#include <algorithm>
#include <cmath>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "../../src/estimators/device.hpp"
#include "../../src/estimators/evaluation.hpp"
#include "simulation.hpp"
#include "strategy_runs.hpp"
#include "study_stream.hpp"

namespace strategy_experiments {

Simulation::Simulation(std::string cop, thesis::H2Table& h2, std::string scenario, unsigned seed, std::vector<double> vff_rates,
                       SimulationOptions options)
    : cop_(std::move(cop)), h2_(h2), scenario_(std::move(scenario)), seed_(seed), vff_rates_(std::move(vff_rates)), options_(options) {}

namespace {

WeightTrace empty_weight_trace(double delta) {
    using thesis::kN;
    return {delta, std::vector<double>(kN * thesis::kK), std::vector<double>(kN), std::vector<double>(kN)};
}

void record_weights(WeightTrace& trace, std::size_t t, const std::vector<double>& weights, double forecast, double truth) {
    std::copy(weights.begin(), weights.end(), trace.weights.begin() + t * thesis::kK);
    const double error = forecast - truth;
    trace.forecast[t] = forecast;
    trace.squared_error[t] = error * error;
}

}  // namespace

SeedResults Simulation::run() const {
    using thesis::kN;
    const StudySeries s = load_study_series(cop_, scenario_, seed_, kN);
    std::vector<double> generating_parameter, log_density;
    if (options_.scores) {
        generating_parameter = generating_parameters(cop_, s);
        const copula::Family family = thesis::family_of(cop_);
        log_density.resize(kN);
        for (std::size_t t = 0; t < kN; ++t) log_density[t] = copula::log_density(family, s.true_pit_x[t], s.true_pit_y[t], generating_parameter[t]);
    }
    const bool mixture = options_.mixture_density;

    SeedResults results{seed_, s.true_spearman, {}};
    for (const auto& estimator : thesis::estimators()) {
        thesis::ForgettingStrategies strategies(estimator, vff_rates_);
        const std::size_t count = strategies.size();
        const auto& weights = strategies.mff_weights();
        const auto mixture_weights = mixture ? std::span<const double>(weights) : std::span<const double>{};
        EstimatorTrace trace{std::vector<StrategyTrace>(count), 0, {}};
        thesis::Mff hedge(copula::Family::Gaussian, thesis::kK, 1.0, 0.0);
        if (options_.weight_paths) trace.weight_paths = {empty_weight_trace(thesis::Mff().delta), empty_weight_trace(hedge.delta)};
        if (options_.scores)
            for (auto& strategy : trace.strategies) {
                strategy.forecast.resize(kN);
                strategy.squared_error.resize(kN);
                strategy.excess.resize(kN);
                strategy.h2.assign(kN, NAN);
            }
        std::vector<double> f(count);
        for (std::size_t t = 0; t < kN; ++t) {
            strategies.forecasts(f);
            if (mixture && strategies.mff_ready(t))
                trace.mixture_log_loss -= thesis::H2Table::mix_logc(f.data(), weights.data(), thesis::kK, s.estimated_pit_x[t], s.estimated_pit_y[t]);
            if (options_.scores) {
                const thesis::ScoringObservation truth{s.true_spearman[t], s.true_pit_x[t], s.true_pit_y[t], generating_parameter[t],
                                                       log_density[t], thesis::checkpoint(scenario_, t)};
                for (std::size_t i = 0; i < count; ++i) {
                    const auto score = thesis::score_forecast(h2_, truth, f, i, mixture_weights);
                    auto& strategy = trace.strategies[i];
                    strategy.forecast[t] = f[i];
                    strategy.squared_error[t] = score.squared_error;
                    strategy.excess[t] = score.excess;
                    strategy.h2[t] = score.h2;
                }
            }
            if (options_.weight_paths) {
                const auto& arms = strategies.arms().estimators();
                record_weights(trace.weight_paths[0], t, weights, f[thesis::kK], s.true_spearman[t]);
                record_weights(trace.weight_paths[1], t, hedge.w, hedge.estimate(arms), s.true_spearman[t]);
                hedge.score_and_update_weights(arms, s.estimated_pit_x[t], s.estimated_pit_y[t], t);
            }
            strategies.update(s.x[t], s.y[t], s.estimated_pit_x[t], s.estimated_pit_y[t], t);
        }
        results.estimators.push_back(std::move(trace));
    }
    return results;
}

}  // namespace strategy_experiments
