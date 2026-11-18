#include <cmath>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "../../src/estimators/device.hpp"
#include "../../src/estimators/evaluation.hpp"
#include "simulation.hpp"
#include "study_stream.hpp"

namespace strategy_experiments {

Simulation::Simulation(const Run& run, std::string scenario, unsigned seed, const std::vector<double>& rates,
                       bool mixture)
    : run_(run), scenario_(std::move(scenario)), seed_(seed), rates_(rates), mixture_(mixture) {}

SeedResults Simulation::run() const {
    using thesis::kN;
    const StudySeries s = load_study_series(run_.cop, scenario_, seed_, kN);
    const auto generating_parameter = generating_parameters(run_.cop, s);
    const copula::Family family = thesis::family_of(run_.cop);
    std::vector<double> log_density(kN);
    for (std::size_t t = 0; t < kN; ++t) log_density[t] = copula::log_density(family, s.true_pit_x[t], s.true_pit_y[t], generating_parameter[t]);

    SeedResults results{seed_, s.true_spearman, {}};
    for (const char* estimator : kEst) {
        thesis::ForgettingStrategies strategies(estimator, rates_);
        const std::size_t count = strategies.size();
        const auto& weights = strategies.mff_weights();
        const auto mixture_weights = mixture_ ? std::span<const double>(weights) : std::span<const double>{};
        EstimatorTrace trace{std::vector<StrategyTrace>(count), 0};
        for (auto& strategy : trace.strategies) {
            strategy.forecast.resize(kN);
            strategy.squared_error.resize(kN);
            strategy.excess.resize(kN);
            strategy.h2.assign(kN, NAN);
        }
        std::vector<double> f(count);
        for (std::size_t t = 0; t < kN; ++t) {
            strategies.forecasts(f);
            if (mixture_ && strategies.mff_ready(t))
                trace.mixture_log_loss -= thesis::H2Table::mix_logc(f.data(), weights.data(), thesis::kK, s.estimated_pit_x[t], s.estimated_pit_y[t]);
            const thesis::ScoringObservation truth{s.true_spearman[t], s.true_pit_x[t], s.true_pit_y[t], generating_parameter[t], log_density[t],
                                                   thesis::checkpoint(scenario_, t)};
            for (std::size_t i = 0; i < count; ++i) {
                const auto score = thesis::score_forecast(run_.h2, truth, f, i, mixture_weights);
                auto& strategy = trace.strategies[i];
                strategy.forecast[t] = f[i];
                strategy.squared_error[t] = score.squared_error;
                strategy.excess[t] = score.excess;
                strategy.h2[t] = score.h2;
            }
            strategies.update(s.x[t], s.y[t], s.estimated_pit_x[t], s.estimated_pit_y[t], t);
        }
        results.estimators.push_back(std::move(trace));
    }
    return results;
}

}  // namespace strategy_experiments
