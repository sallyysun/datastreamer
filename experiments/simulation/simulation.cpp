#include <cmath>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../../src/estimators/device.hpp"
#include "../../src/estimators/evaluation.hpp"
#include "strategy_runs.hpp"
#include "study_stream.hpp"

namespace strategy_experiments {

Stream make_stream(const std::string& family, const std::string& scenario, unsigned seed) {
    Stream stream{load_study_series(family, scenario, seed, thesis::kN), {}};
    const auto generating_family = thesis::family_of(family);
    stream.par.reserve(stream.target.size());
    for (double target : stream.target) stream.par.push_back(copula::param_from_spearman(generating_family, target));
    return stream;
}

WindowScores summarize_window(const thesis::Win& window, const std::vector<double>& squared_error,
                              const std::vector<double>& excess, const std::vector<double>& hellinger) {
    const bool dense = std::string_view(window.name) == "rec_0_100";
    double se_sum = 0, excess_sum = 0, h2_sum = 0;
    std::size_t h2_count = 0;
    for (std::size_t t = window.a; t < window.b; ++t) {
        se_sum += squared_error[t];
        excess_sum += excess[t];
        if (!std::isnan(hellinger[t]) && (dense || (t - thesis::kBurn) % 50 == 0)) {
            h2_sum += hellinger[t];
            ++h2_count;
        }
    }
    const double n = double(window.b - window.a);
    return {std::sqrt(se_sum / n), 1000 * excess_sum / n, h2_count ? 1000 * h2_sum / h2_count : NAN};
}

Simulation::Simulation(const Run& run, std::string scenario, unsigned seed, const std::vector<double>& rates,
                       bool mixture)
    : run_(run), scenario_(std::move(scenario)), seed_(seed), rates_(rates), mixture_(mixture) {}

SeedResults Simulation::run() const {
    using thesis::kN;
    const Stream s = make_stream(run_.cop, scenario_, seed_);
    const copula::Family family = thesis::family_of(run_.cop);
    std::vector<double> log_density(kN);
    for (std::size_t t = 0; t < kN; ++t) log_density[t] = copula::log_density(family, s.pu[t], s.pv[t], s.par[t]);

    SeedResults results{seed_, s.rho, {}};
    for (const char* estimator : kEst) {
        thesis::EstimatorStrategies strategies(estimator, rates_);
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
                trace.mixture_log_loss -= thesis::H2Table::mix_logc(f.data(), weights.data(), thesis::kK, s.u[t], s.v[t]);
            const thesis::ScoringObservation truth{s.rho[t], s.pu[t], s.pv[t], s.par[t], log_density[t],
                                                   thesis::checkpoint(scenario_, t)};
            for (std::size_t i = 0; i < count; ++i) {
                const auto score = thesis::score_forecast(run_.h2, truth, f, i, mixture_weights);
                auto& strategy = trace.strategies[i];
                strategy.forecast[t] = f[i];
                strategy.squared_error[t] = score.squared_error;
                strategy.excess[t] = score.excess;
                strategy.h2[t] = score.h2;
            }
            strategies.update(s.x[t], s.y[t], s.u[t], s.v[t], t);
        }
        results.estimators.push_back(std::move(trace));
    }
    return results;
}

}  // namespace strategy_experiments
