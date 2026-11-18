#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "study_config.hpp"
#include "../../src/copula_family.hpp"
#include "../../src/estimators/dependence/datasketches_kll.hpp"
#include "../../src/special_functions.hpp"
#include "../../src/streamgenerator/datasource/copula.hpp"
#include "run_live_stream.hpp"

// First seed of a run: SEED_BASE if set (e.g. the 6.4 development seeds), else the study's first seed.
inline unsigned first_seed() {
    const char* base = std::getenv("SEED_BASE");
    return base ? unsigned(std::atoi(base)) : thesis::kFirstSeed;
}

inline bool study_verbose() {
    const char* value = std::getenv("VERBOSE");
    return value && std::string_view(value) == "1";
}

struct StudySeries {
    std::vector<double> x, y, true_spearman, target_spearman, estimated_pit_x, estimated_pit_y, true_pit_x, true_pit_y;
};

// Record one seeded series; the estimated PITs are KLL pseudo-observations computed from the preceding pairs.
inline StudySeries load_study_series(const std::string& family, const std::string& scenario, unsigned seed,
                                     std::size_t count) {
    LiveStream live;
    std::unique_ptr<Dependence> dependence;
    std::unique_ptr<dgp::Copula> copula;
    std::unique_ptr<CopulaSource> source;
    if (live_mode()) {
        live = live_stream(family, scenario, seed, count);
    } else {
        dependence = make_scenario(scenario, count);
        copula = dgp::make_copula(family);
        source = std::make_unique<CopulaSource>(*dependence, *copula, seed);
    }

    StudySeries series;
    for (auto* column : {&series.x, &series.y, &series.true_spearman, &series.target_spearman, &series.estimated_pit_x, &series.estimated_pit_y, &series.true_pit_x, &series.true_pit_y})
        column->reserve(count);
    datasketches_kll::seed(1);
    datasketches_kll::PairedKll margins(thesis::kKll, thesis::kClip);
    for (std::size_t t = 0; t < count; ++t) {
        double x, y;
        if (source) {
            const auto sample = source->next();
            x = sample.x;
            y = sample.y;
            series.true_spearman.push_back(sample.true_spearman);
            series.target_spearman.push_back(sample.target_spearman);
            series.true_pit_x.push_back(special::normal_cdf(sample.x));
            series.true_pit_y.push_back(special::normal_cdf(sample.y));
        } else {
            x = live.x[t];
            y = live.y[t];
            series.true_spearman.push_back(live.true_spearman[t]);
            series.target_spearman.push_back(live.target_spearman[t]);
            series.true_pit_x.push_back(live.true_pit_x[t]);
            series.true_pit_y.push_back(live.true_pit_y[t]);
        }
        const auto [u, v] = margins.update(x, y);
        series.x.push_back(x);
        series.y.push_back(y);
        series.estimated_pit_x.push_back(u);
        series.estimated_pit_y.push_back(v);
    }
    return series;
}

// The generating copula's parameter at each tick, from the scenario's target Spearman correlation.
inline std::vector<double> generating_parameters(const std::string& family, const StudySeries& series) {
    const auto generating_family = copula::family_of(family);
    std::vector<double> parameters;
    parameters.reserve(series.target_spearman.size());
    for (double target : series.target_spearman) parameters.push_back(copula::param_from_spearman(generating_family, target));
    return parameters;
}
