#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>

#include "study_config.hpp"
#include "../../src/estimators/dependence/datasketches_kll.hpp"
#include "../../src/special_functions.hpp"
#include "../../src/streamgenerator/datasource/copula_source.hpp"
#include "live_source.hpp"

struct StudyObservation {
    double x, y, rho_s, knob;
    double true_u, true_v;
    double u, v;
};

// Feed matched observations with pseudo-observations computed from prior pairs.
template <typename Observe>
void for_each_study_observation(const std::string& family, const std::string& scenario,
                                unsigned seed, std::size_t count, Observe observe) {
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

    datasketches_kll::seed(1);
    datasketches_kll::Kll x_margin(thesis::kKll), y_margin(thesis::kKll);
    for (std::size_t t = 0; t < count; ++t) {
        StudyObservation observation{};
        if (source) {
            const auto sample = source->next();
            observation.x = sample.x;
            observation.y = sample.y;
            observation.rho_s = sample.rho_s;
            observation.knob = sample.knob;
            observation.true_u = special::normal_cdf(sample.x);
            observation.true_v = special::normal_cdf(sample.y);
        } else {
            observation.x = live.x[t];
            observation.y = live.y[t];
            observation.rho_s = live.rho_s[t];
            observation.knob = live.d[t];
            observation.true_u = live.pu[t];
            observation.true_v = live.pv[t];
        }
        observation.u = std::clamp(x_margin.cdf(observation.x), thesis::kClip, 1 - thesis::kClip);
        observation.v = std::clamp(y_margin.cdf(observation.y), thesis::kClip, 1 - thesis::kClip);
        x_margin.insert(observation.x);
        y_margin.insert(observation.y);
        observe(observation);
    }
}
