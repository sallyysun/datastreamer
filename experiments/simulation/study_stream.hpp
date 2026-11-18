#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>

#include "study_config.hpp"
#include "../../src/estimators/dependence/datasketches_kll.hpp"
#include "../../src/special_functions.hpp"
#include "../../src/streamgenerator/datasource/copula_source.hpp"
#include "run_live_stream.hpp"

inline bool study_verbose() {
    const char* value = std::getenv("VERBOSE");
    return value && std::string_view(value) == "1";
}

struct StudyObservation {
    double x, y, rho_s, target;
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
    datasketches_kll::PairedKll margins(thesis::kKll, thesis::kClip);
    for (std::size_t t = 0; t < count; ++t) {
        StudyObservation observation{};
        if (source) {
            const auto sample = source->next();
            observation.x = sample.x;
            observation.y = sample.y;
            observation.rho_s = sample.rho_s;
            observation.target = sample.target;
            observation.true_u = special::normal_cdf(sample.x);
            observation.true_v = special::normal_cdf(sample.y);
        } else {
            observation.x = live.x[t];
            observation.y = live.y[t];
            observation.rho_s = live.rho_s[t];
            observation.target = live.target[t];
            observation.true_u = live.pu[t];
            observation.true_v = live.pv[t];
        }
        const auto [u, v] = margins.update(observation.x, observation.y);
        observation.u = u;
        observation.v = v;
        observe(observation);
    }
}
