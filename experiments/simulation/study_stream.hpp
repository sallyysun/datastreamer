#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "study_config.hpp"
#include "../../src/estimators/dependence/datasketches_kll.hpp"
#include "../../src/special_functions.hpp"
#include "../../src/streamgenerator/datasource/copula.hpp"
#include "run_live_stream.hpp"

inline bool study_verbose() {
    const char* value = std::getenv("VERBOSE");
    return value && std::string_view(value) == "1";
}

struct StudySeries {
    std::vector<double> x, y, rho, target, u, v, pu, pv;
};

// Record one seeded series; u, v are KLL pseudo-observations computed from the preceding pairs.
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
    for (auto* column : {&series.x, &series.y, &series.rho, &series.target, &series.u, &series.v, &series.pu, &series.pv})
        column->reserve(count);
    datasketches_kll::seed(1);
    datasketches_kll::PairedKll margins(thesis::kKll, thesis::kClip);
    for (std::size_t t = 0; t < count; ++t) {
        double x, y;
        if (source) {
            const auto sample = source->next();
            x = sample.x;
            y = sample.y;
            series.rho.push_back(sample.rho_s);
            series.target.push_back(sample.target);
            series.pu.push_back(special::normal_cdf(sample.x));
            series.pv.push_back(special::normal_cdf(sample.y));
        } else {
            x = live.x[t];
            y = live.y[t];
            series.rho.push_back(live.rho_s[t]);
            series.target.push_back(live.target[t]);
            series.pu.push_back(live.pu[t]);
            series.pv.push_back(live.pv[t]);
        }
        const auto [u, v] = margins.update(x, y);
        series.x.push_back(x);
        series.y.push_back(y);
        series.u.push_back(u);
        series.v.push_back(v);
    }
    return series;
}
