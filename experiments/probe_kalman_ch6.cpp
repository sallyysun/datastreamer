// Kalman and BOCPD methods under the Chapter 6 protocol (seeds 42-101, N = 5000, scored [500, 5000),
// constant/drift/jump/oscillate, 4 families), for comparison with the regenerated bank results
// (results/thesis_bank*.csv; same streams, so paired by seed). See budget_mpl.md §17.
// Usage (from experiments/): ./probe_kalman_ch6 [seeds=60]  (COPULA=...)  -> results/kalman_ch6[_fam].csv
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "../src/estimators/registry.hpp"
#include "../src/streamgenerator/datasource/copula.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"

int main(int argc, char** argv) {
    (void)copula::spearman_from_param(copula::Family::StudentT, 0.5, 4.0);
    const int seeds = argc > 1 ? std::atoi(argv[1]) : 60;
    const std::string cop = std::getenv("COPULA") ? std::getenv("COPULA") : "gaussian";
    const std::vector<std::string> keys{"kalman", "kalman_fast", "kalman_vff", "kalman_bank", "kalman_bank_sticky",
                                        "kalman_bank_quick", "kalman_bank_uniform", "kalman_bocpd", "kalman_bocpd_fast",
                                        "kalman_cusum", "kalman_glr"};
    std::FILE* out = std::fopen(("results/kalman_ch6" + std::string(cop == "gaussian" ? "" : "_" + cop) + ".csv").c_str(), "w");
    std::fprintf(out, "copula,scenario,seed,window,method,rmse\n");
    for (const std::string sc : {"constant", "drift", "jump", "oscillate"})
        for (int s = 0; s < seeds; ++s) {
            std::unique_ptr<Dependence> dep;
            if (sc == "constant") dep = std::make_unique<ConstantDependence>(0.6);
            else if (sc == "drift") dep = std::make_unique<DriftDependence>(-0.5, 0.8, 5000);
            else if (sc == "jump") dep = std::make_unique<JumpDependence>(0.2, 0.8, 2500);
            else dep = std::make_unique<OscillatingDependence>(0.6, 400.0, 0.0);
            auto c = dgp::make_copula(cop);
            CopulaSource src(*dep, *c, 42 + s);
            std::vector<std::unique_ptr<DependenceEstimator>> e;
            for (const auto& k : keys) e.push_back(make_estimator(k));
            std::vector<double> sw(keys.size(), 0), sp(keys.size(), 0), s0(keys.size(), 0);
            for (std::size_t t = 0; t < 5000; ++t) {
                const auto smp = src.next();
                for (std::size_t i = 0; i < keys.size(); ++i) {
                    if (t >= 500) {
                        const double d = e[i]->estimate() - smp.rho_s, d2 = d * d;
                        sw[i] += d2;
                        if (t >= 2500) sp[i] += d2;
                        if (t >= 2500 && t < 2600) s0[i] += d2;
                    }
                    e[i]->update(smp.x, smp.y);
                }
            }
            for (std::size_t i = 0; i < keys.size(); ++i) {
                std::fprintf(out, "%s,%s,%d,whole,%s,%.6f\n", cop.c_str(), sc.c_str(), 42 + s, keys[i].c_str(), std::sqrt(sw[i] / 4500));
                if (sc == "jump") {
                    std::fprintf(out, "%s,%s,%d,post,%s,%.6f\n", cop.c_str(), sc.c_str(), 42 + s, keys[i].c_str(), std::sqrt(sp[i] / 2500));
                    std::fprintf(out, "%s,%s,%d,rec_0_100,%s,%.6f\n", cop.c_str(), sc.c_str(), 42 + s, keys[i].c_str(), std::sqrt(s0[i] / 100));
                }
            }
        }
    std::fclose(out);
}
