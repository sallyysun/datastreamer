// Offline experiment harness for the nonparametric dependence estimators.
//
// Generates a bivariate stream with a known time-varying Spearman rho, feeds it
// to each estimator online, and records estimate-vs-truth to CSV plus an RMSE
// summary. This is the end-to-end evaluation loop the roadmap calls for:
//   Dynamic DGP -> {Xiao, Hermite, ...} -> tracking error.
//
// Usage: ./experiment [scenario]
//   scenario in {constant, drift, jump, stochastic}   (default: jump)

#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "../estimators/registry.hpp"
#include "../streamgenerator/datasource/schedule.hpp"
#include "copula_source.hpp"

namespace {

struct Series {
    std::string name;
    std::vector<double> est;
};

std::unique_ptr<Schedule> make_schedule(const std::string& scenario, std::size_t N, unsigned seed) {
    if (scenario == "constant") return std::make_unique<ConstantSchedule>(0.6);
    if (scenario == "drift") return std::make_unique<DriftSchedule>(-0.5, 0.8, N);
    if (scenario == "jump") return std::make_unique<JumpSchedule>(0.2, 0.8, N / 2);
    if (scenario == "stochastic") return std::make_unique<StochasticSchedule>(0.01, 0.6, seed + 1);
    if (scenario == "oscillate") return std::make_unique<OscillatingSchedule>(0.6, 1000.0, 0.0);
    return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t N = 5000;
    const unsigned seed = 42;
    const std::size_t burn = 500;  // steps to skip before scoring

    const std::string scenario = (argc > 1) ? argv[1] : "jump";

    auto schedule = make_schedule(scenario, N, seed);
    if (!schedule) {
        std::fprintf(stderr,
                     "unknown scenario '%s' (use constant|drift|jump|stochastic|oscillate)\n",
                     scenario.c_str());
        return 1;
    }

    CopulaSource source(*schedule, seed);

    const std::vector<std::string> keys = {"xiao", "xiao_ew", "hermite", "hermite_ew", "windowed_rank"};
    std::vector<std::unique_ptr<DependenceEstimator>> ests;
    for (const auto& k : keys) {
        auto e = make_estimator(k);
        if (!e) {
            std::fprintf(stderr, "unknown estimator '%s'\n", k.c_str());
            return 1;
        }
        ests.push_back(std::move(e));
    }

    std::vector<double> true_s(N);
    std::vector<Series> series(keys.size());
    for (std::size_t e = 0; e < keys.size(); ++e) series[e].name = keys[e];

    for (std::size_t t = 0; t < N; ++t) {
        const CopulaSource::Sample s = source.next();
        true_s[t] = s.rho_s;
        for (std::size_t e = 0; e < ests.size(); ++e) {
            ests[e]->update(s.x, s.y);
            series[e].est.push_back(ests[e]->estimate());
        }
    }

    const std::string path = "experiment_" + scenario + ".csv";
    std::ofstream out(path);
    out << "t,true_rho_s";
    for (const auto& k : keys) out << "," << k;
    out << "\n";
    for (std::size_t t = 0; t < N; ++t) {
        out << t << "," << true_s[t];
        for (const auto& sr : series) out << "," << sr.est[t];
        out << "\n";
    }

    std::printf("scenario=%s  N=%zu  (RMSE vs true Spearman, after burn-in %zu)\n",
                scenario.c_str(), N, burn);
    for (std::size_t e = 0; e < series.size(); ++e) {
        double se = 0.0;
        std::size_t cnt = 0;
        for (std::size_t t = burn; t < N; ++t) {
            const double d = series[e].est[t] - true_s[t];
            se += d * d;
            ++cnt;
        }
        const double rmse = cnt ? std::sqrt(se / static_cast<double>(cnt)) : 0.0;
        std::printf("  %-16s store=%8zu B   RMSE=%.4f\n",
                    series[e].name.c_str(), ests[e]->store_bytes(), rmse);
    }
    std::printf("wrote %s\n", path.c_str());
    return 0;
}
