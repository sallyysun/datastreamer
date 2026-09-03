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

#include "../src/estimators/dependence/kalman_cusum.hpp"
#include "../src/estimators/registry.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"

namespace {

struct Series {
    std::string name;
    std::vector<double> est;
};

std::unique_ptr<Dependence> make_dependence(const std::string& scenario, std::size_t N, unsigned seed) {
    if (scenario == "constant") return std::make_unique<ConstantDependence>(0.6);
    if (scenario == "drift") return std::make_unique<DriftDependence>(-0.5, 0.8, N);
    if (scenario == "jump") return std::make_unique<JumpDependence>(0.2, 0.8, N / 2);
    if (scenario == "stochastic") return std::make_unique<StochasticDependence>(0.01, 0.6, seed + 1);
    // period 400 matches the live server (app/main.cpp) so offline vs online
    // oscillate RMSE is directly comparable.
    if (scenario == "oscillate") return std::make_unique<OscillatingDependence>(0.6, 400.0, 0.0);
    return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t N = 5000;
    const unsigned seed = 42;
    const std::size_t burn = 500;  // steps to skip before scoring

    const std::string scenario = (argc > 1) ? argv[1] : "jump";

    auto dependence = make_dependence(scenario, N, seed);
    if (!dependence) {
        std::fprintf(stderr,
                     "unknown scenario '%s' (use constant|drift|jump|stochastic|oscillate)\n",
                     scenario.c_str());
        return 1;
    }

    CopulaSource source(*dependence, seed);

    const std::vector<std::string> keys = {"xiao", "xiao_ew", "hermite", "hermite_ew",
                                           "windowed_rank", "gaussian_mle", "gaussian_mle_ew",
                                           "kalman", "kalman_fast", "kalman_cusum"};
    std::vector<std::unique_ptr<DependenceEstimator>> ests;
    for (const auto& k : keys) {
        auto e = make_estimator(k);
        if (!e) {
            std::fprintf(stderr, "unknown estimator '%s'\n", k.c_str());
            return 1;
        }
        ests.push_back(std::move(e));
    }

    // Non-owning CUSUM pointers so per-tick alarm state can be logged -- the
    // diagnostic a changepoint detector needs that estimate() alone can't give
    // (estimate() stays Spearman-only so every estimator is comparable in one
    // space, same convention as CopulaSelector::selected() on the other branch).
    std::vector<KalmanCusumDetector*> cusums(keys.size(), nullptr);
    for (std::size_t e = 0; e < keys.size(); ++e) {
        cusums[e] = dynamic_cast<KalmanCusumDetector*>(ests[e].get());
    }
    struct AlarmTrace {
        std::string column;
        std::size_t est_index;
        std::vector<int> vals;
    };
    std::vector<AlarmTrace> alarm_traces;
    for (std::size_t e = 0; e < keys.size(); ++e) {
        if (cusums[e]) alarm_traces.push_back({keys[e] + "_alarm", e, {}});
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
        for (auto& at : alarm_traces) at.vals.push_back(cusums[at.est_index]->alarmed_this_block() ? 1 : 0);
    }

    // Written relative to this directory (run the harness from experiments/).
    const std::string path = "results/experiment_" + scenario + ".csv";
    std::ofstream out(path);
    if (!out.is_open()) {
        std::fprintf(stderr, "cannot open '%s' for writing (does results/ exist? run: mkdir -p results)\n",
                     path.c_str());
        return 1;
    }
    out << "t,true_rho_s";
    for (const auto& k : keys) out << "," << k;
    for (const auto& at : alarm_traces) out << "," << at.column;
    out << "\n";
    for (std::size_t t = 0; t < N; ++t) {
        out << t << "," << true_s[t];
        for (const auto& sr : series) out << "," << sr.est[t];
        for (const auto& at : alarm_traces) out << "," << at.vals[t];
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

    // Changepoint report. `jump` has a single known changepoint at N/2, so
    // detection delay is well defined there; other scenarios only get a raw
    // alarm count (drift/stochastic have no single instant to measure delay
    // against, oscillate changes direction every period so "delay" would be
    // ambiguous by construction, not just unmeasured).
    if (!alarm_traces.empty()) {
        std::printf("\nchangepoint detection (%s):\n", scenario.c_str());
        const bool has_known_jump = (scenario == "jump");
        const std::size_t jump_t = N / 2;
        for (const auto& at : alarm_traces) {
            std::size_t total = 0, before_jump = 0;
            long first_after = -1;
            for (std::size_t t = 0; t < N; ++t) {
                if (!at.vals[t]) continue;
                ++total;
                if (has_known_jump) {
                    if (t < jump_t) ++before_jump;
                    else if (first_after < 0) first_after = static_cast<long>(t);
                }
            }
            std::printf("  %-16s alarms=%3zu", at.column.c_str(), total);
            if (has_known_jump) {
                std::printf("   false_alarms_before_jump=%zu", before_jump);
                if (first_after >= 0) {
                    std::printf("   detection_delay=%ld ticks", first_after - static_cast<long>(jump_t));
                } else {
                    std::printf("   detection_delay=never");
                }
            }
            std::printf("\n");
        }
    }
    return 0;
}
