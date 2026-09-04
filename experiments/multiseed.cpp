// Multi-seed replication with error bars.
//
// notes/todo.md has carried "no number is a thesis result until multi-seed
// with error bars" as a standing caveat since before this Kalman/CUSUM/GLR/
// BOCPD line of work started, and every RMSE and every detector statistic
// produced by ./experiment and calibrate_arl0.cpp so far is a single seed=42
// point estimate. This is that replication: same keys, same scenarios, same
// N and burn-in as experiment.cpp (so the numbers are directly comparable to
// what is already written up), run across several seeds, reported as
// mean +/- standard error.
//
// Usage: ./multiseed <scenario> [n_seeds] [base_seed]   (defaults: 10, 42)
//   scenario in {constant, drift, jump, stochastic, oscillate}

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "../src/estimators/dependence/change_detector.hpp"
#include "../src/estimators/registry.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"

namespace {

std::unique_ptr<Dependence> make_dependence(const std::string& scenario, std::size_t N, unsigned seed) {
    if (scenario == "constant") return std::make_unique<ConstantDependence>(0.6);
    if (scenario == "drift") return std::make_unique<DriftDependence>(-0.5, 0.8, N);
    if (scenario == "jump") return std::make_unique<JumpDependence>(0.2, 0.8, N / 2);
    if (scenario == "stochastic") return std::make_unique<StochasticDependence>(0.01, 0.6, seed + 1);
    // period 400 matches the live server so offline vs online oscillate RMSE
    // is directly comparable (same convention as experiment.cpp).
    if (scenario == "oscillate") return std::make_unique<OscillatingDependence>(0.6, 400.0, 0.0);
    return nullptr;
}

struct SeedResult {
    std::vector<double> rmse;              // per estimator
    std::vector<double> alarm_count;        // per estimator; 0 for non-detectors
    std::vector<double> false_before;       // per estimator; jump scenario only, else 0
    std::vector<long> detection_delay;      // per estimator; -1 = not a detector or never detected
};

SeedResult run_one_seed(const std::string& scenario, const std::vector<std::string>& keys,
                         std::size_t N, std::size_t burn, unsigned seed) {
    auto dependence = make_dependence(scenario, N, seed);
    CopulaSource source(*dependence, seed);

    std::vector<std::unique_ptr<DependenceEstimator>> ests;
    for (const auto& k : keys) ests.push_back(make_estimator(k));

    std::vector<ChangeDetector*> detectors(keys.size(), nullptr);
    for (std::size_t e = 0; e < keys.size(); ++e) detectors[e] = dynamic_cast<ChangeDetector*>(ests[e].get());

    std::vector<std::vector<double>> est_series(keys.size());
    std::vector<std::vector<int>> alarm_series(keys.size());
    std::vector<double> true_s(N);

    for (std::size_t t = 0; t < N; ++t) {
        const CopulaSource::Sample s = source.next();
        true_s[t] = s.rho_s;
        for (std::size_t e = 0; e < ests.size(); ++e) {
            ests[e]->update(s.x, s.y);
            est_series[e].push_back(ests[e]->estimate());
            if (detectors[e]) alarm_series[e].push_back(detectors[e]->alarmed_this_block() ? 1 : 0);
        }
    }

    SeedResult r;
    r.rmse.assign(keys.size(), 0.0);
    r.alarm_count.assign(keys.size(), 0.0);
    r.false_before.assign(keys.size(), 0.0);
    r.detection_delay.assign(keys.size(), -1);

    const bool has_known_jump = (scenario == "jump");
    const std::size_t jump_t = N / 2;

    for (std::size_t e = 0; e < keys.size(); ++e) {
        double se = 0.0;
        std::size_t cnt = 0;
        for (std::size_t t = burn; t < N; ++t) {
            const double d = est_series[e][t] - true_s[t];
            se += d * d;
            ++cnt;
        }
        r.rmse[e] = cnt ? std::sqrt(se / static_cast<double>(cnt)) : 0.0;

        if (detectors[e]) {
            long first_after = -1;
            std::size_t total = 0, before = 0;
            for (std::size_t t = 0; t < N; ++t) {
                if (!alarm_series[e][t]) continue;
                ++total;
                if (has_known_jump) {
                    if (t < jump_t) ++before;
                    else if (first_after < 0) first_after = static_cast<long>(t);
                }
            }
            r.alarm_count[e] = static_cast<double>(total);
            r.false_before[e] = static_cast<double>(before);
            if (has_known_jump && first_after >= 0) {
                r.detection_delay[e] = first_after - static_cast<long>(jump_t);
            }
        }
    }
    return r;
}

struct Stats { double mean = 0.0, se = 0.0; std::size_t n = 0; };

Stats mean_stderr(const std::vector<double>& v) {
    Stats s;
    s.n = v.size();
    if (s.n == 0) return s;
    double sum = 0.0;
    for (double x : v) sum += x;
    s.mean = sum / static_cast<double>(s.n);
    if (s.n < 2) return s;
    double var = 0.0;
    for (double x : v) var += (x - s.mean) * (x - s.mean);
    var /= static_cast<double>(s.n - 1);
    s.se = std::sqrt(var / static_cast<double>(s.n));
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string scenario = (argc > 1) ? argv[1] : "jump";
    const int n_seeds = (argc > 2) ? std::atoi(argv[2]) : 10;
    const unsigned base_seed = (argc > 3) ? static_cast<unsigned>(std::atoi(argv[3])) : 42;
    const std::size_t N = 5000;
    const std::size_t burn = 500;

    if (!make_dependence(scenario, N, base_seed)) {
        std::fprintf(stderr, "unknown scenario '%s' (use constant|drift|jump|stochastic|oscillate)\n",
                     scenario.c_str());
        return 1;
    }

    const std::vector<std::string> keys = {"xiao", "xiao_ew", "hermite", "hermite_ew",
                                           "windowed_rank", "gaussian_mle", "gaussian_mle_ew",
                                           "kalman", "kalman_fast", "kalman_cusum", "kalman_glr",
                                           "kalman_bocpd"};
    // Which keys are detectors, determined once (independent of seed).
    std::vector<bool> is_detector(keys.size(), false);
    for (std::size_t e = 0; e < keys.size(); ++e) {
        auto probe = make_estimator(keys[e]);
        is_detector[e] = dynamic_cast<ChangeDetector*>(probe.get()) != nullptr;
    }

    std::vector<std::vector<double>> rmse(keys.size());
    std::vector<std::vector<double>> alarms(keys.size());
    std::vector<std::vector<double>> false_before(keys.size());
    std::vector<std::vector<double>> delays(keys.size());  // successful detections only
    std::vector<int> misses(keys.size(), 0);                // seeds where a known jump was never caught

    const bool has_known_jump = (scenario == "jump");

    for (int s = 0; s < n_seeds; ++s) {
        const unsigned seed = base_seed + static_cast<unsigned>(s);
        SeedResult r = run_one_seed(scenario, keys, N, burn, seed);
        for (std::size_t e = 0; e < keys.size(); ++e) {
            rmse[e].push_back(r.rmse[e]);
            if (is_detector[e]) {
                alarms[e].push_back(r.alarm_count[e]);
                if (has_known_jump) {
                    false_before[e].push_back(r.false_before[e]);
                    if (r.detection_delay[e] >= 0) delays[e].push_back(static_cast<double>(r.detection_delay[e]));
                    else ++misses[e];
                }
            }
        }
    }

    std::printf("scenario=%s  n_seeds=%d  base_seed=%u  N=%zu  burn=%zu\n\n",
                scenario.c_str(), n_seeds, base_seed, N, burn);
    std::printf("%-16s %16s", "estimator", "RMSE (mean+-se)");
    if (has_known_jump) std::printf("  %14s  %16s  %16s", "alarms/run", "false_before", "delay,ticks(hits)");
    else std::printf("  %14s", "alarms/run");
    std::printf("\n");

    const std::string path = "results/multiseed_" + scenario + ".csv";
    std::ofstream out(path);
    if (!out.is_open()) {
        std::fprintf(stderr, "cannot open '%s' for writing (does results/ exist? run: mkdir -p results)\n",
                     path.c_str());
        return 1;
    }
    out << "estimator,rmse_mean,rmse_se,n_seeds";
    if (has_known_jump) out << ",alarms_mean,alarms_se,false_before_mean,false_before_se,delay_mean,delay_se,delay_n_hits,misses";
    else out << ",alarms_mean,alarms_se";
    out << "\n";

    for (std::size_t e = 0; e < keys.size(); ++e) {
        const Stats rs = mean_stderr(rmse[e]);
        std::printf("%-16s %8.4f +- %6.4f", keys[e].c_str(), rs.mean, rs.se);

        double a_mean = 0.0, a_se = 0.0, fb_mean = 0.0, fb_se = 0.0, d_mean = 0.0, d_se = 0.0;
        std::size_t d_n = 0;
        if (is_detector[e]) {
            const Stats as = mean_stderr(alarms[e]);
            a_mean = as.mean; a_se = as.se;
            std::printf("  %6.2f +- %5.2f", a_mean, a_se);
            if (has_known_jump) {
                const Stats fs = mean_stderr(false_before[e]);
                fb_mean = fs.mean; fb_se = fs.se;
                const Stats ds = mean_stderr(delays[e]);
                d_mean = ds.mean; d_se = ds.se; d_n = ds.n;
                std::printf("  %6.2f +- %5.2f", fb_mean, fb_se);
                if (d_n > 0) {
                    std::printf("  %6.1f +- %5.1f (%zu/%d)", d_mean, d_se, d_n, n_seeds);
                } else {
                    std::printf("  never detected (0/%d)", n_seeds);
                }
            }
        } else if (has_known_jump) {
            std::printf("  %14s  %16s  %16s", "-", "-", "-");
        } else {
            std::printf("  %14s", "-");
        }
        std::printf("\n");

        out << keys[e] << "," << rs.mean << "," << rs.se << "," << rs.n;
        if (has_known_jump) {
            out << "," << a_mean << "," << a_se << "," << fb_mean << "," << fb_se << ","
                << d_mean << "," << d_se << "," << d_n << "," << misses[e];
        } else {
            out << "," << a_mean << "," << a_se;
        }
        out << "\n";
    }
    std::printf("\nwrote %s\n", path.c_str());
    return 0;
}
