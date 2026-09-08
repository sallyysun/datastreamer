// Multi-seed replication with error bars.
//
// Same keys, scenarios, N and burn-in as main.cpp, so the numbers are directly
// comparable to a single-seed ./experiment run -- but repeated across seeds and
// reported as mean +/- standard error. Writes results/multiseed_<scenario>.csv.
//
// Usage: ./multiseed <scenario> [n_seeds] [base_seed]   (defaults: 10, 42)
//   scenario in {constant, drift, jump, stochastic, oscillate}
//   COPULA=gaussian|clayton|gumbel|t selects the DGP family (default gaussian).

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
    if (scenario == "oscillate") return std::make_unique<OscillatingDependence>(0.6, 400.0, 0.0);
    return nullptr;
}

struct SeedResult {
    std::vector<double> rmse;
    std::vector<double> alarm_count;
    std::vector<double> false_before;   // jump scenario only
    std::vector<long> detection_delay;  // -1 = never detected
};

SeedResult run_one_seed(const std::string& scenario, const std::string& family,
                         const std::vector<std::string>& keys,
                         std::size_t N, std::size_t burn, unsigned seed) {
    auto dependence = make_dependence(scenario, N, seed);
    auto copula = dgp::make_copula(family);
    CopulaSource source(*dependence, *copula, seed);

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
    if (!check_registry()) return 1;

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

    // dgp::make_copula falls back to Gaussian on an unrecognised name, which would
    // silently mislabel a whole grid, so reject anything unknown up front.
    const char* copula_env = std::getenv("COPULA");
    const std::string family = copula_env ? copula_env : "gaussian";
    if (family != "gaussian" && family != "clayton" && family != "gumbel" &&
        family != "t" && family != "student_t") {
        std::fprintf(stderr, "unknown COPULA '%s' (use gaussian|clayton|gumbel|t)\n", family.c_str());
        return 1;
    }
    const std::string family_name = dgp::make_copula(family)->name();

    const std::vector<std::string>& keys = all_estimator_keys();
    std::vector<bool> is_detector(keys.size(), false);
    for (std::size_t e = 0; e < keys.size(); ++e) {
        auto probe = make_estimator(keys[e]);
        is_detector[e] = dynamic_cast<ChangeDetector*>(probe.get()) != nullptr;
    }

    std::vector<std::vector<double>> rmse(keys.size());
    std::vector<std::vector<double>> alarms(keys.size());
    std::vector<std::vector<double>> false_before(keys.size());
    std::vector<std::vector<double>> delays(keys.size());
    std::vector<int> misses(keys.size(), 0);

    const bool has_known_jump = (scenario == "jump");

    for (int s = 0; s < n_seeds; ++s) {
        const unsigned seed = base_seed + static_cast<unsigned>(s);
        SeedResult r = run_one_seed(scenario, family, keys, N, burn, seed);
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

    std::printf("scenario=%s  copula=%s  n_seeds=%d  base_seed=%u  N=%zu  burn=%zu\n\n",
                scenario.c_str(), family_name.c_str(), n_seeds, base_seed, N, burn);
    // Distance from the best row, in pooled SEs. A gap under ~2 sigma is noise,
    // and reading point estimates without it is how the einfo ablation was
    // misread (studies/kalman_changepoint_results.md section 13.2).
    std::size_t best_e = 0;
    {
        double best = 1e300;
        for (std::size_t e = 0; e < keys.size(); ++e) {
            double m = 0.0;
            for (double v : rmse[e]) m += v;
            m /= static_cast<double>(rmse[e].size());
            if (m < best) { best = m; best_e = e; }
        }
    }

    std::printf("%-18s %16s %8s", "estimator", "RMSE (mean+-se)", "vs best");
    if (has_known_jump) std::printf("  %14s  %16s  %16s", "alarms/run", "false_before", "delay,ticks(hits)");
    else std::printf("  %14s", "alarms/run");
    std::printf("\n");

    const std::string suffix = (family_name == "gaussian") ? "" : ("_" + family_name);
    const std::string path = "results/multiseed_" + scenario + suffix + ".csv";
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
        std::printf("%-18s %8.4f +- %6.4f", keys[e].c_str(), rs.mean, rs.se);
        if (e == best_e) {
            std::printf(" %8s", "best");
        } else {
            const Stats bs = mean_stderr(rmse[best_e]);
            const double sd = std::sqrt(rs.se * rs.se + bs.se * bs.se);
            const double sigma = (sd > 0.0) ? (rs.mean - bs.mean) / sd : 0.0;
            std::printf(" %7.1f%s", sigma, sigma < 2.0 ? "~" : " ");
        }

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
