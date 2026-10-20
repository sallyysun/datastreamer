// Contract checks over every key in all_estimator_keys():
//   1. reset() restores a fresh estimator: after reset, the estimate sequence on a
//      new stream matches a freshly constructed copy exactly. Reset is tried at
//      several points -- inside rank-PIT calibration and burn-in, mid-block for the
//      block estimators, and after a long run. Keys whose reset is a deliberate soft
//      response or no-op are listed below with the reason.
//   2. store_bytes() does not grow with stream length (bounded memory).
//   3. estimate() is finite and within [-1, 1], for both the reset and the fresh
//      copy, on positive and negative dependence.
//   4. The Kendall side channels (windowed_rank, Hermite) are finite, in [-1, 1],
//      and have the sign of the dependence.
//
// Usage (from experiments/): make check-estimators

#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../src/estimators/dependence/hermiter.hpp"
#include "../src/estimators/dependence/windowed_rank.hpp"
#include "../src/estimators/registry.hpp"
#include "../src/streamgenerator/datasource/copula.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"

namespace {

using Stream = std::vector<std::pair<double, double>>;

Stream draw(std::unique_ptr<Dependence> dep, unsigned seed, std::size_t n) {
    auto cop = dgp::make_copula("gaussian");
    CopulaSource src(*dep, *cop, seed);
    Stream s(n);
    for (auto& p : s) {
        const auto x = src.next();
        p = {x.x, x.y};
    }
    return s;
}

// reset() is not a restart for these, by design.
const std::map<std::string, const char*> kResetExempt = {
    {"kalman_cusum_noreset", "no-op control"},
    {"gaussian_mle_cusum_noreset", "no-op control"},
    {"gaussian_mle_cusum_rob_noreset", "no-op control"},
    {"hermite_cusum_noreset", "no-op control"},
    {"hermite_cusum2_noreset", "no-op control"},
    {"kalman_cusum_soft", "covariance inflation"},
    {"kalman_cusum_inf2", "covariance inflation"},
    {"kalman_cusum_inf5", "covariance inflation"},
    {"kalman_cusum_inf10", "covariance inflation"},
    {"kalman_cusum_inf50", "covariance inflation"},
    {"kalman_periodic_inf", "covariance inflation"},
    {"gaussian_mle_cusum_soft", "information deflation"},
    {"gaussian_mle_cusum_rob", "information deflation"},
    {"gaussian_mle_periodic_defl", "information deflation"},
    {"hermite_cusum_soft", "forgetting boost"},
    {"particle", "RNG continues across reset (contract undecided)"},
    {"particle_fast", "RNG continues across reset (contract undecided)"},
};

bool bad(double e) { return !std::isfinite(e) || std::fabs(e) > 1.0; }

// Prefix lengths fed before reset(): 1, mid rank-PIT calibration (window 30),
// mid 20-tick block, and a long run that ends mid-block.
const std::size_t kResetAt[] = {1, 7, 29, 47, 3007};

template <class E>
int check_kendall(const char* label, E est, const Stream& s, double sign) {
    double last = 0.0;
    for (const auto& [x, y] : s) {
        est.update(x, y);
        last = est.kendall();
        if (bad(last)) {
            std::printf("FAIL %-30s kendall non-finite or outside [-1, 1] (%g)\n", label, last);
            return 1;
        }
    }
    if (last * sign <= 0.0) {
        std::printf("FAIL %-30s kendall %g has the wrong sign\n", label, last);
        return 1;
    }
    return 0;
}

}  // namespace

int main() {
    const Stream before = draw(std::make_unique<JumpDependence>(0.2, 0.8, 1500), 1, 3000);
    const Stream after = draw(std::make_unique<ConstantDependence>(0.6), 2, 3000);
    const Stream longrun = draw(std::make_unique<ConstantDependence>(0.6), 3, 40000);
    const Stream negative = draw(std::make_unique<ConstantDependence>(-0.7), 4, 3000);

    int failures = 0;
    for (const auto& key : all_estimator_keys()) {
        std::vector<std::string> errs;
        auto used = make_estimator(key);
        auto fresh = make_estimator(key);
        if (!used || !fresh) {
            std::printf("FAIL %-30s make_estimator returned null\n", key.c_str());
            ++failures;
            continue;
        }

        const bool exempt = kResetExempt.count(key) > 0;
        bool bad_value = false;
        for (const std::size_t at : kResetAt) {
            used = make_estimator(key);
            fresh = make_estimator(key);
            for (std::size_t i = 0; i < at; ++i) used->update(before[i].first, before[i].second);
            used->reset();
            double max_diff = 0.0;
            bool nan_mismatch = false;
            for (const auto& [x, y] : after) {
                used->update(x, y);
                fresh->update(x, y);
                const double e_fresh = fresh->estimate();
                const double e_used = used->estimate();
                if (bad(e_fresh) || bad(e_used)) bad_value = true;
                // std::max drops a NaN difference, so track a one-sided NaN explicitly.
                if (std::isnan(e_fresh) != std::isnan(e_used)) nan_mismatch = true;
                else if (!std::isnan(e_fresh)) max_diff = std::max(max_diff, std::fabs(e_used - e_fresh));
            }
            if (!exempt && (max_diff > 0.0 || nan_mismatch)) {
                char buf[128];
                std::snprintf(buf, sizeof buf, "reset() after %zu obs differs from fresh (max |diff| %.3g%s)",
                              at, max_diff, nan_mismatch ? ", NaN on one side" : "");
                errs.emplace_back(buf);
            }
        }
        auto neg = make_estimator(key);
        for (const auto& [x, y] : negative) {
            neg->update(x, y);
            if (bad(neg->estimate())) bad_value = true;
        }
        if (bad_value) errs.emplace_back("estimate non-finite or outside [-1, 1]");

        auto grow = make_estimator(key);
        std::size_t bytes_early = 0;
        for (std::size_t i = 0; i < longrun.size(); ++i) {
            grow->update(longrun[i].first, longrun[i].second);
            if (i == 5000) bytes_early = grow->store_bytes();
        }
        if (grow->store_bytes() > bytes_early) {
            char buf[96];
            std::snprintf(buf, sizeof buf, "store_bytes grows %zu -> %zu", bytes_early, grow->store_bytes());
            errs.emplace_back(buf);
        }

        if (!errs.empty()) {
            ++failures;
            std::printf("FAIL %-30s", key.c_str());
            for (const auto& e : errs) std::printf(" | %s", e.c_str());
            std::printf("\n");
        }
    }
    failures += check_kendall("windowed_rank::kendall +", WindowedRankEstimator(256), after, 1.0);
    failures += check_kendall("windowed_rank::kendall -", WindowedRankEstimator(256), negative, -1.0);
    failures += check_kendall("hermite::kendall +", HermiteEstimator({}, "hermite"), after, 1.0);
    failures += check_kendall("hermite::kendall -", HermiteEstimator({}, "hermite"), negative, -1.0);

    std::printf("%zu keys checked, %zu reset-exempt, %d failing\n",
                all_estimator_keys().size(), kResetExempt.size(), failures);
    return failures == 0 ? 0 : 1;
}
