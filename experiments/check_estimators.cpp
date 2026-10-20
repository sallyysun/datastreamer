// Contract checks over every key in all_estimator_keys():
//   1. reset() restores a fresh estimator: after reset, the estimate sequence on a
//      new stream matches a freshly constructed copy exactly. Keys whose reset is a
//      deliberate soft response or no-op are listed below with the reason.
//   2. store_bytes() does not grow with stream length (bounded memory).
//   3. estimate() is finite and within [-1, 1].
//
// Usage (from experiments/): make check-estimators

#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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

}  // namespace

int main() {
    const Stream before = draw(std::make_unique<JumpDependence>(0.2, 0.8, 1500), 1, 3000);
    const Stream after = draw(std::make_unique<ConstantDependence>(0.6), 2, 3000);
    const Stream longrun = draw(std::make_unique<ConstantDependence>(0.6), 3, 40000);

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

        for (const auto& [x, y] : before) used->update(x, y);
        used->reset();
        double max_diff = 0.0;
        bool bad_value = false;
        for (const auto& [x, y] : after) {
            used->update(x, y);
            fresh->update(x, y);
            const double e = fresh->estimate();
            if (!std::isfinite(e) || std::fabs(e) > 1.0) bad_value = true;
            max_diff = std::max(max_diff, std::fabs(used->estimate() - e));
        }
        const bool exempt = kResetExempt.count(key) > 0;
        if (!exempt && max_diff > 0.0) {
            char buf[96];
            std::snprintf(buf, sizeof buf, "reset() differs from fresh (max |diff| %.3g)", max_diff);
            errs.emplace_back(buf);
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
    std::printf("%zu keys checked, %zu reset-exempt, %d failing\n",
                all_estimator_keys().size(), kResetExempt.size(), failures);
    return failures == 0 ? 0 : 1;
}
