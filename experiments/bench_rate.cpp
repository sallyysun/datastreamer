// Throughput of update() against estimate(), per estimator. See notes/studies/rate_budget.md.
//
// The two are timed separately because they have different complexity and only
// one of them has to run on every tick.

#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "../src/estimators/registry.hpp"

namespace {

std::vector<std::pair<double, double>> make_stream(std::size_t n, double rho, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> z(0.0, 1.0);
    std::vector<std::pair<double, double>> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double a = z(rng), b = z(rng);
        out.emplace_back(a, rho * a + std::sqrt(1.0 - rho * rho) * b);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t N = (argc > 1) ? std::stoul(argv[1]) : 200000;
    const auto stream = make_stream(N, 0.6, 42);

    std::printf("%-22s %12s %12s %14s %14s\n", "estimator", "ns/update", "ns/estimate",
                "upd ticks/s", "est+upd tks/s");
    std::printf("%-22s %12s %12s %14s %14s\n", "----------------------", "------------",
                "------------", "--------------", "--------------");

    for (const auto& key : all_estimator_keys()) {
        auto est = make_estimator(key);
        if (!est) continue;

        // Warm the estimator past any calibration/burn-in phase first, so the
        // timed section measures the steady-state path only.
        const std::size_t warm = std::min<std::size_t>(N / 4, 5000);
        for (std::size_t i = 0; i < warm; ++i) est->update(stream[i].first, stream[i].second);

        auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = warm; i < N; ++i) est->update(stream[i].first, stream[i].second);
        auto t1 = std::chrono::steady_clock::now();

        const std::size_t reps = 20000;
        volatile double sink = 0.0;
        auto t2 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < reps; ++i) sink = sink + est->estimate();
        auto t3 = std::chrono::steady_clock::now();

        const double upd_ns =
            std::chrono::duration<double, std::nano>(t1 - t0).count() / double(N - warm);
        const double est_ns = std::chrono::duration<double, std::nano>(t3 - t2).count() / double(reps);

        std::printf("%-22s %12.1f %12.1f %14.0f %14.0f\n", key.c_str(), upd_ns, est_ns,
                    1e9 / upd_ns, 1e9 / (upd_ns + est_ns));
    }
    return 0;
}
