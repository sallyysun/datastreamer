// Does rank-PIT protect Xiao's binned grid from the marginals? Mirrors the
// hermite_pit test of section 9.2 so the two are comparable, adds a mid-stream
// monotone shift (the case Xiao's frozen cutpoints cannot survive), and prices
// the bin count. Section 14.19 of notes/studies/kalman_changepoint_results.md.
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "../src/estimators/registry.hpp"

namespace {

const std::vector<std::string> kKeys = {"xiao_ew", "xiao_pit", "hermite_ew", "hermite_pit"};

void static_transform_test() {
    const std::size_t N = 20000;
    const int seeds = 10;
    const double rho = 0.6;
    const double truth = 6.0 / M_PI * std::asin(rho / 2.0);

    struct Case { const char* name; double (*f)(double); };
    const Case cases[] = {
        {"identity     ", [](double z) { return z; }},
        {"exp (skewed) ", [](double z) { return std::exp(z); }},
        {"cube (heavy) ", [](double z) { return z * z * z; }},
    };

    std::printf("== A. static monotone transform of x (copula identical in every row) ==\n");
    std::printf("true rho_S = %.4f, %d seeds, N=%zu, mean |rho-hat - rho_S|\n\n", truth, seeds, N);
    std::printf("%-14s", "marginal x ->");
    for (const auto& k : kKeys) std::printf("%13s", k.c_str());
    std::printf("\n");

    for (const auto& c : cases) {
        std::vector<double> err(kKeys.size(), 0.0);
        for (int s = 0; s < seeds; ++s) {
            std::mt19937_64 rng(1000 + s);
            std::normal_distribution<double> g;
            std::vector<std::unique_ptr<DependenceEstimator>> ests;
            for (const auto& k : kKeys) ests.push_back(make_estimator(k));
            for (std::size_t i = 0; i < N; ++i) {
                const double z1 = g(rng), z2 = g(rng);
                const double y = rho * z1 + std::sqrt(1 - rho * rho) * z2;
                for (auto& e : ests) e->update(c.f(z1), y);
            }
            for (std::size_t e = 0; e < ests.size(); ++e) {
                err[e] += std::abs(ests[e]->estimate() - truth);
            }
        }
        std::printf("%-14s", c.name);
        for (double v : err) std::printf("%13.4f", v / seeds);
        std::printf("\n");
    }
    std::printf("\n");
}

void midstream_shift_test() {
    const std::size_t N = 5000;
    const int seeds = 20;
    const double rho = 0.6;
    const double truth = 6.0 / M_PI * std::asin(rho / 2.0);
    const std::size_t shift_t = N / 2;
    const std::size_t score_from = shift_t + 500;

    std::printf("== B. monotone marginal shift at t=%zu (x -> 5x+100, y -> 0.1y-50) ==\n", shift_t);
    std::printf("copula held exactly constant; mean |rho-hat - rho_S| over t in [%zu,%zu)\n\n",
                score_from, N);
    std::printf("%-14s", "");
    for (const auto& k : kKeys) std::printf("%13s", k.c_str());
    std::printf("\n");

    std::vector<double> pre(kKeys.size(), 0.0), post(kKeys.size(), 0.0);
    for (int s = 0; s < seeds; ++s) {
        std::mt19937_64 rng(2000 + s);
        std::normal_distribution<double> g;
        std::vector<std::unique_ptr<DependenceEstimator>> ests;
        for (const auto& k : kKeys) ests.push_back(make_estimator(k));
        std::vector<double> acc_pre(kKeys.size(), 0.0), acc_post(kKeys.size(), 0.0);
        std::size_t n_pre = 0, n_post = 0;
        for (std::size_t i = 0; i < N; ++i) {
            const double z1 = g(rng), z2 = g(rng);
            double x = z1;
            double y = rho * z1 + std::sqrt(1 - rho * rho) * z2;
            if (i >= shift_t) { x = 5.0 * x + 100.0; y = 0.1 * y - 50.0; }
            for (std::size_t e = 0; e < ests.size(); ++e) {
                ests[e]->update(x, y);
                const double err = std::abs(ests[e]->estimate() - truth);
                if (i >= 1000 && i < shift_t) acc_pre[e] += err;
                if (i >= score_from) acc_post[e] += err;
            }
            if (i >= 1000 && i < shift_t) ++n_pre;
            if (i >= score_from) ++n_post;
        }
        for (std::size_t e = 0; e < ests.size(); ++e) {
            pre[e] += acc_pre[e] / n_pre;
            post[e] += acc_post[e] / n_post;
        }
    }
    std::printf("%-14s", "before shift ");
    for (double v : pre) std::printf("%13.4f", v / seeds);
    std::printf("\n%-14s", "after shift  ");
    for (double v : post) std::printf("%13.4f", v / seeds);
    std::printf("\n\n");
}

void store_bytes_test() {
    const std::vector<std::string> keys = {"xiao_b8", "xiao", "xiao_b40", "xiao_ew", "xiao_pit"};
    const std::size_t N = 3000;
    std::mt19937_64 rng(7);
    std::normal_distribution<double> g;
    std::vector<std::unique_ptr<DependenceEstimator>> ests;
    for (const auto& k : keys) ests.push_back(make_estimator(k));
    for (std::size_t i = 0; i < N; ++i) {
        const double z1 = g(rng), z2 = g(rng);
        const double y = 0.6 * z1 + std::sqrt(1 - 0.36) * z2;
        for (auto& e : ests) e->update(z1, y);
    }
    std::printf("== C. store_bytes after %zu updates ==\n\n", N);
    std::printf("%-10s %12s %12s\n", "key", "bytes", "vs xiao");
    const double base = static_cast<double>(ests[1]->store_bytes());
    for (std::size_t e = 0; e < keys.size(); ++e) {
        const std::size_t b = ests[e]->store_bytes();
        std::printf("%-10s %12zu %11.2fx\n", keys[e].c_str(), b, b / base);
    }
    std::printf("\n");
}

}  // namespace

int main() {
    static_transform_test();
    midstream_shift_test();
    store_bytes_test();
    return 0;
}
