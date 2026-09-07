// Does rank-PIT actually protect the copula estimate from skewed marginals?
// A strictly monotone transform of a marginal leaves the copula EXACTLY
// unchanged, so any change in rho-hat is pure marginal contamination.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "../src/estimators/registry.hpp"

int main() {
    const std::size_t N = 20000;
    const int seeds = 10;
    const double rho = 0.6;
    const double truth = 6.0 / M_PI * std::asin(rho / 2.0);   // Spearman for a Gaussian copula

    struct Case { const char* name; double (*f)(double); };
    Case cases[] = {
        {"identity      ", [](double z){ return z; }},
        {"exp  (skewed) ", [](double z){ return std::exp(z); }},
        {"cube (heavy)  ", [](double z){ return z * z * z; }},
    };

    std::printf("true rho_S = %.4f   (copula identical in every row)\n\n", truth);
    std::printf("%-16s %14s %14s\n", "marginal x ->", "hermite_ew", "hermite_pit");
    for (auto& c : cases) {
        double err[2] = {0, 0};
        for (int s = 0; s < seeds; ++s) {
            std::mt19937_64 rng(1000 + s);
            std::normal_distribution<double> g;
            auto a = make_estimator("hermite_ew");
            auto b = make_estimator("hermite_pit");
            for (std::size_t i = 0; i < N; ++i) {
                const double z1 = g(rng), z2 = g(rng);
                const double x = z1;
                const double y = rho * z1 + std::sqrt(1 - rho * rho) * z2;
                a->update(c.f(x), y);
                b->update(c.f(x), y);
            }
            err[0] += std::abs(a->estimate() - truth);
            err[1] += std::abs(b->estimate() - truth);
        }
        std::printf("%-16s %14.4f %14.4f\n", c.name, err[0] / seeds, err[1] / seeds);
    }
    std::printf("\n(mean |rho-hat - true rho|, %d seeds, N=%zu)\n", seeds, N);
}
