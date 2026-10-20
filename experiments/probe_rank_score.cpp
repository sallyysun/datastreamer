// What does the rank score-CUSUM's block statistic actually do? It is assumed
// mean 0, variance 1. Section 14.14.
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "../src/estimators/dependence/marginal_rank_pit.hpp"
#include "../src/estimators/registry.hpp"

int main() {
    const std::vector<std::string> inners = {"xiao_ew", "hermite_ew"};
    const std::vector<double> rhos = {0.0, 0.3, 0.6, 0.8};
    const std::size_t N = 300000, burn = 30000, block = 20;

    std::printf("%-12s %6s %10s %10s %12s %12s\n",
                "inner", "rho", "mean[e]", "var[e]", "mean[s]", "var[s]");
    for (const std::string& key : inners) {
        for (double rho : rhos) {
            auto est = make_estimator(key);
            RankPit px(500), py(500);
            std::mt19937 rng(3);
            std::normal_distribution<double> z(0.0, 1.0);

            double sum = 0.0, var = 1.0;
            std::size_t in_block = 0, n = 0, ns = 0;
            double e1 = 0, e2 = 0, s1 = 0, s2 = 0;
            for (std::size_t t = 0; t < N; ++t) {
                const double zx = z(rng);
                const double zy = rho * zx + std::sqrt(1.0 - rho * rho) * z(rng);
                const double rho_hat = est->estimate();
                est->update(zx, zy);
                const double u = px.transform(zx), v = py.transform(zy);
                if (px.size() < 30) continue;
                const double s = 12.0 * (u - 0.5) * (v - 0.5) - rho_hat;
                var += 0.001 * (s * s - var);
                if (t > burn) { s1 += s; s2 += s * s; ++ns; }
                sum += s / std::sqrt(std::max(var, 1e-9));
                if (++in_block < block) continue;
                const double e = sum / std::sqrt(double(block));
                sum = 0.0; in_block = 0;
                if (t > burn) { e1 += e; e2 += e * e; ++n; }
            }
            const double em = e1 / n, ev = e2 / n - em * em;
            const double sm = s1 / ns, sv = s2 / ns - sm * sm;
            std::printf("%-12s %6.1f %10.3f %10.3f %12.4f %12.4f\n",
                        key.c_str(), rho, em, ev, sm, sv);
        }
    }
    std::printf("\ne is the block statistic (assumed mean 0, var 1); s is the per-tick residual.\n");
    return 0;
}
