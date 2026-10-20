// Local estimator noise v_j(rho) for the async memory study. See notes/studies/async_memory.md.
//
// Constant-rho synchronous Gaussian-copula streams; for each memory-grid estimator at ESS m,
// v_j(rho) = m * Var(estimate) at steady state (pooled over time and seeds), i.e. the variance
// of the estimator's effective one-step signal in the heuristic MSE = v/(fM) + (dM/2)^2.
// Two m values check the 1/m scaling the heuristic assumes.
//
// Usage: ./async_noise [n_seeds] [base_seed]   (defaults: 20, 42)
// Writes results/async_noise.csv.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <numbers>
#include <random>
#include <vector>

#include "../src/streamgenerator/datasource/copula.hpp"
#include "memory_grid.hpp"
#include "trajectory_manifest.hpp"

namespace {

constexpr std::size_t T = 20000;
constexpr std::size_t BURN = 3000;
constexpr std::size_t EVAL = 10;
const std::vector<double> kMemories = {80, 320};

std::vector<double> rho_grid() {
    std::vector<double> g;
    for (int i = -8; i <= 8; ++i) g.push_back(0.1 * i);
    return g;
}

// out[(r * families + f) * memories + k] = estimate samples for one seed
std::vector<std::vector<double>> run_seed(unsigned seed) {
    const auto rhos = rho_grid();
    const auto& fams = memgrid::kFamilies;
    std::vector<std::vector<double>> out(rhos.size() * fams.size() * kMemories.size());
    auto copula = dgp::make_copula("gaussian");
    for (std::size_t r = 0; r < rhos.size(); ++r) {
        const double d = 2.0 * std::sin(std::numbers::pi * rhos[r] / 6.0);
        std::mt19937 rng(seed * 101u + static_cast<unsigned>(r));
        std::vector<std::unique_ptr<DependenceEstimator>> ests;
        for (const auto& f : fams)
            for (double m : kMemories) ests.push_back(memgrid::make_at_memory(f, m));
        for (std::size_t t = 0; t < T; ++t) {
            const dgp::CopulaDraw draw = copula->sample(d, rng);
            for (auto& e : ests) e->update(draw.z1, draw.z2);
            if (t >= BURN && t % EVAL == 0)
                for (std::size_t i = 0; i < ests.size(); ++i)
                    out[r * ests.size() + i].push_back(ests[i]->estimate());
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const int n_seeds = (argc > 1) ? std::atoi(argv[1]) : 20;
    const unsigned base = (argc > 2) ? static_cast<unsigned>(std::atoi(argv[2])) : 42;

    std::vector<std::future<std::vector<std::vector<double>>>> jobs;
    for (int s = 0; s < n_seeds; ++s) jobs.push_back(std::async(std::launch::async, run_seed, base + static_cast<unsigned>(s)));
    std::vector<std::vector<std::vector<double>>> seeds;
    for (auto& j : jobs) seeds.push_back(j.get());
    if (traj::on())
        for (int s = 0; s < n_seeds; ++s)
            for (std::size_t i = 0; i < seeds[s].size(); ++i)
                traj::record(traj::tag("async_noise/%u/cell%03zu", base + static_cast<unsigned>(s), i), seeds[s][i]);
    std::vector<double> summary;

    std::ofstream csv("results/async_noise.csv");
    if (!csv.is_open()) {
        std::fprintf(stderr, "cannot open results/async_noise.csv (run: mkdir -p results)\n");
        return 1;
    }
    csv << "estimator,m,rho_s,mean,var,v\n";
    const auto rhos = rho_grid();
    const auto& fams = memgrid::kFamilies;
    std::printf("v_j(rho) = m * Var(estimate), n_seeds=%d\n%-16s %5s", n_seeds, "estimator", "m");
    for (double r : rhos) std::printf(" %6.1f", r);
    std::printf("\n");
    for (std::size_t f = 0; f < fams.size(); ++f)
        for (std::size_t k = 0; k < kMemories.size(); ++k) {
            std::printf("%-16s %5.0f", fams[f].c_str(), kMemories[k]);
            for (std::size_t r = 0; r < rhos.size(); ++r) {
                const std::size_t i = (r * fams.size() + f) * kMemories.size() + k;
                double sum = 0.0, n = 0.0;
                for (const auto& s : seeds) for (double x : s[i]) { sum += x; n += 1.0; }
                const double mean = sum / n;
                double ss = 0.0;
                for (const auto& s : seeds) for (double x : s[i]) ss += (x - mean) * (x - mean);
                const double var = ss / (n - 1.0);
                const double v = kMemories[k] * var;
                summary.insert(summary.end(), {mean, var, v});
                std::printf(" %6.3f", v);
                csv << fams[f] << "," << kMemories[k] << "," << rhos[r] << "," << mean << "," << var << "," << v << "\n";
            }
            std::printf("\n");
        }
    traj::record("async_noise/summary", summary);
    std::printf("wrote results/async_noise.csv\n");
    return 0;
}
