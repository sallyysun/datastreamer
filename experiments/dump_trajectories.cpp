// Bit-exact refactor gate: for every scenario x seed x registered key, hashes the
// input stream and the full per-tick estimate (and alarm) trajectory. Two builds
// produce the same manifest iff every key computes the same bits on the same
// inputs -- stronger than comparing multiseed's rounded summary CSVs.
// Streams match multiseed.cpp (N = 5000, same scenarios and seeding).
//
// Usage (from experiments/):
//   make dump_trajectories && ./dump_trajectories 3 42 > before.txt
//   ...change...
//   make dump_trajectories && ./dump_trajectories 3 42 > after.txt && diff before.txt after.txt
// COPULA=<family> selects the DGP family as in multiseed.
// Probes that build configs outside the registry: TRAJ_MANIFEST=<file> ./probe_memory (see check_probe_trajectories.sh).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

struct Fnv {
    std::uint64_t h = 1469598103934665603ull;
    void add(const void* p, std::size_t n) {
        const auto* b = static_cast<const unsigned char*>(p);
        for (std::size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    }
    void add(double v) { add(&v, sizeof v); }
};

}  // namespace

int main(int argc, char** argv) {
    const int n_seeds = (argc > 1) ? std::atoi(argv[1]) : 3;
    const unsigned base_seed = (argc > 2) ? static_cast<unsigned>(std::atoi(argv[2])) : 42;
    const char* copula_env = std::getenv("COPULA");
    const std::string family = copula_env ? copula_env : "gaussian";
    const std::size_t N = 5000;
    const auto& keys = all_estimator_keys();

    std::printf("# family=%s seeds=%d base_seed=%u N=%zu keys=%zu\n", family.c_str(), n_seeds, base_seed, N,
                keys.size());
    for (const char* scenario : {"constant", "drift", "jump", "stochastic", "oscillate"}) {
        for (int s = 0; s < n_seeds; ++s) {
            const unsigned seed = base_seed + static_cast<unsigned>(s);
            auto dependence = make_dependence(scenario, N, seed);
            auto copula = dgp::make_copula(family);
            CopulaSource source(*dependence, *copula, seed);

            std::vector<std::unique_ptr<DependenceEstimator>> ests;
            for (const auto& k : keys) ests.push_back(make_estimator(k));
            std::vector<Fnv> out(keys.size());
            Fnv in;

            for (std::size_t t = 0; t < N; ++t) {
                const CopulaSource::Sample smp = source.next();
                in.add(smp.x);
                in.add(smp.y);
                in.add(smp.rho_s);
                for (std::size_t e = 0; e < ests.size(); ++e) {
                    ests[e]->update(smp.x, smp.y);
                    out[e].add(ests[e]->estimate());
                    if (auto* d = dynamic_cast<ChangeDetector*>(ests[e].get())) {
                        const unsigned char a = d->alarmed_this_block() ? 1 : 0;
                        out[e].add(&a, 1);
                    }
                }
            }
            for (std::size_t e = 0; e < keys.size(); ++e) {
                std::printf("%s %u %-32s in=%016llx out=%016llx last=%.17g bytes=%zu\n", scenario, seed,
                            keys[e].c_str(), static_cast<unsigned long long>(in.h),
                            static_cast<unsigned long long>(out[e].h), ests[e]->estimate(),
                            ests[e]->store_bytes());
            }
        }
    }
    return 0;
}
