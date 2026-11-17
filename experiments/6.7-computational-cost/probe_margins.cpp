// Section 6.7 diagnostic: why marginal estimation costs RMPL more than Counting Matrix / Hermite Series under constant
// Gaussian dependence. Gaussian copula, constant dependence (as probe_budget's "constant"), n = 15,000, standard-normal margins,
// test seeds 42-141, cumulative estimation (no forgetting), RMSE of rho_S over ticks >= 500 every 5th (as probe_budget).
// Every variant sees the same stream. Margin treatments are applied by transforming the inputs offline:
//   oracle      : true margins (inputs are the true N(0,1) scores)
//   full_emp    : z = probit(rank among ALL n observations / (n + 1))       [uses the future: diagnostic only]
//   seq_emp     : z = probit(#{s < t : x_s < x_t} / (t - 1))                 [exactly the KLL margin's u when nothing is
//                 compressed; t = 1 gives u = 0.5]
//   seq_mid     : z = probit((#{s < t : x_s < x_t} + 0.5) / t)             [mid-rank: avoids u = 0, 1]
//   seq_emp_t0  : seq_emp, copula updates only from observation t0 (margins still accumulate from the start)
// RMPL is fed transformed scores through its ExactNormal margin (u = Phi(z) back again), or uses its own DataSketches KLL margins.
// Counting Matrix and Hermite are fed transformed scores with normal_score_inputs (fixed probit(k/B) cut-points; no standardisation).
// Output: results/margins_seeds.csv (seed, variant, rmse_all, rmse_late), late = ticks >= 10,000.
// Build: ../build.sh
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../../src/estimators/dependence/estimator.hpp"
#include "../../src/estimators/dependence/hermiter.hpp"
#include "../../src/estimators/dependence/marginal_rank_pit.hpp"
#include "../../src/estimators/dependence/recursive_mpl.hpp"
#include "../../src/estimators/dependence/xiao_spearman.hpp"
#include "../../src/streamgenerator/datasource/copula.hpp"
#include "../../src/streamgenerator/datasource/dependence.hpp"
#include "../common/copula_source.hpp"

namespace {
constexpr std::size_t kN = 15000, kBurn = 500, kEvery = 5, kLate = 10000;

double probit(double u) { return RankPit::probit(std::clamp(u, 1e-12, 1.0 - 1e-12)); }

std::vector<double> full_emp(const std::vector<double>& x) {
    std::vector<std::size_t> idx(x.size());
    for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return x[a] < x[b]; });
    std::vector<double> z(x.size());
    for (std::size_t r = 0; r < idx.size(); ++r) z[idx[r]] = probit((r + 1.0) / (x.size() + 1.0));
    return z;
}

std::vector<double> seq_emp(const std::vector<double>& x, bool mid = false) {   // u_t = share of earlier observations strictly below x_t
    std::vector<double> sorted, z(x.size());
    sorted.reserve(x.size());
    for (std::size_t t = 0; t < x.size(); ++t) {
        const double below = static_cast<double>(std::lower_bound(sorted.begin(), sorted.end(), x[t]) - sorted.begin());
        z[t] = (t == 0) ? 0.0 : probit(mid ? (below + 0.5) / (t + 1.0) : below / static_cast<double>(t));
        sorted.insert(std::upper_bound(sorted.begin(), sorted.end(), x[t]), x[t]);
    }
    return z;
}

struct Variant { std::string name; std::function<std::unique_ptr<DependenceEstimator>(unsigned)> make; int input; };   // input: 0 raw, 1 full, 2 seq, 3 seq mid-rank

std::unique_ptr<DependenceEstimator> rmpl(RecursiveMPL::Margin m, std::size_t k, std::size_t warmup, unsigned seed) {
    RecursiveMPL::Config c{};
    c.gain = 1e-12; c.margin = m; c.kll_k = k; c.seed = seed; c.warmup = warmup;
    return std::make_unique<RecursiveMPL>(c, "mpl");
}
std::unique_ptr<DependenceEstimator> xiao(bool ns) {
    XiaoSpearmanEstimator::Config c{}; c.bins = 30; c.lambda = 0.0; c.normal_score_inputs = ns;
    return std::make_unique<XiaoSpearmanEstimator>(c, "xiao");
}
std::unique_ptr<DependenceEstimator> herm(int order, bool ns) {
    HermiteEstimator::Config c{}; c.order = order; c.lambda = 0.0; c.normal_score_inputs = ns;
    return std::make_unique<HermiteEstimator>(c, "hermite");
}

// Spearman's rho from the running mean of 12 (u - 1/2)(v - 1/2): a likelihood-free rank estimator on the same pseudo-observations.
class RankMoment : public DependenceEstimator {
public:
    void update(double x, double y) override {
        const double u = 0.5 * std::erfc(-x / std::sqrt(2.0)), v = 0.5 * std::erfc(-y / std::sqrt(2.0));
        ++n_; s_ += (12.0 * (u - 0.5) * (v - 0.5) - s_) / static_cast<double>(n_);
    }
    double estimate() const override { return s_; }
    void reset() override { n_ = 0; s_ = 0.0; }
    std::size_t store_bytes() const override { return sizeof(*this); }
    const char* name() const override { return "rankmoment"; }
private:
    std::size_t n_ = 0; double s_ = 0.0;
};
}  // namespace

int main() {
    (void)copula::spearman_from_param(copula::Family::StudentT, 0.5, 4.0);
    using M = RecursiveMPL::Margin;
    std::vector<Variant> V = {
        {"rmpl_oracle", [](unsigned s) { return rmpl(M::ExactNormal, 0, 30, s); }, 0},
        {"rmpl_full_emp", [](unsigned s) { return rmpl(M::ExactNormal, 0, 30, s); }, 1},
        {"rmpl_seq_emp", [](unsigned s) { return rmpl(M::ExactNormal, 0, 30, s); }, 2},
        {"rmpl_seq_mid", [](unsigned s) { return rmpl(M::ExactNormal, 0, 30, s); }, 3},
        {"rmpl_seq_emp_t0_500", [](unsigned s) { return rmpl(M::ExactNormal, 0, 500, s); }, 2},
        {"rmpl_seq_emp_t0_2000", [](unsigned s) { return rmpl(M::ExactNormal, 0, 2000, s); }, 2},
    };
    for (std::size_t k : {64, 256, 512, 1024, 4096, 16384})
        V.push_back({"rmpl_kll_" + std::to_string(k), [k](unsigned s) { return rmpl(M::DsKll, k, 30, s); }, 0});
    V.push_back({"rmpl_kll_512_t0_2000", [](unsigned s) { return rmpl(M::DsKll, 512, 2000, s); }, 0});
    V.push_back({"rankmom_oracle", [](unsigned) { return std::make_unique<RankMoment>(); }, 0});
    V.push_back({"rankmom_full_emp", [](unsigned) { return std::make_unique<RankMoment>(); }, 1});
    V.push_back({"rankmom_seq_emp", [](unsigned) { return std::make_unique<RankMoment>(); }, 2});
    V.push_back({"cm_default", [](unsigned) { return xiao(false); }, 0});
    V.push_back({"cm_oracle", [](unsigned) { return xiao(true); }, 0});
    V.push_back({"cm_full_emp", [](unsigned) { return xiao(true); }, 1});
    V.push_back({"cm_seq_emp", [](unsigned) { return xiao(true); }, 2});
    for (int o : {20, 60}) {
        const std::string t = std::to_string(o);
        V.push_back({"hs" + t + "_default", [o](unsigned) { return herm(o, false); }, 0});
        V.push_back({"hs" + t + "_oracle", [o](unsigned) { return herm(o, true); }, 0});
        V.push_back({"hs" + t + "_full_emp", [o](unsigned) { return herm(o, true); }, 1});
        V.push_back({"hs" + t + "_seq_emp", [o](unsigned) { return herm(o, true); }, 2});
    }
    const std::size_t nseed = 100;
    std::vector<std::vector<double>> all(V.size(), std::vector<double>(nseed)), late = all;
    std::mutex mu;
    std::vector<std::thread> pool;
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    for (unsigned w = 0; w < hw; ++w) pool.emplace_back([&, w] {
        for (std::size_t si = w; si < nseed; si += hw) {
            const unsigned seed = 42 + static_cast<unsigned>(si);
            ConstantDependence dep(0.6);
            auto cop = dgp::make_copula("gaussian");
            CopulaSource src(dep, *cop, seed);
            std::vector<double> x(kN), y(kN), tr(kN);
            for (std::size_t t = 0; t < kN; ++t) { const auto s = src.next(); x[t] = s.x; y[t] = s.y; tr[t] = s.rho_s; }
            const std::vector<double> fx = full_emp(x), fy = full_emp(y), sx = seq_emp(x), sy = seq_emp(y), mx = seq_emp(x, true), my = seq_emp(y, true);
            for (std::size_t v = 0; v < V.size(); ++v) {
                auto e = V[v].make(seed);
                const auto& ix = V[v].input == 0 ? x : V[v].input == 1 ? fx : V[v].input == 2 ? sx : mx;
                const auto& iy = V[v].input == 0 ? y : V[v].input == 1 ? fy : V[v].input == 2 ? sy : my;
                double se = 0, sl = 0; std::size_t ne = 0, nl = 0;
                for (std::size_t t = 0; t < kN; ++t) {
                    e->update(ix[t], iy[t]);
                    if (t < kBurn || (t - kBurn) % kEvery != 0) continue;
                    const double err = e->estimate() - tr[t];
                    se += err * err; ++ne;
                    if (t >= kLate) { sl += err * err; ++nl; }
                }
                std::lock_guard<std::mutex> lk(mu);
                all[v][si] = std::sqrt(se / ne); late[v][si] = std::sqrt(sl / nl);
            }
        }
    });
    for (auto& t : pool) t.join();
    std::ofstream out("results/margins_seeds.csv");
    out << "seed,variant,rmse_all,rmse_late\n";
    for (std::size_t v = 0; v < V.size(); ++v)
        for (std::size_t s = 0; s < nseed; ++s) out << 42 + s << ',' << V[v].name << ',' << all[v][s] << ',' << late[v][s] << '\n';
    std::fprintf(stderr, "wrote results/margins_seeds.csv\n");
}
