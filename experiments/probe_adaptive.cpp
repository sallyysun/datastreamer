// Does adaptive memory generalise past the Kalman filter?
//
// Three families (Xiao, Hermite, Kalman) x three ways of choosing memory
// (known_best oracle, VFF, bank), on one tick axis. Protocol is probe_memory.cpp's
// exactly: N=5000, burn-in 500, 40 seeds from 42, Gaussian copula.
// See notes/studies/adaptive_memory.md.
//
// Usage: ./probe_adaptive   -> results/adaptive_memory{,_pairs,_traj}.csv

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../src/estimators/dependence/estimator.hpp"
#include "../src/estimators/dependence/estimator_bank.hpp"
#include "../src/estimators/dependence/hermiter.hpp"
#include "../src/estimators/dependence/kalman_bank.hpp"
#include "../src/estimators/dependence/kalman_local_level.hpp"
#include "../src/estimators/dependence/xiao_spearman.hpp"
#include "../src/streamgenerator/datasource/copula.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"

namespace {

constexpr std::size_t kN = 5000;
constexpr std::size_t kBurn = 500;
constexpr std::size_t kJump = kN / 2;
constexpr std::size_t kBlock = 20;
constexpr std::size_t kSeeds = 40;
constexpr unsigned kBaseSeed = 42;
constexpr int kOrder = 10;
constexpr std::size_t kBins = 20;

double alpha_from_ess(double e) { return 2.0 / (e + 1.0); }
double ess_from_alpha(double a) { return (2.0 - a) / a; }

double kalman_gain(double q) {
    const double sv2 = 1.06 / std::max(static_cast<double>(kBlock) - 3.0, 1.0);
    const double r = q / sv2;
    return 0.5 * (std::sqrt(r * r + 4.0 * r) - r);
}
double kalman_mem_ticks(double q) {
    return ess_from_alpha(kalman_gain(q)) * static_cast<double>(kBlock);
}

std::vector<double> log_grid(double lo, double hi, int n) {
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i) {
        const double f = (n == 1) ? 0.0 : static_cast<double>(i) / (n - 1);
        v[i] = std::exp(std::log(lo) + f * (std::log(hi) - std::log(lo)));
    }
    return v;
}

// The bank arm grid, in TICKS of effective memory. Its span matches the
// incumbent kalman_bank's nominal 22-999, so all three banks reach the same
// memories; its geometric centre (158 ticks) is every VFF's base memory.
const std::vector<double>& arm_ticks() {
    static const std::vector<double> v = log_grid(25.0, 1000.0, 7);
    return v;
}
constexpr double kVffBaseMem = 158.1;

struct Built {
    std::unique_ptr<DependenceEstimator> est;
    std::function<double()> mem;   // posterior-mean / current memory in ticks
};

struct Cfg {
    std::string family;   // xiao | hermite | hermite_db | kalman
    std::string method;   // fixed | bank | vff
    double param = 0.0;
    double mem_ticks = 0.0;
    std::function<Built()> build;
    std::string tag;      // disambiguates two configs sharing a method and param
};

std::unique_ptr<DependenceEstimator> xiao_at(double mem) {
    return std::make_unique<XiaoSpearmanEstimator>(
        XiaoSpearmanEstimator::Config{.bins = kBins, .window = 0,
                                      .lambda = 1.0 - alpha_from_ess(mem),
                                      .calibration = 100}, "m");
}
std::unique_ptr<DependenceEstimator> hermite_at(double mem, bool debias) {
    return std::make_unique<HermiteEstimator>(
        HermiteEstimator::Config{.order = kOrder, .lambda = alpha_from_ess(mem),
                                 .burn_in = 50, .debias = debias}, "m");
}

Built bank_of(const std::string& family) {
    std::vector<std::unique_ptr<DependenceEstimator>> models;
    for (double m : arm_ticks()) {
        if (family == "xiao") models.push_back(xiao_at(m));
        else models.push_back(hermite_at(m, family == "hermite_db"));
    }
    auto b = std::make_unique<EstimatorBank>(std::move(models), EstimatorBank::Config{}, "bank");
    EstimatorBank* p = b.get();
    Built out;
    out.est = std::move(b);
    out.mem = [p] {
        double m = 0.0;
        for (std::size_t k = 0; k < p->size(); ++k) m += p->weight(k) * arm_ticks()[k];
        return m;
    };
    return out;
}

std::vector<Cfg> build_cfgs() {
    std::vector<Cfg> v;
    const std::vector<double> ess = log_grid(25.0, 30000.0, 20);

    for (double e : ess)
        v.push_back({"xiao", "fixed", 1.0 - alpha_from_ess(e), e,
                     [e] { return Built{xiao_at(e), {}}; }});
    for (double e : ess)
        v.push_back({"hermite", "fixed", alpha_from_ess(e), e,
                     [e] { return Built{hermite_at(e, false), {}}; }});
    for (double e : ess)
        v.push_back({"hermite_db", "fixed", alpha_from_ess(e), e,
                     [e] { return Built{hermite_at(e, true), {}}; }});
    for (double q : log_grid(1e-7, 1e0, 10))
        v.push_back({"kalman", "fixed", q, kalman_mem_ticks(q), [q] {
            return Built{std::make_unique<KalmanLocalLevel>(
                KalmanLocalLevel::Config{.block = kBlock, .sigma_omega2 = q,
                                         .burn_in_blocks = 2}, "m"), {}};
        }});

    for (const char* f : {"xiao", "hermite", "hermite_db"})
        v.push_back({f, "bank", 0.0, 0.0, [f] { return bank_of(f); }});
    v.push_back({"kalman", "bank", 0.002, 0.0, [] {
        auto b = std::make_unique<KalmanBank>(KalmanBank::Config{.switch_prob = 0.002}, "bank");
        KalmanBank* p = b.get();
        Built out;
        out.est = std::move(b);
        out.mem = [p] {
            double m = 0.0;
            for (std::size_t k = 0; k < p->size(); ++k)
                m += p->weight(k) * kalman_mem_ticks(p->sigma_omega2(k));
            return m;
        };
        return out;
    }});

    // Both VFF rates, because the rate is a free constant a bank does not have.
    for (double r : {0.05, 0.5}) {
        v.push_back({"xiao", "vff", r, kVffBaseMem, [r] {
            auto e = std::make_unique<XiaoSpearmanEstimator>(
                XiaoSpearmanEstimator::Config{.bins = kBins, .window = 0,
                                              .lambda = 1.0 - alpha_from_ess(kVffBaseMem),
                                              .calibration = 100, .vff_rate = r}, "m");
            XiaoSpearmanEstimator* p = e.get();
            Built out;
            out.est = std::move(e);
            out.mem = [p] { return p->effective_memory_ticks(); };
            return out;
        }, r == 0.05 ? "_slow" : "_fast"});
        for (bool db : {false, true})
            v.push_back({db ? "hermite_db" : "hermite", "vff", r, kVffBaseMem, [db, r] {
                return Built{std::make_unique<HermiteEstimator>(
                    HermiteEstimator::Config{.order = kOrder,
                                             .lambda = alpha_from_ess(kVffBaseMem),
                                             .burn_in = 50, .vff_rate = r,
                                             .debias = db}, "m"), {}};
            }, r == 0.05 ? "_slow" : "_fast"});
    }
    // q = 4.64e-3 is the grid point nearest kVffBaseMem; 1e-3 is the registered
    // kalman_vff, kept so the incumbent row is literally the incumbent.
    for (auto qr : {std::pair{4.64e-3, 0.05}, std::pair{4.64e-3, 0.5}, std::pair{1e-3, 0.05}})
        v.push_back({"kalman", "vff", qr.second, kalman_mem_ticks(qr.first), [qr] {
            const auto [q, r] = qr;
            auto e = std::make_unique<KalmanLocalLevel>(
                KalmanLocalLevel::Config{.block = kBlock, .sigma_omega2 = q,
                                         .burn_in_blocks = 2, .vff_rate = r}, "m");
            KalmanLocalLevel* p = e.get();
            Built out;
            out.est = std::move(e);
            out.mem = [p] { return kalman_mem_ticks(p->sigma_omega2()); };
            return out;
        }, qr.first > 2e-3 ? (qr.second == 0.05 ? "_slow" : "_fast") : "_registered"});
    return v;
}

std::unique_ptr<Dependence> make_dependence(const std::string& s) {
    if (s == "constant") return std::make_unique<ConstantDependence>(0.6);
    if (s == "drift") return std::make_unique<DriftDependence>(-0.5, 0.8, kN);
    if (s == "jump") return std::make_unique<JumpDependence>(0.2, 0.8, kJump);
    if (s == "oscillate") return std::make_unique<OscillatingDependence>(0.6, 400.0, 0.0);
    return nullptr;
}

using Panel = std::vector<std::vector<std::vector<double>>>;

void run_seed(const std::string& scenario, const std::vector<Cfg>& cfgs, std::size_t s,
              Panel& est, Panel& mem, std::vector<double>& truth, bool record_truth) {
    auto dep = make_dependence(scenario);
    auto cop = dgp::make_copula("gaussian");
    CopulaSource src(*dep, *cop, kBaseSeed + static_cast<unsigned>(s));

    std::vector<Built> built;
    built.reserve(cfgs.size());
    for (const Cfg& c : cfgs) built.push_back(c.build());
    for (std::size_t c = 0; c < cfgs.size(); ++c) {
        est[c][s].resize(kN);
        if (built[c].mem) mem[c][s].resize(kN);
    }

    for (std::size_t t = 0; t < kN; ++t) {
        const CopulaSource::Sample smp = src.next();
        if (record_truth) truth[t] = smp.rho_s;
        for (std::size_t c = 0; c < built.size(); ++c) {
            built[c].est->update(smp.x, smp.y);
            est[c][s][t] = built[c].est->estimate();
            if (built[c].mem) mem[c][s][t] = built[c].mem();
        }
    }
}

std::vector<double> seed_rmse(const Panel& est, const std::vector<double>& truth,
                              std::size_t c, std::size_t lo, std::size_t hi) {
    const double n = static_cast<double>(hi - lo);
    std::vector<double> r(kSeeds);
    for (std::size_t s = 0; s < kSeeds; ++s) {
        double acc = 0.0;
        for (std::size_t t = lo; t < hi; ++t) {
            const double d = est[c][s][t] - truth[t];
            acc += d * d;
        }
        r[s] = std::sqrt(acc / n);
    }
    return r;
}

double pooled_rmse(const Panel& est, const std::vector<double>& truth, std::size_t c,
                   std::size_t lo, std::size_t hi) {
    double acc = 0.0;
    for (std::size_t s = 0; s < kSeeds; ++s)
        for (std::size_t t = lo; t < hi; ++t) {
            const double d = est[c][s][t] - truth[t];
            acc += d * d;
        }
    return std::sqrt(acc / (static_cast<double>(kSeeds) * static_cast<double>(hi - lo)));
}

struct Paired { double mean, se, sigma; };

Paired paired(const std::vector<double>& a, const std::vector<double>& b) {
    const double S = static_cast<double>(kSeeds);
    double m = 0.0;
    for (std::size_t s = 0; s < kSeeds; ++s) m += a[s] - b[s];
    m /= S;
    double v = 0.0;
    for (std::size_t s = 0; s < kSeeds; ++s) {
        const double e = (a[s] - b[s]) - m;
        v += e * e;
    }
    const double se = std::sqrt(v / (S - 1.0) / S);
    return {m, se, se > 0.0 ? m / se : 0.0};
}

double se_of(const std::vector<double>& r) {
    const double S = static_cast<double>(kSeeds);
    double m = 0.0;
    for (double v : r) m += v;
    m /= S;
    double v2 = 0.0;
    for (double v : r) v2 += (v - m) * (v - m);
    return std::sqrt(v2 / (S - 1.0) / S);
}

}  // namespace

int main() {
    const std::vector<Cfg> cfgs = build_cfgs();
    const std::vector<std::string> scenarios{"constant", "drift", "jump", "oscillate"};

    std::ofstream out("results/adaptive_memory.csv");
    std::ofstream pout("results/adaptive_memory_pairs.csv");
    std::ofstream tout("results/adaptive_memory_traj.csv");
    if (!out.is_open() || !pout.is_open() || !tout.is_open()) {
        std::fprintf(stderr, "cannot write results/ (run: mkdir -p results)\n");
        return 1;
    }
    out << "family,method,scenario,param,mem_ticks,rmse,rmse_mean_seed,se_rmse\n";
    pout << "family,scenario,method,param,rmse,known_best_mem,known_best_param,"
            "rmse_known_best,diff_mean,diff_se,sigma\n";
    tout << "scenario,family,method,t,mem_post,rho_hat,truth\n";

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());

    for (const std::string& scenario : scenarios) {
        Panel est(cfgs.size(), std::vector<std::vector<double>>(kSeeds));
        Panel mem(cfgs.size(), std::vector<std::vector<double>>(kSeeds));
        std::vector<double> truth(kN);
        run_seed(scenario, cfgs, 0, est, mem, truth, true);

        std::vector<std::thread> pool;
        std::vector<std::vector<double>> ignored(hw, std::vector<double>(kN));
        for (unsigned w = 0; w < hw; ++w) {
            pool.emplace_back([&, w] {
                for (std::size_t s = 1 + w; s < kSeeds; s += hw)
                    run_seed(scenario, cfgs, s, est, mem, ignored[w], false);
            });
        }
        for (auto& th : pool) th.join();

        std::vector<std::vector<double>> per_seed(cfgs.size());
        std::vector<double> pooled(cfgs.size());
        for (std::size_t c = 0; c < cfgs.size(); ++c) {
            per_seed[c] = seed_rmse(est, truth, c, kBurn, kN);
            pooled[c] = pooled_rmse(est, truth, c, kBurn, kN);
            double m = 0.0;
            for (double v : per_seed[c]) m += v;
            m /= static_cast<double>(kSeeds);
            out << cfgs[c].family << "," << cfgs[c].method << cfgs[c].tag << "," << scenario << ","
                << cfgs[c].param << "," << cfgs[c].mem_ticks << "," << pooled[c] << ","
                << m << "," << se_of(per_seed[c]) << "\n";
        }

        // known_best: per family, the single best fixed memory on THIS scenario,
        // chosen after seeing the answer. Same argmin as memory_sweep.csv's rmse
        // column at window=all.
        for (const char* fam : {"xiao", "hermite", "hermite_db", "kalman"}) {
            std::size_t kb = cfgs.size();
            for (std::size_t c = 0; c < cfgs.size(); ++c) {
                if (cfgs[c].family != fam || cfgs[c].method != "fixed") continue;
                if (kb == cfgs.size() || pooled[c] < pooled[kb]) kb = c;
            }
            if (kb == cfgs.size()) continue;
            for (std::size_t c = 0; c < cfgs.size(); ++c) {
                if (cfgs[c].family != fam || cfgs[c].method == "fixed") continue;
                const Paired p = paired(per_seed[c], per_seed[kb]);
                pout << fam << "," << scenario << "," << cfgs[c].method << cfgs[c].tag << ","
                     << cfgs[c].param << "," << pooled[c] << "," << cfgs[kb].mem_ticks << ","
                     << cfgs[kb].param << "," << pooled[kb] << "," << p.mean << ","
                     << p.se << "," << p.sigma << "\n";
            }
            pout << fam << "," << scenario << ",known_best," << cfgs[kb].param << ","
                 << pooled[kb] << "," << cfgs[kb].mem_ticks << "," << cfgs[kb].param << ","
                 << pooled[kb] << ",0,0,0\n";
        }

        for (std::size_t c = 0; c < cfgs.size(); ++c) {
            if (mem[c][0].empty()) continue;
            for (std::size_t t = 0; t < kN; ++t) {
                double mm = 0.0, rr = 0.0;
                for (std::size_t s = 0; s < kSeeds; ++s) {
                    mm += mem[c][s][t];
                    rr += est[c][s][t];
                }
                tout << scenario << "," << cfgs[c].family << ","
                     << cfgs[c].method << cfgs[c].tag << "," << t << "," << mm / kSeeds << "," << rr / kSeeds << ","
                     << truth[t] << "\n";
            }
        }
        std::printf("scenario=%s done (%zu configs x %zu seeds)\n", scenario.c_str(),
                    cfgs.size(), kSeeds);
    }
    std::printf("wrote results/adaptive_memory{,_pairs,_traj}.csv\n");
    return 0;
}
