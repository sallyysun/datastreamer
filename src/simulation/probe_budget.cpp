// Section 6.7 matched-memory-budget comparison: Counting Matrix (B bins) vs Hermite Series (order N) vs RMPL on
// its own KLL margins (mpl_kll: hand-written KLL, mpl_dskll: Apache DataSketches KLL, capacity k), plus mpl_exact
// (true margins, the reference). Storage = store_bytes() at its peak; accuracy = rho_S RMSE over the test seeds.
// Usage: ./probe_budget sweep <n_test> [n_val] [family]  -> results/budget_sweep[_family].csv
//        ./probe_budget time                     -> results/budget_timing.csv

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../estimators/dependence/estimator.hpp"
#include "../estimators/dependence/hermiter.hpp"
#include "../estimators/dependence/rmpl.hpp"
#include "../estimators/dependence/xiao_spearman.hpp"
#include "../streamgenerator/datasource/copula.hpp"
#include "../streamgenerator/datasource/dependence.hpp"
#include "../streamgenerator/datasource/copula_source.hpp"

namespace {

// BUDGET_N overrides the stream length (harness only; default 5000). Drift spans the whole stream; the jump is at kN / 2.
const std::size_t kN = std::getenv("BUDGET_N") ? std::strtoul(std::getenv("BUDGET_N"), nullptr, 10) : 5000;
const std::size_t kJump = kN / 2;
constexpr std::size_t kBurn = 500;
constexpr std::size_t kEvery = 5;          // score every 5th tick
constexpr std::size_t kPost = 1000;        // post-change window
constexpr unsigned kTestSeed = 42;
constexpr unsigned kValSeed = 1000;

// BUDGET_GAINS=g1,g2,... replaces the grid (harness only); g = 0 is each estimator's cumulative (no-forgetting)
// mode: Xiao lambda = 0, Hermite weight 1/m, MPL gain 1/m through the normalised start-up. budget_mpl.md §25.
std::vector<double> gain_grid() {
    const char* e = std::getenv("BUDGET_GAINS");
    if (!e) return {0.005, 0.01, 0.02, 0.05, 0.10};
    std::vector<double> g;
    std::string str(e);
    for (std::size_t a = 0, b; a <= str.size(); a = b + 1) {
        b = str.find(',', a);
        if (b == std::string::npos) b = str.size();
        if (b > a) g.push_back(std::atof(str.substr(a, b - a).c_str()));
    }
    return g;
}
const std::vector<double> kGains = gain_grid();

// Harness only (Section 6.2 extension, 2026-11-15): BUDGET_EST=a,b restricts the estimators; BUDGET_KLL=k1,k2 replaces the
// capacity grid of both KLL implementations (mpl_kll, mpl_dskll); BUDGET_TIME_GAIN sets the gain timed by `time` (default 0.02).
std::vector<std::string> csv_env(const char* name) {
    std::vector<std::string> v;
    const char* e = std::getenv(name);
    if (!e) return v;
    std::string str(e);
    for (std::size_t a = 0, b; a <= str.size(); a = b + 1) {
        b = str.find(',', a);
        if (b == std::string::npos) b = str.size();
        if (b > a) v.push_back(str.substr(a, b - a));
    }
    return v;
}
bool est_on(const std::string& est) {
    const auto v = csv_env("BUDGET_EST");
    return v.empty() || std::find(v.begin(), v.end(), est) != v.end();
}
std::vector<int> ds_kll_grid() {
    const auto v = csv_env("BUDGET_KLL");
    if (v.empty()) return {8, 16, 32, 64, 128, 256, 512, 1024};
    std::vector<int> k;
    for (const auto& x : v) k.push_back(std::atoi(x.c_str()));
    return k;
}

struct Cfg {
    std::string est;    // xiao | hermite | mpl_kll | mpl_dskll | mpl_exact
    int approx;         // B, N, k or W
    double gain;        // per-observation update weight, all estimators
};

std::vector<Cfg> build_cfgs() {
    std::vector<Cfg> v;
    for (double g : kGains) {
        if (est_on("xiao")) for (int b : {10, 20, 30, 45, 65, 90, 130, 180}) v.push_back({"xiao", b, g});
        if (est_on("hermite")) for (int n : {5, 10, 15, 20, 30, 43, 60, 89, 128}) v.push_back({"hermite", n, g});
        if (est_on("mpl_kll")) for (int k : ds_kll_grid()) v.push_back({"mpl_kll", k, g});   // BUDGET_KLL applies to both KLL implementations
        if (est_on("mpl_dskll")) for (int k : ds_kll_grid()) v.push_back({"mpl_dskll", k, g});
        if (est_on("mpl_exact")) v.push_back({"mpl_exact", 0, g});
    }
    return v;
}

std::unique_ptr<DependenceEstimator> build(const Cfg& c, unsigned seed) {
    if (c.est == "xiao") {
        XiaoSpearmanEstimator::Config x{};
        x.bins = static_cast<std::size_t>(c.approx);
        x.lambda = c.gain > 0.0 ? 1.0 - c.gain : 0.0;   // 0 = cumulative
        return std::make_unique<XiaoSpearmanEstimator>(x, "xiao");
    }
    if (c.est == "hermite") {
        HermiteEstimator::Config h{};
        h.order = c.approx;
        h.lambda = c.gain;                               // 0 = cumulative (1/m)
        return std::make_unique<HermiteEstimator>(h, "hermite");
    }
    RecursiveMPL::Config m{};
    m.gain = c.gain > 0.0 ? c.gain : 1e-12;              // normalised start-up: gain/(1-(1-gain)^m) -> 1/m
    m.seed = seed;
    if (c.est == "mpl_kll" || c.est == "mpl_dskll") m.kll_k = static_cast<std::size_t>(c.approx);
    if (c.est == "mpl_dskll") m.margin = RecursiveMPL::Margin::DsKll;
    if (c.est == "mpl_exact") m.margin = RecursiveMPL::Margin::ExactNormal;
    return std::make_unique<RecursiveMPL>(m, "mpl");
}

std::unique_ptr<Dependence> make_dependence(const std::string& s) { return make_scenario(s, kN); }

struct Data { std::vector<double> x, y, truth; };

Data simulate(const std::string& family, const std::string& scenario, unsigned seed) {
    auto dep = make_dependence(scenario);
    auto cop = dgp::make_copula(family);
    CopulaSource src(*dep, *cop, seed);
    Data d;
    d.x.resize(kN); d.y.resize(kN); d.truth.resize(kN);
    for (std::size_t t = 0; t < kN; ++t) {
        const auto s = src.next();
        d.x[t] = s.x; d.y[t] = s.y; d.truth[t] = s.rho_s;
    }
    return d;
}

const std::size_t kPts = (kN - kBurn) / kEvery;

struct Acc {
    std::vector<double> rmse, post;   // per seed
    std::vector<double> path;         // summed estimate at scored ticks
    std::size_t peak = 0;
};

void run_split(const std::string& family, const std::string& scenario, unsigned base,
               std::size_t n_seeds, const std::vector<Cfg>& cfgs, std::vector<Acc>& acc,
               std::vector<double>& truth_path) {
    acc.assign(cfgs.size(), Acc{});
    for (auto& a : acc) { a.rmse.assign(n_seeds, 0.0); a.post.assign(n_seeds, 0.0); a.path.assign(kPts, 0.0); }
    truth_path.assign(kPts, 0.0);
    std::mutex mu;
    // BUDGET_THREADS caps the worker threads (harness only; on a batch cluster set it to the allocated cores).
    const unsigned hw = std::getenv("BUDGET_THREADS") ? std::max(1, std::atoi(std::getenv("BUDGET_THREADS")))
                                                        : std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (unsigned w = 0; w < hw; ++w) {
        pool.emplace_back([&, w] {
            std::vector<std::vector<double>> path(cfgs.size(), std::vector<double>(kPts, 0.0));
            std::vector<std::size_t> peak(cfgs.size(), 0);
            std::vector<double> tp(kPts, 0.0);
            for (std::size_t s = w; s < n_seeds; s += hw) {
                const unsigned seed = base + static_cast<unsigned>(s);
                const Data d = simulate(family, scenario, seed);
                for (std::size_t c = 0; c < cfgs.size(); ++c) {
                    auto e = build(cfgs[c], seed);
                    double se = 0.0, sp = 0.0;
                    std::size_t ne = 0, np = 0;
                    for (std::size_t t = 0; t < kN; ++t) {
                        e->update(d.x[t], d.y[t]);
                        peak[c] = std::max(peak[c], e->store_bytes());   // every tick, start-up included
                        if (t < kBurn || (t - kBurn) % kEvery != 0) continue;
                        const double r = e->estimate(), err = r - d.truth[t];
                        se += err * err; ++ne;
                        if (t >= kJump && t < kJump + kPost) { sp += err * err; ++np; }
                        path[c][(t - kBurn) / kEvery] += r;
                    }
                    acc[c].rmse[s] = std::sqrt(se / ne);
                    acc[c].post[s] = std::sqrt(sp / np);
                }
                for (std::size_t i = 0; i < kPts; ++i) tp[i] += d.truth[kBurn + i * kEvery];
            }
            std::lock_guard<std::mutex> lk(mu);
            for (std::size_t c = 0; c < cfgs.size(); ++c) {
                for (std::size_t i = 0; i < kPts; ++i) acc[c].path[i] += path[c][i];
                acc[c].peak = std::max(acc[c].peak, peak[c]);
            }
            for (std::size_t i = 0; i < kPts; ++i) truth_path[i] += tp[i];
        });
    }
    for (auto& t : pool) t.join();
    for (auto& a : acc) for (double& p : a.path) p /= static_cast<double>(n_seeds);
    for (double& p : truth_path) p /= static_cast<double>(n_seeds);
}

double mean(const std::vector<double>& v) {
    double s = 0.0; for (double x : v) s += x; return s / v.size();
}
double se(const std::vector<double>& v) {
    const double m = mean(v); double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / (v.size() - 1) / v.size());
}

// Ticks after the jump until the mean path covers 90% of its OWN pre->settled change, so a
// biased (misspecified) estimator is not scored as never recovering.
double recovery_delay(const std::vector<double>& path) {
    auto avg = [&](std::size_t t0, std::size_t t1) {
        double s = 0.0; std::size_t n = 0;
        for (std::size_t t = t0; t < t1; t += kEvery) { s += path[(t - kBurn) / kEvery]; ++n; }
        return s / n;
    };
    const double pre = avg(kJump - 1000, kJump), post = avg(kN - 1000, kN);
    const double target = pre + 0.9 * (post - pre);
    for (std::size_t t = kJump; t < kN; t += kEvery) {
        const double r = path[(t - kBurn) / kEvery];
        if ((post > pre && r >= target) || (post <= pre && r <= target)) return double(t - kJump);
    }
    return -1.0;
}

void sweep(std::size_t n_test, std::size_t n_val, const std::vector<std::string>& families) {
    const auto cfgs = build_cfgs();
    const std::string sfx = families.size() == 1 ? "_" + families[0] : "";
    const std::string tg = (std::getenv("TAG") ? std::string("_") + std::getenv("TAG") : std::string());
    std::ofstream out("results/budget_sweep" + tg + sfx + ".csv");
    std::ofstream sout("results/budget_seeds" + tg + sfx + ".csv");   // per-seed test RMSE, for paired comparisons
    sout << "family,scenario,estimator,approx,gain,seed,rmse,post_rmse\n";
    out << "family,scenario,split,estimator,approx,gain,peak_bytes,rmse,se_rmse,post_rmse,se_post,"
           "recovery_delay\n";
    std::ofstream pout("results/budget_paths" + tg + sfx + ".csv");
    pout << "family,scenario,estimator,approx,gain,t,est,truth\n";
    for (const std::string& fam : families) {
        for (const std::string sc : {"constant", "drift", "jump"}) {
            if (const char* only = std::getenv("BUDGET_SCENARIOS"))   // a subset of the same scenarios
                if (("," + std::string(only) + ",").find("," + sc + ",") == std::string::npos) continue;
            for (int split = 0; split < 2; ++split) {
                const unsigned base = split ? kTestSeed : kValSeed;
                const std::size_t ns = split ? n_test : n_val;
                std::vector<Acc> acc;
                std::vector<double> truth;
                const auto t0 = std::chrono::steady_clock::now();
                run_split(fam, sc, base, ns, cfgs, acc, truth);
                const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::fprintf(stderr, "%-8s %-8s %s %3zu seeds  %.1fs\n", fam.c_str(), sc.c_str(),
                             split ? "test" : "val ", ns, secs);
                for (std::size_t c = 0; c < cfgs.size(); ++c) {
                    const auto& a = acc[c];
                    out << fam << ',' << sc << ',' << (split ? "test" : "val") << ',' << cfgs[c].est << ','
                        << cfgs[c].approx << ',' << cfgs[c].gain << ',' << a.peak << ',' << mean(a.rmse) << ','
                        << se(a.rmse) << ',' << mean(a.post) << ',' << se(a.post) << ','
                        << (sc == "jump" ? recovery_delay(a.path) : -1.0) << '\n';
                    if (split)
                        for (std::size_t s = 0; s < ns; ++s)
                            sout << fam << ',' << sc << ',' << cfgs[c].est << ',' << cfgs[c].approx << ',' << cfgs[c].gain
                                 << ',' << base + s << ',' << a.rmse[s] << ',' << a.post[s] << '\n';
                    if (split && sc == "jump")
                        for (std::size_t i = 0; i < kPts; i += 2)
                            pout << fam << ',' << sc << ',' << cfgs[c].est << ',' << cfgs[c].approx << ','
                                 << cfgs[c].gain << ',' << kBurn + i * kEvery << ',' << a.path[i] << ','
                                 << truth[i] << '\n';
                }
            }
        }
    }
    std::fprintf(stderr, "wrote results/budget_sweep%s.csv, results/budget_paths%s.csv\n", sfx.c_str(), sfx.c_str());
}

// Single thread, Gaussian drift, gain 0.02: wall time of update() + estimate() per tick, which
// is the whole online pipeline per observation. Data are generated before the clock starts.
void timing() {
    std::vector<Cfg> cfgs;
    const double tg = std::getenv("BUDGET_TIME_GAIN") ? std::atof(std::getenv("BUDGET_TIME_GAIN")) : 0.02;
    for (Cfg c : build_cfgs()) if (c.gain == kGains.front()) { c.gain = tg; cfgs.push_back(c); }
    std::ofstream out("results/budget_timing" + (std::getenv("TAG") ? std::string("_") + std::getenv("TAG") : std::string()) + ".csv");
    out << "estimator,approx,peak_bytes,mean_ns,p95_ns,update_mean_ns\n";
    for (const Cfg& c : cfgs) {
        std::vector<double> tick, upd;
        std::size_t peak = 0;
        for (unsigned s = 0; s < 5; ++s) {
            const Data d = simulate("gaussian", "drift", kTestSeed + s);
            auto e = build(c, kTestSeed + s);
            volatile double sink = 0.0;
            for (std::size_t t = 0; t < kN; ++t) {
                const auto a = std::chrono::steady_clock::now();
                e->update(d.x[t], d.y[t]);
                const auto b = std::chrono::steady_clock::now();
                sink = e->estimate();
                const auto z = std::chrono::steady_clock::now();
                if (t >= kBurn) {
                    tick.push_back(std::chrono::duration<double, std::nano>(z - a).count());
                    upd.push_back(std::chrono::duration<double, std::nano>(b - a).count());
                }
                peak = std::max(peak, e->store_bytes());
            }
            (void)sink;
        }
        std::vector<double> sorted = tick;
        std::sort(sorted.begin(), sorted.end());
        out << c.est << ',' << c.approx << ',' << peak << ',' << mean(tick) << ','
            << sorted[static_cast<std::size_t>(0.95 * sorted.size())] << ',' << mean(upd) << '\n';
        std::fprintf(stderr, "%-12s %5d  %8zu B  %9.0f ns\n", c.est.c_str(), c.approx, peak, mean(tick));
    }
}

}  // namespace

int main(int argc, char** argv) {
    // copula_family.hpp builds the Student-t rho_S table lazily behind plain statics, which is
    // a data race under threads (it corrupted every t cell once). Build it here, single-threaded.
    (void)copula::spearman_from_param(copula::Family::StudentT, 0.5, 4.0);
    const std::string mode = argc > 1 ? argv[1] : "sweep";
    if (mode == "time") { timing(); return 0; }
    const std::size_t n_test = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 20;
    const std::size_t n_val = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 20;
    std::vector<std::string> fams{"gaussian", "clayton", "gumbel", "t"};
    if (argc > 4) fams = {argv[4]};   // one family only; output files get a _<family> suffix
    sweep(n_test, n_val, fams);
}
