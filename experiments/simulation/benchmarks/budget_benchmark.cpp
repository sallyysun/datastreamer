// Compare estimator accuracy and update time at matched storage budgets.
// Usage: budget_benchmark sweep <n_test> [n_val] [family] | budget_benchmark time

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "../../../src/estimators/dependence/estimator.hpp"
#include "../../../src/estimators/dependence/hermiter.hpp"
#include "../../../src/estimators/dependence/rmpl.hpp"
#include "../../../src/estimators/dependence/count_matrix.hpp"
#include "../../../src/streamgenerator/datasource/copula.hpp"
#include "../../../src/streamgenerator/datasource/dependence.hpp"

namespace {

constexpr std::size_t kBurn = 500;
constexpr std::size_t kEvery = 5;          // score every 5th tick
constexpr std::size_t kPost = 1000;        // post-change window
constexpr unsigned kTestSeed = 42;
constexpr unsigned kValSeed = 1000;

std::vector<std::string> csv_values(std::string_view csv) {
    std::vector<std::string> values;
    for (std::size_t first = 0; first < csv.size();) {
        const auto comma = csv.find(',', first);
        const auto end = comma == std::string_view::npos ? csv.size() : comma;
        if (end > first) values.emplace_back(csv.substr(first, end - first));
        first = end + 1;
    }
    return values;
}

struct BudgetOptions {
    std::size_t observations = 5000;
    unsigned workers = std::max(1u, std::thread::hardware_concurrency());
    std::vector<double> gains{0.005, 0.01, 0.02, 0.05, 0.10};
    std::vector<int> kll_sizes{8, 16, 32, 64, 128, 256, 512, 1024};
    std::vector<std::string> estimators;
    std::optional<std::vector<std::string>> scenarios;
    double timing_gain = 0.02;
    std::string tag_suffix;
};

BudgetOptions read_budget_options() {
    BudgetOptions options;
    if (const char* n = std::getenv("BUDGET_N")) options.observations = std::strtoul(n, nullptr, 10);
    if (const char* workers = std::getenv("BUDGET_THREADS")) options.workers = std::max(1, std::atoi(workers));
    if (const char* gain = std::getenv("BUDGET_TIME_GAIN")) options.timing_gain = std::atof(gain);
    if (const char* tag = std::getenv("TAG")) options.tag_suffix = std::string("_") + tag;
    if (const char* estimators = std::getenv("BUDGET_EST")) options.estimators = csv_values(estimators);
    if (const char* scenarios = std::getenv("BUDGET_SCENARIOS")) options.scenarios = csv_values(scenarios);
    if (const char* gains = std::getenv("BUDGET_GAINS")) {
        options.gains.clear();
        for (const auto& value : csv_values(gains)) options.gains.push_back(std::atof(value.c_str()));
    }
    if (const char* sizes = std::getenv("BUDGET_KLL")) {
        const auto values = csv_values(sizes);
        if (!values.empty()) {
            options.kll_sizes.clear();
            for (const auto& value : values) options.kll_sizes.push_back(std::atoi(value.c_str()));
        }
    }
    return options;
}

struct Cfg {
    std::string est;    // xiao | hermite | mpl_kll | mpl_dskll | mpl_exact
    int approx;         // B, N, k or W
    double gain;        // per-observation update weight, all estimators
};

std::vector<Cfg> build_cfgs(const BudgetOptions& options) {
    auto est_on = [&](std::string_view name) {
        return options.estimators.empty() || std::find(options.estimators.begin(), options.estimators.end(), name) != options.estimators.end();
    };
    std::vector<Cfg> v;
    for (double g : options.gains) {
        if (est_on("xiao")) for (int b : {10, 20, 30, 45, 65, 90, 130, 180}) v.push_back({"xiao", b, g});
        if (est_on("hermite")) for (int n : {5, 10, 15, 20, 30, 43, 60, 89, 128}) v.push_back({"hermite", n, g});
        if (est_on("mpl_kll")) for (int k : options.kll_sizes) v.push_back({"mpl_kll", k, g});
        if (est_on("mpl_dskll")) for (int k : options.kll_sizes) v.push_back({"mpl_dskll", k, g});
        if (est_on("mpl_exact")) v.push_back({"mpl_exact", 0, g});
    }
    return v;
}

std::unique_ptr<DependenceEstimator> build(const Cfg& c, unsigned seed) {
    if (c.est == "xiao") {
        CountingMatrixEstimator::Config x{};
        x.bins = static_cast<std::size_t>(c.approx);
        x.lambda = c.gain > 0.0 ? 1.0 - c.gain : 0.0;   // 0 = cumulative
        return std::make_unique<CountingMatrixEstimator>(x, "xiao");
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

struct Data { std::vector<double> x, y, truth; };

Data simulate(const std::string& family, const std::string& scenario, unsigned seed, std::size_t n) {
    auto dep = make_scenario(scenario, n);
    auto cop = dgp::make_copula(family);
    CopulaSource src(*dep, *cop, seed);
    Data d;
    d.x.resize(n); d.y.resize(n); d.truth.resize(n);
    for (std::size_t t = 0; t < n; ++t) {
        const auto s = src.next();
        d.x[t] = s.x; d.y[t] = s.y; d.truth[t] = s.true_spearman;
    }
    return d;
}

std::size_t scored_index_at_or_after(std::size_t t) {
    return t <= kBurn ? 0 : 1 + (t - kBurn - 1) / kEvery;
}

struct Acc {
    std::vector<double> rmse, post;   // per seed
    std::vector<double> path;         // summed estimate at scored ticks
    std::size_t peak = 0;
};

void run_split(const std::string& family, const std::string& scenario, unsigned base,
               std::size_t n_seeds, const BudgetOptions& options, const std::vector<Cfg>& cfgs, std::vector<Acc>& acc,
               std::vector<double>& truth_path) {
    const auto n = options.observations;
    const auto jump = n / 2;
    const auto points = scored_index_at_or_after(n);
    acc.assign(cfgs.size(), Acc{});
    for (auto& a : acc) { a.rmse.assign(n_seeds, 0.0); a.post.assign(n_seeds, 0.0); a.path.assign(points, 0.0); }
    truth_path.assign(points, 0.0);
    std::mutex mu;
    // BUDGET_THREADS caps the worker threads (harness only; on a batch cluster set it to the allocated cores).
    const unsigned hw = options.workers;
    std::vector<std::thread> pool;
    for (unsigned w = 0; w < hw; ++w) {
        pool.emplace_back([&, w] {
            std::vector<std::vector<double>> path(cfgs.size(), std::vector<double>(points, 0.0));
            std::vector<std::size_t> peak(cfgs.size(), 0);
            std::vector<double> tp(points, 0.0);
            for (std::size_t s = w; s < n_seeds; s += hw) {
                const unsigned seed = base + static_cast<unsigned>(s);
                const Data d = simulate(family, scenario, seed, n);
                for (std::size_t c = 0; c < cfgs.size(); ++c) {
                    auto e = build(cfgs[c], seed);
                    double se = 0.0, sp = 0.0;
                    std::size_t ne = 0, np = 0;
                    for (std::size_t t = 0; t < n; ++t) {
                        e->update(d.x[t], d.y[t]);
                        peak[c] = std::max(peak[c], e->store_bytes());   // every tick, start-up included
                        if (t < kBurn || (t - kBurn) % kEvery != 0) continue;
                        const double r = e->estimate(), err = r - d.truth[t];
                        se += err * err; ++ne;
                        if (t >= jump && t < jump + kPost) { sp += err * err; ++np; }
                        path[c][(t - kBurn) / kEvery] += r;
                    }
                    acc[c].rmse[s] = std::sqrt(se / ne);
                    acc[c].post[s] = std::sqrt(sp / np);
                }
                for (std::size_t i = 0; i < points; ++i) tp[i] += d.truth[kBurn + i * kEvery];
            }
            std::lock_guard<std::mutex> lk(mu);
            for (std::size_t c = 0; c < cfgs.size(); ++c) {
                for (std::size_t i = 0; i < points; ++i) acc[c].path[i] += path[c][i];
                acc[c].peak = std::max(acc[c].peak, peak[c]);
            }
            for (std::size_t i = 0; i < points; ++i) truth_path[i] += tp[i];
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
double recovery_delay(const std::vector<double>& path, std::size_t observations) {
    constexpr std::size_t window = 1000;
    const auto jump_tick = observations / 2;
    const std::size_t jump = scored_index_at_or_after(jump_tick);
    if (jump == 0 || jump >= path.size()) return -1.0;

    const std::size_t pre_start = scored_index_at_or_after(jump_tick > window ? jump_tick - window : 0);
    const std::size_t post_start = scored_index_at_or_after(std::max(jump_tick, observations > window ? observations - window : 0));
    auto avg = [&](std::size_t first, std::size_t last) {
        double sum = 0.0;
        for (std::size_t i = first; i < last; ++i) sum += path[i];
        return sum / (last - first);
    };
    const double pre = avg(pre_start, jump), post = avg(post_start, path.size());
    const double target = pre + 0.9 * (post - pre);
    for (std::size_t i = jump; i < path.size(); ++i) {
        const double r = path[i];
        if ((post > pre && r >= target) || (post <= pre && r <= target))
            return double(kBurn + i * kEvery - jump_tick);
    }
    return -1.0;
}

void sweep(const BudgetOptions& options, const std::vector<Cfg>& cfgs, std::size_t n_test, std::size_t n_val,
           const std::vector<std::string>& families) {
    const std::string sfx = families.size() == 1 ? "_" + families[0] : "";
    const auto& tg = options.tag_suffix;
    std::ofstream out("results/budget_sweep" + tg + sfx + ".csv");
    std::ofstream sout("results/budget_seeds" + tg + sfx + ".csv");   // per-seed test RMSE, for paired comparisons
    sout << "family,scenario,estimator,approx,gain,seed,rmse,post_rmse\n";
    out << "family,scenario,split,estimator,approx,gain,peak_bytes,rmse,se_rmse,post_rmse,se_post,"
           "recovery_delay\n";
    std::ofstream pout("results/budget_paths" + tg + sfx + ".csv");
    pout << "family,scenario,estimator,approx,gain,t,est,truth\n";
    for (const std::string& fam : families) {
        for (const std::string sc : {"constant", "drift", "jump"}) {
            if (options.scenarios &&
                std::find(options.scenarios->begin(), options.scenarios->end(), sc) == options.scenarios->end()) continue;
            for (int split = 0; split < 2; ++split) {
                const unsigned base = split ? kTestSeed : kValSeed;
                const std::size_t ns = split ? n_test : n_val;
                std::vector<Acc> acc;
                std::vector<double> truth;
                run_split(fam, sc, base, ns, options, cfgs, acc, truth);
                for (std::size_t c = 0; c < cfgs.size(); ++c) {
                    const auto& a = acc[c];
                    out << fam << ',' << sc << ',' << (split ? "test" : "val") << ',' << cfgs[c].est << ','
                        << cfgs[c].approx << ',' << cfgs[c].gain << ',' << a.peak << ',' << mean(a.rmse) << ','
                        << se(a.rmse) << ',' << mean(a.post) << ',' << se(a.post) << ','
                        << (sc == "jump" ? recovery_delay(a.path, options.observations) : -1.0) << '\n';
                    if (split)
                        for (std::size_t s = 0; s < ns; ++s)
                            sout << fam << ',' << sc << ',' << cfgs[c].est << ',' << cfgs[c].approx << ',' << cfgs[c].gain
                                 << ',' << base + s << ',' << a.rmse[s] << ',' << a.post[s] << '\n';
                    if (split && sc == "jump")
                        for (std::size_t i = 0; i < a.path.size(); i += 2)
                            pout << fam << ',' << sc << ',' << cfgs[c].est << ',' << cfgs[c].approx << ','
                                 << cfgs[c].gain << ',' << kBurn + i * kEvery << ',' << a.path[i] << ','
                                 << truth[i] << '\n';
                }
            }
        }
    }
}

// Single thread, Gaussian drift, gain 0.02: wall time of update() + estimate() per tick, which
// is the whole online pipeline per observation. Data are generated before the clock starts.
void timing(const BudgetOptions& options, const std::vector<Cfg>& cfgs) {
    std::ofstream out("results/budget_timing" + options.tag_suffix + ".csv");
    out << "estimator,approx,peak_bytes,mean_ns,p95_ns,update_mean_ns\n";
    for (Cfg c : cfgs) {
        if (c.gain != options.gains.front()) continue;
        c.gain = options.timing_gain;
        std::vector<double> tick, upd;
        std::size_t peak = 0;
        for (unsigned s = 0; s < 5; ++s) {
            const Data d = simulate("gaussian", "drift", kTestSeed + s, options.observations);
            auto e = build(c, kTestSeed + s);
            volatile double sink = 0.0;
            for (std::size_t t = 0; t < options.observations; ++t) {
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
    }
}

}  // namespace

int main(int argc, char** argv) {
    // copula_family.hpp builds the Student-t rho_S table lazily behind plain statics, which is
    // a data race under threads (it corrupted every t cell once). Build it here, single-threaded.
    (void)copula::spearman_from_param(copula::Family::StudentT, 0.5, 4.0);
    const auto options = read_budget_options();
    const auto cfgs = build_cfgs(options);
    const std::string mode = argc > 1 ? argv[1] : "sweep";
    if (mode == "time") { timing(options, cfgs); return 0; }
    const std::size_t n_test = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 20;
    const std::size_t n_val = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 20;
    std::vector<std::string> fams{"gaussian", "clayton", "gumbel", "t"};
    if (argc > 4) fams = {argv[4]};   // one family only; output files get a _<family> suffix
    sweep(options, cfgs, n_test, n_val, fams);
}
