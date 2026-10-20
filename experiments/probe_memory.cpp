// Memory sweep: RMSE decomposed into bias^2 and variance against effective
// memory length, on one tick axis shared by Xiao, Hermite and the Kalman filter.
// See notes/studies/memory_sweep.md.
//
// Usage: ./probe_memory          -- the sweep -> results/memory_sweep[_gap].csv
//        ./probe_memory ess      -- ESS-vs-half-life check -> memory_sweep_ess.csv
//
// Configs are built here rather than registered: 50 keys in registry.hpp would
// be run by every other harness too.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../src/estimators/dependence/estimator.hpp"
#include "../src/estimators/dependence/hermiter.hpp"
#include "../src/estimators/dependence/kalman_local_level.hpp"
#include "../src/estimators/dependence/xiao_spearman.hpp"
#include "../src/streamgenerator/datasource/copula.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"
#include "trajectory_manifest.hpp"

namespace {

constexpr std::size_t kN = 5000;      // same as main.cpp / multiseed.cpp
constexpr std::size_t kBurn = 500;
constexpr std::size_t kJump = kN / 2;
constexpr std::size_t kBlock = 20;
constexpr std::size_t kSeeds = 40;
constexpr unsigned kBaseSeed = 42;
constexpr std::size_t kRecovery = 1000;

// Variance-sense effective sample size of an exponential filter with update
// weight a is (2-a)/a observations; inverting it is how the two mirror-image
// lambda conventions are put on one axis.
double alpha_from_ess(double ess) { return 2.0 / (ess + 1.0); }
double ess_from_alpha(double a) { return (2.0 - a) / a; }

double kalman_gain(double sigma_omega2) {
    const double sigma_v2 = 1.06 / std::max(static_cast<double>(kBlock) - 3.0, 1.0);
    const double q = sigma_omega2 / sigma_v2;
    return 0.5 * (std::sqrt(q * q + 4.0 * q) - q);
}

struct Cfg {
    std::string family;  // xiao_ew | hermite_ew | kalman
    double param;        // lambda (xiao: retention, hermite: update weight) or sigma_omega2
    double gain;         // per-observation update weight
    double obs_ticks;    // ticks consumed per observation
    double mem_ticks;
    double halflife_pred;
};

std::string cfg_name(const Cfg& c) { return traj::tag("%s@%.17g", c.family.c_str(), c.param); }

Cfg make_cfg(const std::string& family, double param) {
    Cfg c;
    c.family = family;
    c.param = param;
    if (family == "xiao_ew") {
        c.gain = 1.0 - param;  // registry.hpp: Xiao lambda is RETENTION
        c.obs_ticks = 1.0;
    } else if (family == "hermite_ew") {
        c.gain = param;  // registry.hpp: Hermite lambda is the UPDATE WEIGHT
        c.obs_ticks = 1.0;
    } else {
        c.gain = kalman_gain(param);
        c.obs_ticks = static_cast<double>(kBlock);
    }
    c.mem_ticks = ess_from_alpha(c.gain) * c.obs_ticks;
    c.halflife_pred = std::log(0.5) / std::log(1.0 - c.gain) * c.obs_ticks;
    return c;
}

std::unique_ptr<DependenceEstimator> build(const Cfg& c) {
    if (c.family == "xiao_ew") {
        return std::make_unique<XiaoSpearmanEstimator>(
            XiaoSpearmanEstimator::Config{.bins = 20, .window = 0, .lambda = c.param,
                                          .calibration = 100},
            "xiao_ew");
    }
    if (c.family == "hermite_ew") {
        return std::make_unique<HermiteEstimator>(
            HermiteEstimator::Config{.order = 10, .lambda = c.param, .burn_in = 50},
            "hermite_ew");
    }
    return std::make_unique<KalmanLocalLevel>(
        KalmanLocalLevel::Config{.block = kBlock, .sigma_omega2 = c.param, .burn_in_blocks = 2},
        "kalman");
}

std::vector<double> log_grid(double lo, double hi, int n) {
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i) {
        const double f = (n == 1) ? 0.0 : static_cast<double>(i) / (n - 1);
        v[i] = std::exp(std::log(lo) + f * (std::log(hi) - std::log(lo)));
    }
    return v;
}

// Xiao and Hermite share the ESS grid exactly, so Figure 2's gap is a paired
// comparison at matched memory rather than an interpolation.
const std::vector<double>& ess_targets() {
    static const std::vector<double> v = log_grid(25.0, 30000.0, 20);
    return v;
}

std::vector<Cfg> build_cfgs() {
    std::vector<Cfg> v;
    for (double e : ess_targets()) v.push_back(make_cfg("xiao_ew", 1.0 - alpha_from_ess(e)));
    for (double e : ess_targets()) v.push_back(make_cfg("hermite_ew", alpha_from_ess(e)));
    for (double q : log_grid(1e-7, 1e0, 10)) v.push_back(make_cfg("kalman", q));
    return v;
}

std::unique_ptr<Dependence> make_dependence(const std::string& scenario) {
    if (scenario == "constant") return std::make_unique<ConstantDependence>(0.6);
    if (scenario == "drift") return std::make_unique<DriftDependence>(-0.5, 0.8, kN);
    if (scenario == "jump") return std::make_unique<JumpDependence>(0.2, 0.8, kJump);
    if (scenario == "oscillate") return std::make_unique<OscillatingDependence>(0.6, 400.0, 0.0);
    return nullptr;
}

// est[cfg][seed][t]
using Panel = std::vector<std::vector<std::vector<double>>>;

void run_seed(const std::string& scenario, const std::vector<Cfg>& cfgs, std::size_t s,
              Panel& est, std::vector<double>& truth, bool record_truth) {
    auto dependence = make_dependence(scenario);
    auto copula = dgp::make_copula("gaussian");
    CopulaSource source(*dependence, *copula, kBaseSeed + static_cast<unsigned>(s));

    std::vector<std::unique_ptr<DependenceEstimator>> ests;
    ests.reserve(cfgs.size());
    for (const Cfg& c : cfgs) ests.push_back(build(c));
    for (std::size_t c = 0; c < cfgs.size(); ++c) est[c][s].resize(kN);

    for (std::size_t t = 0; t < kN; ++t) {
        const CopulaSource::Sample smp = source.next();
        if (record_truth) truth[t] = smp.rho_s;
        for (std::size_t c = 0; c < ests.size(); ++c) {
            ests[c]->update(smp.x, smp.y);
            est[c][s][t] = ests[c]->estimate();
        }
    }
}

struct Decomp {
    double bias2 = 0.0, var = 0.0, mse = 0.0, rmse = 0.0, se_rmse = 0.0;
    std::vector<double> seed_rmse;
};

std::size_t g_assert_checks = 0;
double g_assert_worst = 0.0;

Decomp decompose(const Panel& est, const std::vector<double>& truth, std::size_t c,
                 std::size_t lo, std::size_t hi) {
    const double S = static_cast<double>(kSeeds);
    Decomp d;
    const std::size_t n = hi - lo;
    for (std::size_t t = lo; t < hi; ++t) {
        double sum = 0.0;
        for (std::size_t s = 0; s < kSeeds; ++s) sum += est[c][s][t];
        const double mean_t = sum / S;
        const double bias = mean_t - truth[t];
        double var_t = 0.0, mse_t = 0.0;
        for (std::size_t s = 0; s < kSeeds; ++s) {
            const double dv = est[c][s][t] - mean_t;
            const double dm = est[c][s][t] - truth[t];
            var_t += dv * dv;
            mse_t += dm * dm;
        }
        var_t /= S;
        mse_t /= S;

        // The decomposition is an identity, so a mismatch means the accumulator
        // is wrong, not that the estimator is interesting.
        const double resid = std::fabs(mse_t - bias * bias - var_t);
        const double tol = 1e-9 * std::max(1.0, mse_t);
        g_assert_worst = std::max(g_assert_worst, resid / std::max(1.0, mse_t));
        ++g_assert_checks;
        if (resid > tol) {
            std::fprintf(stderr, "decomposition identity failed at t=%zu: mse=%.17g "
                                 "bias2=%.17g var=%.17g resid=%.3g\n",
                         t, mse_t, bias * bias, var_t, resid);
            std::exit(1);
        }

        d.bias2 += bias * bias;
        d.var += var_t;
        d.mse += mse_t;
    }
    d.bias2 /= static_cast<double>(n);
    d.var /= static_cast<double>(n);
    d.mse /= static_cast<double>(n);
    d.rmse = std::sqrt(d.mse);

    d.seed_rmse.resize(kSeeds);
    for (std::size_t s = 0; s < kSeeds; ++s) {
        double se = 0.0;
        for (std::size_t t = lo; t < hi; ++t) {
            const double dm = est[c][s][t] - truth[t];
            se += dm * dm;
        }
        d.seed_rmse[s] = std::sqrt(se / static_cast<double>(n));
    }
    double m = 0.0;
    for (double v : d.seed_rmse) m += v;
    m /= S;
    double v2 = 0.0;
    for (double v : d.seed_rmse) v2 += (v - m) * (v - m);
    d.se_rmse = std::sqrt(v2 / (S - 1.0) / S);
    return d;
}

struct Window {
    std::string name;
    std::size_t lo, hi;
};

std::vector<Window> windows_for(const std::string& scenario) {
    std::vector<Window> w{{"all", kBurn, kN}};
    if (scenario == "jump") {
        w.push_back({"settled", kJump + kRecovery, kN});
        w.push_back({"recovery", kJump, kJump + kRecovery});
    }
    return w;
}

void sweep() {
    const std::vector<Cfg> cfgs = build_cfgs();
    const std::vector<std::string> scenarios{"constant", "jump", "drift", "oscillate"};

    std::ofstream out("results/memory_sweep.csv");
    std::ofstream gout("results/memory_sweep_gap.csv");
    if (!out.is_open() || !gout.is_open()) {
        std::fprintf(stderr, "cannot write results/ (run: mkdir -p results)\n");
        std::exit(1);
    }
    out << "estimator,scenario,window,param,mem_ticks,bias2,var,mse,rmse,se_rmse\n";
    gout << "scenario,window,mem_ticks,param_xiao,param_hermite,rmse_xiao,rmse_hermite,"
            "gap_mean,gap_se,t_stat\n";

    const std::size_t n_ess = ess_targets().size();
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());

    for (const std::string& scenario : scenarios) {
        Panel est(cfgs.size(), std::vector<std::vector<double>>(kSeeds));
        std::vector<double> truth(kN);
        run_seed(scenario, cfgs, 0, est, truth, true);

        std::vector<std::thread> pool;
        std::vector<std::vector<double>> ignored(hw, std::vector<double>(kN));
        for (unsigned w = 0; w < hw; ++w) {
            pool.emplace_back([&, w] {
                for (std::size_t s = 1 + w; s < kSeeds; s += hw)
                    run_seed(scenario, cfgs, s, est, ignored[w], false);
            });
        }
        for (auto& th : pool) th.join();
        if (traj::on()) {
            traj::record("probe_memory/" + scenario + "/truth", truth);
            for (std::size_t c = 0; c < cfgs.size(); ++c)
                for (std::size_t s = 0; s < kSeeds; ++s)
                    traj::record(traj::tag("probe_memory/%s/%s/%zu", scenario.c_str(), cfg_name(cfgs[c]).c_str(), s),
                                 est[c][s]);
        }

        std::vector<std::vector<Decomp>> dec(cfgs.size());
        const std::vector<Window> wins = windows_for(scenario);
        for (std::size_t c = 0; c < cfgs.size(); ++c) {
            for (const Window& w : wins) {
                Decomp d = decompose(est, truth, c, w.lo, w.hi);
                if (traj::on()) {
                    std::vector<double> v{d.bias2, d.var, d.mse, d.rmse, d.se_rmse};
                    v.insert(v.end(), d.seed_rmse.begin(), d.seed_rmse.end());
                    traj::record(traj::tag("probe_memory/%s/%s/decomp_%s", scenario.c_str(),
                                           cfg_name(cfgs[c]).c_str(), w.name.c_str()), v);
                }
                out << cfgs[c].family << "," << scenario << "," << w.name << ","
                    << cfgs[c].param << "," << cfgs[c].mem_ticks << "," << d.bias2 << ","
                    << d.var << "," << d.mse << "," << d.rmse << "," << d.se_rmse << "\n";
                dec[c].push_back(std::move(d));
            }
        }

        for (std::size_t wi = 0; wi < wins.size(); ++wi) {
            for (std::size_t i = 0; i < n_ess; ++i) {
                const Decomp& dx = dec[i][wi];
                const Decomp& dh = dec[n_ess + i][wi];
                double m = 0.0;
                for (std::size_t s = 0; s < kSeeds; ++s) m += dx.seed_rmse[s] - dh.seed_rmse[s];
                m /= static_cast<double>(kSeeds);
                double v = 0.0;
                for (std::size_t s = 0; s < kSeeds; ++s) {
                    const double e = (dx.seed_rmse[s] - dh.seed_rmse[s]) - m;
                    v += e * e;
                }
                const double se = std::sqrt(v / (kSeeds - 1.0) / static_cast<double>(kSeeds));
                gout << scenario << "," << wins[wi].name << "," << cfgs[i].mem_ticks << ","
                     << cfgs[i].param << "," << cfgs[n_ess + i].param << "," << dx.rmse << ","
                     << dh.rmse << "," << m << "," << se << "," << (se > 0.0 ? m / se : 0.0)
                     << "\n";
            }
        }
        std::printf("scenario=%s done (%zu configs x %zu seeds)\n", scenario.c_str(),
                    cfgs.size(), kSeeds);
    }
    std::printf("\ndecomposition identity mse = bias^2 + var checked %zu times, "
                "worst relative residual %.3g -- PASS\n", g_assert_checks, g_assert_worst);
    std::printf("wrote results/memory_sweep.csv and results/memory_sweep_gap.csv\n");
}

// Step response: the algebra above predicts a half-life; this measures one.
void ess_check() {
    const std::vector<Cfg> cfgs = build_cfgs();
    const std::size_t S = 24;
    std::ofstream out("results/memory_sweep_ess.csv");
    if (!out.is_open()) {
        std::fprintf(stderr, "cannot write results/ (run: mkdir -p results)\n");
        std::exit(1);
    }
    out << "estimator,param,gain,mem_ticks,halflife_pred,halflife_rho,halflife_z,"
           "ratio_rho,ratio_z,ess_meas\n";
    std::printf("%-12s %12s %10s %10s %10s %10s %8s %8s\n", "estimator", "param", "mem_ticks",
                "hl_pred", "hl_rho", "hl_z", "r_rho", "r_z");

    for (const Cfg& c : cfgs) {
        const std::size_t pre = std::min<std::size_t>(
            200000, std::max<std::size_t>(2000, static_cast<std::size_t>(8.0 * c.halflife_pred)));
        const std::size_t post = std::min<std::size_t>(
            300000, std::max<std::size_t>(3000, static_cast<std::size_t>(8.0 * c.halflife_pred)));
        const std::size_t T = pre + post;

        std::vector<std::vector<double>> mean(2, std::vector<double>(T, 0.0));
        std::vector<std::thread> pool;
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::vector<std::vector<double>>> acc(
            hw, std::vector<std::vector<double>>(2, std::vector<double>(T, 0.0)));
        for (unsigned w = 0; w < hw; ++w) {
            pool.emplace_back([&, w] {
                for (std::size_t s = w; s < S; s += hw) {
                    JumpDependence dep(0.2, 0.8, pre);
                    auto cop = dgp::make_copula("gaussian");
                    CopulaSource src(dep, *cop, 1000 + static_cast<unsigned>(s));
                    auto e = build(c);
                    traj::Series ser;
                    for (std::size_t t = 0; t < T; ++t) {
                        const CopulaSource::Sample smp = src.next();
                        e->update(smp.x, smp.y);
                        const double r = e->estimate();
                        ser.add(r);
                        acc[w][0][t] += r;
                        acc[w][1][t] += std::atanh(std::clamp(r, -0.999, 0.999));
                    }
                    if (traj::on()) ser.commit(traj::tag("probe_memory/ess/%s/%zu", cfg_name(c).c_str(), s));
                }
            });
        }
        for (auto& th : pool) th.join();
        for (unsigned w = 0; w < hw; ++w)
            for (int k = 0; k < 2; ++k)
                for (std::size_t t = 0; t < T; ++t) mean[k][t] += acc[w][k][t] / static_cast<double>(S);

        double hl[2] = {-1.0, -1.0};
        for (int k = 0; k < 2; ++k) {
            const std::size_t pre_win = std::max<std::size_t>(1, pre / 4);
            double lo_level = 0.0;
            for (std::size_t t = pre - pre_win; t < pre; ++t) lo_level += mean[k][t];
            lo_level /= static_cast<double>(pre_win);
            const std::size_t post_win = std::max<std::size_t>(1, post / 5);
            double hi_level = 0.0;
            for (std::size_t t = T - post_win; t < T; ++t) hi_level += mean[k][t];
            hi_level /= static_cast<double>(post_win);
            const double half = lo_level + 0.5 * (hi_level - lo_level);
            for (std::size_t t = pre; t < T; ++t) {
                if (mean[k][t] >= half) {
                    hl[k] = static_cast<double>(t - pre + 1);
                    break;
                }
            }
        }
        // Seed-mean summed per worker, so its bits depend on hardware_concurrency.
        if (traj::on())
            traj::record(traj::tag("probe_memory/ess/%s/halflife_hwdep", cfg_name(c).c_str()), hl, 2);
        // Invert the half-life back to an ESS so the measurement, not the
        // algebra, can carry the x-axis if the two disagree. Kalman is linear in
        // Fisher-z, so its z-column is the one to trust.
        const double hl_lin = (c.family == "kalman") ? hl[1] : hl[0];
        const double k_meas = 1.0 - std::pow(0.5, c.obs_ticks / std::max(hl_lin, 1e-9));
        const double ess_meas = ess_from_alpha(std::clamp(k_meas, 1e-12, 1.0)) * c.obs_ticks;
        std::printf("%-12s %12.6g %10.1f %10.1f %10.1f %10.1f %8.3f %8.3f\n", c.family.c_str(),
                    c.param, c.mem_ticks, c.halflife_pred, hl[0], hl[1],
                    hl[0] / c.halflife_pred, hl[1] / c.halflife_pred);
        out << c.family << "," << c.param << "," << c.gain << "," << c.mem_ticks << ","
            << c.halflife_pred << "," << hl[0] << "," << hl[1] << ","
            << hl[0] / c.halflife_pred << "," << hl[1] / c.halflife_pred << "," << ess_meas << "\n";
    }
    std::printf("\nwrote results/memory_sweep_ess.csv\n");
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = (argc > 1) ? argv[1] : "sweep";
    if (mode == "ess") {
        ess_check();
        return 0;
    }
    if (mode != "sweep") {
        std::fprintf(stderr, "usage: ./probe_memory [sweep|ess]\n");
        return 1;
    }
    sweep();
    return 0;
}
