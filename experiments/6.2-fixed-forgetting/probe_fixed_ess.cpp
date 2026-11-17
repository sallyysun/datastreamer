// Chapter 6.2 fixed-forgetting experiments (constant, drift, jump; four families; 60 seeds 42-101; matched paths).
// One stream per (family, scenario, seed), generated once and reused by every estimator and memory. RMPL margins are
// two cumulative DataSketches KLL sketches (k = 256), computed once per stream from the state before each pair and
// fed to every RMPL arm (identical pseudo-observations across arms; the arm's ExactNormal input maps z back to u).
// Counting matrix and Hermite arms are thesis::arm (src/estimators/registry.hpp), raw inputs.
// Forecast held before tick t; window t = 500-4999 (jump also pre, 0-100, 100-500, post).
// Memories: main {39, 99, 199, 399, 1999, no forgetting}; drift adds a fine grid (21 log-spaced, 200-600) and the
// memory-sweep grid (20 log-spaced, 25-30000) for the bias-variance analysis and the approximation's variance scale.
//
// Usage: COPULA=<family> ./probe_fixed_ess [seeds=60]
//   -> results/fixed_ess_seeds[_fam].csv   copula,scenario,estimator,grid,neff,seed,window,rmse,mean_err
//      results/fixed_ess_decomp[_fam].csv  copula,scenario,estimator,grid,neff,window,bias2,var,mse
//      results/fixed_ess_jumppath[_fam].csv copula,estimator,neff,t,mse   (main memories, jump path)
//   VALIDATE=1: seed 42, Gaussian drift: a RecursiveMPL with its own DsKll k = 256 margins against the cached feed.
//   JUMPBIAS=1: the jump scenario only, writing just results/fixed_ess_jumpbias[_fam].csv
//               (copula,estimator,neff,t,mean_err,mse: per-tick mean over seeds of the error and squared error).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "../../src/estimators/dependence/ds_margin.hpp"
#include "../../src/estimators/dependence/marginal_rank_pit.hpp"
#include "../../src/estimators/registry.hpp"
#include "../../src/streamgenerator/datasource/copula.hpp"
#include "../../src/streamgenerator/datasource/dependence.hpp"
#include "../common/copula_source.hpp"
#include "../common/live_source.hpp"

namespace {
constexpr std::size_t kN = 5000, kBurn = 500, kChange = 2500;
constexpr double kDelta = 1e-3;          // RecursiveMPL's clip
constexpr std::uint16_t kK = 256;
constexpr double kNoForget = -1.0;       // neff marker for the no-forgetting arm

struct Arm { std::string est, grid; double neff; };

std::vector<double> log_grid(double lo, double hi, int n) {
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i) v[i] = std::exp(std::log(lo) + (std::log(hi) - std::log(lo)) * i / (n - 1));
    return v;
}

std::vector<Arm> arms_for(const std::string& sc) {
    std::vector<Arm> a;
    for (const char* e : {"xiao", "hermite", "rmpl"}) {
        for (double n : {39.0, 99.0, 199.0, 399.0, 1999.0}) a.push_back({e, "main", n});
        a.push_back({e, "main", kNoForget});
        if (sc == "drift") {
            for (double n : log_grid(200, 600, 21)) a.push_back({e, "fine", n});
            for (double n : log_grid(25, 30000, 20)) a.push_back({e, "sweep", n});
        }
    }
    return a;
}

std::unique_ptr<DependenceEstimator> build(const Arm& a) {
    return thesis::arm(a.est, a.neff == kNoForget ? 0.0 : a.neff);
}

std::unique_ptr<Dependence> make_dep(const std::string& s) { return make_scenario(s, kN); }

struct Win { const char* name; std::size_t a, b; };
std::vector<Win> windows(const std::string& s) {
    std::vector<Win> w{{"whole", kBurn, kN}};
    if (s == "jump") {
        w.push_back({"pre", kBurn, kChange}); w.push_back({"rec_0_100", kChange, kChange + 100});
        w.push_back({"rec_100_500", kChange + 100, kChange + 500}); w.push_back({"post", kChange, kN});
    }
    return w;
}

struct Stream { std::vector<double> x, y, rho, zx, zy; };

Stream make_stream(const std::string& cop, const std::string& sc, unsigned seed) {
    LiveStream L;
    std::unique_ptr<Dependence> dep;
    std::unique_ptr<dgp::Copula> c;
    std::unique_ptr<CopulaSource> src;
    if (live_mode()) L = live_stream(cop, sc, seed, kN);
    else { dep = make_dep(sc); c = dgp::make_copula(cop); src = std::make_unique<CopulaSource>(*dep, *c, seed); }
    Stream s;
    ds_margin::seed(1);                       // as RecursiveMPL's DsKll margin (seed 1), so VALIDATE can match it
    ds_margin::Kll kx(kK), ky(kK);
    for (std::size_t t = 0; t < kN; ++t) {
        double x, y, rho_s;
        if (src) { const auto smp = src->next(); x = smp.x; y = smp.y; rho_s = smp.rho_s; }
        else { x = L.x[t]; y = L.y[t]; rho_s = L.rho_s[t]; }
        const double u = std::clamp(kx.cdf(x), kDelta, 1 - kDelta), v = std::clamp(ky.cdf(y), kDelta, 1 - kDelta);
        kx.insert(x); ky.insert(y);   // pseudo-observation from the sketch BEFORE this pair
        s.x.push_back(x); s.y.push_back(y); s.rho.push_back(rho_s);
        s.zx.push_back(RankPit::probit(u)); s.zy.push_back(RankPit::probit(v));
    }
    return s;
}

std::string label(double n) { return n == kNoForget ? "nocum" : std::to_string(n); }
}  // namespace

int main(int argc, char** argv) {
    const int seeds = argc > 1 ? std::atoi(argv[1]) : 60;
    const char* env = std::getenv("COPULA");
    const std::string cop = env ? env : "gaussian";
    const std::string sfx = cop == "gaussian" ? "" : "_" + cop;

    if (std::getenv("VALIDATE")) {
        const Stream s = make_stream(cop, "drift", 42);
        for (double n : {39.0, 399.0, kNoForget}) {
            auto fed = build({"rmpl", "main", n});
            RecursiveMPL::Config m = thesis::rmpl(n == kNoForget ? 1e-12 : 2 / (n + 1));
            m.margin = RecursiveMPL::Margin::DsKll; m.kll_k = kK;
            RecursiveMPL own(m, "own");          // constructor reseeds the coin with seed 1, as make_stream does
            double worst = 0;
            for (std::size_t t = 0; t < kN; ++t) {
                worst = std::max(worst, std::fabs(fed->estimate() - own.estimate()));
                fed->update(s.zx[t], s.zy[t]); own.update(s.x[t], s.y[t]);
            }
            std::printf("VALIDATE %s N=%s: max |cached-feed - own DsKll| over 5000 ticks = %.3g\n", cop.c_str(), label(n).c_str(), worst);
        }
        return 0;
    }

    const bool jb = std::getenv("JUMPBIAS") != nullptr;   // only the jump bias paths; the other outputs are left alone
    const std::string dn = jb ? "/dev/null" : "";
    std::FILE* fs = std::fopen(jb ? dn.c_str() : ("results/fixed_ess_seeds" + sfx + ".csv").c_str(), "w");
    std::FILE* fd = std::fopen(jb ? dn.c_str() : ("results/fixed_ess_decomp" + sfx + ".csv").c_str(), "w");
    std::FILE* fj = std::fopen(jb ? dn.c_str() : ("results/fixed_ess_jumppath" + sfx + ".csv").c_str(), "w");
    std::FILE* fb = jb ? std::fopen(("results/fixed_ess_jumpbias" + sfx + ".csv").c_str(), "w") : nullptr;
    if (!fs || !fd || !fj || (jb && !fb)) { std::fprintf(stderr, "run from the experiment folder (results/ must exist)\n"); return 1; }
    if (fb) std::fprintf(fb, "copula,estimator,neff,t,mean_err,mse\n");
    std::fprintf(fs, "copula,scenario,estimator,grid,neff,seed,window,rmse,mean_err\n");
    std::fprintf(fd, "copula,scenario,estimator,grid,neff,window,bias2,var,mse\n");
    std::fprintf(fj, "copula,estimator,neff,t,mse\n");
    std::size_t checks = 0; double worst_resid = 0;

    for (const std::string sc : {"constant", "drift", "jump"}) {
        if (jb && sc != "jump") continue;
        const auto arms = arms_for(sc);
        const auto wins = windows(sc);
        std::vector<std::vector<double>> s1(arms.size(), std::vector<double>(kN)), s2 = s1;   // sums over seeds
        for (int sd = 0; sd < seeds; ++sd) {
            const unsigned seed = 42u + sd;
            const Stream s = make_stream(cop, sc, seed);
            std::vector<double> e(kN);
            for (std::size_t k = 0; k < arms.size(); ++k) {
                auto est = build(arms[k]);
                const bool rmpl = arms[k].est == "rmpl";
                for (std::size_t t = 0; t < kN; ++t) {
                    e[t] = est->estimate() - s.rho[t];   // forecast held before tick t
                    if (rmpl) est->update(s.zx[t], s.zy[t]); else est->update(s.x[t], s.y[t]);
                    s1[k][t] += e[t]; s2[k][t] += e[t] * e[t];
                }
                for (const auto& w : wins) {
                    double a = 0, b = 0;
                    for (std::size_t t = w.a; t < w.b; ++t) { a += e[t] * e[t]; b += e[t]; }
                    const double n = double(w.b - w.a);
                    std::fprintf(fs, "%s,%s,%s,%s,%s,%u,%s,%.8f,%.8f\n", cop.c_str(), sc.c_str(), arms[k].est.c_str(),
                                 arms[k].grid.c_str(), label(arms[k].neff).c_str(), seed, w.name, std::sqrt(a / n), b / n);
                }
            }
        }
        const double S = seeds;
        for (std::size_t k = 0; k < arms.size(); ++k) {
            for (const auto& w : wins) {
                double b2 = 0, v = 0, m = 0;
                for (std::size_t t = w.a; t < w.b; ++t) {
                    const double mu = s1[k][t] / S, ms = s2[k][t] / S, var = ms - mu * mu;
                    const double resid = std::fabs(ms - mu * mu - var);   // identity, kept as a guard on the accumulators
                    worst_resid = std::max(worst_resid, resid); ++checks;
                    b2 += mu * mu; v += var; m += ms;
                }
                const double n = double(w.b - w.a);
                std::fprintf(fd, "%s,%s,%s,%s,%s,%s,%.10g,%.10g,%.10g\n", cop.c_str(), sc.c_str(), arms[k].est.c_str(),
                             arms[k].grid.c_str(), label(arms[k].neff).c_str(), w.name, b2 / n, v / n, m / n);
            }
            if (fb && sc == "jump")
                for (std::size_t t = kBurn; t < kN; ++t)
                    std::fprintf(fb, "%s,%s,%s,%zu,%.10g,%.10g\n", cop.c_str(), arms[k].est.c_str(), label(arms[k].neff).c_str(), t,
                                 s1[k][t] / S, s2[k][t] / S);
            if (sc == "jump" && arms[k].grid == "main")
                for (std::size_t t = kBurn; t < kN; ++t)
                    std::fprintf(fj, "%s,%s,%s,%zu,%.8g\n", cop.c_str(), arms[k].est.c_str(), label(arms[k].neff).c_str(), t, s2[k][t] / S);
        }
        std::printf("%s %s done (%zu arms x %d seeds)\n", cop.c_str(), sc.c_str(), arms.size(), seeds);
    }
    std::printf("decomposition guard: %zu checks, worst residual %.3g\n", checks, worst_resid);
    std::fclose(fs); std::fclose(fd); std::fclose(fj);
    if (fb) std::fclose(fb);
}
