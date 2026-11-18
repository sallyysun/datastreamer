// Compare fixed memories on matched streams. RMPL shares KLL margins fitted on prior observations.
// Usage: COPULA=<family> fixed_ess_sweep [seeds=60]
// VALIDATE=1 compares shared and internal margins; JUMPBIAS=1 writes only the jump bias path.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "../../src/estimators/dependence/datasketches_kll.hpp"
#include "../../src/special_functions.hpp"
#include "../../src/estimators/evaluation.hpp"
#include "../../src/estimators/registry.hpp"
#include "study_stream.hpp"

namespace {
using thesis::kN; using thesis::kBurn;
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
        for (double n : thesis::kMem) a.push_back({e, "main", n});
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

struct Stream : StudySeries { std::vector<double> zx, zy; };

Stream make_stream(const std::string& family, const std::string& scenario, unsigned seed) {
    Stream stream{load_study_series(family, scenario, seed, kN), {}, {}};
    for (std::size_t t = 0; t < stream.estimated_pit_x.size(); ++t) {
        stream.zx.push_back(special::probit(stream.estimated_pit_x[t]));
        stream.zy.push_back(special::probit(stream.estimated_pit_y[t]));
    }
    return stream;
}

std::string label(double n) { return n == kNoForget ? "nocum" : std::to_string(n); }
}  // namespace

int main(int argc, char** argv) {
    const int seeds = argc > 1 ? std::atoi(argv[1]) : thesis::kSeedCount;
    const char* env = std::getenv("COPULA");
    const std::string cop = env ? env : "gaussian";
    const std::string sfx = cop == "gaussian" ? "" : "_" + cop;

    if (std::getenv("VALIDATE")) {
        const Stream s = make_stream(cop, "drift", thesis::kFirstSeed);
        for (double n : {39.0, 399.0, kNoForget}) {
            auto fed = build({"rmpl", "main", n});
            RecursiveMPL::Config m = thesis::rmpl(n == kNoForget ? 1e-12 : 2 / (n + 1));
            m.margin = RecursiveMPL::Margin::DsKll; m.kll_k = thesis::kKll;
            RecursiveMPL own(m, "own");          // constructor reseeds the coin with seed 1, as make_stream does
            double worst = 0;
            for (std::size_t t = 0; t < kN; ++t) {
                worst = std::max(worst, std::fabs(fed->estimate() - own.estimate()));
                fed->update(s.zx[t], s.zy[t]); own.update(s.x[t], s.y[t]);
            }
            std::printf("VALIDATE %s N=%s: max |cached-feed - own DsKll| over %zu ticks = %.3g\n",
                        cop.c_str(), label(n).c_str(), kN, worst);
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
        const auto wins = thesis::windows(sc);
        std::vector<std::vector<double>> s1(arms.size(), std::vector<double>(kN)), s2 = s1;   // sums over seeds
        for (int sd = 0; sd < seeds; ++sd) {
            const unsigned seed = thesis::kFirstSeed + sd;
            const Stream s = make_stream(cop, sc, seed);
            std::vector<double> e(kN);
            for (std::size_t k = 0; k < arms.size(); ++k) {
                auto est = build(arms[k]);
                const bool rmpl = arms[k].est == "rmpl";
                for (std::size_t t = 0; t < kN; ++t) {
                    e[t] = est->estimate() - s.true_spearman[t];   // forecast held before tick t
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
    }
    if (study_verbose())
        std::printf("Decomposition residual max %.3g (%zu checks)\n", worst_resid, checks);
    std::fclose(fs); std::fclose(fd); std::fclose(fj);
    if (fb) std::fclose(fb);
}
