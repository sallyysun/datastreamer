// Chapter 6.3.2 forgetting-strategy comparison on the Section 6.2 construction (probe_fixed_ess.cpp): one stream per
// (family, scenario, seed 42-101), one cumulative DataSketches KLL k = 256 pseudo-observation sequence per stream
// (clipped to [1e-3, 1 - 1e-3]) that feeds every margin-dependent part, for every strategy:
//   fixed  the five arms N_eff 39 ... 1999 (identical to probe_fixed_ess's main arms);
//   MFF    ForgettingBank's log-loss / fixed-share rule (eta = 1, delta = 1e-4) over those same five arms, scored on the
//          KLL pseudo-observations (replica of ForgettingBank::update; VALIDATE checks it against the class);
//   VFF    CommonVff's rule (VffController, rates 0.02 and 0.1) with the KLL p_n = 12(U - 1/2)(V - 1/2)
//          (replica of CommonVff::update; VALIDATE checks it).
// Counting matrix and Hermite arms read the raw data (their own margins); RMPL arms read z = probit(U_KLL).
// Metrics on the forecast held before each tick: rho_S RMSE; excess log loss log c_true(U,V) - log c_G(rho_hat)(U,V) at
// the true PIT; squared Hellinger H^2(c_true, c_G(rho_hat)) every 50 ticks from 500 (and every 10 in [2500, 2600) on the
// jump path, counted only in rec_0_100), read from a per-truth table over rho_hat (VALIDATE checks it). c_G(rho) is the
// Gaussian copula at Spearman rho (RMPL's own density; a plug-in working model for the counting matrix and Hermite).
//
// Usage (from experiments/): SCENARIOS=matched COPULA=<family> ./probe_strategies [seeds=60]
//   -> results/strategies_seeds[_fam].csv  copula,scenario,estimator,strategy,seed,window,rmse,excess_x1000,h2_x1000
//      results/strategies_jumppath[_fam].csv copula,estimator,strategy,t,mse
//   VALIDATE=1: replica MFF / VFF vs ForgettingBank / CommonVff on EW margins (seed 42), and the H^2 table vs direct.
//   VFF_RATES=0.005,0.01,... TAG=<tag>: VFF at those rates; outputs results/strategies_{seeds,jumppath}_<tag>[_fam].csv and
//            results/strategies_extremes_<tag>[_fam].csv (per scenario x estimator x strategy, whole window, all seeds x
//            ticks: excess mean / median / max, worst-1% share of the summed excess, seeds and seed-ticks with |rho_hat| > 0.98).
//   PATHS=1: the same runs, writing only results/strategies_paths[_fam].csv, per scenario x estimator x strategy x tick
//            (t = 500-4999): mse (mean over seeds of the squared error), ex_mean / ex_median / ex_max (excess log loss
//            over seeds, x 1000), h2_mean (x 1000, checkpoints only), n_extreme (seeds with |rho_hat| > 0.98).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../src/estimators/dependence/ds_margin.hpp"
#include "../src/estimators/dependence/ew_marginal.hpp"
#include "../src/estimators/dependence/marginal_rank_pit.hpp"
#include "../src/estimators/dependence/vff.hpp"
#include "../src/estimators/registry.hpp"
#include "../src/streamgenerator/datasource/copula.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"
#include "density_metrics.hpp"
#include "scenarios.hpp"

namespace {
constexpr std::size_t kN = 5000, kBurn = 500, kChange = 2500, kK = 5;
constexpr double kDelta = 1e-3;
const double kMem[kK] = {39, 99, 199, 399, 1999};
const char* kEst[3] = {"xiao", "hermite", "rmpl"};
// Strategies: five fixed arms, MFF, then one VFF per rate. Default rates 0.02 / 0.1 named vff_slow / vff_fast (the
// original outputs); VFF_RATES=a,b,... names them vff_<a>, ... (sensitivity runs; write under TAG=...).
std::vector<std::string> kStrat{"fixed_39", "fixed_99", "fixed_199", "fixed_399", "fixed_1999", "mff", "vff_slow", "vff_fast"};
std::vector<double> kRates{0.02, 0.1};

std::unique_ptr<DependenceEstimator> arm(const std::string& e, double n) {
    if (e == "xiao") return std::make_unique<XiaoSpearmanEstimator>(bank_experiment::xiao((n - 1) / (n + 1)), "x");
    if (e == "hermite") return std::make_unique<HermiteEstimator>(bank_experiment::hermite(2 / (n + 1)), "h");
    RecursiveMPL::Config m{};
    m.margin = RecursiveMPL::Margin::ExactNormal;
    m.gain = 2 / (n + 1);
    bank_experiment::rmpl(m);
    return std::make_unique<RecursiveMPL>(m, "m");
}

// the inner estimator of probe_thesis_bank's make_vff
std::unique_ptr<DependenceEstimator> vff_inner(const std::string& e) {
    if (e == "xiao") return std::make_unique<XiaoSpearmanEstimator>(bank_experiment::xiao(0.99), "x");
    if (e == "hermite") return std::make_unique<HermiteEstimator>(bank_experiment::hermite(0.01), "h");
    RecursiveMPL::Config m{};
    m.margin = RecursiveMPL::Margin::ExactNormal;
    m.gain = 0.01;
    bank_experiment::rmpl(m);
    return std::make_unique<RecursiveMPL>(m, "m");
}

// ForgettingBank::update with Loss::Log, the margins supplied from outside.
struct Mff {
    double eta = 1.0, delta = 1e-4;
    std::vector<double> w = std::vector<double>(kK, 1.0 / kK), loss = std::vector<double>(kK), lw = std::vector<double>(kK);
    double estimate(const std::vector<std::unique_ptr<DependenceEstimator>>& arms) const {
        double r = 0;
        for (std::size_t k = 0; k < kK; ++k) r += w[k] * arms[k]->estimate();
        return std::clamp(r, -1.0, 1.0);
    }
    // n_seen: margin observations held before this tick (the bank's mx_.size() before transform)
    void score(const std::vector<std::unique_ptr<DependenceEstimator>>& arms, double u, double v, std::size_t n_seen) {
        bool scorable = n_seen >= 30;
        for (const auto& a : arms) scorable = scorable && a->ready();
        if (!scorable) return;
        for (std::size_t k = 0; k < kK; ++k) {
            const double par = copula::param_from_spearman(copula::Family::Gaussian, arms[k]->estimate());
            loss[k] = -copula::log_density(copula::Family::Gaussian, u, v, par);
        }
        double worst = 0;
        for (double l : loss) if (std::isfinite(l)) worst = std::max(worst, l);
        for (double& l : loss) if (!std::isfinite(l)) l = worst;
        double top = -INFINITY;
        for (std::size_t k = 0; k < kK; ++k) {
            lw[k] = (w[k] > 0 ? std::log(w[k]) : -INFINITY) - eta * loss[k];
            top = std::max(top, lw[k]);
        }
        double tot = 0;
        for (std::size_t k = 0; k < kK; ++k) { w[k] = std::exp(lw[k] - top); tot += w[k]; }
        for (double& x : w) x = (1 - delta) * x / tot + delta / kK;
    }
};

// CommonVff::update, the margins supplied from outside.
struct Vff {
    std::unique_ptr<DependenceEstimator> inner;
    MemoryControlled* mc;
    VffController ctl;
    Vff(const std::string& e, double rate) : inner(vff_inner(e)), mc(dynamic_cast<MemoryControlled*>(inner.get())),
                                             ctl(VffController::Config{.rate = rate}) {}
    // n_after: margin observations held after absorbing this tick (CommonVff's mx_.size() after transform)
    void update(double x, double y, double u, double v, std::size_t n_after, bool raw) {
        if (n_after > 30) ctl.observe(12.0 * (u - 0.5) * (v - 0.5));
        if (mc) mc->set_neff(ctl.neff());
        if (raw) inner->update(x, y); else inner->update(RankPit::probit(u), RankPit::probit(v));
    }
};

std::unique_ptr<Dependence> make_dep(const std::string& s) {
    if (s == "constant") return std::make_unique<ConstantDependence>(0.6);
    if (s == "drift") return scenario_drift(kN);
    return std::make_unique<JumpDependence>(0.2, 0.8, kChange);
}

copula::Family family_of(const std::string& f) {
    if (f == "clayton") return copula::Family::Clayton;
    if (f == "gumbel") return copula::Family::Gumbel;
    if (f == "t") return copula::Family::StudentT;
    return copula::Family::Gaussian;
}

struct Win { const char* name; std::size_t a, b; };
std::vector<Win> windows(const std::string& s) {
    std::vector<Win> w{{"whole", kBurn, kN}};
    if (s == "jump") {
        w.push_back({"pre", kBurn, kChange}); w.push_back({"rec_0_100", kChange, kChange + 100});
        w.push_back({"rec_100_500", kChange + 100, kChange + 500}); w.push_back({"post", kChange, kN});
    }
    return w;
}
bool checkpoint(const std::string& sc, std::size_t t) {
    return t >= kBurn && ((t - kBurn) % 50 == 0 || (sc == "jump" && t >= kChange && t < kChange + 100 && (t - kChange) % 10 == 0));
}

struct Stream { std::vector<double> x, y, rho, par, u, v, pu, pv; };
Stream make_stream(const std::string& cop, const std::string& sc, unsigned seed) {
    auto dep = make_dep(sc);
    auto c = dgp::make_copula(cop);
    CopulaSource src(*dep, *c, seed);
    const copula::Family F = family_of(cop);
    Stream s;
    ds_margin::seed(1);
    ds_margin::Kll kx(256), ky(256);
    for (std::size_t t = 0; t < kN; ++t) {
        const auto smp = src.next();
        s.u.push_back(std::clamp(kx.cdf(smp.x), kDelta, 1 - kDelta)); s.v.push_back(std::clamp(ky.cdf(smp.y), kDelta, 1 - kDelta));
        kx.insert(smp.x); ky.insert(smp.y);
        s.x.push_back(smp.x); s.y.push_back(smp.y); s.rho.push_back(smp.rho_s);
        s.par.push_back(copula::param_from_spearman(F, dgp::detail::target_spearman(smp.rho)));
        s.pu.push_back(special::normal_cdf(smp.x)); s.pv.push_back(special::normal_cdf(smp.y));
    }
    return s;
}

double gauss_logc(double rho, double u, double v) {
    rho = std::clamp(rho, -0.999, 0.999);
    return copula::Gaussian::log_density(u, v, copula::param_from_spearman(copula::Family::Gaussian, rho));
}

// H^2(c_true(par), c_G(rho)) tabulated over rho in [-0.99, 0.99] (step 0.0025) once per truth parameter.
struct H2Table {
    const dmetric::Grid g = dmetric::Grid::full(81, 8.0);
    copula::Family F;
    std::map<double, std::vector<double>> cache;
    static constexpr double lo = -0.99, step = 0.0025;
    static constexpr int n = 793;
    double direct(double par, double rho) const {
        const dmetric::Table truth(g, [&](double a, double b) { const double l = copula::log_density(F, a, b, par); return std::isfinite(l) ? std::exp(l) : 0.0; });
        const dmetric::Table f(g, [&](double a, double b) { return std::exp(gauss_logc(rho, a, b)); });
        return dmetric::hellinger2(g, truth, f);
    }
    double operator()(double par, double rho) {
        auto it = cache.find(par);
        if (it == cache.end()) {
            const dmetric::Table truth(g, [&](double a, double b) { const double l = copula::log_density(F, a, b, par); return std::isfinite(l) ? std::exp(l) : 0.0; });
            std::vector<double> tab(n);
            for (int i = 0; i < n; ++i) {
                const double r = lo + step * i;
                const dmetric::Table f(g, [&](double a, double b) { return std::exp(gauss_logc(r, a, b)); });
                tab[i] = dmetric::hellinger2(g, truth, f);
            }
            it = cache.emplace(par, std::move(tab)).first;
        }
        const double x = (std::clamp(rho, lo, lo + step * (n - 1)) - lo) / step;
        const int i = std::min(n - 2, int(x));
        return it->second[i] + (x - i) * (it->second[i + 1] - it->second[i]);
    }
};
}  // namespace

int main(int argc, char** argv) {
    const int seeds = argc > 1 ? std::atoi(argv[1]) : 60;
    const char* env = std::getenv("COPULA");
    const std::string cop = env ? env : "gaussian";
    const char* tag = std::getenv("TAG");
    const std::string sfx = (tag ? std::string("_") + tag : std::string()) + (cop == "gaussian" ? "" : "_" + cop);
    if (const char* r = std::getenv("VFF_RATES")) {
        kRates.clear(); kStrat.resize(6);
        std::string rs(r);
        for (std::size_t a = 0, b; a <= rs.size(); a = b + 1) {
            b = rs.find(',', a); if (b == std::string::npos) b = rs.size();
            if (b > a) { kRates.push_back(std::atof(rs.substr(a, b - a).c_str())); kStrat.push_back("vff_" + rs.substr(a, b - a)); }
        }
    }
    H2Table h2{.F = family_of(cop)};

    if (std::getenv("VALIDATE")) {
        // Replicas against the registry classes, both on the EW margins the classes compute internally.
        for (const std::string sc : {"jump", "drift"}) {
            for (const char* e : kEst) {
                const std::string es(e), key = (es == "rmpl" ? std::string("mle") : es);
                auto bank = make_estimator(key + "_bank_log");
                std::vector<std::unique_ptr<DependenceEstimator>> arms;
                for (double n : kMem) arms.push_back(arm(es, n));
                Mff mff;
                VffController::Config vc{.rate = 0.1};
                const bool raw = es != "rmpl";
                CommonVff cls(vff_inner(es), vc, raw, "vff");
                Vff rep(es, 0.1);
                EwMarginal mx(500), my(500), vx(500), vy(500);
                double db = 0, dv = 0;
                auto dep2 = make_dep(sc); auto cop2 = dgp::make_copula(cop);
                CopulaSource s2(*dep2, *cop2, 42);
                for (std::size_t t = 0; t < kN; ++t) {
                    const auto smp = s2.next();
                    db = std::max(db, std::fabs(bank->estimate() - mff.estimate(arms)));
                    dv = std::max(dv, std::fabs(cls.estimate() - rep.inner->estimate()));
                    const std::size_t seen = mx.size();
                    const double u = mx.transform(smp.x), v = my.transform(smp.y);
                    mff.score(arms, u, v, seen);
                    for (auto& a : arms) { if (raw) a->update(smp.x, smp.y); else a->update(RankPit::probit(u), RankPit::probit(v)); }
                    bank->update(smp.x, smp.y);
                    const double uu = vx.transform(smp.x), vv = vy.transform(smp.y);
                    rep.update(smp.x, smp.y, uu, vv, vx.size(), raw);
                    cls.update(smp.x, smp.y);
                }
                std::printf("VALIDATE %s %s %s: max |replica - ForgettingBank| %.3g, max |replica - CommonVff| %.3g\n",
                            cop.c_str(), sc.c_str(), e, db, dv);
            }
        }
        const double par = copula::param_from_spearman(h2.F, 0.6 * 0.97);
        for (double r : {0.123, 0.4567, 0.58, 0.7891, 0.9})
            std::printf("VALIDATE H2 %s rho_hat %.4f: table %.6e direct %.6e\n", cop.c_str(), r, h2(par, r), h2.direct(par, r));
        return 0;
    }

    const bool paths = std::getenv("PATHS") != nullptr;   // only the per-tick paths; the other outputs are left alone
    std::FILE* fs = std::fopen(paths ? "/dev/null" : ("results/strategies_seeds" + sfx + ".csv").c_str(), "w");
    std::FILE* fj = std::fopen(paths ? "/dev/null" : ("results/strategies_jumppath" + sfx + ".csv").c_str(), "w");
    std::FILE* fp = paths ? std::fopen(("results/strategies_paths" + sfx + ".csv").c_str(), "w") : nullptr;
    std::FILE* fx = (tag && !paths) ? std::fopen(("results/strategies_extremes" + sfx + ".csv").c_str(), "w") : nullptr;
    if (!fs || !fj || (paths && !fp) || (tag && !paths && !fx)) { std::fprintf(stderr, "run from experiments/\n"); return 1; }
    if (fx) std::fprintf(fx, "copula,scenario,estimator,strategy,ex_mean,ex_median,ex_max,top1pct_share,seeds_extreme,seedticks_extreme\n");
    if (fp) std::fprintf(fp, "copula,scenario,estimator,strategy,t,mse,ex_mean,ex_median,ex_max,h2_mean,n_extreme\n");
    std::fprintf(fs, "copula,scenario,estimator,strategy,seed,window,rmse,excess_x1000,h2_x1000\n");
    std::fprintf(fj, "copula,estimator,strategy,t,mse\n");
    const copula::Family F = family_of(cop);
    const std::size_t S = kStrat.size();

    for (const std::string sc : {"constant", "drift", "jump"}) {
        const auto wins = windows(sc);
        std::vector<std::vector<std::vector<double>>> jsum(3, std::vector<std::vector<double>>(S, std::vector<double>(kN)));
        auto at = [&](int ei, std::size_t i, int sd, std::size_t t) { return ((std::size_t(ei) * S + i) * seeds + sd) * kN + t; };
        std::vector<float> exs(paths ? 3 * S * seeds * kN : 0);
        std::vector<double> h2s(paths ? 3 * S * kN : 0);
        std::vector<int> nex(paths ? 3 * S * kN : 0);
        std::vector<std::vector<float>> xall(fx ? 3 * S : 0);          // whole-window excess, all seeds x ticks (x 1000)
        std::vector<int> xseeds(fx ? 3 * S : 0), xticks(fx ? 3 * S : 0);
        for (int sd = 0; sd < seeds; ++sd) {
            const unsigned seed = 42u + sd;
            const Stream s = make_stream(cop, sc, seed);
            std::vector<double> lc(kN);
            for (std::size_t t = 0; t < kN; ++t) {
                const double l = copula::log_density(F, s.pu[t], s.pv[t], s.par[t]);
                lc[t] = l;
            }
            for (int ei = 0; ei < 3; ++ei) {
                const std::string e = kEst[ei];
                const bool raw = e != "rmpl";
                std::vector<std::unique_ptr<DependenceEstimator>> arms;
                for (double n : kMem) arms.push_back(arm(e, n));
                Mff mff;
                std::vector<Vff> vffs;
                vffs.reserve(kRates.size());
                for (double r : kRates) vffs.emplace_back(e, r);
                std::vector<char> seed_ext(S, 0);
                std::vector<std::vector<double>> se(S, std::vector<double>(kN)), ex(S, std::vector<double>(kN)), hh(S, std::vector<double>(kN, NAN));
                std::vector<double> f(S);
                for (std::size_t t = 0; t < kN; ++t) {
                    for (std::size_t k = 0; k < kK; ++k) f[k] = arms[k]->estimate();   // forecasts held before tick t
                    f[5] = mff.estimate(arms);
                    for (std::size_t j = 0; j < vffs.size(); ++j) f[6 + j] = vffs[j].inner->estimate();
                    const bool cp = checkpoint(sc, t);
                    for (std::size_t i = 0; i < S; ++i) {
                        const double d = f[i] - s.rho[t];
                        se[i][t] = d * d;
                        ex[i][t] = lc[t] - gauss_logc(f[i], s.pu[t], s.pv[t]);
                        if (cp) hh[i][t] = h2(s.par[t], f[i]);
                        jsum[ei][i][t] += d * d;
                        if (paths) {
                            exs[at(ei, i, sd, t)] = float(1000 * ex[i][t]);
                            if (cp) h2s[(ei * S + i) * kN + t] += 1000 * hh[i][t];
                            if (std::fabs(f[i]) > 0.98) ++nex[(ei * S + i) * kN + t];
                        }
                    }
                    mff.score(arms, s.u[t], s.v[t], t);
                    const double zx = RankPit::probit(s.u[t]), zy = RankPit::probit(s.v[t]);
                    for (auto& a : arms) { if (raw) a->update(s.x[t], s.y[t]); else a->update(zx, zy); }
                    for (auto& v : vffs) v.update(s.x[t], s.y[t], s.u[t], s.v[t], t + 1, raw);
                    if (fx && t >= kBurn)
                        for (std::size_t i = 0; i < S; ++i) {
                            xall[ei * S + i].push_back(float(1000 * ex[i][t]));
                            if (std::fabs(f[i]) > 0.98) { ++xticks[ei * S + i]; seed_ext[i] = 1; }
                        }
                }
                if (fx) for (std::size_t i = 0; i < S; ++i) xseeds[ei * S + i] += seed_ext[i];
                for (const auto& w : wins) {
                    const bool dense = std::string(w.name) == "rec_0_100";
                    for (std::size_t i = 0; i < S; ++i) {
                        double a = 0, b = 0, h = 0; int nh = 0;
                        for (std::size_t t = w.a; t < w.b; ++t) {
                            a += se[i][t]; b += ex[i][t];
                            if (!std::isnan(hh[i][t]) && (dense || (t - kBurn) % 50 == 0)) { h += hh[i][t]; ++nh; }
                        }
                        const double n = double(w.b - w.a);
                        std::fprintf(fs, "%s,%s,%s,%s,%u,%s,%.8f,%.6f,%.6f\n", cop.c_str(), sc.c_str(), e.c_str(), kStrat[i].c_str(),
                                     seed, w.name, std::sqrt(a / n), 1000 * b / n, nh ? 1000 * h / nh : NAN);
                    }
                }
            }
        }
        if (fx)
            for (int ei = 0; ei < 3; ++ei)
                for (std::size_t i = 0; i < S; ++i) {
                    auto v = xall[ei * S + i];
                    double sum = 0, mx = -INFINITY;
                    for (float x : v) { sum += x; mx = std::max(mx, double(x)); }
                    std::sort(v.begin(), v.end());
                    const double med = 0.5 * (v[(v.size() - 1) / 2] + v[v.size() / 2]);
                    const std::size_t k = std::max<std::size_t>(1, v.size() / 100);
                    double top = 0;
                    for (std::size_t j = v.size() - k; j < v.size(); ++j) top += v[j];
                    std::fprintf(fx, "%s,%s,%s,%s,%.6g,%.6g,%.6g,%.6g,%d,%d\n", cop.c_str(), sc.c_str(), kEst[ei], kStrat[i].c_str(),
                                 sum / v.size(), med, mx, sum != 0 ? top / sum : NAN, xseeds[ei * S + i], xticks[ei * S + i]);
                }
        if (paths)
            for (int ei = 0; ei < 3; ++ei)
                for (std::size_t i = 0; i < S; ++i)
                    for (std::size_t t = kBurn; t < kN; ++t) {
                        std::vector<float> v(seeds);
                        double sum = 0, mx = -INFINITY;
                        for (int sd = 0; sd < seeds; ++sd) { v[sd] = exs[at(ei, i, sd, t)]; sum += v[sd]; mx = std::max(mx, double(v[sd])); }
                        std::nth_element(v.begin(), v.begin() + seeds / 2, v.end());
                        double med = v[seeds / 2];
                        if (seeds % 2 == 0) med = 0.5 * (med + *std::max_element(v.begin(), v.begin() + seeds / 2));
                        const bool cp = checkpoint(sc, t);
                        std::fprintf(fp, "%s,%s,%s,%s,%zu,%.8g,%.6g,%.6g,%.6g,", cop.c_str(), sc.c_str(), kEst[ei], kStrat[i].c_str(), t,
                                     jsum[ei][i][t] / seeds, sum / seeds, med, mx);
                        if (cp) std::fprintf(fp, "%.6g", h2s[(ei * S + i) * kN + t] / seeds);
                        std::fprintf(fp, ",%d\n", nex[(ei * S + i) * kN + t]);
                    }
        if (sc == "jump")
            for (int ei = 0; ei < 3; ++ei)
                for (std::size_t i = 0; i < S; ++i)
                    for (std::size_t t = kBurn; t < kN; ++t)
                        std::fprintf(fj, "%s,%s,%s,%zu,%.8g\n", cop.c_str(), kEst[ei], kStrat[i].c_str(), t, jsum[ei][i][t] / seeds);
        std::printf("%s %s done\n", cop.c_str(), sc.c_str());
    }
    std::fclose(fs); std::fclose(fj);
    if (fp) std::fclose(fp);
    if (fx) std::fclose(fx);
}
