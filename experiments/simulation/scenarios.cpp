// Chapter 6 simulation probe for 6.3-6.5: per (family, scenario, seed) one stream and one cumulative KLL k = 256
// pseudo-observation sequence that feeds every margin-dependent part; the five fixed arms, MFF (thesis::Mff) and VFF
// (thesis::Vff) per estimator, scored on the forecast held before each tick (rho_S RMSE; excess log loss at the true PIT;
// H^2 on a grid, src/estimators/evaluation.hpp). One mode per run:
//   (default)  results/strategies_seeds[_tag][_fam].csv, strategies_jumppath; PATHS=1 per-tick paths; DUMP_SEED=<s> one stream
//   MIXDENS=1  MFF density = the mixture of its candidates' densities (with TAG=mix)
//   VFF_RATES=a,b,... TAG=<tag>  VFF at those rates (sensitivity), plus strategies_extremes
//   WEIGHTS / REGRET / REGRETFAM / SCORES / SCORESFAM / FAMSWITCH  the 6.3-6.5 studies (see each run_* function)
//   VALIDATE=1 checks the H^2 tables and mixture densities against direct computation.
// Usage: COPULA=<family> ./scenarios [seeds=60]; LIVE=1 takes the streams over TCP (live_source.hpp).

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../../src/estimators/dependence/ds_margin.hpp"
#include "../../src/estimators/special_functions.hpp"
#include "../../src/estimators/evaluation.hpp"
#include "../../src/estimators/registry.hpp"
#include "../../src/streamgenerator/datasource/copula.hpp"
#include "../../src/streamgenerator/datasource/dependence.hpp"
#include "../../src/streamgenerator/datasource/copula_source.hpp"
#include "live_source.hpp"

namespace {
using thesis::family_of; using thesis::gauss_logc; using thesis::H2Table; using thesis::Mff; using thesis::Vff;
using thesis::Win; using thesis::windows; using thesis::checkpoint;
using thesis::kN; using thesis::kBurn; using thesis::kChange; using thesis::kK; using thesis::kMem;
constexpr double kDelta = thesis::kClip;
const char* kEst[3] = {"xiao", "hermite", "rmpl"};
// Strategies: five fixed arms, MFF, then one VFF per rate. Default rates 0.02 / 0.1 named vff_slow / vff_fast (the
// original outputs); VFF_RATES=a,b,... names them vff_<a>, ... (sensitivity runs; write under TAG=...).
std::vector<std::string> kStrat = thesis::strategy_names();
std::vector<double> kRates = thesis::vff_rates();

std::unique_ptr<DependenceEstimator> arm(const std::string& e, double n) { return thesis::arm(e, n, "m"); }

std::unique_ptr<Dependence> make_dep(const std::string& s) { return make_scenario(s, kN); }

struct Stream { std::vector<double> x, y, rho, par, u, v, pu, pv; };
Stream make_stream(const std::string& cop, const std::string& sc, unsigned seed) {
    const copula::Family F = family_of(cop);
    LiveStream L;
    std::unique_ptr<Dependence> dep;
    std::unique_ptr<dgp::Copula> c;
    std::unique_ptr<CopulaSource> src;
    if (live_mode()) L = live_stream(cop, sc, seed, kN);
    else { dep = make_dep(sc); c = dgp::make_copula(cop); src = std::make_unique<CopulaSource>(*dep, *c, seed); }
    Stream s;
    ds_margin::seed(1);
    ds_margin::Kll kx(256), ky(256);
    for (std::size_t t = 0; t < kN; ++t) {
        double x, y, rho_s, d, pu, pv;
        if (src) {
            const auto smp = src->next();
            x = smp.x; y = smp.y; rho_s = smp.rho_s; d = smp.rho;
            pu = special::normal_cdf(smp.x); pv = special::normal_cdf(smp.y);
        } else {
            x = L.x[t]; y = L.y[t]; rho_s = L.rho_s[t]; d = L.d[t]; pu = L.pu[t]; pv = L.pv[t];
        }
        s.u.push_back(std::clamp(kx.cdf(x), kDelta, 1 - kDelta)); s.v.push_back(std::clamp(ky.cdf(y), kDelta, 1 - kDelta));
        kx.insert(x); ky.insert(y);
        s.x.push_back(x); s.y.push_back(y); s.rho.push_back(rho_s);
        s.par.push_back(copula::param_from_spearman(F, dgp::detail::target_spearman(d)));
        s.pu.push_back(pu); s.pv.push_back(pv);
    }
    return s;
}

// rho_S -> family parameter, tabulated on 40001 points between the family's parameter bounds (exact for the Gaussian):
// how each RMPL arm's rho_S becomes a working-family density in 6.4-6.5.
struct FamilyInverse {
    copula::Family F;
    double lo = 0, hi = 0;
    std::vector<double> th;
    explicit FamilyInverse(copula::Family f) : F(f) {
        if (F == copula::Family::Gaussian) return;
        const double plo = F == copula::Family::Clayton ? copula::Clayton::kMinParam : F == copula::Family::Gumbel ? copula::Gumbel::kMinParam : copula::StudentT::kMinParam;
        const double phi = F == copula::Family::Clayton ? copula::Clayton::kMaxParam : F == copula::Family::Gumbel ? copula::Gumbel::kMaxParam : copula::StudentT::kMaxParam;
        lo = copula::spearman_from_param(F, plo);
        hi = copula::spearman_from_param(F, phi);
        for (int i = 0; i < 40001; ++i) th.push_back(copula::param_from_spearman(F, lo + (hi - lo) * i / 40000.0));
    }
    bool clamped(double rho) const { return F != copula::Family::Gaussian && (rho <= lo || rho >= hi); }
    double operator()(double rho) const {
        if (F == copula::Family::Gaussian) return copula::param_from_spearman(F, rho);
        const double x = (std::clamp(rho, lo, hi) - lo) / (hi - lo) * (th.size() - 1.0);
        const std::size_t i = std::min(th.size() - 2, std::size_t(x));
        return th[i] + (x - i) * (th[i + 1] - th[i]);
    }
};

struct Run { int seeds; std::string cop, sfx; const char* tag; H2Table& h2; };

int run_validate(const Run& run) {
    const std::string& cop = run.cop;
    H2Table& h2 = run.h2;
    const double par = copula::param_from_spearman(h2.F, 0.6 * 0.97);
    for (double r : {0.123, 0.4567, 0.58, 0.7891, 0.9})
        std::printf("VALIDATE H2 %s rho_hat %.4f: table %.6e direct %.6e\n", cop.c_str(), r, h2(par, r), h2.direct(par, r));
    {   // mixture H^2: closed-form grid vs a dmetric::Table of the same mixture through copula::log_density
        const double rs[5] = {0.2, 0.45, 0.6, 0.75, 0.9}, ws[5] = {0.1, 0.3, 0.2, 0.25, 0.15}, w1[5] = {0, 0, 1, 0, 0};
        const dmetric::Table truth(h2.g, [&](double a, double b) { const double l = copula::log_density(h2.F, a, b, par); return std::isfinite(l) ? std::exp(l) : 0.0; });
        const dmetric::Table mt(h2.g, [&](double a, double b) { double m = 0; for (int k = 0; k < 5; ++k) m += ws[k] * std::exp(gauss_logc(rs[k], a, b)); return m; });
        std::printf("VALIDATE H2mix %s 5-comp: grid %.6e dmetric %.6e | 1-comp (rho 0.6): grid %.6e direct %.6e\n", cop.c_str(),
                    h2.mix(par, rs, ws, 5), dmetric::hellinger2(h2.g, truth, mt), h2.mix(par, rs, w1, 5), h2.direct(par, 0.6));
        const double lm = H2Table::mix_logc(rs, ws, 5, 0.3, 0.8);
        double m = 0; for (int k = 0; k < 5; ++k) m += ws[k] * std::exp(gauss_logc(rs[k], 0.3, 0.8));
        std::printf("VALIDATE mix_logc: %.10f vs %.10f\n", lm, std::log(m));
    }
    return 0;
}

// Section 6.5 on the Chapter 6 configuration: RMPL arms N_eff 39-1999 on the shared cumulative KLL (k = 256)
// pseudo-observations; family-augmented MFF = 20 models (Gaussian, t4, Clayton, Gumbel) x 5 memories, each model's
// density c_W(theta_W(rho_k)) with theta_W = param_from_spearman(W, rho_k) (the SCORESFAM inversion); Gaussian-only
// MFF = the 5 Gaussian models. Both: log loss on the KLL (U, V) before the arms update, eta = 1, delta = 1e-4, Mff's
// warm-up (t >= 30, arms ready) and non-finite guard; forecast = mixture of the models' densities (weights before the
// tick), rho = sum_k w_k rho_k. Truth: matched d = 0.6 (rho_S ~ 0.582); copula = COPULA-independent list below,
// switch copulas change family at tick 2500. Excess log loss at the true PIT, windows from tick 500.
int run_famswitch(const Run& run) {
    const int seeds = run.seeds;
    H2Table& h2 = run.h2;
    const char* GN[7] = {"gaussian", "t", "clayton", "gumbel", "gauss2t", "gauss2clayton", "clayton2gumbel"};
    const int G0[7] = {0, 1, 2, 3, 0, 0, 2}, G1[7] = {0, 1, 2, 3, 1, 2, 3};
    const copula::Family WF[4] = {copula::Family::Gaussian, copula::Family::StudentT, copula::Family::Clayton, copula::Family::Gumbel};
    const char* WN[4] = {"gaussian", "t", "clayton", "gumbel"};
    const FamilyInverse inv[4] = {FamilyInverse(WF[0]), FamilyInverse(WF[1]), FamilyInverse(WF[2]), FamilyInverse(WF[3])};
    auto theta_of = [&](int wi, double rho) { return inv[wi](rho); };
    const std::string fx = std::getenv("FS_TAG") ? std::string("_") + std::getenv("FS_TAG") : std::string();
    std::FILE* fo = std::fopen(("results/famsw_seeds" + fx + ".csv").c_str(), "w");
    std::FILE* fi = std::fopen(("results/famsw_ident" + fx + ".csv").c_str(), "w");
    std::FILE* fp = std::fopen(("results/famsw_path" + fx + ".csv").c_str(), "w");
    if (!fo || !fi || !fp) { std::fprintf(stderr, "run from the experiment folder (results/ must exist)\n"); return 1; }
    // FS_H2=1: also squared Hellinger H^2(c_true, mixture) at the Chapter 6 checkpoints (every 50 from tick 500; every
    // 10 in [2500, 2600) on switch paths, counted only in rec_0_100), on the 81 x 81 grid of the 6.3 evaluation:
    // Gaussian components in closed form (as H2Table::mix), t4 / Clayton / Gumbel components tabulated on a rho_S grid
    // (step 0.0025) and interpolated (checked against exact integration at start). Truth table per (family, parameter).
    const bool fh2 = std::getenv("FS_H2") != nullptr;
    std::fprintf(fo, fh2 ? "gen,seed,method,window,rmse,excess_x1000,h2_x1000\n" : "gen,seed,method,window,rmse,excess_x1000\n");
    const int ng = h2.g.size(); const int nr = 799; const double rlo = -0.995, rst = 0.0025;
    std::vector<std::vector<float>> ctab(4);
    std::map<std::pair<int, double>, std::vector<double>> truthc;
    if (fh2)
        for (int wi = 1; wi < 4; ++wi) {
            ctab[wi].resize(std::size_t(nr) * ng * ng);
            for (int a = 0; a < nr; ++a) {
                const double thv = theta_of(wi, rlo + rst * a);
                for (int i = 0; i < ng; ++i)
                    for (int j = 0; j < ng; ++j) {
                        const double l = copula::log_density(WF[wi], h2.g.u[i], h2.g.u[j], thv);
                        ctab[wi][(std::size_t(a) * ng + i) * ng + j] = float(std::isfinite(l) ? std::exp(l) : 0.0);
                    }
            }
        }
    // H^2 of the truth (family gt, parameter par) against sum_m w_m c_{wi_m}(rho_m)
    auto h2mix = [&](int gt, double par, const int* wis, const double* rhos, const double* ws, int nm) {
        auto key = std::make_pair(gt, par);
        auto it = truthc.find(key);
        if (it == truthc.end()) {
            const dmetric::Table tt(h2.g, [&](double a, double b) { const double l = copula::log_density(WF[gt], a, b, par); return std::isfinite(l) ? std::exp(l) : 0.0; });
            it = truthc.emplace(key, tt.v).first;
        }
        std::vector<double> r(nm), c0(nm), c1(nm), fr(nm); std::vector<std::size_t> i0(nm);
        for (int m = 0; m < nm; ++m) {
            if (wis[m] == 0) {
                r[m] = copula::param_from_spearman(copula::Family::Gaussian, std::clamp(rhos[m], -0.999, 0.999));
                c0[m] = 1 / std::sqrt(1 - r[m] * r[m]); c1[m] = 1 / (2 * (1 - r[m] * r[m]));
            } else {
                const double x = (std::clamp(rhos[m], rlo, rlo + rst * (nr - 1)) - rlo) / rst;
                i0[m] = std::min(std::size_t(nr - 2), std::size_t(x)); fr[m] = x - i0[m];
            }
        }
        double sacc = 0;
        for (int i = 0; i < ng; ++i)
            for (int j = 0; j < ng; ++j) {
                const double x = h2.g.z[i], y = h2.g.z[j];
                const std::size_t o = std::size_t(i) * ng + j;
                double mm = 0;
                for (int m = 0; m < nm; ++m) {
                    if (ws[m] <= 0) continue;
                    if (wis[m] == 0) mm += ws[m] * c0[m] * std::exp(-(r[m] * r[m] * (x * x + y * y) - 2 * r[m] * x * y) * c1[m]);
                    else { const float* T0 = &ctab[wis[m]][i0[m] * ng * ng]; mm += ws[m] * ((1 - fr[m]) * T0[o] + fr[m] * T0[o + ng * ng]); }
                }
                sacc += std::sqrt(std::max(it->second[o], 0.0) * mm) * h2.g.w[i] * h2.g.w[j];
            }
        return 1.0 - sacc;
    };
    if (fh2) {   // tabulated components vs exact dmetric integration, one mixture per non-Gaussian family
        double worst = 0;
        for (int wi = 1; wi < 4; ++wi) {
            const int wis[3] = {wi, wi, 0}; const double rhs[3] = {0.31, 0.6123, 0.55}, wss[3] = {0.3, 0.5, 0.2};
            const double par = copula::param_from_spearman(WF[wi], 0.582);
            const dmetric::Table tt(h2.g, [&](double a, double b) { const double l = copula::log_density(WF[wi], a, b, par); return std::isfinite(l) ? std::exp(l) : 0.0; });
            const dmetric::Table mt(h2.g, [&](double a, double b) { double m = 0; for (int k = 0; k < 3; ++k) { const double th = wis[k] == 0 ? copula::param_from_spearman(copula::Family::Gaussian, rhs[k]) : copula::param_from_spearman(WF[wis[k]], rhs[k]); const double l = copula::log_density(WF[wis[k]], a, b, th); m += wss[k] * (std::isfinite(l) ? std::exp(l) : 0.0); } return m; });
            const double ex = dmetric::hellinger2(h2.g, tt, mt), tb = h2mix(wi, par, wis, rhs, wss, 3);
            worst = std::max(worst, std::fabs(tb - ex) / ex);
        }
        std::printf("CHECK FAMSWITCH H2 mixture: max relative error vs exact %.2e\n", worst);
    }
    std::fprintf(fi, "gen,seed,kind,target,t50,t90,share_ge50_after,incorrect_majority,first_incorrect,pre_weight_target,n_guard\n");
    std::fprintf(fp, "gen,t,family,mean,sd\n");
    const unsigned base = std::getenv("SEED_BASE") ? unsigned(std::atoi(std::getenv("SEED_BASE"))) : 42u;
    // FS_EST = rmpl (default) | xiao | hermite: the five arms of that estimator (counting matrix / Hermite read the raw
    // data with their own margins, as in Sections 6.3-6.4; RMPL reads probit of the KLL pseudo-observations).
    const std::string fest = std::getenv("FS_EST") ? std::getenv("FS_EST") : "rmpl";
    std::vector<int> gens;
    if (const char* gs = std::getenv("FS_GENS")) {   // comma-separated subset of GN
        std::string s(gs); std::size_t p = 0;
        while (p <= s.size()) {
            std::size_t q = s.find(',', p); if (q == std::string::npos) q = s.size();
            const std::string tok = s.substr(p, q - p);
            for (int g = 0; g < 7; ++g) if (tok == GN[g]) gens.push_back(g);
            p = q + 1;
        }
    } else for (int g = 0; g < 7; ++g) gens.push_back(g);
    for (int g : gens) {
        const bool sw = G0[g] != G1[g];
        std::vector<double> ps(4 * kN, 0.0), pq(4 * kN, 0.0);
        for (int sd = 0; sd < seeds; ++sd) {
            const unsigned seed = base + sd;
            const Stream s = make_stream(GN[g], "constant", seed);
            std::vector<std::unique_ptr<DependenceEstimator>> arms;
            for (double n : kMem) arms.push_back(arm(fest, n));
            // instance 0: family (20 models, index wi * 5 + k); instance 1: Gaussian-only (5 models)
            std::vector<double> wf(20, 1.0 / 20), wg(kK, 1.0 / kK);
            std::vector<double> sef(kN), seg(kN), exf(kN), exg(kN), hhf(kN, NAN), hhg(kN, NAN);
            std::vector<std::array<double, 4>> fw(kN);
            long nguard = 0;
            double th[4][kK], te[4][kK], ld[4][kK], ls[4][kK], f[kK];
            for (std::size_t t = 0; t < kN; ++t) {
                bool ready = t >= 30;
                for (std::size_t k = 0; k < kK; ++k) { f[k] = arms[k]->estimate(); ready = ready && arms[k]->ready(); }
                const int gt = t < kChange ? G0[g] : G1[g];
                const double part = copula::param_from_spearman(WF[gt], dgp::detail::target_spearman(0.6));
                const double lc = copula::log_density(WF[gt], s.pu[t], s.pv[t], part);
                for (int wi = 0; wi < 4; ++wi)
                    for (std::size_t k = 0; k < kK; ++k) {
                        th[wi][k] = theta_of(wi, f[k]);
                        te[wi][k] = wi == 0 ? copula::param_from_spearman(copula::Family::Gaussian, std::clamp(f[k], -0.999, 0.999)) : th[wi][k];
                        ld[wi][k] = copula::log_density(WF[wi], s.pu[t], s.pv[t], te[wi][k]);
                    }
                // forecasts held before tick t
                auto mixlog = [&](const double* w, int nw, bool fam) {
                    double top = -INFINITY;
                    for (int i = 0; i < nw; ++i) if (w[i] > 0) top = std::max(top, std::log(w[i]) + (fam ? ld[i / kK][i % kK] : ld[0][i]));
                    double m = 0;
                    for (int i = 0; i < nw; ++i) if (w[i] > 0) m += std::exp(std::log(w[i]) + (fam ? ld[i / kK][i % kK] : ld[0][i]) - top);
                    return top + std::log(m);
                };
                double rf = 0, rg = 0;
                for (int i = 0; i < 20; ++i) rf += wf[i] * f[i % kK];
                for (std::size_t k = 0; k < kK; ++k) rg += wg[k] * f[k];
                rf = std::clamp(rf, -1.0, 1.0); rg = std::clamp(rg, -1.0, 1.0);
                sef[t] = (rf - s.rho[t]) * (rf - s.rho[t]); seg[t] = (rg - s.rho[t]) * (rg - s.rho[t]);
                exf[t] = lc - mixlog(wf.data(), 20, true); exg[t] = lc - mixlog(wg.data(), kK, false);
                if (fh2 && checkpoint(sw ? "jump" : "constant", t)) {
                    int wis[20]; double rhs[20];
                    for (int i = 0; i < 20; ++i) { wis[i] = i / int(kK); rhs[i] = f[i % kK]; }
                    hhf[t] = h2mix(gt, part, wis, rhs, wf.data(), 20);
                    hhg[t] = h2mix(gt, part, wis, rhs, wg.data(), int(kK));   // first five entries: the Gaussian models
                }
                for (int wi = 0; wi < 4; ++wi) { double a = 0; for (std::size_t k = 0; k < kK; ++k) a += wf[wi * kK + k]; fw[t][wi] = a; }
                if (ready) {
                    for (int wi = 0; wi < 4; ++wi)
                        for (std::size_t k = 0; k < kK; ++k) ls[wi][k] = -copula::log_density(WF[wi], s.u[t], s.v[t], th[wi][k]);
                    auto hedge = [&](std::vector<double>& w, bool fam) {
                        const int nw = int(w.size());
                        std::vector<double> L(nw), lw(nw);
                        double worst = 0;
                        for (int i = 0; i < nw; ++i) { L[i] = fam ? ls[i / kK][i % kK] : ls[0][i]; if (std::isfinite(L[i])) worst = std::max(worst, L[i]); else ++nguard; }
                        for (double& x : L) if (!std::isfinite(x)) x = worst;
                        double top = -INFINITY, tot = 0;
                        for (int i = 0; i < nw; ++i) { lw[i] = (w[i] > 0 ? std::log(w[i]) : -INFINITY) - L[i]; top = std::max(top, lw[i]); }
                        for (int i = 0; i < nw; ++i) { w[i] = std::exp(lw[i] - top); tot += w[i]; }
                        for (double& x : w) x = (1 - 1e-4) * x / tot + 1e-4 / nw;
                    };
                    hedge(wf, true); hedge(wg, false);
                }
                const double zx = special::probit(s.u[t]), zy = special::probit(s.v[t]);
                for (auto& a : arms) { if (fest == "rmpl") a->update(zx, zy); else a->update(s.x[t], s.y[t]); }   // as in 6.3
            }
            // windows
            std::vector<Win> wins{{"whole", kBurn, kN}};
            if (sw) { wins.push_back({"pre", kBurn, kChange}); wins.push_back({"rec_0_100", kChange, kChange + 100});
                      wins.push_back({"late", kChange + 100, kN}); wins.push_back({"post", kChange, kN}); }
            for (const auto& w : wins)
                for (int m = 0; m < 2; ++m) {
                    double a = 0, b = 0, hsum = 0; int nh = 0;
                    const bool dense = std::string(w.name) == "rec_0_100";
                    for (std::size_t t = w.a; t < w.b; ++t) {
                        a += m ? seg[t] : sef[t]; b += m ? exg[t] : exf[t];
                        const double hv = m ? hhg[t] : hhf[t];
                        if (!std::isnan(hv) && (dense || (t - kBurn) % 50 == 0)) { hsum += hv; ++nh; }
                    }
                    const double n = double(w.b - w.a);
                    if (fh2) std::fprintf(fo, "%s,%u,%s,%s,%.8f,%.6f,%.6f\n", GN[g], seed, m ? "gaussian_mff" : "family_mff", w.name, std::sqrt(a / n), 1000 * b / n, nh ? 1000 * hsum / nh : NAN);
                    else std::fprintf(fo, "%s,%u,%s,%s,%.8f,%.6f\n", GN[g], seed, m ? "gaussian_mff" : "family_mff", w.name, std::sqrt(a / n), 1000 * b / n);
                }
            // identification: constant = true family from t = 0; switch = new family from 2500 (times relative)
            const int tgt = G1[g];
            const std::size_t t0 = sw ? kChange : 0;
            long t50 = -1, t90 = -1, firstinc = -1; long after = 0, ge = 0; bool inc = false;
            for (std::size_t t = t0; t < kN; ++t) {
                const double wt = fw[t][tgt];
                if (t50 < 0 && wt >= 0.5) t50 = long(t - t0);
                if (t90 < 0 && wt >= 0.9) t90 = long(t - t0);
                if (t50 >= 0 && long(t - t0) >= t50) {
                    ++after; if (wt >= 0.5) ++ge;
                    for (int wi = 0; wi < 4; ++wi) if (wi != tgt && fw[t][wi] >= 0.5 && !inc) { inc = true; firstinc = long(t - t0); }
                }
            }
            std::fprintf(fi, "%s,%u,%s,%s,%ld,%ld,%.6f,%d,%ld,%.6f,%ld\n", GN[g], seed, sw ? "switch" : "initial", WN[tgt], t50, t90,
                         after ? double(ge) / after : NAN, int(inc), firstinc, sw ? fw[kChange - 1][tgt] : fw[0][tgt], nguard);
            for (std::size_t t = 0; t < kN; ++t) for (int wi = 0; wi < 4; ++wi) { ps[wi * kN + t] += fw[t][wi]; pq[wi * kN + t] += fw[t][wi] * fw[t][wi]; }
        }
        for (std::size_t t = 0; t < kN; t += 10)
            for (int wi = 0; wi < 4; ++wi) {
                const double mu = ps[wi * kN + t] / seeds, var = std::max(0.0, pq[wi * kN + t] / seeds - mu * mu) * seeds / (seeds - 1.0);
                std::fprintf(fp, "%s,%zu,%s,%.6f,%.6f\n", GN[g], t, WN[wi], mu, std::sqrt(var));
            }
        std::printf("%s famswitch done\n", GN[g]); std::fflush(stdout);
    }
    std::fclose(fo); std::fclose(fi); std::fclose(fp);
    return 0;
}

// MFF scoring function (squared vs log) x working copula family W, generating family = COPULA. Candidates:
//   inversion estimators rmpl / xiao / hermite: each arm's rho_S -> theta_W = param_from_spearman(W, rho) (the
//     estimator is unchanged; only its candidate density c_W(theta_W) depends on W);
// Squared loss (12(U-1/2)(V-1/2) - rho_k)^2 needs no W; log loss -log c_W(theta_k)(U, V); both on the KLL (U, V)
// before the arms update, warm-up t >= 30 (+ arms ready), eta = 1 (ETAS=... for the dev grid), delta = 1e-4,
// Mff's non-finite guard. MFF: rho = sum w rho_k; density = mixture sum w c_W(theta_k) (weights before the tick).
// Excess at the true PIT; H^2 (NOH2 skips it) on the validated 81x81 grid: W = Gaussian exact (h2.mix), else the
// component densities tabulated on a rho_S grid (step 0.0025) and interpolated (checked against exact at start).
int run_scoresfam(const Run& run) {
    const int seeds = run.seeds;
    const std::string& cop = run.cop;
    const std::string& sfx = run.sfx;
    const char* tag = run.tag;
    H2Table& h2 = run.h2;
    const unsigned base = std::getenv("SEED_BASE") ? unsigned(std::atoi(std::getenv("SEED_BASE"))) : 42u;
    const bool noh2 = std::getenv("NOH2") != nullptr;
    std::vector<double> etas{1.0};
    if (const char* es = std::getenv("ETAS")) {
        etas.clear(); std::string s(es); std::size_t p = 0;
        while (p < s.size()) { std::size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); etas.push_back(std::atof(s.substr(p, q - p).c_str())); p = q + 1; }
    }
    if (!tag) { std::fprintf(stderr, "SCORESFAM needs TAG\n"); return 1; }
    const copula::Family WF[4] = {copula::Family::Gaussian, copula::Family::StudentT, copula::Family::Clayton, copula::Family::Gumbel};
    const char* WN[4] = {"gaussian", "t", "clayton", "gumbel"};
    // rho_S -> theta inverse tables (fine grid, linear interpolation) for t / Clayton / Gumbel
    struct Inv { double lo, hi; std::vector<double> th; };
    std::vector<Inv> inv(4);
    for (int wi = 1; wi < 4; ++wi) {
        const auto F = WF[wi];
        const double plo = F == copula::Family::Clayton ? copula::Clayton::kMinParam : F == copula::Family::Gumbel ? copula::Gumbel::kMinParam : copula::StudentT::kMinParam;
        const double phi = F == copula::Family::Clayton ? copula::Clayton::kMaxParam : F == copula::Family::Gumbel ? copula::Gumbel::kMaxParam : copula::StudentT::kMaxParam;
        inv[wi].lo = copula::spearman_from_param(F, plo); inv[wi].hi = copula::spearman_from_param(F, phi);
        const int n = 40001;
        for (int i = 0; i < n; ++i) inv[wi].th.push_back(copula::param_from_spearman(F, inv[wi].lo + (inv[wi].hi - inv[wi].lo) * i / (n - 1.0)));
    }
    long clamp_hits = 0;
    auto theta_of = [&](int wi, double rho, bool& clamped) {
        if (wi == 0) { const double r = copula::param_from_spearman(copula::Family::Gaussian, rho); clamped = std::fabs(r) >= copula::Gaussian::kMaxParam; return r; }
        const Inv& I = inv[wi];
        clamped = rho <= I.lo || rho >= I.hi;
        const double x = (std::clamp(rho, I.lo, I.hi) - I.lo) / (I.hi - I.lo) * (I.th.size() - 1.0);
        const std::size_t i = std::min(I.th.size() - 2, std::size_t(x));
        return I.th[i] + (x - i) * (I.th[i + 1] - I.th[i]);
    };
    {   // inverse-table check against bisection
        double worst = 0;
        for (int wi = 1; wi < 4; ++wi)
            for (double r = -0.2; r < 0.995; r += 0.0137) { bool c; const double a = theta_of(wi, r, c), b = copula::param_from_spearman(WF[wi], r); worst = std::max(worst, std::fabs(a - b) / std::max(1e-3, std::fabs(b))); }
        std::printf("CHECK inverse table: max relative theta error %.2e\n", worst);
    }
    // component density tables on a rho_S grid for H^2 (t / Clayton / Gumbel)
    const int nr = 799; const double rlo = -0.995, rst = 0.0025;
    const int ng = h2.g.size();
    std::vector<std::vector<float>> ctab(4);
    auto build = [&](int wi) {
        if (!ctab[wi].empty() || wi == 0 || noh2) return;
        ctab[wi].resize(std::size_t(nr) * ng * ng);
        for (int a = 0; a < nr; ++a) {
            bool c; const double th = theta_of(wi, rlo + rst * a, c);
            for (int i = 0; i < ng; ++i)
                for (int j = 0; j < ng; ++j) {
                    const double l = copula::log_density(WF[wi], h2.g.u[i], h2.g.u[j], th);
                    ctab[wi][(std::size_t(a) * ng + i) * ng + j] = float(std::isfinite(l) ? std::exp(l) : 0.0);
                }
        }
    };
    auto mixW = [&](int wi, double par, const double* rho, const double* w, std::size_t K) {
        if (wi == 0) return h2.mix(par, rho, w, K);
        auto it = h2.truth.find(par);
        if (it == h2.truth.end()) {
            const dmetric::Table t(h2.g, [&](double a, double b) { const double l = copula::log_density(h2.F, a, b, par); return std::isfinite(l) ? std::exp(l) : 0.0; });
            it = h2.truth.emplace(par, t.v).first;
        }
        std::vector<std::size_t> i0(K); std::vector<double> fr(K);
        for (std::size_t k = 0; k < K; ++k) {
            const double x = (std::clamp(rho[k], rlo, rlo + rst * (nr - 1)) - rlo) / rst;
            i0[k] = std::min(std::size_t(nr - 2), std::size_t(x)); fr[k] = x - i0[k];
        }
        double s = 0;
        for (int i = 0; i < ng; ++i)
            for (int j = 0; j < ng; ++j) {
                const std::size_t o = std::size_t(i) * ng + j;
                double m = 0;
                for (std::size_t k = 0; k < K; ++k) {
                    if (w[k] <= 0) continue;
                    const float* T0 = &ctab[wi][i0[k] * ng * ng], *T1 = T0 + ng * ng;
                    m += w[k] * ((1 - fr[k]) * T0[o] + fr[k] * T1[o]);
                }
                s += std::sqrt(std::max(it->second[o], 0.0) * m) * h2.g.w[i] * h2.g.w[j];
            }
        return 1.0 - s;
    };
    if (!noh2) {   // tabulated mixture H^2 vs an exact dmetric table of the same mixture
        for (int wi = 1; wi < 4; ++wi) build(wi);
        const double par = copula::param_from_spearman(h2.F, 0.55);
        double worst = 0;
        for (int wi = 1; wi < 4; ++wi)
            for (const auto& rs : std::vector<std::array<double, 5>>{{0.21, 0.4567, 0.6, 0.78, 0.91}, {0.05, 0.33, 0.52, 0.5501, 0.97}, {-0.2, 0.1, 0.3, 0.7, 0.85}}) {
                const double ws[5] = {0.1, 0.3, 0.2, 0.25, 0.15};
                double th[5]; bool c;
                for (int k = 0; k < 5; ++k) th[k] = theta_of(wi, rs[k], c);
                const dmetric::Table truth(h2.g, [&](double a, double b) { const double l = copula::log_density(h2.F, a, b, par); return std::isfinite(l) ? std::exp(l) : 0.0; });
                const dmetric::Table mt(h2.g, [&](double a, double b) { double m = 0; for (int k = 0; k < 5; ++k) { const double l = copula::log_density(WF[wi], a, b, th[k]); m += ws[k] * (std::isfinite(l) ? std::exp(l) : 0.0); } return m; });
                const double ex = dmetric::hellinger2(h2.g, truth, mt), tb = mixW(wi, par, rs.data(), ws, 5);
                worst = std::max(worst, std::fabs(tb - ex) / ex);
                std::printf("CHECK H2mix %s gen, W=%s: table %.6e exact %.6e\n", cop.c_str(), WN[wi], tb, ex);
            }
        std::printf("CHECK H2mix max relative error %.2e\n", worst);
    }
    std::FILE* fo = std::fopen(("results/scoresfam_seeds" + sfx + ".csv").c_str(), "w");
    std::FILE* fd = std::fopen(("results/scoresfam_diag" + sfx + ".csv").c_str(), "w");
    if (!fo || !fd) { std::fprintf(stderr, "run from the experiment folder (results/ must exist)\n"); return 1; }
    std::fprintf(fo, "gen,scenario,estimator,working,score,eta,seed,window,rmse,excess_x1000,h2_x1000\n");
    std::fprintf(fd, "gen,scenario,estimator,working,seed,arm_clamp_ticks,score_nonfinite,eval_nonfinite\n");
    const char* EN[3] = {"rmpl", "xiao", "hermite"};
    for (const std::string sc : {"constant", "drift", "jump"}) {
        const auto wins = windows(sc);
        for (int sd = 0; sd < seeds; ++sd) {
            const unsigned seed = base + sd;
            const Stream s = make_stream(cop, sc, seed);
            std::vector<double> lc(kN);
            for (std::size_t t = 0; t < kN; ++t) lc[t] = copula::log_density(h2.F, s.pu[t], s.pv[t], s.par[t]);
            // inversion estimators: arm forecasts once
            std::vector<std::vector<std::vector<double>>> FC(3, std::vector<std::vector<double>>(kK, std::vector<double>(kN)));
            std::vector<std::vector<char>> RD(3, std::vector<char>(kN));
            for (int ei = 0; ei < 3; ++ei) {
                const bool raw = ei != 0;
                std::vector<std::unique_ptr<DependenceEstimator>> arms;
                for (double n : kMem) arms.push_back(arm(kEst[ei == 0 ? 2 : ei - 1], n));
                for (std::size_t t = 0; t < kN; ++t) {
                    bool ready = true;
                    for (std::size_t k = 0; k < kK; ++k) { FC[ei][k][t] = arms[k]->estimate(); ready = ready && arms[k]->ready(); }
                    RD[ei][t] = ready;
                    const double zx = special::probit(s.u[t]), zy = special::probit(s.v[t]);
                    for (auto& a : arms) { if (raw) a->update(s.x[t], s.y[t]); else a->update(zx, zy); }
                }
            }
            for (int wi = 0; wi < 4; ++wi) {
                build(wi);
                for (int ei = 0; ei < 3; ++ei) {
                    std::vector<std::vector<double>> RH(kK, std::vector<double>(kN)), TH(kK, std::vector<double>(kN)), TE(kK, std::vector<double>(kN));
                    std::vector<char> ready(kN, 1);
                    long aclamp = 0;
                    for (std::size_t k = 0; k < kK; ++k)
                        for (std::size_t t = 0; t < kN; ++t) {
                            bool c; RH[k][t] = FC[ei][k][t]; TH[k][t] = theta_of(wi, RH[k][t], c);
                            if (c && t >= kBurn) ++aclamp;
                            TE[k][t] = wi == 0 ? copula::param_from_spearman(copula::Family::Gaussian, std::clamp(RH[k][t], -0.999, 0.999)) : TH[k][t];
                        }
                    ready = RD[ei];
                    struct In { int score; double eta; std::vector<double> w; std::vector<double> se, ex, hh; };
                    std::vector<In> in;
                    for (double h : etas) for (int sco = 0; sco < 2; ++sco)
                        in.push_back({sco, h, std::vector<double>(kK, 1.0 / kK), std::vector<double>(kN), std::vector<double>(kN), std::vector<double>(kN, NAN)});
                    long snf = 0, enf = 0;
                    std::vector<double> lsq(kK), llog(kK), ld(kK), lw(kK), rk(kK);
                    for (std::size_t t = 0; t < kN; ++t) {
                        for (std::size_t k = 0; k < kK; ++k) {
                            rk[k] = RH[k][t];
                            ld[k] = copula::log_density(WF[wi], s.pu[t], s.pv[t], TE[k][t]);
                            if (!std::isfinite(ld[k])) { ++enf; ld[k] = -INFINITY; }
                        }
                        const bool cp = !noh2 && checkpoint(sc, t);
                        for (auto& q : in) {
                            double r = 0, top = -INFINITY;
                            for (std::size_t k = 0; k < kK; ++k) { r += q.w[k] * rk[k]; if (q.w[k] > 0) top = std::max(top, std::log(q.w[k]) + ld[k]); }
                            double m = 0;
                            for (std::size_t k = 0; k < kK; ++k) if (q.w[k] > 0) m += std::exp(std::log(q.w[k]) + ld[k] - top);
                            r = std::clamp(r, -1.0, 1.0);
                            q.se[t] = (r - s.rho[t]) * (r - s.rho[t]);
                            q.ex[t] = lc[t] - (top + std::log(m));
                            if (cp) q.hh[t] = mixW(wi, s.par[t], rk.data(), q.w.data(), kK);
                        }
                        if (t >= 30 && ready[t]) {
                            const double p = 12.0 * (s.u[t] - 0.5) * (s.v[t] - 0.5);
                            double worst = 0;
                            for (std::size_t k = 0; k < kK; ++k) {
                                lsq[k] = (p - rk[k]) * (p - rk[k]);
                                llog[k] = -copula::log_density(WF[wi], s.u[t], s.v[t], TH[k][t]);
                                if (std::isfinite(llog[k])) worst = std::max(worst, llog[k]); else ++snf;
                            }
                            for (double& x : llog) if (!std::isfinite(x)) x = worst;
                            for (auto& q : in) {
                                const auto& L = q.score ? llog : lsq;
                                double top = -INFINITY, tot = 0;
                                for (std::size_t k = 0; k < kK; ++k) { lw[k] = (q.w[k] > 0 ? std::log(q.w[k]) : -INFINITY) - q.eta * L[k]; top = std::max(top, lw[k]); }
                                for (std::size_t k = 0; k < kK; ++k) { q.w[k] = std::exp(lw[k] - top); tot += q.w[k]; }
                                for (double& x : q.w) x = (1 - 1e-4) * x / tot + 1e-4 / kK;
                            }
                        }
                    }
                    for (const auto& q : in)
                        for (const auto& w : wins) {
                            const bool dense = std::string(w.name) == "rec_0_100";
                            double a = 0, b = 0, h = 0; int nh = 0;
                            for (std::size_t t = w.a; t < w.b; ++t) {
                                a += q.se[t]; b += q.ex[t];
                                if (!std::isnan(q.hh[t]) && (dense || (t - kBurn) % 50 == 0)) { h += q.hh[t]; ++nh; }
                            }
                            const double n = double(w.b - w.a);
                            std::fprintf(fo, "%s,%s,%s,%s,%s,%g,%u,%s,%.8f,%.6f,%.6f\n", cop.c_str(), sc.c_str(), EN[ei], WN[wi], q.score ? "log" : "squared",
                                         q.eta, seed, w.name, std::sqrt(a / n), 1000 * b / n, nh ? 1000 * h / nh : NAN);
                        }
                    std::fprintf(fd, "%s,%s,%s,%s,%u,%ld,%ld,%ld\n", cop.c_str(), sc.c_str(), EN[ei], WN[wi], seed, aclamp, snf, enf);
                    clamp_hits += aclamp;
                }
            }
        }
        std::printf("%s %s scoresfam done\n", cop.c_str(), sc.c_str());
        std::fflush(stdout);
    }
    std::fclose(fo); std::fclose(fd);
    return 0;
}

// Regret robustness over working families, generating family = COPULA. Working model W: the five RMPL arms with each
// arm's rho_S inverted into W's parameter (FamilyInverse, as in 6.5), loss -log c_W(theta_W(rho_k)) at the KLL (U, V);
// W = Gaussian is the REGRET loss.
// Log loss, eta = 1, delta = 0 / 1e-4, mixture forecaster, Mff's non-finite guard; comparators best arm and
// best <= 1 / <= 2 switch sequence (exact DP) on the same guarded losses. Diagnostics per run: guarded
// (non-finite) arm losses, scored ticks with an arm at its parameter clamp.
int run_regretfam(const Run& run) {
    const int seeds = run.seeds;
    const std::string& cop = run.cop;
    const std::string& sfx = run.sfx;
    const copula::Family WF[4] = {copula::Family::Gaussian, copula::Family::StudentT, copula::Family::Clayton, copula::Family::Gumbel};
    const char* WN[4] = {"gaussian", "t", "clayton", "gumbel"};
    const double deltas[2] = {0.0, 1e-4};
    std::FILE* fr = std::fopen(("results/regretfam_seeds" + sfx + ".csv").c_str(), "w");
    if (!fr) { std::fprintf(stderr, "run from the experiment folder (results/ must exist)\n"); return 1; }
    std::fprintf(fr, "gen,scenario,working,delta,seed,T,t0,loss_mix,best0,arm0,best1,best2,sw_t,sw_from,sw_to,n_nonfinite,n_clamp\n");
    for (const std::string sc : {"constant", "drift", "jump"}) {
        for (int sd = 0; sd < seeds; ++sd) {
            const unsigned seed = 42u + sd;
            const Stream s = make_stream(cop, sc, seed);
            for (int wi = 0; wi < 4; ++wi) {
                const FamilyInverse inv(WF[wi]);
                std::vector<std::unique_ptr<DependenceEstimator>> arms;
                for (double n : kMem) arms.push_back(arm("rmpl", n));
                std::vector<std::vector<double>> w(2, std::vector<double>(kK, 1.0 / kK));
                std::vector<std::vector<double>> L;
                std::vector<std::array<double, 2>> mf;
                std::vector<std::size_t> tt;
                std::vector<double> l(kK), lw(kK);
                long nonfin = 0, nclamp = 0;
                for (std::size_t t = 0; t < kN; ++t) {
                    bool ready = t >= 30;
                    for (const auto& a : arms) ready = ready && a->ready();
                    if (ready) {
                        double worst = 0; bool clamp = false;
                        for (std::size_t k = 0; k < kK; ++k) {
                            const double rho = arms[k]->estimate();
                            l[k] = -copula::log_density(WF[wi], s.u[t], s.v[t], inv(rho));
                            clamp = clamp || inv.clamped(rho);
                            if (std::isfinite(l[k])) worst = std::max(worst, l[k]); else ++nonfin;
                        }
                        nclamp += clamp;
                        for (double& x : l) if (!std::isfinite(x)) x = worst;
                        std::array<double, 2> row{};
                        for (int di = 0; di < 2; ++di) {
                            double lo = INFINITY, mix = 0;
                            for (std::size_t k = 0; k < kK; ++k) lo = std::min(lo, l[k]);
                            for (std::size_t k = 0; k < kK; ++k) mix += w[di][k] * std::exp(-(l[k] - lo));
                            row[di] = lo - std::log(mix);
                            double top = -INFINITY, tot = 0;
                            for (std::size_t k = 0; k < kK; ++k) { lw[k] = (w[di][k] > 0 ? std::log(w[di][k]) : -INFINITY) - l[k]; top = std::max(top, lw[k]); }
                            for (std::size_t k = 0; k < kK; ++k) { w[di][k] = std::exp(lw[k] - top); tot += w[di][k]; }
                            for (double& x : w[di]) x = (1 - deltas[di]) * x / tot + deltas[di] / kK;
                        }
                        L.push_back(l); mf.push_back(row); tt.push_back(t);
                    }
                    const double zx = special::probit(s.u[t]), zy = special::probit(s.v[t]);
                    for (auto& a : arms) a->update(zx, zy);
                }
                const std::size_t T = L.size();
                constexpr int MS = 3;
                std::vector<double> D(MS * kK, INFINITY);
                double b2 = INFINITY;
                for (std::size_t i = 0; i < T; ++i) {
                    std::vector<double> N(MS * kK);
                    for (int m = 0; m < MS; ++m) {
                        double prev = INFINITY;
                        if (m > 0) for (std::size_t j = 0; j < kK; ++j) prev = std::min(prev, D[(m - 1) * kK + j]);
                        for (std::size_t k = 0; k < kK; ++k) {
                            double stay = i == 0 ? (m == 0 ? 0.0 : INFINITY) : D[m * kK + k];
                            if (i > 0 && m > 0) stay = std::min(stay, prev);
                            N[m * kK + k] = L[i][k] + stay;
                        }
                    }
                    D = N;
                }
                for (double x : D) b2 = std::min(b2, x);
                std::vector<std::vector<double>> C(kK, std::vector<double>(T + 1, 0.0));
                for (std::size_t k = 0; k < kK; ++k) for (std::size_t i = 0; i < T; ++i) C[k][i + 1] = C[k][i] + L[i][k];
                double b1 = INFINITY; std::size_t swt = 0, sa = 0, sb = 0;
                for (std::size_t tau = 0; tau <= T; ++tau)
                    for (std::size_t a = 0; a < kK; ++a)
                        for (std::size_t b = 0; b < kK; ++b) {
                            const double v = C[a][tau] + C[b][T] - C[b][tau];
                            if (v < b1 - 1e-12) { b1 = v; swt = tau; sa = a; sb = b; }
                        }
                std::size_t a0 = 0;
                for (std::size_t k = 1; k < kK; ++k) if (C[k][T] < C[a0][T]) a0 = k;
                for (int di = 0; di < 2; ++di) {
                    double cm = 0;
                    for (const auto& row : mf) cm += row[di];
                    std::fprintf(fr, "%s,%s,%s,%g,%u,%zu,%zu,%.8f,%.8f,%g,%.8f,%.8f,%zu,%g,%g,%ld,%ld\n", cop.c_str(), sc.c_str(), WN[wi],
                                 deltas[di], seed, T, tt[0], cm, C[a0][T], kMem[a0], b1, b2, sa == sb ? 0 : tt[std::min(swt, T - 1)],
                                 kMem[sa], kMem[sb], nonfin, nclamp);
                }
            }
        }
        std::printf("%s %s regretfam done\n", cop.c_str(), sc.c_str());
    }
    std::fclose(fr);
    return 0;
}

// Static and switching regret of the log-loss MFF (eta = 1), delta = 0 (Hedge) and 1e-4 (fixed share), on the
// scored ticks (t >= 30, every arm ready: Mff::score). Arm loss l_k = -log c_G(rho_k)(U, V) at the KLL (U, V)
// with Mff's non-finite guard; the comparators use the same guarded losses. MFF loss: mix = -log sum_k w_k
// exp(-l_k) (the forecaster of the Chapter 5 bound) and point = -log c_G(sum_k w_k rho_k)(U, V) (what the
// Chapter 6 MFF forecasts). Comparators: best arm, best sequence with <= 1 and <= 2 switches (exact forward DP,
// prefix optimum at every t). Weights are those held before the tick.
int run_regret(const Run& run) {
    const int seeds = run.seeds;
    const std::string& cop = run.cop;
    const std::string& sfx = run.sfx;
    constexpr int MS = 3;   // switch budgets 0, 1, 2
    const double deltas[2] = {0.0, 1e-4};
    std::FILE* fr = std::fopen(("results/regret_seeds" + sfx + ".csv").c_str(), "w");
    std::FILE* fc = std::fopen(("results/regret_paths" + sfx + ".csv").c_str(), "w");
    if (!fr || !fc) { std::fprintf(stderr, "run from the experiment folder (results/ must exist)\n"); return 1; }
    std::fprintf(fr, "copula,scenario,estimator,delta,seed,T,t0,loss_mix,loss_point,best0,arm0,best1,best2,sw_t,sw_from,sw_to\n");
    std::fprintf(fc, "copula,scenario,estimator,delta,loss,m,t,mean,sd\n");
    for (const std::string sc : {"constant", "drift", "jump"}) {
        // path sums over seeds: [ei][di][loss 0 mix / 1 point][m][t]
        const std::size_t P = 3 * 2 * 2 * MS * kN;
        std::vector<double> s1(P, 0.0), s2(P, 0.0);
        auto pi = [&](int ei, int di, int l, int m, std::size_t t) { return ((((std::size_t(ei) * 2 + di) * 2 + l) * MS + m) * kN) + t; };
        std::vector<char> scored(kN, 0);
        for (int sd = 0; sd < seeds; ++sd) {
            const unsigned seed = 42u + sd;
            const Stream s = make_stream(cop, sc, seed);
            for (int ei = 0; ei < 3; ++ei) {
                const std::string e = kEst[ei];
                const bool raw = e != "rmpl";
                std::vector<std::unique_ptr<DependenceEstimator>> arms;
                for (double n : kMem) arms.push_back(arm(e, n));
                std::vector<std::vector<double>> w(2, std::vector<double>(kK, 1.0 / kK));
                std::vector<std::vector<double>> L;          // [scored tick][k] guarded arm loss
                std::vector<std::array<double, 4>> mf;       // mix / point loss for delta 0, 1e-4
                std::vector<std::size_t> tt;
                std::vector<double> f(kK), l(kK), lw(kK);
                for (std::size_t t = 0; t < kN; ++t) {
                    for (std::size_t k = 0; k < kK; ++k) f[k] = arms[k]->estimate();
                    bool ready = t >= 30;
                    for (const auto& a : arms) ready = ready && a->ready();
                    if (ready) {
                        double worst = 0;
                        for (std::size_t k = 0; k < kK; ++k) {
                            const double par = copula::param_from_spearman(copula::Family::Gaussian, f[k]);
                            l[k] = -copula::log_density(copula::Family::Gaussian, s.u[t], s.v[t], par);
                            if (std::isfinite(l[k])) worst = std::max(worst, l[k]);
                        }
                        for (double& x : l) if (!std::isfinite(x)) x = worst;
                        std::array<double, 4> row{};
                        for (int di = 0; di < 2; ++di) {
                            double lo = INFINITY, r = 0;
                            for (std::size_t k = 0; k < kK; ++k) { lo = std::min(lo, l[k]); r += w[di][k] * f[k]; }
                            double mix = 0;
                            for (std::size_t k = 0; k < kK; ++k) mix += w[di][k] * std::exp(-(l[k] - lo));
                            row[2 * di] = lo - std::log(mix);
                            r = std::clamp(r, -1.0, 1.0);
                            row[2 * di + 1] = -copula::log_density(copula::Family::Gaussian, s.u[t], s.v[t],
                                                                   copula::param_from_spearman(copula::Family::Gaussian, r));
                            // Hedge (eta = 1) then fixed share, as Mff::score
                            double top = -INFINITY, tot = 0;
                            for (std::size_t k = 0; k < kK; ++k) { lw[k] = (w[di][k] > 0 ? std::log(w[di][k]) : -INFINITY) - l[k]; top = std::max(top, lw[k]); }
                            for (std::size_t k = 0; k < kK; ++k) { w[di][k] = std::exp(lw[k] - top); tot += w[di][k]; }
                            for (double& x : w[di]) x = (1 - deltas[di]) * x / tot + deltas[di] / kK;
                        }
                        L.push_back(l); mf.push_back(row); tt.push_back(t); scored[t] = 1;
                    }
                    const double zx = special::probit(s.u[t]), zy = special::probit(s.v[t]);
                    for (auto& a : arms) { if (raw) a->update(s.x[t], s.y[t]); else a->update(zx, zy); }
                }
                const std::size_t T = L.size();
                // forward DP: D[m][k] = min loss of a sequence ending at k with exactly m switches
                std::vector<std::array<double, MS>> best(T);
                std::vector<double> D(MS * kK, INFINITY);
                for (std::size_t i = 0; i < T; ++i) {
                    std::vector<double> N(MS * kK);
                    for (int m = 0; m < MS; ++m) {
                        double prev = INFINITY;
                        if (m > 0) for (std::size_t j = 0; j < kK; ++j) prev = std::min(prev, D[(m - 1) * kK + j]);
                        for (std::size_t k = 0; k < kK; ++k) {
                            double stay = i == 0 ? (m == 0 ? 0.0 : INFINITY) : D[m * kK + k];
                            if (i > 0 && m > 0) stay = std::min(stay, prev);
                            N[m * kK + k] = L[i][k] + stay;
                        }
                    }
                    D = N;
                    for (int m = 0; m < MS; ++m) {
                        double b = INFINITY;
                        for (int q = 0; q <= m; ++q) for (std::size_t k = 0; k < kK; ++k) b = std::min(b, D[q * kK + k]);
                        best[i][m] = b;
                    }
                }
                // best one-switch sequence over the whole run: switch time and arms
                std::vector<std::vector<double>> C(kK, std::vector<double>(T + 1, 0.0));
                for (std::size_t k = 0; k < kK; ++k) for (std::size_t i = 0; i < T; ++i) C[k][i + 1] = C[k][i] + L[i][k];
                double b1 = INFINITY; std::size_t swt = 0, sa = 0, sb = 0;
                for (std::size_t tau = 0; tau <= T; ++tau)
                    for (std::size_t a = 0; a < kK; ++a)
                        for (std::size_t b = 0; b < kK; ++b) {
                            const double v = C[a][tau] + C[b][T] - C[b][tau];
                            if (v < b1 - 1e-12) { b1 = v; swt = tau; sa = a; sb = b; }
                        }
                std::size_t a0 = 0;
                for (std::size_t k = 1; k < kK; ++k) if (C[k][T] < C[a0][T]) a0 = k;
                for (int di = 0; di < 2; ++di) {
                    double cm = 0, cp = 0;
                    for (std::size_t i = 0; i < T; ++i) {
                        cm += mf[i][2 * di]; cp += mf[i][2 * di + 1];
                        for (int m = 0; m < MS; ++m) {
                            const double rm = cm - best[i][m], rp = cp - best[i][m];
                            s1[pi(ei, di, 0, m, tt[i])] += rm; s2[pi(ei, di, 0, m, tt[i])] += rm * rm;
                            s1[pi(ei, di, 1, m, tt[i])] += rp; s2[pi(ei, di, 1, m, tt[i])] += rp * rp;
                        }
                    }
                    const bool same = sa == sb;
                    std::fprintf(fr, "%s,%s,%s,%g,%u,%zu,%zu,%.8f,%.8f,%.8f,%g,%.8f,%.8f,%zu,%g,%g\n", cop.c_str(), sc.c_str(), e.c_str(),
                                 deltas[di], seed, T, tt.empty() ? 0 : tt[0], cm, cp, C[a0][T], kMem[a0], b1, best[T - 1][2],
                                 same ? 0 : tt[std::min(swt, T - 1)], kMem[sa], kMem[sb]);
                }
            }
        }
        for (int ei = 0; ei < 3; ++ei)
            for (int di = 0; di < 2; ++di)
                for (int lo = 0; lo < 2; ++lo)
                    for (int m = 0; m < MS; ++m)
                        for (std::size_t t = 0; t < kN; ++t) {
                            if (!scored[t] || (t % 10 != 9 && t != kN - 1)) continue;
                            const double mu = s1[pi(ei, di, lo, m, t)] / seeds;
                            const double var = std::max(0.0, s2[pi(ei, di, lo, m, t)] / seeds - mu * mu) * seeds / (seeds - 1.0);
                            std::fprintf(fc, "%s,%s,%s,%g,%s,%d,%zu,%.6g,%.6g\n", cop.c_str(), sc.c_str(), kEst[ei], deltas[di],
                                         lo ? "point" : "mix", m, t, mu, std::sqrt(var));
                        }
        std::printf("%s %s regret done\n", cop.c_str(), sc.c_str());
    }
    std::fclose(fr); std::fclose(fc);
    return 0;
}

// MFF scoring rule x learning rate: the squared loss (p = 12(U-1/2)(V-1/2) against each arm's
// rho) and log loss (Gaussian copula at each arm's rho), both scored on the KLL (U, V) before the arms see
// the tick, delta = 1e-4, over the same five arms per stream; plus the fixed arms. SEED_BASE (default 42).
int run_scores(const Run& run) {
    const int seeds = run.seeds;
    const std::string& cop = run.cop;
    const std::string& sfx = run.sfx;
    const char* tag = run.tag;
    H2Table& h2 = run.h2;
    const unsigned base = std::getenv("SEED_BASE") ? unsigned(std::atoi(std::getenv("SEED_BASE"))) : 42u;
    const bool smix = std::getenv("MIXDENS") != nullptr;   // MFF density = mixture (excess, H^2); needs TAG
    if (smix && !tag) { std::fprintf(stderr, "MIXDENS needs TAG\n"); return 1; }
    const std::vector<double> eta_sq{0.1, 0.25, 0.5, 1, 2, 4, 8, 16}, eta_log{0.25, 0.5, 1, 2, 4, 8, 16};
    struct Inst { int score; double eta; std::vector<double> w; };   // score 0 squared, 1 log
    const std::string out = "results/scores_seeds" + sfx + ".csv", outj = "results/scores_jumppath" + sfx + ".csv";
    std::FILE* fo = std::fopen(out.c_str(), "w");
    std::FILE* fj2 = std::fopen(outj.c_str(), "w");
    if (!fo || !fj2) { std::fprintf(stderr, "run from the experiment folder (results/ must exist)\n"); return 1; }
    std::fprintf(fo, "copula,scenario,estimator,method,eta,seed,window,rmse,excess_x1000,h2_x1000\n");
    std::fprintf(fj2, "copula,estimator,method,eta,t,mse,ex_mean\n");
    const copula::Family Fs = family_of(cop);
    for (const std::string sc : {"constant", "drift", "jump"}) {
        const auto wins = windows(sc);
        std::vector<std::vector<double>> jse, jex;   // [ei * M + i][t], jump only
        std::size_t M = 0;
        for (int sd = 0; sd < seeds; ++sd) {
            const unsigned seed = base + sd;
            const Stream s = make_stream(cop, sc, seed);
            std::vector<double> lc(kN);
            for (std::size_t t = 0; t < kN; ++t) lc[t] = copula::log_density(Fs, s.pu[t], s.pv[t], s.par[t]);
            for (int ei = 0; ei < 3; ++ei) {
                const std::string e = kEst[ei];
                const bool raw = e != "rmpl";
                std::vector<std::unique_ptr<DependenceEstimator>> arms;
                for (double n : kMem) arms.push_back(arm(e, n));
                std::vector<Inst> in;
                for (double h : eta_sq) in.push_back({0, h, std::vector<double>(kK, 1.0 / kK)});
                for (double h : eta_log) in.push_back({1, h, std::vector<double>(kK, 1.0 / kK)});
                const std::size_t I = in.size();
                M = kK + I;   // methods: five fixed arms, then the MFF instances
                if (sc == "jump" && jse.empty()) { jse.assign(3 * M, std::vector<double>(kN)); jex.assign(3 * M, std::vector<double>(kN)); }
                std::vector<std::vector<double>> se(M, std::vector<double>(kN)), ex(M, std::vector<double>(kN)),
                    hh(M, std::vector<double>(kN, NAN));
                std::vector<double> f(M), lsq(kK), llog(kK), lw(kK);
                for (std::size_t t = 0; t < kN; ++t) {
                    for (std::size_t k = 0; k < kK; ++k) f[k] = arms[k]->estimate();
                    for (std::size_t i = 0; i < I; ++i) {
                        double r = 0;
                        for (std::size_t k = 0; k < kK; ++k) r += in[i].w[k] * f[k];
                        f[kK + i] = std::clamp(r, -1.0, 1.0);
                    }
                    const bool cp = checkpoint(sc, t);
                    for (std::size_t i = 0; i < M; ++i) {
                        const double d = f[i] - s.rho[t];
                        se[i][t] = d * d;
                        if (smix && i >= kK) {   // in[].w: the weights held before tick t (updated below)
                            ex[i][t] = lc[t] - H2Table::mix_logc(f.data(), in[i - kK].w.data(), kK, s.pu[t], s.pv[t]);
                            if (cp) hh[i][t] = h2.mix(s.par[t], f.data(), in[i - kK].w.data(), kK);
                        } else {
                            ex[i][t] = lc[t] - gauss_logc(f[i], s.pu[t], s.pv[t]);
                            if (cp) hh[i][t] = h2(s.par[t], f[i]);
                        }
                        if (sc == "jump") { jse[ei * M + i][t] += d * d; jex[ei * M + i][t] += ex[i][t]; }
                    }
                    if (t >= 30) {   // MFF warm-up (Mff::score: n_seen >= 30 and every arm ready)
                        bool ready = true;
                        for (const auto& a : arms) ready = ready && a->ready();
                        if (ready) {
                            const double p = 12.0 * (s.u[t] - 0.5) * (s.v[t] - 0.5);
                            double worst = 0;
                            for (std::size_t k = 0; k < kK; ++k) {
                                lsq[k] = (p - f[k]) * (p - f[k]);
                                const double par = copula::param_from_spearman(copula::Family::Gaussian, f[k]);
                                llog[k] = -copula::log_density(copula::Family::Gaussian, s.u[t], s.v[t], par);
                                if (std::isfinite(llog[k])) worst = std::max(worst, llog[k]);
                            }
                            for (double& l : llog) if (!std::isfinite(l)) l = worst;
                            for (auto& q : in) {
                                const auto& L = q.score ? llog : lsq;
                                double top = -INFINITY, tot = 0;
                                for (std::size_t k = 0; k < kK; ++k) {
                                    lw[k] = (q.w[k] > 0 ? std::log(q.w[k]) : -INFINITY) - q.eta * L[k];
                                    top = std::max(top, lw[k]);
                                }
                                for (std::size_t k = 0; k < kK; ++k) { q.w[k] = std::exp(lw[k] - top); tot += q.w[k]; }
                                for (double& x : q.w) x = (1 - 1e-4) * x / tot + 1e-4 / kK;
                            }
                        }
                    }
                    const double zx = special::probit(s.u[t]), zy = special::probit(s.v[t]);
                    for (auto& a : arms) { if (raw) a->update(s.x[t], s.y[t]); else a->update(zx, zy); }
                }
                for (const auto& w : wins) {
                    const bool dense = std::string(w.name) == "rec_0_100";
                    for (std::size_t i = 0; i < M; ++i) {
                        double a = 0, b = 0, h = 0; int nh = 0;
                        for (std::size_t t = w.a; t < w.b; ++t) {
                            a += se[i][t]; b += ex[i][t];
                            if (!std::isnan(hh[i][t]) && (dense || (t - kBurn) % 50 == 0)) { h += hh[i][t]; ++nh; }
                        }
                        const double n = double(w.b - w.a);
                        const char* meth = i < kK ? "fixed" : (in[i - kK].score ? "log" : "squared");
                        const double par_ = i < kK ? kMem[i] : in[i - kK].eta;
                        std::fprintf(fo, "%s,%s,%s,%s,%g,%u,%s,%.8f,%.6f,%.6f\n", cop.c_str(), sc.c_str(), e.c_str(), meth, par_,
                                     seed, w.name, std::sqrt(a / n), 1000 * b / n, nh ? 1000 * h / nh : NAN);
                    }
                }
            }
        }
        if (sc == "jump")
            for (int ei = 0; ei < 3; ++ei)
                for (std::size_t i = kK; i < M; ++i) {
                    const bool lg = (i - kK) >= eta_sq.size();
                    const double eta = lg ? eta_log[i - kK - eta_sq.size()] : eta_sq[i - kK];
                    for (std::size_t t = kBurn; t < kN; ++t)
                        std::fprintf(fj2, "%s,%s,%s,%g,%zu,%.8g,%.6g\n", cop.c_str(), kEst[ei], lg ? "log" : "squared", eta, t,
                                     jse[ei * M + i][t] / seeds, 1000 * jex[ei * M + i][t] / seeds);
                }
        std::printf("%s %s scores done\n", cop.c_str(), sc.c_str());
    }
    std::fclose(fo); std::fclose(fj2);
    return 0;
}

// MFF weight paths: delta = 1e-4 (the main MFF) and plain Hedge delta = 0, same arms
int run_weights(const Run& run) {
    const int seeds = run.seeds;
    const std::string& cop = run.cop;
    const std::string& sfx = run.sfx;
    std::FILE* fw = std::fopen(("results/strategies_weights" + sfx + ".csv").c_str(), "w");
    std::FILE* fc = std::fopen(("results/strategies_weights_seeds" + sfx + ".csv").c_str(), "w");
    if (!fw || !fc) { std::fprintf(stderr, "run from the experiment folder (results/ must exist)\n"); return 1; }
    std::fprintf(fw, "copula,scenario,estimator,delta,t,w39,w99,w199,w399,w1999,mse,rho_true,est_mean\n");
    std::fprintf(fc, "copula,scenario,estimator,delta,seed,window,rmse\n");
    const double deltas[2] = {1e-4, 0.0};
    for (const std::string sc : {"constant", "drift", "jump"}) {
        const auto wins = windows(sc);
        // [ei][di][t][k]: summed weights held before tick t; [ei][di][t]: summed squared error and forecast
        std::vector<double> ws(3 * 2 * kN * kK), es(3 * 2 * kN), fsum(3 * 2 * kN), rt(kN);
        for (int sd = 0; sd < seeds; ++sd) {
            const unsigned seed = 42u + sd;
            const Stream s = make_stream(cop, sc, seed);
            for (int ei = 0; ei < 3; ++ei) {
                const std::string e = kEst[ei];
                const bool raw = e != "rmpl";
                std::vector<std::unique_ptr<DependenceEstimator>> arms;
                for (double n : kMem) arms.push_back(arm(e, n));
                Mff m[2];
                for (int di = 0; di < 2; ++di) m[di].delta = deltas[di];
                std::vector<std::vector<double>> se(2, std::vector<double>(kN));
                for (std::size_t t = 0; t < kN; ++t) {
                    rt[t] = s.rho[t];
                    for (int di = 0; di < 2; ++di) {
                        const double f = m[di].estimate(arms), d = f - s.rho[t];
                        se[di][t] = d * d;
                        const std::size_t b = (std::size_t(ei) * 2 + di) * kN + t;
                        es[b] += d * d; fsum[b] += f;
                        for (std::size_t k = 0; k < kK; ++k) ws[b * kK + k] += m[di].w[k];
                    }
                    for (int di = 0; di < 2; ++di) m[di].score(arms, s.u[t], s.v[t], t);
                    const double zx = special::probit(s.u[t]), zy = special::probit(s.v[t]);
                    for (auto& a : arms) { if (raw) a->update(s.x[t], s.y[t]); else a->update(zx, zy); }
                }
                for (int di = 0; di < 2; ++di)
                    for (const auto& w : wins) {
                        double a = 0;
                        for (std::size_t t = w.a; t < w.b; ++t) a += se[di][t];
                        std::fprintf(fc, "%s,%s,%s,%g,%u,%s,%.8f\n", cop.c_str(), sc.c_str(), e.c_str(), deltas[di], seed, w.name,
                                     std::sqrt(a / double(w.b - w.a)));
                    }
            }
        }
        for (int ei = 0; ei < 3; ++ei)
            for (int di = 0; di < 2; ++di)
                for (std::size_t t = 0; t < kN; ++t) {
                    const std::size_t b = (std::size_t(ei) * 2 + di) * kN + t;
                    std::fprintf(fw, "%s,%s,%s,%g,%zu", cop.c_str(), sc.c_str(), kEst[ei], deltas[di], t);
                    for (std::size_t k = 0; k < kK; ++k) std::fprintf(fw, ",%.6g", ws[b * kK + k] / seeds);
                    std::fprintf(fw, ",%.8g,%.6g,%.6g\n", es[b] / seeds, rt[t], fsum[b] / seeds);
                }
        std::printf("%s %s weights done\n", cop.c_str(), sc.c_str());
    }
    std::fclose(fw); std::fclose(fc);
    return 0;
}

// MIXDENS=1: MFF's density forecast is the mixture sum_k w_k c_G(rho_k) (weights held before the tick) for the
// excess log loss and H^2; its rho_S forecast (sum_k w_k rho_k), weights and every other strategy are unchanged.
// Needs TAG (new files only). Also writes results/strategies_mixcheck<sfx>.csv: the mixture's cumulative log loss at
// the KLL (U, V) over the regret window (t >= 30, arms ready), for comparison with REGRET's loss_mix (delta = 1e-4).
int run_strategies(const Run& run) {
    const int seeds = run.seeds;
    const std::string& cop = run.cop;
    const std::string& sfx = run.sfx;
    const char* tag = run.tag;
    H2Table& h2 = run.h2;
    const bool mixdens = std::getenv("MIXDENS") != nullptr;
    if (mixdens && !tag) { std::fprintf(stderr, "MIXDENS needs TAG\n"); return 1; }
    std::FILE* fmc = mixdens ? std::fopen(("results/strategies_mixcheck" + sfx + ".csv").c_str(), "w") : nullptr;
    if (fmc) std::fprintf(fmc, "copula,scenario,estimator,seed,mixloss_kll\n");
    const char* dseed = std::getenv("DUMP_SEED");
    const std::string dscen = std::getenv("DUMP_SCEN") ? std::getenv("DUMP_SCEN") : "jump";
    const bool paths = std::getenv("PATHS") != nullptr || dseed;   // per-tick paths / dump only; the other outputs are left alone
    std::FILE* fs = std::fopen(paths ? "/dev/null" : ("results/strategies_seeds" + sfx + ".csv").c_str(), "w");
    std::FILE* fj = std::fopen(paths ? "/dev/null" : ("results/strategies_jumppath" + sfx + ".csv").c_str(), "w");
    std::FILE* fp = (paths && !dseed) ? std::fopen(("results/strategies_paths" + sfx + ".csv").c_str(), "w") : nullptr;
    std::FILE* fdump = dseed ? std::fopen(("results/strategies_dump_" + dscen + "_s" + dseed + sfx + ".csv").c_str(), "w") : nullptr;
    if (fdump) std::fprintf(fdump, "estimator,strategy,t,forecast,rho_true\n");
    std::FILE* fx = (tag && !paths) ? std::fopen(("results/strategies_extremes" + sfx + ".csv").c_str(), "w") : nullptr;
    if (!fs || !fj || (paths && !dseed && !fp) || (dseed && !fdump) || (tag && !paths && !fx)) { std::fprintf(stderr, "run from the experiment folder (results/ must exist)\n"); return 1; }
    if (fx) std::fprintf(fx, "copula,scenario,estimator,strategy,ex_mean,ex_median,ex_max,top1pct_share,seeds_extreme,seedticks_extreme\n");
    if (fp) std::fprintf(fp, "copula,scenario,estimator,strategy,t,mse,ex_mean,ex_median,ex_max,h2_mean,n_extreme,rho_true,est_mean,est_q10,est_q90\n");
    std::fprintf(fs, "copula,scenario,estimator,strategy,seed,window,rmse,excess_x1000,h2_x1000\n");
    std::fprintf(fj, "copula,estimator,strategy,t,mse\n");
    const copula::Family F = family_of(cop);
    const std::size_t S = kStrat.size();


    for (const std::string sc : {"constant", "drift", "jump"}) {
        if (dseed && sc != dscen) continue;
        const auto wins = windows(sc);
        std::vector<std::vector<std::vector<double>>> jsum(3, std::vector<std::vector<double>>(S, std::vector<double>(kN)));
        auto at = [&](int ei, std::size_t i, int sd, std::size_t t) { return ((std::size_t(ei) * S + i) * seeds + sd) * kN + t; };
        std::vector<float> exs(paths ? 3 * S * seeds * kN : 0), fcs(paths ? 3 * S * seeds * kN : 0);   // excess, forecasts
        std::vector<double> rho_true(kN);
        std::vector<double> h2s(paths ? 3 * S * kN : 0);
        std::vector<int> nex(paths ? 3 * S * kN : 0);
        std::vector<std::vector<float>> xall(fx ? 3 * S : 0);          // whole-window excess, all seeds x ticks (x 1000)
        std::vector<int> xseeds(fx ? 3 * S : 0), xticks(fx ? 3 * S : 0);
        for (int sd = 0; sd < seeds; ++sd) {
            const unsigned seed = 42u + sd;
            if (dseed && int(seed) != std::atoi(dseed)) continue;
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
                double mixkll = 0;
                for (std::size_t t = 0; t < kN; ++t) {
                    for (std::size_t k = 0; k < kK; ++k) f[k] = arms[k]->estimate();   // forecasts held before tick t
                    f[5] = mff.estimate(arms);
                    if (mixdens && t >= 30) {
                        bool ready = true;
                        for (const auto& a : arms) ready = ready && a->ready();
                        if (ready) mixkll -= H2Table::mix_logc(f.data(), mff.w.data(), kK, s.u[t], s.v[t]);
                    }
                    for (std::size_t j = 0; j < vffs.size(); ++j) f[6 + j] = vffs[j].inner->estimate();
                    if (fdump)
                        for (std::size_t i = 0; i < S; ++i)
                            std::fprintf(fdump, "%s,%s,%zu,%.10g,%.10g\n", e.c_str(), kStrat[i].c_str(), t, f[i], s.rho[t]);
                    const bool cp = checkpoint(sc, t);
                    for (std::size_t i = 0; i < S; ++i) {
                        const double d = f[i] - s.rho[t];
                        se[i][t] = d * d;
                        if (mixdens && i == 5) {   // mff.w are still the weights held before tick t (score() comes below)
                            ex[i][t] = lc[t] - H2Table::mix_logc(f.data(), mff.w.data(), kK, s.pu[t], s.pv[t]);
                            if (cp) hh[i][t] = h2.mix(s.par[t], f.data(), mff.w.data(), kK);
                        } else {
                            ex[i][t] = lc[t] - gauss_logc(f[i], s.pu[t], s.pv[t]);
                            if (cp) hh[i][t] = h2(s.par[t], f[i]);
                        }
                        jsum[ei][i][t] += d * d;
                        if (paths) {
                            exs[at(ei, i, sd, t)] = float(1000 * ex[i][t]);
                            fcs[at(ei, i, sd, t)] = float(f[i]);
                            rho_true[t] = s.rho[t];
                            if (cp) h2s[(ei * S + i) * kN + t] += 1000 * hh[i][t];
                            if (std::fabs(f[i]) > 0.98) ++nex[(ei * S + i) * kN + t];
                        }
                    }
                    mff.score(arms, s.u[t], s.v[t], t);
                    const double zx = special::probit(s.u[t]), zy = special::probit(s.v[t]);
                    for (auto& a : arms) { if (raw) a->update(s.x[t], s.y[t]); else a->update(zx, zy); }
                    for (auto& v : vffs) v.update(s.x[t], s.y[t], s.u[t], s.v[t], t + 1, raw);
                    if (fx && t >= kBurn)
                        for (std::size_t i = 0; i < S; ++i) {
                            xall[ei * S + i].push_back(float(1000 * ex[i][t]));
                            if (std::fabs(f[i]) > 0.98) { ++xticks[ei * S + i]; seed_ext[i] = 1; }
                        }
                }
                if (fx) for (std::size_t i = 0; i < S; ++i) xseeds[ei * S + i] += seed_ext[i];
                if (fmc) std::fprintf(fmc, "%s,%s,%s,%u,%.8f\n", cop.c_str(), sc.c_str(), e.c_str(), seed, mixkll);
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
        if (fp)
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
                        std::vector<float> fv(seeds);
                        double fs_ = 0;
                        for (int sd = 0; sd < seeds; ++sd) { fv[sd] = fcs[at(ei, i, sd, t)]; fs_ += fv[sd]; }
                        std::sort(fv.begin(), fv.end());
                        auto q = [&](double p) { const double x = p * (seeds - 1); const int k = int(x); return fv[k] + (x - k) * (fv[std::min(k + 1, seeds - 1)] - fv[k]); };
                        std::fprintf(fp, ",%d,%.6g,%.6g,%.6g,%.6g\n", nex[(ei * S + i) * kN + t], rho_true[t], fs_ / seeds, q(0.1), q(0.9));
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
    if (fdump) std::fclose(fdump);
    if (fx) std::fclose(fx);
    if (fmc) std::fclose(fmc);
    return 0;
}

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
    const Run run{seeds, cop, sfx, tag, h2};
    if (std::getenv("VALIDATE")) return run_validate(run);
    if (std::getenv("FAMSWITCH")) return run_famswitch(run);
    if (std::getenv("SCORESFAM")) return run_scoresfam(run);
    if (std::getenv("REGRETFAM")) return run_regretfam(run);
    if (std::getenv("REGRET")) return run_regret(run);
    if (std::getenv("SCORES")) return run_scores(run);
    if (std::getenv("WEIGHTS")) return run_weights(run);
    return run_strategies(run);
}
