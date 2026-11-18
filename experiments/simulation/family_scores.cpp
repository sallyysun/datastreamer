// Squared and log scoring rules across working copula families.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "../../src/estimators/evaluation.hpp"
#include "../../src/special_functions.hpp"
#include "strategy_runs.hpp"
#include "study_stream.hpp"

namespace strategy_experiments {

using thesis::H2Table;
using thesis::checkpoint;
using thesis::kBurn;
using thesis::kK;
using thesis::kMem;
using thesis::kN;
using thesis::windows;

// Compare squared and log loss across working copulas, using the same KLL inputs and arms.
int run_scoresfam(const Run& run) {
    const int seeds = run.seeds;
    const std::string& cop = run.cop;
    const std::string& sfx = run.sfx;
    const char* tag = run.tag;
    H2Table& h2 = run.h2;
    const unsigned base = std::getenv("SEED_BASE") ? unsigned(std::atoi(std::getenv("SEED_BASE"))) : thesis::kFirstSeed;
    const bool noh2 = std::getenv("NOH2") != nullptr;
    std::vector<double> etas{1.0};
    if (const char* es = std::getenv("ETAS")) {
        etas.clear(); std::string s(es); std::size_t p = 0;
        while (p < s.size()) { std::size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); etas.push_back(std::atof(s.substr(p, q - p).c_str())); p = q + 1; }
    }
    if (!tag) { std::fprintf(stderr, "SCORESFAM needs TAG\n"); return 1; }
    const copula::Family WF[4] = {copula::Family::Gaussian, copula::Family::StudentT, copula::Family::Clayton, copula::Family::Gumbel};
    const char* WN[4] = {"gaussian", "t", "clayton", "gumbel"};
    const FamilyInverse inv[4] = {FamilyInverse(WF[0]), FamilyInverse(WF[1]), FamilyInverse(WF[2]), FamilyInverse(WF[3])};
    {   // Compare the fine inverse grid with the shared family conversion.
        double worst = 0;
        for (int wi = 1; wi < 4; ++wi)
            for (double r = -0.2; r < 0.995; r += 0.0137) {
                const double a = inv[wi](r), b = copula::param_from_spearman(WF[wi], r);
                worst = std::max(worst, std::fabs(a - b) / std::max(1e-3, std::fabs(b)));
            }
        if (study_verbose())
            std::printf("Inverse table max relative error %.2e\n", worst);
    }
    // component density tables on a rho_S grid for H^2 (t / Clayton / Gumbel)
    const int nr = 799; const double rlo = -0.995, rst = 0.0025;
    const int ng = h2.g.size();
    std::vector<std::vector<float>> ctab(4);
    auto build = [&](int wi) {
        if (!ctab[wi].empty() || wi == 0 || noh2) return;
        ctab[wi].resize(std::size_t(nr) * ng * ng);
        for (int a = 0; a < nr; ++a) {
            const double th = inv[wi](rlo + rst * a);
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
                double th[5];
                for (int k = 0; k < 5; ++k) th[k] = inv[wi](rs[k]);
                const dmetric::Table truth(h2.g, [&](double a, double b) { const double l = copula::log_density(h2.F, a, b, par); return std::isfinite(l) ? std::exp(l) : 0.0; });
                const dmetric::Table mt(h2.g, [&](double a, double b) { double m = 0; for (int k = 0; k < 5; ++k) { const double l = copula::log_density(WF[wi], a, b, th[k]); m += ws[k] * (std::isfinite(l) ? std::exp(l) : 0.0); } return m; });
                const double ex = dmetric::hellinger2(h2.g, truth, mt), tb = mixW(wi, par, rs.data(), ws, 5);
                worst = std::max(worst, std::fabs(tb - ex) / ex);
            }
        if (study_verbose())
            std::printf("Mixture H2 max relative error %.2e\n", worst);
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
                            RH[k][t] = FC[ei][k][t];
                            TH[k][t] = inv[wi](RH[k][t]);
                            const bool clamped = wi == 0 ? std::fabs(TH[k][t]) >= copula::Gaussian::kMaxParam
                                                         : inv[wi].clamped(RH[k][t]);
                            if (clamped && t >= kBurn) ++aclamp;
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
                            for (std::size_t k = 0; k < kK; ++k) {
                                lsq[k] = (p - rk[k]) * (p - rk[k]);
                                llog[k] = -copula::log_density(WF[wi], s.u[t], s.v[t], TH[k][t]);
                            }
                            snf += thesis::guard_mff_losses(llog);
                            for (auto& q : in) {
                                thesis::apply_fixed_share(q.w, q.score ? llog : lsq, lw, q.eta, 1e-4);
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
                }
            }
        }
        std::fflush(stdout);
    }
    std::fclose(fo); std::fclose(fd);
    return 0;
}

}  // namespace strategy_experiments
