// Online confidence intervals for the EW corner counter chi_hat(q) = b/a (tail_chi): coverage, width and
// target. Baseline here is the analytic (independent-observation) interval
//     chi_hat +- z * sqrt( S2 * chi_hat (1 - chi_hat) / (a * W1) ),   alpha = 2/(M+1),
// the delta method for b/a with indicator inputs (per-observation variance of b - chi a is b (1 - chi)),
// with W1 = sum of the EW weights and S2 = sum of their squares, tracked recursively, so it is exact
// during warm-up; in steady state S2/W1 -> alpha/(2-alpha), the chi(1-chi)/(qM) law.
// It needs no state beyond the counter's. The online bootstrap of Nagler, Brock and Palm (2026) is to be
// added as a second interval once the paper is read; this probe is its harness.
// Coverage is reported against two targets: the instantaneous chi_t(q) of the regime in force, and the
// EW-smoothed target chi_bar_t = EW_alpha(chi_s(q)), which is what an EW estimator estimates.
// DGPs (copula space, rank-PIT margins as every Chapter 6 estimator, rho_S = 0.582, switch at TAU):
//   independent: gauss2clayton, clayton2gumbel, gauss2t, gaussian (no switch)
//   serially dependent: gauss_ar   Gaussian copula, latent pair AR(1) with phi = 0.7 (copula unchanged);
//                       t_persist  t4 copula whose chi-square mixing variable is redrawn with prob 0.05 per
//                                  tick (copula unchanged; joint extremes cluster in time);
//                       gauss2t_dep  gauss_ar -> t_persist at TAU.
// Note: notes/studies/thesis_tail_ci.md.
//
// Usage (from experiments/): ./probe_tail_ci <regime> <first_seed> <seeds>  -> results/tail_ci_<regime>_<first_seed>.csv

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "../src/estimators/dependence/family_bank.hpp"
#include "../src/estimators/dependence/marginal_rank_pit.hpp"
#include "tail_common.hpp"

namespace {

constexpr std::size_t T = 10000, TAU = 5000, BURN = 1000, WARM = 30;
const std::vector<double> kQ{0.05, 0.10};
const std::vector<int> kMem{399, 1999, 4999};
constexpr double kZ90 = 1.6448536269514722;

// Serially dependent samplers with the same marginal copula as tail_common's.
class GaussAR {
public:
    GaussAR(double rho, double phi, unsigned seed) : rho_(rho), phi_(phi), rng_(seed) {}
    std::pair<double, double> next() {
        const double e1 = n_(rng_), e2 = rho_ * e1 + std::sqrt(1 - rho_ * rho_) * n_(rng_);
        const double s = std::sqrt(1 - phi_ * phi_);
        x_ = first_ ? e1 : phi_ * x_ + s * e1;
        y_ = first_ ? e2 : phi_ * y_ + s * e2;
        first_ = false;
        return {special::normal_cdf(x_), special::normal_cdf(y_)};
    }
private:
    double rho_, phi_, x_ = 0, y_ = 0;
    bool first_ = true;
    std::mt19937 rng_;
    std::normal_distribution<double> n_{0.0, 1.0};
};

class TPersist {
public:
    TPersist(double rho, double renew, unsigned seed) : rho_(rho), renew_(renew), rng_(seed) {}
    std::pair<double, double> next() {
        if (first_ || u_(rng_) < renew_) { s_ = std::sqrt(chi2_(rng_) / tail::NU); first_ = false; }
        const double z1 = n_(rng_), z2 = rho_ * z1 + std::sqrt(1 - rho_ * rho_) * n_(rng_);
        return {special::t_cdf(z1 / s_, tail::NU), special::t_cdf(z2 / s_, tail::NU)};
    }
private:
    double rho_, renew_, s_ = 1;
    bool first_ = true;
    std::mt19937 rng_;
    std::normal_distribution<double> n_{0.0, 1.0};
    std::uniform_real_distribution<double> u_{0.0, 1.0};
    std::gamma_distribution<double> chi2_{tail::NU / 2.0, 2.0};
};

struct Source {
    std::unique_ptr<tail::Sampler> s;
    std::unique_ptr<GaussAR> g;
    std::unique_ptr<TPersist> p;
    Source(const std::string& f, unsigned seed) {
        if (f == "gauss_ar") g = std::make_unique<GaussAR>(0.6, 0.7, seed);
        else if (f == "t_persist") p = std::make_unique<TPersist>(tail::fix("t", 0.6).param, 0.05, seed);
        else s = std::make_unique<tail::Sampler>(tail::fix(f, 0.6), seed);
    }
    std::pair<double, double> next() { return s ? s->next() : g ? g->next() : p->next(); }
};

std::pair<std::string, std::string> regime_of(const std::string& r) {
    if (r == "gauss2clayton") return {"gaussian", "clayton"};
    if (r == "clayton2gumbel") return {"clayton", "gumbel"};
    if (r == "gauss2t") return {"gaussian", "t"};
    if (r == "gauss_ar") return {"gauss_ar", "gauss_ar"};
    if (r == "t_persist") return {"t_persist", "t_persist"};
    if (r == "gauss2t_dep") return {"gauss_ar", "t_persist"};
    return {"gaussian", "gaussian"};
}

// Exact chi(q) of the marginal copula of a source.
std::pair<double, double> target(const std::string& f, double q) {
    const std::string fam = f == "gauss_ar" ? "gaussian" : f == "t_persist" ? "t" : f;
    const copula::Shape s = fam == "t" ? copula::Shape::StudentT : fam == "clayton" ? copula::Shape::Clayton
                          : fam == "gumbel" ? copula::Shape::Gumbel : copula::Shape::Gaussian;
    const auto t = copula::exact_tails(s, tail::fix(fam, 0.6).param, tail::NU, q);
    return {t.chi_L, t.chi_U};
}

struct Win { const char* name; std::size_t a, b; };
const std::vector<Win> kWins{{"whole", BURN, T}, {"pre", BURN, TAU}, {"post_0_500", TAU, TAU + 500},
                             {"post_500_2000", TAU + 500, TAU + 2000}, {"post_2000_5000", TAU + 2000, T}};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: probe_tail_ci <regime> <first_seed> <seeds>\n"); return 1; }
    const std::string rname = argv[1];
    const unsigned first = static_cast<unsigned>(std::atoi(argv[2]));
    const int seeds = std::atoi(argv[3]);
    const auto [before, after] = regime_of(rname);
    std::FILE* out = std::fopen(("results/tail_ci_" + rname + "_" + std::to_string(first) + ".csv").c_str(), "w");
    if (!out) { std::fprintf(stderr, "cannot write results/ (run from experiments/)\n"); return 1; }
    std::fprintf(out, "regime,seed,q,tail,M,interval,target,window,coverage,width\n");

    const std::size_t NQ = kQ.size(), NM = kMem.size();
    std::vector<std::array<std::pair<double, double>, 2>> tgts(NQ);   // [q][before/after], computed once
    for (std::size_t qi = 0; qi < NQ; ++qi) tgts[qi] = {target(before, kQ[qi]), target(after, kQ[qi])};
    for (int s = 0; s < seeds; ++s) {
        const unsigned seed = first + s;
        Source pre(before, seed), post(after, seed + 100000u);
        RankPit px{500}, py{500};
        // counters [q][M]: aL, bL, aU, bU; smoothed targets [q][M][tail]
        std::vector<std::array<double, 4>> cnt(NQ * NM, {0, 0, 0, 0});
        std::vector<std::array<double, 2>> sm(NQ * NM, {0, 0});
        std::vector<double> W1(NM, 0.0), S2(NM, 0.0);
        bool sm_init = false;
        // hits[q][M][tail][target][t], width[q][M][tail][t]
        std::vector<std::vector<char>> hit(NQ * NM * 2 * 2, std::vector<char>(T, 0));
        std::vector<std::vector<float>> wid(NQ * NM * 2, std::vector<float>(T, 0));
        for (std::size_t t = 0; t < T; ++t) {
            const auto [u0, v0] = t < TAU ? pre.next() : post.next();
            for (std::size_t qi = 0; qi < NQ; ++qi) {
                const auto& tg = tgts[qi][t < TAU ? 0 : 1];
                for (std::size_t mi = 0; mi < NM; ++mi) {
                    const double al = 2.0 / (kMem[mi] + 1.0);
                    const std::size_t k = qi * NM + mi;
                    auto& sk = sm[k];
                    if (!sm_init) sk = {tg.first, tg.second};
                    else { sk[0] += al * (tg.first - sk[0]); sk[1] += al * (tg.second - sk[1]); }
                    for (int tl = 0; tl < 2; ++tl) {
                        const double a = cnt[k][2 * tl], b = cnt[k][2 * tl + 1];
                        const double chi = a > 1e-12 ? b / a : 0.0;
                        const double se = a > 1e-12 ? std::sqrt(S2[mi] * chi * (1 - chi) / (a * W1[mi])) : 1.0;
                        const double lo = chi - kZ90 * se, hi = chi + kZ90 * se;
                        const double inst = tl ? tg.second : tg.first, smt = sk[tl];
                        const std::size_t h = ((k * 2 + tl) * 2);
                        hit[h][t] = inst >= lo && inst <= hi;
                        hit[h + 1][t] = smt >= lo && smt <= hi;
                        wid[k * 2 + tl][t] = float(hi - lo);
                    }
                }
            }
            sm_init = true;
            const bool scorable = px.size() >= WARM;
            const double u = px.transform(special::normal_ppf(u0)), v = py.transform(special::normal_ppf(v0));
            if (scorable) {
                for (std::size_t mi = 0; mi < NM; ++mi) {
                    const double al = 2.0 / (kMem[mi] + 1.0);
                    W1[mi] = (1 - al) * W1[mi] + al;
                    S2[mi] = (1 - al) * (1 - al) * S2[mi] + al * al;
                }
                for (std::size_t qi = 0; qi < NQ; ++qi)
                    for (std::size_t mi = 0; mi < NM; ++mi) {
                        const double al = 2.0 / (kMem[mi] + 1.0), q = kQ[qi];
                        auto& c = cnt[qi * NM + mi];
                        const bool lu = u <= q, uu = u > 1 - q;
                        c[0] += al * (lu - c[0]);
                        c[1] += al * ((lu && v <= q) - c[1]);
                        c[2] += al * (uu - c[2]);
                        c[3] += al * ((uu && v > 1 - q) - c[3]);
                    }
            }
        }
        for (std::size_t qi = 0; qi < NQ; ++qi)
            for (std::size_t mi = 0; mi < NM; ++mi)
                for (int tl = 0; tl < 2; ++tl)
                    for (const auto& w : kWins) {
                        const std::size_t k = qi * NM + mi, h = (k * 2 + tl) * 2;
                        double ci = 0, cs = 0, wd = 0;
                        for (std::size_t t = w.a; t < w.b; ++t) { ci += hit[h][t]; cs += hit[h + 1][t]; wd += wid[k * 2 + tl][t]; }
                        const double n = double(w.b - w.a);
                        for (int g = 0; g < 2; ++g)
                            std::fprintf(out, "%s,%u,%.2f,%s,%d,analytic,%s,%s,%.5f,%.5f\n", rname.c_str(), seed, kQ[qi],
                                         tl ? "upper" : "lower", kMem[mi], g ? "smoothed" : "instant", w.name,
                                         (g ? cs : ci) / n, wd / n);
                    }
        std::fprintf(stderr, "%s seed %u\n", rname.c_str(), seed);
    }
    std::fclose(out);
    return 0;
}
