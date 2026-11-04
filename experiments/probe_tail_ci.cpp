// Online confidence intervals for the EW corner counter chi_hat(q) = b/a (tail_chi): coverage, width and
// target. Baseline here is the analytic (independent-observation) interval
//     chi_hat +- z * sqrt( S2 * chi_hat (1 - chi_hat) / (a * W1) ),   alpha = 2/(M+1),
// the delta method for b/a with indicator inputs (per-observation variance of b - chi a is b (1 - chi)),
// with W1 = sum of the EW weights and S2 = sum of their squares, tracked recursively, so it is exact
// during warm-up; in steady state S2/W1 -> alpha/(2-alpha), the chi(1-chi)/(qM) law.
// It needs no state beyond the counter's.
// Bootstrap: Algorithm 1 of Nagler, Brock and Palm (2026, UAI, PMLR 337), applied to the linearised residual of
// b/a around the lagged smoother, (J_t - b_{t-1}) - chi_{t-1} (K_t - a_{t-1}), divided by a. Multipliers
// V = T(Z), Z AR(1) with rho = 1 - nu^(-1/3) ("ar", chi = 1/3) or rho = 0 ("iid", their iid baseline),
// T = t_{2+nu^(1/3)}^{-1} o Phi. Pointwise intervals: +- z sigma_hat* (sigma_hat* from B1 = 20 replicates).
// Uniform band: +- q_hat sigma_hat*, q_hat the (1 - alpha/K) quantile over B2 = 80 replicates of the running
// max |delta*|/sigma_hat*, recalibrated at t0 + 2^(k-1)(t1 - t0). Intervals are evaluated after tick t is
// absorbed, pairing chi_hat(t) with chi_bar(t) as the paper pairs mu_hat(t) with mu_eta(t).
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
// Usage (from experiments/): ./probe_tail_ci <regime> <first_seed> <seeds>
//   -> results/tail_ci_<regime>_<first_seed>.csv           pointwise coverage and width by window
//      results/tail_ci_uniform_<regime>_<first_seed>.csv   uniform-in-time coverage of the bands

#include <algorithm>
#include <array>
#include <tuple>
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
// Algorithm 1 of Nagler, Brock and Palm (2026): burn-in t0, first calibration end t1, B = B1 + B2 replicates
// (B1 for the variance, B2 for the sup-quantile), alpha = 0.1, heavy-tail transform tabulated on [-kTSpan, kTSpan].
constexpr std::size_t T0 = 600, T1 = 1000;
constexpr int kB = 100, kB1 = 20, kTGrid = 4000;
constexpr double kAlpha = 0.1, kTSpan = 8.0;
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
    std::FILE* uout = std::fopen(("results/tail_ci_uniform_" + rname + "_" + std::to_string(first) + ".csv").c_str(), "w");
    std::fprintf(uout, "regime,seed,q,tail,M,interval,target,horizon,covered,mean_width\n");

    // Per memory: EWMA weight, effective sample size nu = (2 - alpha)/alpha, multiplier persistence for the
    // AR bootstrap rho = 1 - nu^(-1/3) (chi = 1/3, the paper's default) and for its iid baseline rho = 0, and
    // the heavy-tail transform T = t_{2 + nu^(1/3)}^{-1} o Phi, tabulated in z.
    struct Mem { double al, nu, rho; std::vector<double> T; };
    std::vector<Mem> mem;
    for (int M : kMem) {
        Mem m{2.0 / (M + 1.0), 0, 0, {}};
        m.nu = (2 - m.al) / m.al;
        m.rho = 1 - std::pow(m.nu, -1.0 / 3.0);
        const double df = 2 + std::cbrt(m.nu);
        for (int g = 0; g <= kTGrid; ++g) m.T.push_back(special::t_ppf(special::normal_cdf(-kTSpan + 2 * kTSpan * g / kTGrid), df));
        mem.push_back(m);
    }
    auto Tz = [&](const Mem& m, double z) {
        const double x = (std::clamp(z, -kTSpan, kTSpan) + kTSpan) / (2 * kTSpan) * kTGrid;
        const int g = std::min(int(x), kTGrid - 1);
        return m.T[g] + (x - g) * (m.T[g + 1] - m.T[g]);
    };
    // Calibration blocks: [t0, t1] is burn-in + first calibration, then blocks t0 + 2^(k-1)(t1 - t0).
    std::vector<std::size_t> bounds;
    for (std::size_t k = 1;; ++k) {
        const std::size_t b = T0 + (std::size_t(1) << (k - 1)) * (T1 - T0);
        bounds.push_back(b);
        if (b >= T) break;
    }
    const double Kblk = double(bounds.size() - 1);   // blocks after t1 that intersect (t1, T]
    const char* kInt[] = {"analytic", "iid_point", "ar_point", "iid_band", "ar_band"};
    constexpr int NI = 5;

    for (int s = 0; s < seeds; ++s) {
        const unsigned seed = first + s;
        Source pre(before, seed), post(after, seed + 100000u);
        std::mt19937 brng(seed * 7919u + 13u);
        std::normal_distribution<double> nrm(0.0, 1.0);
        RankPit px{500}, py{500};
        std::vector<std::array<double, 4>> cnt(NQ * NM, {0, 0, 0, 0});   // aL, bL, aU, bU
        std::vector<std::array<double, 2>> smn(NQ * NM, {0, 0});        // EW numerators of the smoothed target
        std::vector<double> W1(NM, 0.0), S2(NM, 0.0);
        // Bootstrap state. Multipliers per [variant][M][b]; error replicates per [variant][q][M][tail][b].
        std::vector<double> Z(2 * NM * kB, 0.0);
        std::vector<double> D(2 * NQ * NM * 2 * kB, 0.0), mx(2 * NQ * NM * 2 * kB, 0.0), qhat(2 * NQ * NM * 2, NAN);
        auto zi = [&](int var, std::size_t mi, int b) -> double& { return Z[(var * NM + mi) * kB + b]; };
        auto ci = [&](int var, std::size_t qi, std::size_t mi, int tl) { return ((var * NQ + qi) * NM + mi) * 2 + tl; };
        // hit[interval][target][q][M][tail][t], width[interval][q][M][tail][t]
        const std::size_t NC = NQ * NM * 2;
        std::vector<std::vector<char>> hit(NI * 2 * NC, std::vector<char>(T, 0));
        std::vector<std::vector<float>> wid(NI * NC, std::vector<float>(T, NAN));
        std::vector<double> V(kB);
        for (std::size_t t = 0; t < T; ++t) {
            const auto [u0, v0] = t < TAU ? pre.next() : post.next();
            const bool scorable = px.size() >= WARM;
            const double u = px.transform(special::normal_ppf(u0)), v = py.transform(special::normal_ppf(v0));
            if (!scorable) continue;
            for (std::size_t mi = 0; mi < NM; ++mi) {
                const auto& m = mem[mi];
                W1[mi] = (1 - m.al) * W1[mi] + m.al;
                S2[mi] = (1 - m.al) * (1 - m.al) * S2[mi] + m.al * m.al;
                for (int var = 0; var < 2; ++var) {
                    const double rho = var ? m.rho : 0.0;
                    for (int b = 0; b < kB; ++b) {
                        double& z = zi(var, mi, b);
                        z = rho * z + std::sqrt(1 - rho * rho) * nrm(brng);
                    }
                }
            }
            for (std::size_t qi = 0; qi < NQ; ++qi) {
                const double q = kQ[qi];
                const auto& tg = tgts[qi][t < TAU ? 0 : 1];
                const bool lu = u <= q, uu = u > 1 - q;
                const double J[2] = {double(lu && v <= q), double(uu && v > 1 - q)}, Kk[2] = {double(lu), double(uu)};
                for (std::size_t mi = 0; mi < NM; ++mi) {
                    const auto& m = mem[mi];
                    const std::size_t k = qi * NM + mi;
                    auto& c = cnt[k];
                    for (int tl = 0; tl < 2; ++tl) {
                        const double a0 = c[2 * tl], b0 = c[2 * tl + 1];
                        const double chi0 = a0 > 1e-12 ? b0 / a0 : 0.0;
                        // Linearised residual of b/a around the lagged smoother (Algorithm 1's X_t - mu_hat(t-1)).
                        const double r = (J[tl] - b0) - chi0 * (Kk[tl] - a0);
                        for (int var = 0; var < 2; ++var) {
                            double* d = &D[ci(var, qi, mi, tl) * kB];
                            for (int b = 0; b < kB; ++b) d[b] += m.al * (Tz(m, zi(var, mi, b)) * r - d[b]);
                        }
                        c[2 * tl] += m.al * (Kk[tl] - a0);
                        c[2 * tl + 1] += m.al * (J[tl] - b0);
                        smn[k][tl] += m.al * ((tl ? tg.second : tg.first) - smn[k][tl]);
                    }
                    for (int tl = 0; tl < 2; ++tl) {
                        const double a = c[2 * tl], b = c[2 * tl + 1];
                        if (a <= 1e-12) continue;
                        const double chi = b / a;
                        const double inst = tl ? tg.second : tg.first, smt = smn[k][tl] / W1[mi];
                        const std::size_t cc = k * 2 + tl;
                        auto mark = [&](int iv, double half) {
                            hit[(iv * 2 + 0) * NC + cc][t] = std::fabs(chi - inst) <= half;
                            hit[(iv * 2 + 1) * NC + cc][t] = std::fabs(chi - smt) <= half;
                            wid[iv * NC + cc][t] = float(2 * half);
                        };
                        mark(0, kZ90 * std::sqrt(S2[mi] * chi * (1 - chi) / (a * W1[mi])));
                        for (int var = 0; var < 2; ++var) {
                            const std::size_t id = ci(var, qi, mi, tl);
                            const double* d = &D[id * kB];
                            double s1 = 0, s2 = 0;
                            for (int b = 0; b < kB1; ++b) { s1 += d[b] / a; s2 += (d[b] / a) * (d[b] / a); }
                            const double sd = std::sqrt(std::max(s2 / kB1 - (s1 / kB1) * (s1 / kB1), 1e-18));
                            mark(1 + var, kZ90 * sd);
                            if (t > T0)
                                for (int b = kB1; b < kB; ++b) mx[id * kB + b] = std::max(mx[id * kB + b], std::fabs(d[b] / a) / sd);
                            if (std::find(bounds.begin(), bounds.end(), t) != bounds.end() && t >= T1) {
                                std::vector<double> mm(mx.begin() + id * kB + kB1, mx.begin() + (id + 1) * kB);
                                std::sort(mm.begin(), mm.end());
                                const double lev = 1 - kAlpha / Kblk;
                                qhat[id] = mm[std::min<std::size_t>(mm.size() - 1, std::size_t(std::ceil(lev * mm.size())) - 1)];
                            }
                            if (t > T1) mark(3 + var, qhat[id] * sd);
                        }
                    }
                }
            }
        }
        for (std::size_t qi = 0; qi < NQ; ++qi)
            for (std::size_t mi = 0; mi < NM; ++mi)
                for (int tl = 0; tl < 2; ++tl) {
                    const std::size_t cc = (qi * NM + mi) * 2 + tl;
                    for (int iv = 0; iv < NI; ++iv) {
                        for (const auto& w : kWins) {
                            const std::size_t a = std::max(w.a, iv >= 3 ? T1 + 1 : w.a);
                            double h0 = 0, h1 = 0, wd = 0;
                            for (std::size_t t = a; t < w.b; ++t) { h0 += hit[(iv * 2) * NC + cc][t]; h1 += hit[(iv * 2 + 1) * NC + cc][t]; wd += wid[iv * NC + cc][t]; }
                            const double n = double(w.b - a);
                            for (int g = 0; g < 2; ++g)
                                std::fprintf(out, "%s,%u,%.2f,%s,%d,%s,%s,%s,%.5f,%.5f\n", rname.c_str(), seed, kQ[qi],
                                             tl ? "upper" : "lower", kMem[mi], kInt[iv], g ? "smoothed" : "instant", w.name,
                                             (g ? h1 : h0) / n, wd / n);
                        }
                        if (iv < 3) continue;
                        // Uniform coverage: every tick of the horizon inside the band.
                        for (const auto& [hn, a, b] : {std::tuple{"t1_T", T1 + 1, T}, std::tuple{"t1_tau", T1 + 1, TAU}})
                            for (int g = 0; g < 2; ++g) {
                                bool all = true;
                                double wd = 0;
                                for (std::size_t t = a; t < b; ++t) { all = all && hit[(iv * 2 + g) * NC + cc][t]; wd += wid[iv * NC + cc][t]; }
                                std::fprintf(uout, "%s,%u,%.2f,%s,%d,%s,%s,%s,%d,%.5f\n", rname.c_str(), seed, kQ[qi],
                                             tl ? "upper" : "lower", kMem[mi], kInt[iv], g ? "smoothed" : "instant", hn,
                                             int(all), wd / double(b - a));
                            }
                    }
                }
        std::fprintf(stderr, "%s seed %u\n", rname.c_str(), seed);
    }
    std::fclose(uout);
    std::fclose(out);
    return 0;
}
