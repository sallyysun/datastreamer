// Finite-q tail dependence as a streaming quantity: the family bank's mixture chi(q) = sum_k w_k chi_k(q)
// against the model-free EW corner counter tail_chi (M = 399, 1999, 4999) and the bank's fixed-M
// single-shape arms, all on a 500-tick rank PIT. Scored against the exact finite-q chi(q) of the true
// regime (not lambda), q = 0.05 and 0.10. One known-margin corner counter (true PIT, M = 1999) is the
// check against tail-study step 1, which used known uniform margins.
// Nonparametric competitors read chi(q) off their own predictive density (PredictiveDensity), as
// P(U, V both in the corner) / P(U in the tail), by a midpoint rule on the unit square, every 20 ticks
// (held between): the Hermite own-density bank, and single Hermite (N = 10) and Xiao (20 x 20, pooled
// density on the same 20 x 20 grid, prior mass 12.5 pseudo-observations as the default 5 x 5 pooling's)
// estimators at M = 399 and 1999. The corner counter is the corner
// cell of a Xiao-style histogram read directly.
// DGP: tail_common.hpp samplers at matched rho_S = 0.5819 (d = 0.6), switch at TAU.
// Regimes: gauss2clayton, clayton2gumbel, gauss2t (nu = 4), mix (Clayton/Gumbel mixture, pi 0.75 -> 0.25),
// gaussian (no switch; sanity check that finite-q chi is not 0 when lambda = 0).
// Note: notes/studies/thesis_bank_tails.md.
//
// Usage (from experiments/):  ./probe_tail_bank <regime> <first_seed> <seeds>
//   -> results/tail_bank_<regime>_<first_seed>.csv        seed x q x tail x window x method: rmse, bias
//      results/tail_bank_path_<regime>_<first_seed>.csv   seed means per tick (every 20): chi, shape weights
//      results/tail_bank_cross_<regime>_<first_seed>.csv  ticks after TAU for the seed-mean chi to cross
//                                                         halfway from the old target to the new

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "../src/estimators/registry.hpp"
#include "tail_common.hpp"

namespace {

constexpr std::size_t T = 10000, TAU = 5000, BURN = 1000, STRIDE = 20;
const std::vector<double> kQ{0.05, 0.10};
const char* kShapeName[] = {"gaussian", "t", "clayton", "clayton90", "gumbel", "gumbel90", "clayton270", "gumbel270"};
constexpr double kPi0 = 0.75;

struct Regime {
    std::string before, after;  // tail_common family names, or "mix0" / "mix1" for the mixture at pi0 / 1 - pi0
};

Regime regime_of(const std::string& r) {
    if (r == "gauss2clayton") return {"gaussian", "clayton"};
    if (r == "clayton2gumbel") return {"clayton", "gumbel"};
    if (r == "gauss2t") return {"gaussian", "t"};
    if (r == "mix") return {"mix0", "mix1"};
    return {"gaussian", "gaussian"};
}

tail::Mix mix_of(const std::string& f) {
    return {f == "mix0" ? kPi0 : 1.0 - kPi0, tail::fix("clayton", 0.6).param, tail::fix("gumbel", 0.6).param};
}

// Exact chi(q) of a regime: closed form for Clayton/Gumbel and the mixture, 64-point quadrature for
// Gaussian/t (copula::exact_tails; checked against tail::exact_corners in main).
std::pair<double, double> target(const std::string& f, double q) {
    if (f == "mix0" || f == "mix1") { const auto m = mix_of(f); return {m.chi_L(q), m.chi_U(q)}; }
    const auto fx = tail::fix(f, 0.6);
    const copula::Shape s = f == "t" ? copula::Shape::StudentT : f == "clayton" ? copula::Shape::Clayton
                          : f == "gumbel" ? copula::Shape::Gumbel : copula::Shape::Gaussian;
    const auto t = copula::exact_tails(s, fx.param, tail::NU, q);
    return {t.chi_L, t.chi_U};
}

struct Source {
    std::unique_ptr<tail::Sampler> s;
    std::unique_ptr<tail::MixSampler> m;
    Source(const std::string& f, unsigned seed) {
        if (f == "mix0" || f == "mix1") m = std::make_unique<tail::MixSampler>(mix_of(f), seed);
        else s = std::make_unique<tail::Sampler>(tail::fix(f, 0.6), seed);
    }
    std::pair<double, double> next() { return s ? s->next() : m->next(); }
};

// A density estimator fed the bank's inputs: probit of a 500-tick rank PIT.
struct PitFed {
    std::unique_ptr<DependenceEstimator> e;
    RankPit px{500}, py{500};
    void update(double x, double y) { e->update(RankPit::probit(px.transform(x)), RankPit::probit(py.transform(y))); }
};

// chi_L(q), chi_U(q) of a density on the unit square: corner mass over tail-strip mass, midpoint rule
// with 16 nodes across each tail strip and 32 across the middle.
template <typename F>
std::pair<double, double> chi_density(F&& f, double q) {
    constexpr int kE = 16, kM = 32;
    std::vector<std::pair<double, double>> vs;  // (node, width)
    for (int i = 0; i < kE; ++i) vs.push_back({q * (i + 0.5) / kE, q / kE});
    for (int i = 0; i < kM; ++i) vs.push_back({q + (1 - 2 * q) * (i + 0.5) / kM, (1 - 2 * q) / kM});
    for (int i = 0; i < kE; ++i) vs.push_back({1 - q + q * (i + 0.5) / kE, q / kE});
    double nl = 0, dl = 0, nu = 0, du = 0;
    for (int i = 0; i < kE; ++i) {
        const double ul = q * (i + 0.5) / kE, uh = 1 - q + q * (i + 0.5) / kE;
        for (std::size_t j = 0; j < vs.size(); ++j) {
            const auto [v, w] = vs[j];
            const double a = f(ul, v) * w, b = f(uh, v) * w;
            dl += a;
            du += b;
            if (j < std::size_t(kE)) nl += a;
            if (j >= vs.size() - kE) nu += b;
        }
    }
    return {dl > 0 ? nl / dl : 0.0, du > 0 ? nu / du : 0.0};
}

struct Win { const char* name; std::size_t a, b; };
const std::vector<Win> kWins{{"whole", BURN, T}, {"pre", BURN, TAU}, {"post_0_500", TAU, TAU + 500},
                             {"post_500_2000", TAU + 500, TAU + 2000}, {"post_2000_5000", TAU + 2000, T}};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: probe_tail_bank <regime> <first_seed> <seeds>\n"); return 1; }
    const std::string rname = argv[1];
    const unsigned first = static_cast<unsigned>(std::atoi(argv[2]));
    const int seeds = std::atoi(argv[3]);
    const Regime rg = regime_of(rname);
    const std::string tag = rname + "_" + std::to_string(first) + ".csv";

    // Targets [q][before/after] = (chi_L, chi_U), with the Monte Carlo check.
    std::vector<std::array<std::pair<double, double>, 2>> tgt(kQ.size());
    for (std::size_t i = 0; i < kQ.size(); ++i) {
        tgt[i][0] = target(rg.before, kQ[i]);
        tgt[i][1] = target(rg.after, kQ[i]);
    }
    for (const auto& f : {rg.before, rg.after}) {
        if (f.rfind("mix", 0) == 0) continue;
        const auto c = tail::exact_corners(tail::fix(f, 0.6), kQ, 4000000);
        for (std::size_t i = 0; i < kQ.size(); ++i) {
            const auto t = target(f, kQ[i]);
            std::fprintf(stderr, "target %s q=%.2f chi_L %.4f (MC %.4f) chi_U %.4f (MC %.4f)\n", f.c_str(), kQ[i],
                         t.first, c.pL[i] / kQ[i], t.second, c.pU[i] / kQ[i]);
        }
    }

    std::FILE* out = std::fopen(("results/tail_bank_" + tag).c_str(), "w");
    std::FILE* pout = std::fopen(("results/tail_bank_path_" + tag).c_str(), "w");
    std::FILE* cout_ = std::fopen(("results/tail_bank_cross_" + tag).c_str(), "w");
    if (!out || !pout || !cout_) { std::fprintf(stderr, "cannot write results/ (run from experiments/)\n"); return 1; }
    std::fprintf(out, "regime,seed,q,tail,window,method,rmse,bias,mean\n");
    std::fprintf(pout, "regime,t,series,value\n");
    std::fprintf(cout_, "regime,q,tail,method,old,new,ticks_to_half\n");

    std::vector<std::string> methods{"family_bank", "tail_chi_399", "tail_chi_1999", "tail_chi_4999", "tail_chi_1999_known",
                                     "hermite_bank_own", "hermite_399", "hermite_1999", "xiao_399", "xiao_1999"};
    constexpr std::size_t kDens0 = 5, kNDens = 5;   // methods[kDens0 ..] read chi off a density
    std::vector<std::string> arm_names;
    {
        FamilyBank probe(FamilyBank::Config{}, "x");
        for (std::size_t k = 0; k < probe.size(); ++k)
            arm_names.push_back(std::string("arm_") + kShapeName[int(probe.arm(k).shape)] + "_" + std::to_string(probe.arm(k).mem));
    }
    const std::size_t nmain = methods.size();
    for (const auto& a : arm_names) methods.push_back(a);
    const std::size_t M = methods.size(), NQ = kQ.size();
    // Seed-mean estimate per [q][tail][main method][t], and shape weights per tick.
    std::vector<std::vector<double>> mean_est(NQ * 2 * nmain, std::vector<double>(T, 0.0));
    std::vector<std::vector<double>> mean_w(copula::kShapes, std::vector<double>(T, 0.0));

    for (int s = 0; s < seeds; ++s) {
        const unsigned seed = first + s;
        Source pre(rg.before, seed), post(rg.after, seed + 100000u);
        auto e_fb = make_estimator("family_bank");
        auto* fb = dynamic_cast<FamilyBank*>(e_fb.get());
        std::vector<std::unique_ptr<DependenceEstimator>> tc;
        for (const char* k : {"tail_chi_399", "tail_chi_1999", "tail_chi_4999"}) tc.push_back(make_estimator(k));
        TailChi known(TailChi::Config{.memory = 1999}, "known");
        auto e_hb = make_estimator("hermite_bank_own");
        auto* hb = dynamic_cast<ForgettingBank*>(e_hb.get());
        std::vector<PitFed> dens;
        for (double lam : {0.005, 0.001})
            dens.push_back({std::make_unique<HermiteEstimator>(
                HermiteEstimator::Config{.order = 10, .lambda = lam, .burn_in = 50}, "h")});
        for (double lam : {0.995, 0.999})
            dens.push_back({std::make_unique<XiaoSpearmanEstimator>(
                XiaoSpearmanEstimator::Config{.bins = 20, .lambda = lam, .calibration = 100, .density_bins = 20,
                                              .predictive_prior = 0.5 / 16.0}, "x")});
        std::vector<std::array<double, 4>> dchi(kNDens, {0, 0, 0, 0});   // [estimator] = L q05, U q05, L q10, U q10
        // est[q][tail][method][t]
        std::vector<std::vector<float>> est(NQ * 2 * M, std::vector<float>(T));
        auto at = [&](std::size_t qi, int tl, std::size_t m) -> std::vector<float>& { return est[(qi * 2 + tl) * M + m]; };

        for (std::size_t t = 0; t < T; ++t) {
            const auto [u, v] = t < TAU ? pre.next() : post.next();
            if (t % 20 == 0)
                for (std::size_t qi = 0; qi < NQ; ++qi) {
                    const auto c0 = chi_density([&](double a, double b) { return std::exp(hb->log_density(a, b)); }, kQ[qi]);
                    dchi[0][2 * qi] = c0.first;
                    dchi[0][2 * qi + 1] = c0.second;
                    for (std::size_t k = 0; k < dens.size(); ++k) {
                        const auto* pd = dynamic_cast<const PredictiveDensity*>(dens[k].e.get());
                        const auto c = chi_density([&](double a, double b) { return std::exp(pd->log_predictive(a, b)); }, kQ[qi]);
                        dchi[1 + k][2 * qi] = c.first;
                        dchi[1 + k][2 * qi + 1] = c.second;
                    }
                }
            for (std::size_t qi = 0; qi < NQ; ++qi) {
                const double q = kQ[qi];
                std::vector<TailEstimate> te{fb->tails(q)};
                for (auto& e : tc) te.push_back(dynamic_cast<const TailEstimator*>(e.get())->tails(q));
                te.push_back(known.tails(q));
                for (std::size_t k = 0; k < kNDens; ++k) te.push_back({dchi[k][2 * qi], dchi[k][2 * qi + 1], NAN, NAN});
                for (std::size_t m = 0; m < nmain; ++m) {
                    at(qi, 0, m)[t] = float(te[m].chi_L);
                    at(qi, 1, m)[t] = float(te[m].chi_U);
                    mean_est[(qi * 2 + 0) * nmain + m][t] += te[m].chi_L / seeds;
                    mean_est[(qi * 2 + 1) * nmain + m][t] += te[m].chi_U / seeds;
                }
                for (std::size_t k = 0; k < fb->size(); ++k) {
                    const auto a = fb->arm_tails(k, q);
                    at(qi, 0, nmain + k)[t] = float(a.chi_L);
                    at(qi, 1, nmain + k)[t] = float(a.chi_U);
                }
            }
            for (std::size_t k = 0; k < fb->size(); ++k) mean_w[int(fb->arm(k).shape)][t] += fb->weight(k) / seeds;

            const double x = special::normal_ppf(u), y = special::normal_ppf(v);
            fb->update(x, y);
            for (auto& e : tc) e->update(x, y);
            hb->update(x, y);
            for (auto& d : dens) d.update(x, y);
            if (t >= 30) known.update_uv(u, v);   // same warm-up as the rank-PIT counters
        }
        for (std::size_t qi = 0; qi < NQ; ++qi)
            for (int tl = 0; tl < 2; ++tl)
                for (std::size_t m = 0; m < M; ++m)
                    for (const auto& w : kWins) {
                        double se = 0, b = 0, mu = 0;
                        for (std::size_t t = w.a; t < w.b; ++t) {
                            const auto& tg = tgt[qi][t < TAU ? 0 : 1];
                            const double d = at(qi, tl, m)[t] - (tl ? tg.second : tg.first);
                            se += d * d;
                            b += d;
                            mu += at(qi, tl, m)[t];
                        }
                        const double n = double(w.b - w.a);
                        std::fprintf(out, "%s,%u,%.2f,%s,%s,%s,%.6f,%.6f,%.6f\n", rname.c_str(), seed, kQ[qi],
                                     tl ? "upper" : "lower", w.name, methods[m].c_str(), std::sqrt(se / n), b / n, mu / n);
                    }
        std::fprintf(stderr, "%s seed %u\n", rname.c_str(), seed);
    }

    for (std::size_t qi = 0; qi < NQ; ++qi)
        for (int tl = 0; tl < 2; ++tl) {
            const double o = tl ? tgt[qi][0].second : tgt[qi][0].first, nw = tl ? tgt[qi][1].second : tgt[qi][1].first;
            for (std::size_t m = 0; m < nmain; ++m) {
                const auto& p = mean_est[(qi * 2 + tl) * nmain + m];
                long cross = -1;
                if (std::fabs(nw - o) > 1e-3)
                    for (std::size_t t = TAU; t < T; ++t)
                        if ((p[t] - o) / (nw - o) >= 0.5) { cross = long(t - TAU); break; }
                std::fprintf(cout_, "%s,%.2f,%s,%s,%.4f,%.4f,%ld\n", rname.c_str(), kQ[qi], tl ? "upper" : "lower",
                             methods[m].c_str(), o, nw, cross);
            }
        }
    for (std::size_t t = 0; t < T; t += STRIDE) {
        for (std::size_t qi = 0; qi < NQ; ++qi)
            for (int tl = 0; tl < 2; ++tl) {
                const auto& tg = tgt[qi][t < TAU ? 0 : 1];
                const std::string pre = std::string(tl ? "chiU" : "chiL") + (qi ? "_q10_" : "_q05_");
                std::fprintf(pout, "%s,%zu,%strue,%.5f\n", rname.c_str(), t, pre.c_str(), tl ? tg.second : tg.first);
                for (std::size_t m = 0; m < nmain; ++m)
                    std::fprintf(pout, "%s,%zu,%s%s,%.5f\n", rname.c_str(), t, pre.c_str(), methods[m].c_str(),
                                 mean_est[(qi * 2 + tl) * nmain + m][t]);
            }
        for (int j = 0; j < copula::kShapes; ++j)
            std::fprintf(pout, "%s,%zu,w_%s,%.5f\n", rname.c_str(), t, kShapeName[j], mean_w[j][t]);
    }
    std::fclose(out);
    std::fclose(pout);
    std::fclose(cout_);
    return 0;
}
