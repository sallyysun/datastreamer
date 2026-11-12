// Self-checks for density_metrics.hpp (evaluation only; notes/studies/chapter6_metrics.md §8). No estimator is
// changed. Parts A-D are deterministic quadratures of known densities. Part E builds one fixed estimator state
// per case from a single fixed stream (seed 42, 2000 ticks): a unit-test fixture of the density output's
// form, not an experiment, and no results are drawn from it.
// Usage (from experiments/): ./check_density_metrics  -> stdout
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "../src/estimators/registry.hpp"
#include "copula_source.hpp"
#include "density_metrics.hpp"

using F = copula::Family;
using dmetric::Dens;
using dmetric::Grid;
using dmetric::Table;

namespace {
double ts(double d) { return dgp::detail::target_spearman(d); }
double par(F f, double rho_s) { return copula::param_from_spearman(f, rho_s, 4.0); }
Dens fam(F f, double p) { return [=](double u, double v) { return std::exp(copula::log_density(f, u, v, p, 4.0)); }; }
Dens mix(std::vector<std::pair<double, Dens>> parts) {
    return [=](double u, double v) { double s = 0; for (const auto& [w, c] : parts) s += w * c(u, v); return s; };
}
Dens own(const PredictiveDensity* pd) {
    return [=](double u, double v) { const double l = pd->log_predictive(u, v); return std::isfinite(l) ? std::exp(l) : 0.0; };
}
}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const bool only_d = argc > 1 && std::string(argv[1]) == "D";   // skip A-C (E is still built: D uses its fixtures)

    (void)copula::spearman_from_param(F::StudentT, 0.5, 4.0);
    const double r6 = par(F::Gaussian, ts(0.6));
    if (argc > 1 && std::string(argv[1]) == "F") {
        // F. Grid size for the in-probe Hellinger (L = 8): error against n = 801 for representative forecast pairs.
        std::printf("F. H2 grid convergence at L = 8 (abs. error vs n = 801)\n");
        const double rs = ts(0.6);
        const Dens G = fam(F::Gaussian, r6), C = fam(F::Clayton, par(F::Clayton, rs)), Gu = fam(F::Gumbel, par(F::Gumbel, rs)),
                   T = fam(F::StudentT, par(F::StudentT, rs));
        const Dens FB = mix({{0.90, fam(F::Clayton, par(F::Clayton, 1.05 * rs))}, {0.04, T}, {0.03, Gu}, {0.03, G}});
        const Dens MIX = mix({{0.75, C}, {0.25, Gu}});
        struct P { const char* n; Dens c, ch; };
        for (const P& p : std::vector<P>{{"clayton vs family-like mixture", C, FB}, {"t4 vs t4 +5%", T, fam(F::StudentT, par(F::StudentT, 1.05 * rs))},
                                        {"mixture vs t4", MIX, T}, {"gumbel vs Gaussian r*", Gu, fam(F::Gaussian, 0.6046)},
                                        {"gaussian vs Gaussian +5%", G, fam(F::Gaussian, par(F::Gaussian, 1.05 * rs))}}) {
            const Grid g0 = Grid::full(801, 8.0);
            const double ref = dmetric::hellinger2(g0, Table(g0, p.c), Table(g0, p.ch));
            std::printf("   %-32s ref %.8f |", p.n, ref);
            for (int n : {31, 41, 61, 81, 101}) {
                const Grid g = Grid::full(n, 8.0);
                std::printf("  n=%d %.1e", n, std::abs(dmetric::hellinger2(g, Table(g, p.c), Table(g, p.ch)) - ref));
            }
            std::printf("\n");
        }
        return 0;
    }

    // ---------------- A. quadrature vs closed forms (two Gaussian copulas) ----------------
    if (!only_d) {
    std::printf("A. Quadrature vs closed form, Gaussian copulas (full grid [-L, L]^2, n per axis)\n");
    for (auto [r, s] : {std::pair{r6, par(F::Gaussian, 1.05 * ts(0.6))}, std::pair{0.2, 0.8}}) {
        std::printf("   r = %.4f, s = %.4f: closed H2 %.8f  KL %.8f  ISE %.8f\n", r, s, dmetric::gauss_h2(r, s),
                    dmetric::gauss_kl(r, s), dmetric::gauss_ise(r, s));
        for (double L : {6.0, 8.0})
            for (int n : {101, 201, 401, 801}) {
                const Grid g = Grid::full(n, L);
                const Table a(g, fam(F::Gaussian, r)), b(g, fam(F::Gaussian, s));
                std::printf("     L=%.0f n=%3d  mass %.8f  H2 %.8f  KL %.8f  ISE %.8f\n", L, n, dmetric::mass(g, a),
                            dmetric::hellinger2(g, a, b), dmetric::kl(g, a, b), dmetric::ise(g, a, b));
            }
    }

    // ---------------- B. normalisation of the true densities over the tested range ----------------
    std::printf("\nB. Mass of the true densities on the full grid (L = 8, n = 801), d in the tested range [0.2, 0.8]\n");
    {
        const Grid g = Grid::full(801, 8.0);
        for (F f : {F::Gaussian, F::Clayton, F::Gumbel, F::StudentT}) {
            std::printf("   %-9s", f == F::Gaussian ? "gaussian" : f == F::Clayton ? "clayton" : f == F::Gumbel ? "gumbel" : "t4");
            for (double d : {0.2, 0.6, 0.8}) std::printf("  d=%.1f mass %.6f", d, dmetric::mass(g, Table(g, fam(f, par(f, ts(d))))));
            std::printf("\n");
        }
    }

    // ---------------- C. Hellinger convergence, tail pairs, and the corner contribution ----------------
    std::printf("\nC. Squared Hellinger convergence (n, L) and the share of 1 - H2's integrand outside max(|x|,|y|) > 4\n");
    {
        const double rs = ts(0.6);
        struct P { const char* name; Dens c, ch; };
        std::vector<P> pairs{
            {"clayton vs clayton (rho_S +5%)", fam(F::Clayton, par(F::Clayton, rs)), fam(F::Clayton, par(F::Clayton, 1.05 * rs))},
            {"clayton vs gaussian (same rho_S)", fam(F::Clayton, par(F::Clayton, rs)), fam(F::Gaussian, r6)},
            {"gumbel vs gumbel (rho_S +5%)", fam(F::Gumbel, par(F::Gumbel, rs)), fam(F::Gumbel, par(F::Gumbel, 1.05 * rs))},
            {"t4 vs gaussian (same rho_S)", fam(F::StudentT, par(F::StudentT, rs)), fam(F::Gaussian, r6)},
            {"mix(0.75 C + 0.25 G) vs clayton", mix({{0.75, fam(F::Clayton, par(F::Clayton, rs))}, {0.25, fam(F::Gumbel, par(F::Gumbel, rs))}}),
             fam(F::Clayton, par(F::Clayton, rs))},
            {"gaussian vs 0.97 G + 0.01 C/Gu/t", fam(F::Gaussian, r6),
             mix({{0.97, fam(F::Gaussian, r6)}, {0.01, fam(F::Clayton, par(F::Clayton, rs))}, {0.01, fam(F::Gumbel, par(F::Gumbel, rs))},
                  {0.01, fam(F::StudentT, par(F::StudentT, rs))}})},
        };
        for (const auto& p : pairs) {
            std::printf("   %s\n", p.name);
            for (double L : {6.0, 7.0, 8.0, 9.0}) {
                std::printf("     L=%.0f:", L);
                for (int n : {101, 201, 401, 801}) {
                    const Grid g = Grid::full(n, L);
                    const Table a(g, p.c), b(g, p.ch);
                    std::printf("  n=%3d %.8f", n, dmetric::hellinger2(g, a, b));
                    if (n == 801 && L == 8.0) {
                        double tot = 0, out = 0;
                        for (int i = 0; i < n; ++i)
                            for (int j = 0; j < n; ++j) {
                                const double x = std::sqrt(a(i, j) * b(i, j)) * g.w[i] * g.w[j];
                                tot += x;
                                if (std::max(std::abs(g.z[i]), std::abs(g.z[j])) > 4.0) out += x;
                            }
                        std::printf("  [corner share %.2e]", out / tot);
                    }
                }
                std::printf("\n");
            }
        }
    }

    }   // !only_d

    // ---------------- E (built first: D uses the fixtures). Xiao and Hermite densities as output ----------------
    std::printf("\nE. Xiao and Hermite predictive densities as they are output (one fixed state each: seed 42, 2000 ticks, raw inputs,\n"
                "   bank arm configs). Grid L = 8, n = 401. 'series zero area' = share of [0,1]^2 where Hermite's series was clipped.\n");
    struct Fix { std::string name; std::unique_ptr<DependenceEstimator> est; bool hermite; double e; };
    std::vector<Fix> fixtures;
    auto run_fixture = [&](const std::string& cop, double d, std::unique_ptr<DependenceEstimator> est) {
        ConstantDependence dep(d);
        auto c = dgp::make_copula(cop);
        CopulaSource src(dep, *c, 42);
        for (int t = 0; t < 2000; ++t) { const auto s = src.next(); est->update(s.x, s.y); }
        return est;
    };
    for (const std::string cop : {"gaussian", "clayton"})
        for (double lam : {0.95, 0.999})
            fixtures.push_back({"xiao " + cop + " lambda=" + std::to_string(lam).substr(0, 5),
                                run_fixture(cop, 0.6, std::make_unique<XiaoSpearmanEstimator>(bank_experiment::xiao(lam), "x")), false, 0});
    for (const std::string cop : {"gaussian", "clayton"})
        for (double lam : {0.05, 0.001}) {
            auto cfg = bank_experiment::hermite(lam);
            fixtures.push_back({"hermite " + cop + " lambda=" + std::to_string(lam).substr(0, 5),
                                run_fixture(cop, 0.6, std::make_unique<HermiteEstimator>(cfg, "h")), true, cfg.predictive_floor});
        }
    std::vector<Dens> fixture_dens(fixtures.size());
    {
        for (int n : {201, 401, 801}) {
            const Grid g = Grid::full(n, 8.0);
            std::printf("   grid n = %d\n", n);
            for (std::size_t k = 0; k < fixtures.size(); ++k) {
                const auto* pd = dynamic_cast<const PredictiveDensity*>(fixtures[k].est.get());
                const Table a(g, own(pd));
                double mn = 1e300, zero_area = 0;
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j) {
                        mn = std::min(mn, a(i, j));
                        if (fixtures[k].hermite && a(i, j) <= fixtures[k].e * (1 + 1e-9)) zero_area += g.w[i] * g.w[j];
                    }
                const double m = dmetric::mass(g, a);
                const auto mc = dmetric::margins(g, a);
                std::printf("     %-28s min %.4f  mass %.5f  margin max|m-1| %.4f  L1 %.4f", fixtures[k].name.c_str(), mn, m, mc.max_dev, mc.l1_dev);
                if (fixtures[k].hermite) {
                    // Renormalised in evaluation code only: g = (exp(lp) - e)/(1 - e) is the clipped series density.
                    const double e = fixtures[k].e;
                    const double Z = dmetric::integrate(g, [&](int i, int j) { return (a(i, j) - e) / (1 - e); });
                    Table b = a;
                    for (auto& x : b.v) x = (1 - e) * ((x - e) / (1 - e)) / Z + e;
                    const auto mb = dmetric::margins(g, b);
                    std::printf("  series zero area %.3f | renormalised: Z %.5f mass %.5f margin max|m-1| %.4f L1 %.4f",
                                zero_area, Z, dmetric::mass(g, b), mb.max_dev, mb.l1_dev);
                    if (n == 401) {
                        const double ee = e, ZZ = Z;
                        fixture_dens[k] = [pd, ee, ZZ](double u, double v) {
                            const double l = pd->log_predictive(u, v);
                            return std::isfinite(l) ? (1 - ee) * ((std::exp(l) - ee) / (1 - ee)) / ZZ + ee : 0.0;
                        };
                    }
                } else if (n == 401) {
                    fixture_dens[k] = own(pd);
                }
                std::printf("\n");
            }
        }
    }

    // ---------------- D. full-domain ISE finiteness for the actual true / estimated pairs ----------------
    std::printf("\nD. Interior ISE on [eps, 1-eps]^2 for eps = 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, and its increments. Full-domain ISE is the\n"
                "   eps -> 0 limit: finite if the increments shrink geometrically, infinite (log-divergent) if they stay roughly constant.\n");
    {
        const double rs = ts(0.6);
        const Dens G = fam(F::Gaussian, r6), C = fam(F::Clayton, par(F::Clayton, rs)), Gu = fam(F::Gumbel, par(F::Gumbel, rs)),
                   T = fam(F::StudentT, par(F::StudentT, rs));
        const Dens Gmix = mix({{0.25, fam(F::Gaussian, par(F::Gaussian, 0.95 * rs))}, {0.5, fam(F::Gaussian, par(F::Gaussian, 1.02 * rs))},
                               {0.25, fam(F::Gaussian, par(F::Gaussian, 1.08 * rs))}});
        auto fb_like = [&](Dens main) {   // a family-bank-like mixture: one dominant arm, every other arm at fixed-share weight
            return mix({{0.97, main}, {0.01, C}, {0.01, Gu}, {0.01, T}});
        };
        const Dens MIX75 = mix({{0.75, C}, {0.25, Gu}});
        struct P { const char* truth; const char* est; Dens c, ch; double closed = -1; };
        auto gpair = [&](double d) {
            const double a = par(F::Gaussian, ts(d)), b = par(F::Gaussian, 1.05 * ts(d));
            return P{"", "", fam(F::Gaussian, a), fam(F::Gaussian, b), dmetric::gauss_ise(a, b)};
        };
        P g2 = gpair(0.2), g6 = gpair(0.6), g8 = gpair(0.8);
        g2.truth = "gaussian d=0.2"; g6.truth = "gaussian d=0.6"; g8.truth = "gaussian d=0.8";
        g2.est = g6.est = g8.est = "Gaussian, rho_S +5% (RMPL / single Gaussian)";
        std::vector<P> pairs{g6, g2, g8,
            {"gaussian d=0.6", "Gaussian mixture (log-bank style)", G, Gmix},
            {"gaussian d=0.6", "family-bank-like mixture", G, fb_like(fam(F::Gaussian, par(F::Gaussian, 1.05 * rs)))},
            {"gaussian d=0.6", "Xiao histogram (fixture, gaussian, 0.999)", G, fixture_dens[1]},
            {"gaussian d=0.6", "Hermite renormalised (fixture, gaussian, 0.001)", G, fixture_dens[5]},
            {"clayton d=0.6", "Clayton, rho_S +5% (best arm)", C, fam(F::Clayton, par(F::Clayton, 1.05 * rs))},
            {"clayton d=0.2", "Clayton, rho_S +5%", fam(F::Clayton, par(F::Clayton, ts(0.2))), fam(F::Clayton, par(F::Clayton, 1.05 * ts(0.2)))},
            {"clayton d=0.6", "Gaussian at r* (single Gaussian)", C, fam(F::Gaussian, 0.5949)},
            {"clayton d=0.6", "family-bank-like mixture", C, fb_like(fam(F::Clayton, par(F::Clayton, 1.05 * rs)))},
            {"clayton d=0.6", "Xiao histogram (fixture, clayton, 0.999)", C, fixture_dens[3]},
            {"clayton d=0.6", "Hermite renormalised (fixture, clayton, 0.001)", C, fixture_dens[7]},
            {"gumbel d=0.6", "Gumbel, rho_S +5%", Gu, fam(F::Gumbel, par(F::Gumbel, 1.05 * rs))},
            {"gumbel d=0.6", "Gaussian at r*", Gu, fam(F::Gaussian, 0.6046)},
            {"t4 d=0.6", "t4, rho_S +5%", T, fam(F::StudentT, par(F::StudentT, 1.05 * rs))},
            {"t4 d=0.6", "Gaussian at r*", T, fam(F::Gaussian, 0.6056)},
            {"mixture pi=0.75", "Clayton at the same rho_S", MIX75, C},
            {"mixture pi=0.75", "t4 at the same rho_S", MIX75, T},
            {"mixture pi=0.75", "Gaussian at r*", MIX75, fam(F::Gaussian, 0.5973)},
        };
        for (const auto& p : pairs) {
            std::vector<double> v;
            for (double eps : {1e-2, 1e-4, 1e-6, 1e-8, 1e-10}) {
                const Grid g = Grid::interior(601, eps);
                v.push_back(dmetric::ise(g, Table(g, p.c), Table(g, p.ch)));
            }
            std::printf("   %-16s | %-48s ISE_eps:", p.truth, p.est);
            for (double x : v) std::printf(" %.6f", x);
            std::printf(" | increments:");
            for (std::size_t i = 1; i < v.size(); ++i) std::printf(" %.2e", v[i] - v[i - 1]);
            // Equal log-steps in eps: a log-divergent integral gains a constant amount per step (ratio ~1).
            const double r1 = (v[3] - v[2]) / (v[2] - v[1]), r2 = (v[4] - v[3]) / (v[3] - v[2]);
            const char* verdict = p.closed >= 0 ? "finite (closed form)"
                                : (r1 > 0.85 && r2 > 0.85) ? "INFINITE (log-divergent)"
                                : (r2 < r1 && r2 < 0.85) ? "decreasing increments: see theory" : "irregular: see theory";
            std::printf(" | step ratios %.2f %.2f -> %s", r1, r2, verdict);
            if (p.closed >= 0) std::printf(" (closed form %.6f)", p.closed);
            std::printf("\n");
        }
    }
    // D2. Lower corner only, where u can go far below double's 1 - eps limit: Gaussian truth vs 0.99 Gaussian + 0.01
    // Clayton (the family bank's fixed-share floor on one tail-dependent arm). int over [eps, 1/2]^2.
    std::printf("\nD2. Lower-corner interior ISE on [eps, 1/2]^2, Gaussian truth (d = 0.6) vs 0.99 Gaussian + 0.01 Clayton:\n   ");
    {
        const double rs = ts(0.6);
        const Dens G = fam(F::Gaussian, r6), C = fam(F::Clayton, par(F::Clayton, rs)), M = mix({{0.99, G}, {0.01, C}});
        double prev = 0;
        for (double eps : {1e-2, 1e-6, 1e-10, 1e-20, 1e-40, 1e-80}) {
            const Grid g(1201, dmetric::probit(eps), 0.0);
            const double v = dmetric::ise(g, Table(g, G), Table(g, M));
            std::printf(" eps=%.0e %.3e (+%.2e)", eps, v, v - prev);
            prev = v;
        }
        std::printf("\n   Clayton's own corner: c ~ K/s on the diagonal, so (0.01 c_C)^2 adds a constant per decade of eps.\n");
    }
    return 0;
}
