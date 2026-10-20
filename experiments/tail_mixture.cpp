// Clayton-Gumbel mixture: feasibility and calibration at matched rho_S. See notes/studies/tail_dependence.md.
//
// C = pi * Clayton(theta) + (1 - pi) * Gumbel(zeta)   (Shamiri, Hamzah & Pirmoradian 2011), sampled
// generatively: Clayton with probability pi, else Gumbel, via the step-1 validated samplers.
// Because C is a convex combination, every functional LINEAR in C is the same mixture of the
// component values: C(q,q), so chi_L(q) and chi_U(q); lambda_L, lambda_U; and rho_S = 12 int C - 3.
// Kendall's tau = 4 E[C(U,V)] - 1 is quadratic in C, so it is not. Section I checks all of this by
// Monte Carlo; section II maps what (lambda_L, lambda_U) and (chi_L, chi_U) are reachable at the
// target rho_S; section III verifies candidate regimes.
//
// Usage: ./tail_mixture [d] [seed_base]   (defaults 0.6 -> rho_S 0.58192, 5150)
// Writes results/tail_mixture_feasible.csv.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <numbers>
#include <string>
#include <vector>

#include "tail_common.hpp"

namespace {

using copula::Family;
unsigned g_seed_base = 5150u;

const std::vector<double> kQ = {0.01, 0.02, 0.05, 0.10};

using tail::Mix;
using tail::MixSampler;
double rho_clayton(double th) { return tail::rho_clayton(th); }
double rho_gumbel(double ze) { return tail::rho_gumbel(ze); }

struct MC {
    double rho_s, rho_s_se, tau;
    std::vector<double> chiL, chiU;
};

// rho_S and chi by Monte Carlo; tau = 4 E[C(U,V)] - 1 evaluated with the exact mixture CDF.
MC monte_carlo(const Mix& m, std::size_t n) {
    const unsigned threads = 8;
    std::vector<std::future<std::vector<double>>> jobs;
    for (unsigned k = 0; k < threads; ++k)
        jobs.push_back(std::async(std::launch::async, [&m, n, k] {
            MixSampler s(m, g_seed_base + k);
            std::vector<double> acc(3 + 2 * kQ.size(), 0.0);
            for (std::size_t i = 0; i < n / threads; ++i) {
                const auto [u, v] = s.next();
                acc[0] += u * v;
                acc[1] += u * v * u * v;
                acc[2] += m.C(u, v);
                for (std::size_t j = 0; j < kQ.size(); ++j) {
                    acc[3 + j] += (u <= kQ[j] && v <= kQ[j]);
                    acc[3 + kQ.size() + j] += (u > 1.0 - kQ[j] && v > 1.0 - kQ[j]);
                }
            }
            return acc;
        }));
    std::vector<double> acc(3 + 2 * kQ.size(), 0.0);
    for (auto& j : jobs) { auto a = j.get(); for (std::size_t i = 0; i < a.size(); ++i) acc[i] += a[i]; }
    const double N = static_cast<double>(n / threads * threads);
    const double e = acc[0] / N, var = acc[1] / N - e * e;
    MC r{12.0 * e - 3.0, 12.0 * std::sqrt(var / N), 4.0 * acc[2] / N - 1.0, {}, {}};
    for (std::size_t j = 0; j < kQ.size(); ++j) {
        r.chiL.push_back(acc[3 + j] / N / kQ[j]);
        r.chiU.push_back(acc[3 + kQ.size() + j] / N / kQ[j]);
    }
    return r;
}

double tau_component(Family f, double p) {
    // Closed forms, for the "weighted average of tau" comparison only.
    return f == Family::Clayton ? p / (p + 2.0) : 1.0 - 1.0 / p;
}

void report(const char* label, const Mix& m, std::size_t n) {
    const MC mc = monte_carlo(m, n);
    const double tau_avg = m.pi * tau_component(Family::Clayton, m.theta) + (1.0 - m.pi) * tau_component(Family::Gumbel, m.zeta);
    std::printf("%-26s pi=%.3f theta=%7.4f zeta=%7.4f | rho_S exact %.4f mc %.4f (z=%+.1f) | tau mc %.4f vs avg %.4f | "
                "lam %.3f/%.3f\n",
                label, m.pi, m.theta, m.zeta, m.rho_s(), mc.rho_s, (mc.rho_s - m.rho_s()) / mc.rho_s_se, mc.tau,
                tau_avg, m.lam_L(), m.lam_U());
    std::printf("%-26s   chi_L exact/mc:", "");
    for (std::size_t j = 0; j < kQ.size(); ++j) std::printf("  q=%.2f %.3f/%.3f", kQ[j], m.chi_L(kQ[j]), mc.chiL[j]);
    std::printf("\n%-26s   chi_U exact/mc:", "");
    for (std::size_t j = 0; j < kQ.size(); ++j) std::printf("  q=%.2f %.3f/%.3f", kQ[j], m.chi_U(kQ[j]), mc.chiU[j]);
    std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
    const double d = (argc > 1) ? std::atof(argv[1]) : 0.6;
    if (argc > 2) g_seed_base = static_cast<unsigned>(std::atoi(argv[2]));
    const double R0 = 6.0 / std::numbers::pi * std::asin(d / 2.0);
    const double th0 = dgp::param_from_spearman(Family::Clayton, R0);
    const double ze0 = dgp::param_from_spearman(Family::Gumbel, R0);
    std::printf("target rho_S = %.5f; Clayton theta0 = %.4f, Gumbel zeta0 = %.4f at that rho_S\n\n", R0, th0, ze0);

    std::printf("== I. linearity checks (4M draws; z for rho_S uses its own MC SE)\n");
    report("equal-rho, pi=0.5", {0.5, th0, ze0}, 4000000);
    report("unequal, pi=0.3", {0.3, dgp::param_from_spearman(Family::Clayton, 0.85),
                               dgp::param_from_spearman(Family::Gumbel, (R0 - 0.3 * 0.85) / 0.7)}, 4000000);
    report("unequal, pi=0.7", {0.7, dgp::param_from_spearman(Family::Clayton, 0.45),
                               dgp::param_from_spearman(Family::Gumbel, (R0 - 0.7 * 0.45) / 0.3)}, 4000000);

    // II. constraint surface: pi grid x rho_C grid, rho_G fixed by pi*rho_C + (1-pi)*rho_G = R0.
    const double rc_lo = rho_clayton(copula::Clayton::kMinParam), rc_hi = rho_clayton(copula::Clayton::kMaxParam);
    const double rg_lo = rho_gumbel(copula::Gumbel::kMinParam), rg_hi = rho_gumbel(copula::Gumbel::kMaxParam);
    std::printf("\n== II. feasible set at rho_S = %.5f (Clayton theta <= %.0f -> rho_C <= %.4f; Gumbel zeta <= %.0f -> rho_G <= %.4f)\n",
                R0, copula::Clayton::kMaxParam, rc_hi, copula::Gumbel::kMaxParam, rg_hi);
    std::ofstream csv("results/tail_mixture_feasible.csv");
    if (!csv.is_open()) { std::fprintf(stderr, "cannot open results/ (run: mkdir -p results)\n"); return 1; }
    csv << "pi,theta,zeta,rho_C,rho_G,lam_L,lam_U";
    for (double q : kQ) csv << ",chiL_" << q << ",chiU_" << q;
    csv << "\n";
    struct Pt { double pi, rc, rg, th, ze, lL, lU, cL, cU; };
    std::vector<Pt> pts;
    for (int ip = 0; ip <= 100; ++ip) {
        const double pi = ip / 100.0;
        for (int ir = 0; ir <= 250; ++ir) {
            const double rc = rc_lo + (rc_hi - rc_lo) * ir / 250.0;
            double rg;
            if (ip == 100) { if (ir) continue; rg = R0; }
            else if (ip == 0) { if (ir) continue; rg = R0; }
            else rg = (R0 - pi * rc) / (1.0 - pi);
            if (rg < rg_lo || rg > rg_hi) continue;
            const double th = ip == 0 ? th0 : (ip == 100 ? th0 : dgp::param_from_spearman(Family::Clayton, rc));
            const double ze = ip == 100 ? ze0 : dgp::param_from_spearman(Family::Gumbel, rg);
            const Mix m{pi, th, ze};
            pts.push_back({pi, ip == 100 || ip == 0 ? R0 : rc, rg, th, ze, m.lam_L(), m.lam_U(), m.chi_L(0.05), m.chi_U(0.05)});
            csv << pi << "," << th << "," << ze << "," << pts.back().rc << "," << rg << "," << m.lam_L() << "," << m.lam_U();
            for (double q : kQ) csv << "," << m.chi_L(q) << "," << m.chi_U(q);
            csv << "\n";
        }
    }
    std::printf("%zu feasible grid points\n", pts.size());
    std::printf("lambda envelope:  lam_L bin   |  min lam_U  max lam_U   (and the pi giving the max)\n");
    for (int b = 0; b < 20; ++b) {
        const double lo = b * 0.05, hi = lo + 0.05;
        double mn = 9, mx = -1, pimx = NAN;
        for (const auto& p : pts)
            if (p.lL >= lo && p.lL < hi) { mn = std::min(mn, p.lU); if (p.lU > mx) { mx = p.lU; pimx = p.pi; } }
        if (mx >= 0) std::printf("                 [%.2f,%.2f)  |   %.3f      %.3f      pi=%.2f\n", lo, hi, mn, mx, pimx);
    }
    std::printf("chi(q=.05) envelope: chi_L bin |  min chi_U  max chi_U\n");
    for (int b = 0; b < 20; ++b) {
        const double lo = b * 0.05, hi = lo + 0.05;
        double mn = 9, mx = -1;
        for (const auto& p : pts)
            if (p.cL >= lo && p.cL < hi) { mn = std::min(mn, p.cU); mx = std::max(mx, p.cU); }
        if (mx >= 0) std::printf("                 [%.2f,%.2f)  |   %.3f      %.3f\n", lo, hi, mn, mx);
    }
    auto arg = [&pts](auto score) { return *std::max_element(pts.begin(), pts.end(), [&](const Pt& a, const Pt& b) { return score(a) < score(b); }); };
    for (auto [name, p] : {std::pair{"max lam_L", arg([](const Pt& p) { return p.lL; })},
                           std::pair{"max lam_U", arg([](const Pt& p) { return p.lU; })},
                           std::pair{"max lam_L - lam_U", arg([](const Pt& p) { return p.lL - p.lU; })},
                           std::pair{"max lam_U - lam_L", arg([](const Pt& p) { return p.lU - p.lL; })},
                           std::pair{"max lam_L + lam_U", arg([](const Pt& p) { return p.lL + p.lU; })}})
        std::printf("  %-18s pi=%.2f rho_C=%.3f rho_G=%.3f theta=%7.3f zeta=%7.3f -> lam %.3f/%.3f  chi(.05) %.3f/%.3f\n",
                    name, p.pi, p.rc, p.rg, p.th, p.ze, p.lL, p.lU, p.cL, p.cU);

    std::printf("\n== III. candidate path: both components held at rho_S = %.5f, only pi moves\n", R0);
    for (double pi : {1.0, 0.9, 0.75, 0.6, 0.5, 0.4, 0.25, 0.1, 0.0}) {
        char label[32];
        std::snprintf(label, sizeof label, "path pi=%.2f", pi);
        report(label, {pi, th0, ze0}, 4000000);
    }
    std::printf("\nwrote results/tail_mixture_feasible.csv\n");
    return 0;
}
