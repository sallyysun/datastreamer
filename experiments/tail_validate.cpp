// Stationary validation of online tail-dependence estimators. See notes/studies/tail_dependence.md.
//
// Copula space, known uniform margins. For Gaussian / Clayton / Gumbel / t(4) at one matched
// Spearman rho_S (the DGP's convention), reports:
//   A. exact lambda_L, lambda_U and the finite-threshold targets chi_L(q), chi_U(q);
//   B. bias / SD / RMSE of exponentially-forgotten SEC, SEC-ratio and LOG estimators over a
//      memory grid, against their own finite-q target and against lambda;
//   C. signal-to-noise of the matched-rho_S transitions G->Clayton, G->Gumbel, G->t, Clayton->Gumbel.
//
// Samplers fix the family parameter once (copula.hpp re-solves it per draw: 225 us/draw for t)
// and reproduce copula.hpp's formulas; section V cross-checks them against the DGP classes.
//
// Usage: ./tail_validate [d] [n_seeds]   (d = Gaussian-equivalent knob, default 0.6 -> rho_S 0.582)
// Writes results/tail_validate_targets.csv and results/tail_validate_stream.csv.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include "../src/estimators/copula_family.hpp"
#include "../src/estimators/special_functions.hpp"
#include "../src/streamgenerator/datasource/copula.hpp"
#include "tail_common.hpp"

namespace {

using tail::Fixed;
using tail::Lambdas;
using tail::NU;
using tail::Sampler;
using tail::closed_C;
using tail::exact_lambda;
using tail::fix;

const std::vector<std::string> kFamilies = {"gaussian", "clayton", "gumbel", "t"};
const std::vector<double> kQ = {0.01, 0.02, 0.05, 0.10};
const std::vector<double> kMem = {250, 500, 1000, 2000, 5000, 10000, 20000, 50000};
const std::vector<std::string> kEst = {"sec", "ratio", "log"};
constexpr std::size_t T_STREAM = 300000, BURN = 150000, EVAL = 100;
constexpr std::size_t N_EXACT = 20000000;

// Population target of each estimator from the joint corner probability pj = P(corner) at q.
double target_of(const std::string& est, double pj, double q) {
    if (est == "log") return 2.0 - std::log(1.0 - 2.0 * q + pj) / std::log(1.0 - q);
    return pj / q;
}

struct Exact {
    double rho_s;
    std::vector<double> pL, pU;  // P(U<=q,V<=q), P(U>1-q,V>1-q) per q
    std::vector<double> pL_closed, pU_closed;
};

Exact exact_targets(const Fixed& f) {
    const tail::Corners c = tail::exact_corners(f, kQ, N_EXACT);
    Exact e;
    e.rho_s = c.rho_s;
    for (std::size_t j = 0; j < kQ.size(); ++j) {
        const double q = kQ[j];
        e.pL.push_back(c.pL[j]);
        e.pU.push_back(c.pU[j]);
        e.pL_closed.push_back(closed_C(f, q));
        const double cc = closed_C(f, 1.0 - q);
        e.pU_closed.push_back(std::isnan(cc) ? NAN : 1.0 - 2.0 * (1.0 - q) + cc);
    }
    return e;
}

std::size_t sidx(std::size_t tail, std::size_t qi, std::size_t mi) {
    return (tail * kQ.size() + qi) * kMem.size() + mi;
}

// Exponentially forgotten corner fractions, normalised by accumulated weight (no startup bias).
// Per seed: estimate samples [tail][q][m][est].
std::vector<std::vector<double>> run_stream(const Fixed& f, unsigned seed) {
    Sampler s(f, seed);
    const std::size_t cells = 2 * kQ.size() * kMem.size();
    std::vector<double> joint(cells, 0.0), marg(cells, 0.0), total(cells, 0.0);
    std::vector<double> keep(kMem.size());
    for (std::size_t mi = 0; mi < kMem.size(); ++mi) keep[mi] = 1.0 - 2.0 / (kMem[mi] + 1.0);
    std::vector<std::vector<double>> out(cells * kEst.size());
    for (std::size_t t = 0; t < T_STREAM; ++t) {
        const auto [u, v] = s.next();
        for (std::size_t tail = 0; tail < 2; ++tail)
            for (std::size_t qi = 0; qi < kQ.size(); ++qi) {
                const double q = kQ[qi];
                const bool mu = tail == 0 ? u <= q : u > 1.0 - q;
                const bool mv = tail == 0 ? v <= q : v > 1.0 - q;
                for (std::size_t mi = 0; mi < kMem.size(); ++mi) {
                    const std::size_t i = sidx(tail, qi, mi);
                    const double l = keep[mi];
                    joint[i] = l * joint[i] + (1.0 - l) * (mu && mv);
                    marg[i] = l * marg[i] + (1.0 - l) * mu;
                    total[i] = l * total[i] + (1.0 - l);
                }
            }
        if (t < BURN || t % EVAL) continue;
        for (std::size_t i = 0; i < cells; ++i) {
            const double q = kQ[(i / kMem.size()) % kQ.size()];
            const double pj = joint[i] / total[i];
            out[i * kEst.size() + 0].push_back(pj / q);
            out[i * kEst.size() + 1].push_back(marg[i] > 0.0 ? joint[i] / marg[i] : 0.0);
            out[i * kEst.size() + 2].push_back(target_of("log", pj, q));
        }
    }
    return out;
}

void cross_check(const std::string& fam, double d) {
    // DGP class vs local sampler: chi_L, chi_U at q = 0.05 and rho_S, each as a two-sample z.
    const std::size_t n = fam == "t" ? 160000 : 1000000;
    auto cop = dgp::make_copula(fam);
    std::mt19937 rng(777);
    Sampler loc(fix(fam, d), 778);
    double a[4] = {0, 0, 0, 0}, b[4] = {0, 0, 0, 0};
    for (std::size_t i = 0; i < n; ++i) {
        const auto dr = cop->sample(d, rng);
        const double u = special::normal_cdf(dr.z1), v = special::normal_cdf(dr.z2);
        a[0] += (u <= 0.05 && v <= 0.05); a[1] += (u > 0.95 && v > 0.95); a[2] += u * v; a[3] += u * v * u * v;
        const auto [x, y] = loc.next();
        b[0] += (x <= 0.05 && y <= 0.05); b[1] += (x > 0.95 && y > 0.95); b[2] += x * y; b[3] += x * y * x * y;
    }
    const double N = static_cast<double>(n);
    auto zp = [N](double c1, double c2) {
        const double p1 = c1 / N, p2 = c2 / N;
        return (p1 - p2) / std::sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / N);
    };
    const double va = a[3] / N - (a[2] / N) * (a[2] / N), vb = b[3] / N - (b[2] / N) * (b[2] / N);
    const double zr = (a[2] - b[2]) / N / std::sqrt((va + vb) / N);
    std::printf("  %-9s n=%-8zu chi_L(.05) %.3f vs %.3f (z=%+.1f)  chi_U(.05) %.3f vs %.3f (z=%+.1f)  "
                "rho_S %.4f vs %.4f (z=%+.1f)\n",
                fam.c_str(), n, a[0] / N / 0.05, b[0] / N / 0.05, zp(a[0], b[0]), a[1] / N / 0.05, b[1] / N / 0.05,
                zp(a[1], b[1]), 12 * a[2] / N - 3, 12 * b[2] / N - 3, zr);
}

}  // namespace

int main(int argc, char** argv) {
    const double d = (argc > 1) ? std::atof(argv[1]) : 0.6;
    const int n_seeds = (argc > 2) ? std::atoi(argv[2]) : 20;

    std::printf("== V. sampler cross-check (DGP class vs fixed-parameter sampler)\n");
    for (const auto& f : kFamilies) cross_check(f, d);

    std::ofstream tcsv("results/tail_validate_targets.csv"), scsv("results/tail_validate_stream.csv");
    if (!tcsv.is_open() || !scsv.is_open()) {
        std::fprintf(stderr, "cannot open results/ (run: mkdir -p results)\n");
        return 1;
    }
    tcsv << "family,param,rho_s_mc,lambda_L,lambda_U,q,pL_mc,pL_closed,pU_mc,pU_closed,chiL,chiU\n";
    scsv << "family,tail,q,est,m,target,lambda,mean,sd,rmse_target,rmse_lambda\n";

    std::vector<Fixed> fixed;
    std::vector<Exact> exact;
    std::printf("\n== A. exact targets (d=%.2f, MC n=%zu)\n", d, N_EXACT);
    std::printf("%-9s %8s %8s %7s %7s |", "family", "param", "rho_S", "lam_L", "lam_U");
    for (double q : kQ) std::printf("  chiL(%.2f) chiU(%.2f)", q, q);
    std::printf("\n");
    for (const auto& fam : kFamilies) {
        fixed.push_back(fix(fam, d));
        exact.push_back(exact_targets(fixed.back()));
        const auto& e = exact.back();
        const Lambdas lam = exact_lambda(fixed.back());
        std::printf("%-9s %8.4f %8.4f %7.3f %7.3f |", fam.c_str(), fixed.back().param, e.rho_s, lam.lower, lam.upper);
        for (std::size_t j = 0; j < kQ.size(); ++j) {
            std::printf("  %9.3f %10.3f", e.pL[j] / kQ[j], e.pU[j] / kQ[j]);
            tcsv << fam << "," << fixed.back().param << "," << e.rho_s << "," << lam.lower << "," << lam.upper << ","
                 << kQ[j] << "," << e.pL[j] << "," << e.pL_closed[j] << "," << e.pU[j] << "," << e.pU_closed[j] << ","
                 << e.pL[j] / kQ[j] << "," << e.pU[j] / kQ[j] << "\n";
        }
        std::printf("\n");
        for (std::size_t j = 0; j < kQ.size(); ++j)
            if (!std::isnan(e.pL_closed[j]))
                std::printf("           closed-form check q=%.2f: chiL %.4f (mc %.4f)  chiU %.4f (mc %.4f)\n", kQ[j],
                            e.pL_closed[j] / kQ[j], e.pL[j] / kQ[j], e.pU_closed[j] / kQ[j], e.pU[j] / kQ[j]);
    }

    // B. streaming estimators
    // sd[f][tail][q][m][est]
    std::vector<double> SD(kFamilies.size() * 2 * kQ.size() * kMem.size() * kEst.size());
    std::vector<double> TGT(SD.size());
    auto bidx = [](std::size_t f, std::size_t tail, std::size_t qi, std::size_t mi, std::size_t e) {
        return (((f * 2 + tail) * kQ.size() + qi) * kMem.size() + mi) * kEst.size() + e;
    };
    for (std::size_t fi = 0; fi < kFamilies.size(); ++fi) {
        std::vector<std::future<std::vector<std::vector<double>>>> jobs;
        for (int s = 0; s < n_seeds; ++s)
            jobs.push_back(std::async(std::launch::async, run_stream, fixed[fi], 42u + static_cast<unsigned>(s)));
        std::vector<std::vector<std::vector<double>>> seeds;
        for (auto& j : jobs) seeds.push_back(j.get());
        const Lambdas lam = exact_lambda(fixed[fi]);
        for (std::size_t tail = 0; tail < 2; ++tail)
            for (std::size_t qi = 0; qi < kQ.size(); ++qi)
                for (std::size_t mi = 0; mi < kMem.size(); ++mi)
                    for (std::size_t e = 0; e < kEst.size(); ++e) {
                        const std::size_t cell = sidx(tail, qi, mi) * kEst.size() + e;
                        double s = 0, n = 0;
                        for (const auto& sd : seeds) for (double x : sd[cell]) { s += x; n += 1; }
                        const double mean = s / n;
                        double ss = 0;
                        for (const auto& sd : seeds) for (double x : sd[cell]) ss += (x - mean) * (x - mean);
                        const double sdv = std::sqrt(ss / (n - 1));
                        const double pj = tail == 0 ? exact[fi].pL[qi] : exact[fi].pU[qi];
                        const double tgt = target_of(kEst[e], pj, kQ[qi]);
                        const double lm = tail == 0 ? lam.lower : lam.upper;
                        const double rt = std::sqrt(sdv * sdv + (mean - tgt) * (mean - tgt));
                        const double rl = std::sqrt(sdv * sdv + (mean - lm) * (mean - lm));
                        SD[bidx(fi, tail, qi, mi, e)] = sdv;
                        TGT[bidx(fi, tail, qi, mi, e)] = tgt;
                        scsv << kFamilies[fi] << "," << (tail ? "upper" : "lower") << "," << kQ[qi] << "," << kEst[e]
                             << "," << kMem[mi] << "," << tgt << "," << lm << "," << mean << "," << sdv << "," << rt
                             << "," << rl << "\n";
                    }
    }

    std::printf("\n== B. streaming SD of the SEC-ratio estimator (bias vs own target is in the CSV)\n");
    for (std::size_t fi = 0; fi < kFamilies.size(); ++fi)
        for (std::size_t tail = 0; tail < 2; ++tail) {
            std::printf("%-9s %-5s", kFamilies[fi].c_str(), tail ? "upper" : "lower");
            for (std::size_t qi = 0; qi < kQ.size(); ++qi) {
                std::printf("  q=%.2f:", kQ[qi]);
                for (std::size_t mi : {2u, 4u, 6u}) std::printf(" %.3f", SD[bidx(fi, tail, qi, mi, 1)]);
            }
            std::printf("    (m = 1000 / 5000 / 20000)\n");
        }

    std::printf("\n== C. transitions: |delta target| / max(SD_before, SD_after), SEC-ratio\n");
    struct Tr { std::size_t a, b; std::size_t tail; const char* label; };
    const std::vector<Tr> trs = {{0, 1, 0, "gauss->clayton  lower"}, {0, 2, 1, "gauss->gumbel   upper"},
                                 {0, 3, 0, "gauss->t        lower"}, {1, 2, 0, "clayton->gumbel lower"},
                                 {1, 2, 1, "clayton->gumbel upper"}};
    std::printf("%-22s %5s %8s %8s %8s |", "transition", "q", "before", "after", "lam a->b");
    for (double m : kMem) std::printf(" m=%-6.0f", m);
    std::printf("\n");
    for (const auto& tr : trs)
        for (std::size_t qi = 0; qi < kQ.size(); ++qi) {
            const double ta = TGT[bidx(tr.a, tr.tail, qi, 0, 1)], tb = TGT[bidx(tr.b, tr.tail, qi, 0, 1)];
            const Lambdas la = exact_lambda(fixed[tr.a]), lb = exact_lambda(fixed[tr.b]);
            std::printf("%-22s %5.2f %8.3f %8.3f  %.2f->%.2f|", tr.label, kQ[qi], ta, tb,
                        tr.tail ? la.upper : la.lower, tr.tail ? lb.upper : lb.lower);
            for (std::size_t mi = 0; mi < kMem.size(); ++mi) {
                const double sd = std::max(SD[bidx(tr.a, tr.tail, qi, mi, 1)], SD[bidx(tr.b, tr.tail, qi, mi, 1)]);
                std::printf(" %8.1f", std::fabs(tb - ta) / sd);
            }
            std::printf("\n");
        }
    std::printf("\nwrote results/tail_validate_targets.csv, results/tail_validate_stream.csv\n");
    return 0;
}
