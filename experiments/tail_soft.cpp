// Step 3b: how much old-regime information to keep after a known tail-structure change.
// See notes/studies/tail_dependence.md.
//
// DGP: Clayton-Gumbel mixture with both components at rho_S = 0.58192, switching pi0 -> 1 - pi0 at
// TAU; rho_S is matched exactly and Delta chi(q) = Delta pi * (chi_C(q) - chi_G(q)).
// Estimator: ratio chi_hat = J / K, with J, K exponentially weighted sums of the joint-corner and
// marginal-corner indicators. Soft oracle restart at the true TAU: J <- gamma J, K <- gamma K.
// gamma = 1 is no restart, gamma = 0 is the hard restart (the estimate holds its last value until
// the first post-change marginal-corner event); between, old evidence is kept but aged by
// ln(1/gamma)/alpha ticks. Each gamma is swept over the full memory grid. The bank is step 2's.
// Scored against chi_t(q).
//
// n0 = gamma * E[K at TAU] is the number of old marginal-corner events the soft reset keeps; the
// shrinkage argument in the note predicts n0* = chi_post(1 - chi_post) / Delta chi^2.
//
// Usage: ./tail_soft [n_seeds] [seed_base]   (defaults 40, 6060)
// Writes results/tail_soft.csv (per cell) and results/tail_soft_gamma.csv (gamma* bootstrap).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include "tail_common.hpp"
#include "trajectory_manifest.hpp"

namespace {

constexpr std::size_t TAU = 20000, T = 40000, PRE0 = 10000;
constexpr std::size_t PATH0 = TAU, STRIDE = 100;
const std::vector<double> kQ = {0.01, 0.02, 0.05, 0.10};
const std::vector<double> kPi0 = {0.55, 0.60, 0.70, 0.75, 0.90, 1.00};  // switch pi0 -> 1 - pi0
const std::vector<double> kGamma = {0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0};

std::vector<double> mem_grid() {
    std::vector<double> g;
    for (int k = 0; k <= 18; ++k) g.push_back(100.0 * std::pow(2.0, k / 2.0));
    g.push_back(INFINITY);
    return g;
}
const std::vector<double> kMem = mem_grid();
const std::vector<std::size_t> kBankArms = {2, 4, 6, 8, 10, 12, 14, 16};
constexpr std::size_t BANK_BLOCK = 20;
constexpr double BANK_SWITCH = 0.002;

const std::size_t G = kGamma.size(), M = kMem.size(), NQ = kQ.size();
std::size_t per_tq() { return G * M + 1; }  // gamma x memory, then bank
std::size_t n_cells() { return 2 * NQ * per_tq(); }
std::size_t cell(std::size_t tail, std::size_t qi, std::size_t g, std::size_t m) {
    return (tail * NQ + qi) * per_tq() + g * M + m;
}
std::size_t bank_cell(std::size_t tail, std::size_t qi) { return (tail * NQ + qi) * per_tq() + G * M; }
std::size_t n_path() { return (T - PATH0) / STRIDE; }

struct SeedOut {
    std::vector<double> se_pre, se_post, path;
};

SeedOut run_seed(const tail::Mix& ma, const tail::Mix& mb, const std::vector<double>& chi_a,
                 const std::vector<double>& chi_b, unsigned seed) {
    tail::MixSampler sa(ma, seed), sb(mb, seed + 7777u);
    std::vector<double> keep(M);
    for (std::size_t k = 0; k < M; ++k) keep[k] = std::isinf(kMem[k]) ? 1.0 : 1.0 - 2.0 / (kMem[k] + 1.0);
    // State per (tail, q, gamma, m). Before TAU every gamma shares the gamma = 1 state.
    const std::size_t S = 2 * NQ * G * M;
    std::vector<double> J(S, 0.0), K(S, 0.0), last(S, 0.0);
    auto sidx = [](std::size_t tail, std::size_t qi, std::size_t g, std::size_t m) { return ((tail * NQ + qi) * G + g) * M + m; };
    const std::size_t B = kBankArms.size(), g1 = G - 1;
    std::vector<double> bw(2 * NQ * B, 1.0 / static_cast<double>(B)), bll(bw.size(), 0.0);

    const std::size_t C = n_cells();
    SeedOut out{std::vector<double>(C, 0.0), std::vector<double>(C, 0.0), std::vector<double>(C * n_path(), 0.0)};
    std::vector<double> est(C, 0.0);
    traj::Series ser;

    for (std::size_t t = 0; t < T; ++t) {
        const bool post = t >= TAU;
        const auto [u, v] = post ? sb.next() : sa.next();
        if (t == TAU)
            for (std::size_t tail = 0; tail < 2; ++tail)
                for (std::size_t qi = 0; qi < NQ; ++qi)
                    for (std::size_t g = 0; g + 1 < G; ++g)
                        for (std::size_t m = 0; m < M; ++m) {
                            const std::size_t s1 = sidx(tail, qi, g1, m), s = sidx(tail, qi, g, m);
                            J[s] = kGamma[g] * J[s1];
                            K[s] = kGamma[g] * K[s1];
                            last[s] = last[s1];
                        }
        const std::size_t g_lo = post ? 0 : g1;  // pre-change only the shared gamma = 1 state is live
        for (std::size_t tail = 0; tail < 2; ++tail)
            for (std::size_t qi = 0; qi < NQ; ++qi) {
                const double q = kQ[qi];
                const bool mu = tail == 0 ? u <= q : u > 1.0 - q;
                const bool mv = tail == 0 ? v <= q : v > 1.0 - q;
                double* w = &bw[(tail * NQ + qi) * B];
                double* ll = &bll[(tail * NQ + qi) * B];
                if (mu)
                    for (std::size_t b = 0; b < B; ++b) {
                        const double p = std::clamp(last[sidx(tail, qi, g1, kBankArms[b])], 0.005, 0.995);
                        ll[b] += mv ? std::log(p) : std::log(1.0 - p);
                    }
                if (t % BANK_BLOCK == BANK_BLOCK - 1) {
                    const double top = *std::max_element(ll, ll + B);
                    double tot = 0.0;
                    for (std::size_t b = 0; b < B; ++b) { w[b] *= std::exp(ll[b] - top); ll[b] = 0.0; tot += w[b]; }
                    for (std::size_t b = 0; b < B; ++b)
                        w[b] = (1.0 - BANK_SWITCH) * w[b] / tot + BANK_SWITCH / static_cast<double>(B);
                }
                for (std::size_t g = g_lo; g < G; ++g)
                    for (std::size_t m = 0; m < M; ++m) {
                        const std::size_t s = sidx(tail, qi, g, m);
                        const double l = keep[m];
                        J[s] = l * J[s] + (mu && mv);
                        K[s] = l * K[s] + mu;
                        if (K[s] > 0.0) last[s] = J[s] / K[s];
                    }
                for (std::size_t g = 0; g < G; ++g)
                    for (std::size_t m = 0; m < M; ++m)
                        est[cell(tail, qi, g, m)] = last[sidx(tail, qi, post ? g : g1, m)];
                double bank = 0.0;
                for (std::size_t b = 0; b < B; ++b) bank += w[b] * last[sidx(tail, qi, g1, kBankArms[b])];
                est[bank_cell(tail, qi)] = bank;
            }
        ser.add(est);
        if (t >= PRE0)
            for (std::size_t tail = 0; tail < 2; ++tail)
                for (std::size_t qi = 0; qi < NQ; ++qi) {
                    const double truth = post ? chi_b[tail * NQ + qi] : chi_a[tail * NQ + qi];
                    const std::size_t c0 = (tail * NQ + qi) * per_tq();
                    for (std::size_t c = c0; c < c0 + per_tq(); ++c) {
                        const double e = est[c] - truth;
                        (post ? out.se_post : out.se_pre)[c] += e * e;
                    }
                }
        if (t >= PATH0 && (t - PATH0) % STRIDE == 0) {
            const std::size_t p = (t - PATH0) / STRIDE;
            for (std::size_t c = 0; c < C; ++c) out.path[c * n_path() + p] = est[c];
        }
    }
    if (traj::on()) {
        const std::string base = traj::tag("tail_soft/%.17g->%.17g/%u/", ma.pi, mb.pi, seed);
        ser.commit(base + "est");
        traj::record(base + "se_pre", out.se_pre);
        traj::record(base + "se_post", out.se_post);
        traj::record(base + "path", out.path);
    }
    return out;
}

// Expected marginal-corner EW count at TAU for memory m (cumulative: q * TAU).
double k_at_tau(double q, double m) {
    if (std::isinf(m)) return q * static_cast<double>(TAU);
    const double l = 1.0 - 2.0 / (m + 1.0);
    return q * (1.0 - std::pow(l, static_cast<double>(TAU))) / (1.0 - l);
}

double mean(const std::vector<double>& v) { double s = 0; for (double x : v) s += x; return s / v.size(); }

}  // namespace

int main(int argc, char** argv) {
    const int n_seeds = (argc > 1) ? std::atoi(argv[1]) : 40;
    const unsigned seed_base = (argc > 2) ? static_cast<unsigned>(std::atoi(argv[2])) : 6060u;
    const double R0 = 6.0 / std::numbers::pi * std::asin(0.6 / 2.0);
    const double th0 = dgp::param_from_spearman(copula::Family::Clayton, R0);
    const double ze0 = dgp::param_from_spearman(copula::Family::Gumbel, R0);
    const double W_PRE = static_cast<double>(TAU - PRE0), W_POST = static_cast<double>(T - TAU);

    std::ofstream csv("results/tail_soft.csv"), gcsv("results/tail_soft_gamma.csv");
    if (!csv.is_open() || !gcsv.is_open()) { std::fprintf(stderr, "cannot open results/ (run: mkdir -p results)\n"); return 1; }
    csv << "pi0,pi1,tail,q,chi_pre,chi_post,lam_pre,lam_post,arm,gamma,m,mse_pre,mse_post,mse_total,recovery\n";
    gcsv << "pi0,pi1,tail,q,dchi,v_bar,v_post,gamma_star,m_star,n0_star,n0_pred,boot_gamma_lo,boot_gamma_hi,"
            "boot_n0_lo,boot_n0_hi,boot_frac_star,best_fixed,hard,soft,bank,soft_vs_fixed_paired_se,fixed_m_star\n";

    std::printf("Clayton(theta=%.4f) / Gumbel(zeta=%.4f) mixture, both at rho_S %.5f; tau=%zu T=%zu, scored from %zu, %d seeds\n",
                th0, ze0, R0, TAU, T, PRE0, n_seeds);
    std::printf("columns: total MSE (x1e-3) of best fixed (gamma=1), hard restart (gamma=0), best soft (any gamma), bank;\n"
                "gamma* = argmin over gamma of min over m; bootstrap = 16-84%% range of gamma* over 400 seed resamples\n");
    std::mt19937 boot(2024);

    for (double pi0 : kPi0) {
        const double pi1 = 1.0 - pi0;
        const tail::Mix ma{pi0, th0, ze0}, mb{pi1, th0, ze0};
        std::vector<double> chi_a(2 * NQ), chi_b(2 * NQ);
        for (std::size_t qi = 0; qi < NQ; ++qi) {
            chi_a[qi] = ma.chi_L(kQ[qi]); chi_a[NQ + qi] = ma.chi_U(kQ[qi]);
            chi_b[qi] = mb.chi_L(kQ[qi]); chi_b[NQ + qi] = mb.chi_U(kQ[qi]);
        }
        std::vector<SeedOut> seeds;
        for (int s0 = 0; s0 < n_seeds; s0 += 8) {
            std::vector<std::future<SeedOut>> jobs;
            for (int s = s0; s < std::min(n_seeds, s0 + 8); ++s)
                jobs.push_back(std::async(std::launch::async, run_seed, ma, mb, chi_a, chi_b, seed_base + 37u * static_cast<unsigned>(s)));
            for (auto& j : jobs) seeds.push_back(j.get());
        }
        const std::size_t NS = seeds.size();

        std::printf("\n=== pi %.2f -> %.2f   (lambda_L %.3f -> %.3f, lambda_U %.3f -> %.3f)\n", pi0, pi1, ma.lam_L(), mb.lam_L(),
                    ma.lam_U(), mb.lam_U());
        std::printf("%-5s %5s %6s %6s | %8s %8s %8s %8s | %6s %6s %-15s | %6s %6s %-13s | %s\n", "tail", "q", "dchi", "qd2/v",
                    "fixed", "hard", "soft", "bank", "gamma*", "m*", "boot gamma*", "n0*", "pred", "boot n0*",
                    "soft/fixed (paired SE)");
        for (std::size_t tail = 0; tail < 2; ++tail)
            for (std::size_t qi = 0; qi < NQ; ++qi) {
                const double xa = chi_a[tail * NQ + qi], xb = chi_b[tail * NQ + qi];
                const double dchi = std::fabs(xb - xa), vbar = 0.5 * (xa * (1 - xa) + xb * (1 - xb));
                const double lam_a = tail ? ma.lam_U() : ma.lam_L(), lam_b = tail ? mb.lam_U() : mb.lam_L();
                // per-seed total MSE per (gamma, m) and bank
                std::vector<std::vector<double>> tot(per_tq(), std::vector<double>(NS));
                for (std::size_t k = 0; k < per_tq(); ++k) {
                    const std::size_t c = (tail * NQ + qi) * per_tq() + k;
                    for (std::size_t s = 0; s < NS; ++s) tot[k][s] = (seeds[s].se_pre[c] + seeds[s].se_post[c]) / (W_PRE + W_POST);
                    double pre = 0, post = 0;
                    for (const auto& sd : seeds) { pre += sd.se_pre[c] / W_PRE; post += sd.se_post[c] / W_POST; }
                    double rec = NAN;
                    for (std::size_t p = 0; p < n_path(); ++p) {
                        double mp = 0;
                        for (const auto& sd : seeds) mp += sd.path[c * n_path() + p];
                        if (std::fabs(mp / NS - xb) <= 0.1 * std::fabs(xb - xa)) { rec = static_cast<double>(p * STRIDE); break; }
                    }
                    const bool is_bank = k == G * M;
                    csv << pi0 << "," << pi1 << "," << (tail ? "upper" : "lower") << "," << kQ[qi] << "," << xa << "," << xb
                        << "," << lam_a << "," << lam_b << "," << (is_bank ? "bank" : "gamma") << ","
                        << (is_bank ? NAN : kGamma[k / M]) << "," << (is_bank ? NAN : kMem[k % M]) << "," << pre / NS << ","
                        << post / NS << "," << mean(tot[k]) << "," << rec << "\n";
                }
                // gamma* = argmin_g min_m mean total MSE; on a resample of seeds, recompute.
                auto best_of = [&](const std::vector<std::size_t>& idx, std::size_t g_from, std::size_t g_to,
                                   std::size_t* gb, std::size_t* mb_) {
                    double best = INFINITY;
                    for (std::size_t g = g_from; g < g_to; ++g)
                        for (std::size_t m = 0; m < M; ++m) {
                            double s = 0;
                            for (std::size_t i : idx) s += tot[g * M + m][i];
                            if (s < best) { best = s; *gb = g; *mb_ = m; }
                        }
                    return best / idx.size();
                };
                std::vector<std::size_t> all(NS);
                for (std::size_t i = 0; i < NS; ++i) all[i] = i;
                std::size_t gs = 0, ms = 0, gf = 0, mf = 0, gh = 0, mh = 0;
                const double soft = best_of(all, 0, G, &gs, &ms);
                const double fixed = best_of(all, G - 1, G, &gf, &mf);
                const double hard = best_of(all, 0, 1, &gh, &mh);
                const double bank = mean(tot[G * M]);
                std::vector<double> diff(NS);
                for (std::size_t i = 0; i < NS; ++i) diff[i] = tot[gs * M + ms][i] - tot[gf * M + mf][i];
                const double dm = mean(diff);
                double dv = 0; for (double x : diff) dv += (x - dm) * (x - dm);
                const double dse = std::sqrt(dv / (NS - 1) / NS);
                const double vpost = xb * (1.0 - xb);
                const double n0_star = kGamma[gs] * k_at_tau(kQ[qi], kMem[ms]);
                const double n0_pred = vpost / (dchi * dchi);
                std::vector<double> bg, bn;
                std::size_t same = 0;
                for (int r = 0; r < 400; ++r) {
                    std::vector<std::size_t> idx(NS);
                    for (auto& i : idx) i = boot() % NS;
                    std::size_t g, m;
                    best_of(idx, 0, G, &g, &m);
                    bg.push_back(kGamma[g]);
                    bn.push_back(kGamma[g] * k_at_tau(kQ[qi], kMem[m]));
                    same += (g == gs);
                }
                std::sort(bg.begin(), bg.end());
                std::sort(bn.begin(), bn.end());
                const double blo = bg[bg.size() * 16 / 100], bhi = bg[bg.size() * 84 / 100];
                const double nlo = bn[bn.size() * 16 / 100], nhi = bn[bn.size() * 84 / 100];
                if (traj::on()) {
                    const std::string base = traj::tag("tail_soft/%.17g->%.17g/%zu/%zu/", pi0, pi1, tail, qi);
                    std::vector<double> flat;
                    for (const auto& r : tot) flat.insert(flat.end(), r.begin(), r.end());
                    traj::record(base + "tot", flat);
                    traj::record(base + "boot_gamma", bg);
                    traj::record(base + "boot_n0", bn);
                    traj::record(base + "summary",
                                 std::vector<double>{fixed, hard, soft, bank, dm, dse, n0_star, n0_pred,
                                                     static_cast<double>(gs), static_cast<double>(ms),
                                                     static_cast<double>(gf), static_cast<double>(mf),
                                                     static_cast<double>(same)});
                }
                char bstr[32], nstr[32];
                std::snprintf(bstr, sizeof bstr, "[%.4f,%.4f]", blo, bhi);
                std::snprintf(nstr, sizeof nstr, "[%.1f,%.1f]", nlo, nhi);
                std::printf("%-5s %5.2f %6.3f %6.4f | %8.4f %8.4f %8.4f %8.4f | %6.4f %6.0f %-15s | %6.1f %6.1f %-13s | %.3f (%+.4f +- %.4f)\n",
                            tail ? "upper" : "lower", kQ[qi], dchi, kQ[qi] * dchi * dchi / vbar, fixed * 1e3, hard * 1e3,
                            soft * 1e3, bank * 1e3, kGamma[gs], kMem[ms], bstr, n0_star, n0_pred, nstr, soft / fixed,
                            dm * 1e3, dse * 1e3);
                gcsv << pi0 << "," << pi1 << "," << (tail ? "upper" : "lower") << "," << kQ[qi] << "," << dchi << "," << vbar
                     << "," << vpost << "," << kGamma[gs] << "," << kMem[ms] << "," << n0_star << "," << n0_pred << "," << blo
                     << "," << bhi << "," << nlo << "," << nhi << "," << static_cast<double>(same) / 400.0 << "," << fixed
                     << "," << hard << "," << soft << "," << bank << "," << dse << "," << kMem[mf] << "\n";
            }
    }
    std::printf("\nwrote results/tail_soft.csv, results/tail_soft_gamma.csv\n");
    return 0;
}
