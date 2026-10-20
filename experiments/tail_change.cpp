// Step 2: online tail dependence through a known copula change. See notes/studies/tail_dependence.md.
//
// Matched-rho_S family switch at TAU, copula space, known margins. Ratio estimator
// chi_hat = EW joint-corner count / EW marginal-corner count, over a memory grid x threshold grid.
// Arms per (tail, q):
//   fixed  -- EW at each memory (and cumulative), never reset
//   oracle -- same, but every count discarded at the true changepoint
//   bank   -- estimator_bank's device on the fixed arms, scored by a Bernoulli log-likelihood
//             of the tail event given the marginal corner (predict before absorb), 20-tick blocks,
//             Markov switch 0.002 -- the only change from estimator_bank is the score
// Scored against chi_t(q), not lambda_t; chi(q) - lambda is reported separately.
//
// Usage: ./tail_change [n_seeds]   (default 40)
// Writes results/tail_change.csv.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <string>
#include <vector>

#include "tail_common.hpp"

namespace {

constexpr double D = 0.6;  // matched target rho_S = 0.58192
constexpr std::size_t TAU = 20000, T = 40000, PRE0 = 10000;
constexpr std::size_t PATH0 = TAU - 2000, STRIDE = 20;
constexpr std::size_t N_EXACT = 20000000;
const std::vector<double> kQ = {0.01, 0.025, 0.05, 0.10};

struct Transition { const char* a; const char* b; };
const std::vector<Transition> kTr = {{"gaussian", "t"}, {"gaussian", "clayton"}, {"gaussian", "gumbel"},
                                     {"clayton", "gumbel"}, {"gumbel", "clayton"}};

std::vector<double> mem_grid() {
    std::vector<double> g;
    for (int k = 0; k <= 18; ++k) g.push_back(100.0 * std::pow(2.0, k / 2.0));
    g.push_back(INFINITY);  // cumulative
    return g;
}
const std::vector<double> kMem = mem_grid();
const std::vector<std::size_t> kBankArms = {2, 4, 6, 8, 10, 12, 14, 16};  // 200 .. 25600
constexpr std::size_t BANK_BLOCK = 20;
constexpr double BANK_SWITCH = 0.002;

std::size_t n_cells() { return 2 * kQ.size() * (2 * kMem.size() + 1); }
// cell layout per (tail, q): [0, M) fixed, [M, 2M) oracle, 2M bank
std::size_t cell(std::size_t tail, std::size_t qi, std::size_t k) {
    return (tail * kQ.size() + qi) * (2 * kMem.size() + 1) + k;
}

struct SeedOut {
    std::vector<double> se_pre, se_post;   // per cell, summed squared error
    std::vector<double> path;              // per cell x path point, estimate
};

std::size_t n_path() { return (T - PATH0) / STRIDE; }

SeedOut run_seed(const tail::Fixed& fa, const tail::Fixed& fb, const std::vector<double>& chi_a,
                 const std::vector<double>& chi_b, unsigned seed) {
    // chi_a/chi_b: [tail * nQ + qi]
    tail::Sampler sa(fa, seed), sb(fb, seed + 7777u);
    const std::size_t M = kMem.size(), nQ = kQ.size(), C = n_cells();
    std::vector<double> keep(M);
    for (std::size_t k = 0; k < M; ++k) keep[k] = std::isinf(kMem[k]) ? 1.0 : 1.0 - 2.0 / (kMem[k] + 1.0);

    // joint/marg EW sums for fixed (k < M) and oracle (M <= k < 2M) arms, per (tail, q)
    std::vector<double> joint(2 * nQ * 2 * M, 0.0), marg(joint.size(), 0.0), last(joint.size(), 0.0);
    auto aidx = [M, nQ](std::size_t tail, std::size_t qi, std::size_t k) { return (tail * nQ + qi) * 2 * M + k; };
    const std::size_t B = kBankArms.size();
    std::vector<double> bw(2 * nQ * B, 1.0 / static_cast<double>(B)), bll(bw.size(), 0.0);

    SeedOut out{std::vector<double>(C, 0.0), std::vector<double>(C, 0.0), std::vector<double>(C * n_path(), 0.0)};
    std::vector<double> est(C, 0.0);

    for (std::size_t t = 0; t < T; ++t) {
        const auto [u, v] = (t < TAU) ? sa.next() : sb.next();
        if (t == TAU)
            for (std::size_t tail = 0; tail < 2; ++tail)
                for (std::size_t qi = 0; qi < nQ; ++qi)
                    for (std::size_t k = M; k < 2 * M; ++k) joint[aidx(tail, qi, k)] = marg[aidx(tail, qi, k)] = 0.0;

        for (std::size_t tail = 0; tail < 2; ++tail)
            for (std::size_t qi = 0; qi < nQ; ++qi) {
                const double q = kQ[qi];
                const bool mu = tail == 0 ? u <= q : u > 1.0 - q;
                const bool mv = tail == 0 ? v <= q : v > 1.0 - q;

                // Bank: score fixed arms' current predictions before absorbing.
                double* w = &bw[(tail * nQ + qi) * B];
                double* ll = &bll[(tail * nQ + qi) * B];
                if (mu)
                    for (std::size_t b = 0; b < B; ++b) {
                        const double p = std::clamp(last[aidx(tail, qi, kBankArms[b])], 0.005, 0.995);
                        ll[b] += mv ? std::log(p) : std::log(1.0 - p);
                    }
                if (t % BANK_BLOCK == BANK_BLOCK - 1) {
                    const double top = *std::max_element(ll, ll + B);
                    double tot = 0.0;
                    for (std::size_t b = 0; b < B; ++b) { w[b] *= std::exp(ll[b] - top); ll[b] = 0.0; tot += w[b]; }
                    for (std::size_t b = 0; b < B; ++b)
                        w[b] = (1.0 - BANK_SWITCH) * w[b] / tot + BANK_SWITCH / static_cast<double>(B);
                }

                for (std::size_t k = 0; k < 2 * M; ++k) {
                    const std::size_t i = aidx(tail, qi, k);
                    const double l = keep[k % M];
                    joint[i] = l * joint[i] + (mu && mv);
                    marg[i] = l * marg[i] + mu;
                    // Hold the last value until the first marginal-corner event (only bites after a reset).
                    if (marg[i] > 0.0) last[i] = joint[i] / marg[i];
                }
                double bank = 0.0;
                for (std::size_t b = 0; b < B; ++b) bank += w[b] * last[aidx(tail, qi, kBankArms[b])];

                for (std::size_t k = 0; k < 2 * M; ++k) est[cell(tail, qi, k)] = last[aidx(tail, qi, k)];
                est[cell(tail, qi, 2 * M)] = bank;
            }

        if (t >= PRE0) {
            for (std::size_t tail = 0; tail < 2; ++tail)
                for (std::size_t qi = 0; qi < nQ; ++qi) {
                    const double truth = (t < TAU) ? chi_a[tail * nQ + qi] : chi_b[tail * nQ + qi];
                    for (std::size_t k = 0; k <= 2 * M; ++k) {
                        const std::size_t c = cell(tail, qi, k);
                        const double e = est[c] - truth;
                        (t < TAU ? out.se_pre : out.se_post)[c] += e * e;
                    }
                }
        }
        if (t >= PATH0 && (t - PATH0) % STRIDE == 0) {
            const std::size_t p = (t - PATH0) / STRIDE;
            for (std::size_t c = 0; c < C; ++c) out.path[c * n_path() + p] = est[c];
        }
    }
    return out;
}

double mean(const std::vector<double>& v) { double s = 0; for (double x : v) s += x; return s / v.size(); }
double se(const std::vector<double>& v) {
    const double m = mean(v); double s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / (v.size() - 1) / v.size());
}

// Log-parabolic argmin over the finite memory grid; edge flag if the minimum is at either end.
double argmin_mem(const std::vector<double>& y, bool* edge) {
    const std::size_t n = kMem.size() - 1;  // exclude cumulative
    std::size_t k = 0;
    for (std::size_t i = 1; i < n; ++i) if (y[i] < y[k]) k = i;
    *edge = (k == 0 || k + 1 == n);
    if (*edge) return kMem[k];
    const double x0 = std::log2(kMem[k - 1]), x1 = std::log2(kMem[k]), x2 = std::log2(kMem[k + 1]);
    const double den = (x0 - x1) * (x0 - x2) * (x1 - x2);
    const double A = (x2 * (y[k] - y[k - 1]) + x1 * (y[k - 1] - y[k + 1]) + x0 * (y[k + 1] - y[k])) / den;
    const double Bc = (x2 * x2 * (y[k - 1] - y[k]) + x1 * x1 * (y[k + 1] - y[k - 1]) + x0 * x0 * (y[k] - y[k + 1])) / den;
    if (A <= 0.0) return kMem[k];
    return std::exp2(std::clamp(-Bc / (2.0 * A), x0, x2));
}

}  // namespace

int main(int argc, char** argv) {
    const int n_seeds = (argc > 1) ? std::atoi(argv[1]) : 40;
    const std::size_t M = kMem.size(), nQ = kQ.size(), C = n_cells();
    const double W_PRE = static_cast<double>(TAU - PRE0), W_POST = static_cast<double>(T - TAU);

    std::ofstream csv("results/tail_change.csv");
    if (!csv.is_open()) { std::fprintf(stderr, "cannot open results/ (run: mkdir -p results)\n"); return 1; }
    csv << "transition,tail,q,arm,m,chi_pre,chi_post,lam_pre,lam_post,mse_pre,mse_post,mse_total,mse_total_se,recovery\n";

    std::printf("tau=%zu T=%zu pre-score from %zu, n_seeds=%d, rho_S target 0.58192\n", TAU, T, PRE0, n_seeds);
    std::printf("recovery = ticks until the seed-mean estimate is within 10%% of the jump of chi_post\n");

    for (const auto& tr : kTr) {
        const tail::Fixed fa = tail::fix(tr.a, D), fb = tail::fix(tr.b, D);
        const tail::Corners ca = tail::exact_corners(fa, kQ, N_EXACT), cb = tail::exact_corners(fb, kQ, N_EXACT);
        const tail::Lambdas la = tail::exact_lambda(fa), lb = tail::exact_lambda(fb);
        std::vector<double> chi_a(2 * nQ), chi_b(2 * nQ);
        for (std::size_t qi = 0; qi < nQ; ++qi) {
            chi_a[qi] = ca.pL[qi] / kQ[qi]; chi_a[nQ + qi] = ca.pU[qi] / kQ[qi];
            chi_b[qi] = cb.pL[qi] / kQ[qi]; chi_b[nQ + qi] = cb.pU[qi] / kQ[qi];
        }

        std::vector<SeedOut> seeds;
        for (int s0 = 0; s0 < n_seeds; s0 += 8) {
            std::vector<std::future<SeedOut>> jobs;
            for (int s = s0; s < std::min(n_seeds, s0 + 8); ++s)
                jobs.push_back(std::async(std::launch::async, run_seed, fa, fb, chi_a, chi_b, 4242u + 31u * static_cast<unsigned>(s)));
            for (auto& j : jobs) seeds.push_back(j.get());
        }

        std::printf("\n=== %s -> %s   (achieved rho_S %.4f -> %.4f)\n", tr.a, tr.b, ca.rho_s, cb.rho_s);
        std::printf("%-5s %5s %6s %6s %10s | %-24s | %-20s | %-20s | %8s %8s\n", "tail", "q", "chi_a", "chi_b",
                    "lam a->b", "best fixed: M* MSEtot post", "oracle(cum): tot post", "bank: tot post", "fix/orc", "bank/orc");
        for (std::size_t tail = 0; tail < 2; ++tail)
            for (std::size_t qi = 0; qi < nQ; ++qi) {
                const double xa = chi_a[tail * nQ + qi], xb = chi_b[tail * nQ + qi];
                const double lam_a = tail ? la.upper : la.lower, lam_b = tail ? lb.upper : lb.lower;
                std::vector<double> tot_mean(2 * M + 1), post_mean(2 * M + 1), pre_mean(2 * M + 1), rec(2 * M + 1);
                for (std::size_t k = 0; k <= 2 * M; ++k) {
                    const std::size_t c = cell(tail, qi, k);
                    std::vector<double> tot, pre, post;
                    for (const auto& s : seeds) {
                        pre.push_back(s.se_pre[c] / W_PRE);
                        post.push_back(s.se_post[c] / W_POST);
                        tot.push_back((s.se_pre[c] + s.se_post[c]) / (W_PRE + W_POST));
                    }
                    tot_mean[k] = mean(tot); post_mean[k] = mean(post); pre_mean[k] = mean(pre);
                    rec[k] = NAN;
                    for (std::size_t p = (TAU - PATH0) / STRIDE; p < n_path(); ++p) {
                        double m = 0.0;
                        for (const auto& s : seeds) m += s.path[c * n_path() + p];
                        m /= seeds.size();
                        if (std::fabs(m - xb) <= 0.1 * std::fabs(xb - xa)) { rec[k] = static_cast<double>(p * STRIDE + PATH0 - TAU); break; }
                    }
                    const char* arm = k < M ? "fixed" : k < 2 * M ? "oracle" : "bank";
                    const double mm = k == 2 * M ? NAN : kMem[k % M];
                    csv << tr.a << "->" << tr.b << "," << (tail ? "upper" : "lower") << "," << kQ[qi] << "," << arm << ","
                        << mm << "," << xa << "," << xb << "," << lam_a << "," << lam_b << "," << pre_mean[k] << ","
                        << post_mean[k] << "," << tot_mean[k] << "," << se(tot) << "," << rec[k] << "\n";
                }
                bool edge = false;
                std::vector<double> fixed_tot(tot_mean.begin(), tot_mean.begin() + M);
                const double mstar = argmin_mem(fixed_tot, &edge);
                std::size_t kbest = 0;
                for (std::size_t k = 1; k < M; ++k) if (tot_mean[k] < tot_mean[kbest]) kbest = k;
                std::size_t obest = M;
                for (std::size_t k = M + 1; k < 2 * M; ++k) if (tot_mean[k] < tot_mean[obest]) obest = k;
                char b1[64], b2[64], b3[64];
                std::snprintf(b1, sizeof b1, "%6.0f%s %.5f %.5f", mstar, edge ? "|" : " ", tot_mean[kbest], post_mean[kbest]);
                std::snprintf(b2, sizeof b2, "%.5f %.5f%s", tot_mean[2 * M - 1], post_mean[2 * M - 1],
                              obest == 2 * M - 1 ? "" : "*");
                std::snprintf(b3, sizeof b3, "%.5f %.5f", tot_mean[2 * M], post_mean[2 * M]);
                std::printf("%-5s %5.3f %6.3f %6.3f %4.2f->%4.2f | %-24s | %-20s | %-20s | %8.2f %8.2f\n",
                            tail ? "upper" : "lower", kQ[qi], xa, xb, lam_a, lam_b, b1, b2, b3,
                            tot_mean[kbest] / tot_mean[obest], tot_mean[2 * M] / tot_mean[obest]);
            }
    }
    std::printf("\n'|' = M* on the grid edge; '*' = oracle's best memory is not cumulative (ratio uses the best oracle).\n");
    std::printf("wrote results/tail_change.csv\n");
    return 0;
}
