// After fair tuning, does a discounted-loss bank improve jump recovery without paying too much
// during drift or stationary periods? Bank vs fixed share vs VFF vs fixed memory, all over ONE
// density estimator (an 8x8 forgetting histogram copula), all scored at the TRUE PIT.
// Design, pre-specified windows and results: notes/studies/memory_tradeoff.md.
//
// Usage (from experiments/):  ./probe_memory_tradeoff [out_dir]
//   default out_dir = ../notes/studies/memory_tradeoff
//   ./probe_memory_tradeoff <out_dir> xiao   -> components are fixed-memory XiaoSpearmanEstimator
//     (rank-PIT mode, kBins, kPrior) read through log_density_uv; the VFF keeps its own Hist.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../src/estimators/copula_family.hpp"
#include "../src/estimators/dependence/ew_marginal.hpp"
#include "../src/estimators/dependence/estimator_bank.hpp"
#include "../src/estimators/dependence/marginal_rank_pit.hpp"
#include "../src/estimators/dependence/xiao_spearman.hpp"
#include "../src/streamgenerator/datasource/copula.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"

namespace {

constexpr std::size_t kN = 5000, kBurn = 500, kChange = 2500, kBlock = 20, kArms = 10;
constexpr int kBins = 8;
constexpr double kPrior = 0.5;            // Laplace pseudo-count per cell
constexpr double kMemMin = 25.0, kMemMax = 8000.0;
constexpr unsigned kTuneSeed = 42, kEvalSeed = 5042;
constexpr int kTuneSeeds = 30, kEvalSeeds = 60;
constexpr double kT975 = 2.001;           // t_{0.975, 59}

bool g_xiao_arms = false;

double Phi(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }
double lam_of(double mem) { return 1.0 - 2.0 / (mem + 1.0); }

const std::vector<double>& arm_mem() {
    static const std::vector<double> v = [] {
        std::vector<double> m(kArms);
        for (std::size_t i = 0; i < kArms; ++i)
            m[i] = std::exp(std::log(kMemMin) + i / double(kArms - 1) * (std::log(kMemMax) - std::log(kMemMin)));
        return m;
    }();
    return v;
}

// Histogram copula with exponential forgetting; counts in observation units, so total -> 1/(1-lambda).
struct Hist {
    double lambda;
    std::vector<double> c = std::vector<double>(kBins * kBins, 0.0);
    double total = 0.0;
    static int cell(double u) { return std::clamp(static_cast<int>(u * kBins), 0, kBins - 1); }
    void update(double u, double v) {
        for (auto& x : c) x *= lambda;
        total = lambda * total + 1.0;
        c[cell(u) * kBins + cell(v)] += 1.0;
    }
    double dens(int i, int j) const {
        return (c[i * kBins + j] + kPrior) / (total + kPrior * kBins * kBins) * kBins * kBins;
    }
    double logpdf(double u, double v) const { return std::log(dens(cell(u), cell(v))); }
    // Weighted Pearson correlation of cell indices: a rank correlation on the grid.
    double rho() const {
        if (total <= 0.0) return 0.0;
        double mu = 0, mv = 0, cuv = 0, cuu = 0, cvv = 0;
        for (int i = 0; i < kBins; ++i)
            for (int j = 0; j < kBins; ++j) {
                const double w = c[i * kBins + j] / total;
                mu += w * (i + 0.5); mv += w * (j + 0.5);
            }
        for (int i = 0; i < kBins; ++i)
            for (int j = 0; j < kBins; ++j) {
                const double w = c[i * kBins + j] / total, du = i + 0.5 - mu, dv = j + 0.5 - mv;
                cuv += w * du * dv; cuu += w * du * du; cvv += w * dv * dv;
            }
        return (cuu > 0 && cvv > 0) ? std::clamp(cuv / std::sqrt(cuu * cvv), -1.0, 1.0) : 0.0;
    }
    // Mean of the two squared Rosenblatt residuals (V|U and U|V) of the predictive histogram.
    // Under a correct predictive copula each conditional PIT is uniform, so E = 1.
    double rosenblatt_e2(double u, double v) const {
        auto one = [&](bool row, double a, double b) {
            const int ia = cell(a), jb = cell(b);
            double tot = 0, below = 0;
            for (int j = 0; j < kBins; ++j) {
                const double d = row ? dens(ia, j) : dens(j, ia);
                tot += d;
                if (j < jb) below += d;
            }
            const double d = row ? dens(ia, jb) : dens(jb, ia);
            const double H = std::clamp((below + d * (b * kBins - jb)) / tot, 1e-6, 1.0 - 1e-6);
            const double z = RankPit::probit(H);
            return z * z;
        };
        return 0.5 * (one(true, u, v) + one(false, v, u));
    }
};

// ---- weighting rules over shared arms -------------------------------------------------------

struct FixedShare {      // log score, block 20, mixing step (EstimatorBank's rule)
    double alpha, eta;
    std::vector<double> w = std::vector<double>(kArms, 1.0 / kArms), ll = std::vector<double>(kArms, 0.0);
    std::size_t in_block = 0;
    void learn(const std::vector<double>& logp, bool scorable) {
        if (!scorable) return;
        for (std::size_t k = 0; k < kArms; ++k) ll[k] += eta * logp[k];
        if (++in_block < kBlock) return;
        in_block = 0;
        const double top = *std::max_element(ll.begin(), ll.end());
        double tot = 0;
        for (std::size_t k = 0; k < kArms; ++k) { w[k] *= std::exp(ll[k] - top); ll[k] = 0; tot += w[k]; }
        for (double& x : w) x = (1 - alpha) * x / tot + alpha / kArms;
    }
};

struct Discounted {      // L = beta L + (1-beta) s,  w = softmax(-eta L),  eta = c / (1-beta)
    double beta, c;
    std::vector<double> L = std::vector<double>(kArms, 0.0), w = std::vector<double>(kArms, 1.0 / kArms);
    void learn(const std::vector<double>& logp, bool scorable) {
        if (!scorable) return;
        const double eta = c / (1 - beta);
        for (std::size_t k = 0; k < kArms; ++k) L[k] = beta * L[k] + (1 - beta) * (-logp[k]);
        const double lo = *std::min_element(L.begin(), L.end());
        double tot = 0;
        for (std::size_t k = 0; k < kArms; ++k) { w[k] = std::exp(-eta * (L[k] - lo)); tot += w[k]; }
        for (double& x : w) x /= tot;
    }
};

struct RhoFixedShare {   // Spearman-scored, exactly EstimatorBank's update (reference only)
    std::vector<double> w = std::vector<double>(kArms, 1.0 / kArms), ll = std::vector<double>(kArms, 0.0);
    double var = 1.0;
    std::size_t in_block = 0;
    void learn(double point, const std::vector<double>& rho, double mix_rho, bool scorable) {
        if (!scorable) return;
        const double s = point - mix_rho;
        var += 0.001 * (s * s - var);
        for (std::size_t k = 0; k < kArms; ++k) { const double e = point - rho[k]; ll[k] -= e * e / (2 * std::max(var, 1e-6)); }
        if (++in_block < kBlock) return;
        in_block = 0;
        const double top = *std::max_element(ll.begin(), ll.end());
        double tot = 0;
        for (std::size_t k = 0; k < kArms; ++k) { w[k] *= std::exp(ll[k] - top); ll[k] = 0; tot += w[k]; }
        for (double& x : w) x = (1 - 0.002) * x / tot + 0.002 / kArms;
    }
};

// VFF: XiaoSpearmanEstimator's rule (C <- C + r(e2 - C); alpha <- alpha C^r; clamp) driven by the
// histogram's own Rosenblatt residual, null scale 1.
struct Vff {
    double rate, base;
    Hist h{lam_of(base)};
    double C = 1.0, acc = 0.0, last_e2 = 1.0;
    std::size_t nb = 0, blocks = 0;
    void observe(double u, double v, bool scorable) {
        if (scorable) {
            acc += h.rosenblatt_e2(u, v);
            if (++nb == kBlock) {
                last_e2 = acc / kBlock;
                acc = 0; nb = 0;
                if (++blocks > 2) {
                    C += rate * (last_e2 - C);
                    double a = (1 - h.lambda) * std::pow(std::max(C, 1e-6), rate);
                    a = std::clamp(a, 2.0 / (kMemMax + 1.0), 2.0 / (kMemMin + 1.0));
                    h.lambda = 1 - a;
                }
            }
        }
        h.update(u, v);
    }
    double mem() const { const double a = 1 - h.lambda; return (2 - a) / a; }
};

// ---- scenarios and the true copula density --------------------------------------------------

struct Scenario { std::string name, dep, before, after; double d; bool primary; };
const std::vector<Scenario>& scenarios() {
    static const std::vector<Scenario> s{
        {"jump", "jump", "gaussian", "gaussian", 0, true},
        {"drift", "drift", "gaussian", "gaussian", 0, true},
        {"constant", "constant", "gaussian", "gaussian", 0.6, false},
        {"oscillate", "oscillate", "gaussian", "gaussian", 0, false},
        {"gauss2t_.6", "constant", "gaussian", "t", 0.6, false},
        {"clay2gumb_.6", "constant", "clayton", "gumbel", 0.6, false},
        {"clay2gumb_.8", "constant", "clayton", "gumbel", 0.8, false}};
    return s;
}
std::unique_ptr<Dependence> make_dep(const Scenario& s) {
    if (s.dep == "constant") return std::make_unique<ConstantDependence>(s.d);
    if (s.dep == "drift") return std::make_unique<DriftDependence>(-0.5, 0.8, kN);
    if (s.dep == "jump") return std::make_unique<JumpDependence>(0.2, 0.8, kChange);
    return std::make_unique<OscillatingDependence>(0.6, 400.0, 0.0);
}
std::unique_ptr<dgp::Copula> one_cop(const std::string& f) { return dgp::make_copula(f == "t" ? "t" : f); }
std::unique_ptr<dgp::Copula> make_cop(const Scenario& s) {
    if (s.before == s.after) return one_cop(s.before);
    return std::make_unique<dgp::SwitchingCopula>(one_cop(s.before), one_cop(s.after), kChange, "switch");
}

// log c_true at the true PIT. x, y are the DGP's normal scores; d the dependence knob.
double true_logc(const std::string& fam, double d, double x, double y) {
    const double u = Phi(x), v = Phi(y);
    if (fam == "gaussian") {
        const double r = std::clamp(d, -0.999, 0.999), q = 1 - r * r;
        return -0.5 * std::log(q) - (r * r * (x * x + y * y) - 2 * r * x * y) / (2 * q);
    }
    static std::map<std::pair<std::string, double>, double> cache;
    const double target = dgp::detail::target_spearman(d);
    auto key = std::make_pair(fam, target);
    if (!cache.count(key)) {
        const auto F = fam == "clayton" ? copula::Family::Clayton
                     : fam == "gumbel"  ? copula::Family::Gumbel : copula::Family::StudentT;
        cache[key] = dgp::param_from_spearman(F, target, 4.0);
    }
    const double p = cache[key];
    if (fam == "clayton") return copula::Clayton::log_density(u, v, p);
    if (fam == "gumbel") return copula::Gumbel::log_density(u, v, p);
    return copula::StudentT::log_density(u, v, p, 4.0);
}

// ---- methods --------------------------------------------------------------------------------

struct Method { std::string kind, label; double p1 = 0, p2 = 0; };

std::vector<Method> all_methods() {
    std::vector<Method> m;
    for (std::size_t k = 0; k < kArms; ++k) {
        char b[32]; std::snprintf(b, sizeof b, "fixed_%.0f", arm_mem()[k]);
        m.push_back({"fixed", b, double(k)});
    }
    for (double a : {0.0005, 0.002, 0.008, 0.03}) for (double e : {0.5, 1.0, 2.0}) {
        char b[48]; std::snprintf(b, sizeof b, "fs_a%g_eta%g", a, e); m.push_back({"fs", b, a, e});
    }
    for (double be : {0.98, 0.99, 0.995, 0.998}) for (double c : {1.0, 2.0, 4.0}) {
        char b[48]; std::snprintf(b, sizeof b, "disc_b%g_c%g", be, c); m.push_back({"disc", b, be, c});
    }
    for (double r : {0.02, 0.05, 0.2, 0.5}) for (double base : {250.0, 1000.0, 4000.0}) {
        char b[48]; std::snprintf(b, sizeof b, "vff_r%g_m%.0f", r, base); m.push_back({"vff", b, r, base});
    }
    m.push_back({"rhofs", "rho_fixed_share"});
    return m;
}

// Per-tick output of one run, per method.
struct Trace {
    std::vector<std::vector<float>> excess, mem, sqerr;  // [method][t]
    std::vector<std::vector<float>> e2;                  // [method][t] VFF's last block e2, else 0
};

Trace run(const Scenario& sc, unsigned seed, const std::vector<Method>& M) {
    auto dep = make_dep(sc);
    auto cop = make_cop(sc);
    CopulaSource src(*dep, *cop, seed);
    EwMarginal px{default_marginal_neff()}, py{default_marginal_neff()};
    std::vector<Hist> arms;
    for (double m : arm_mem()) arms.push_back(Hist{lam_of(m)});
    std::vector<XiaoSpearmanEstimator> xarms;
    if (g_xiao_arms)
        for (double m : arm_mem())
            xarms.emplace_back(XiaoSpearmanEstimator::Config{.bins = kBins, .lambda = lam_of(m),
                                                             .rank_pit_window = 500, .density_prior = kPrior}, "arm");
    std::vector<FixedShare> fs; std::vector<Discounted> ds; std::vector<Vff> vf; RhoFixedShare rf;
    std::vector<std::pair<char, std::size_t>> slot;   // method -> (kind, index)
    for (const auto& m : M) {
        if (m.kind == "fixed") slot.push_back({'x', std::size_t(m.p1)});
        else if (m.kind == "fs") { slot.push_back({'f', fs.size()}); fs.push_back(FixedShare{m.p1, m.p2}); }
        else if (m.kind == "disc") { slot.push_back({'d', ds.size()}); ds.push_back(Discounted{m.p1, m.p2}); }
        else if (m.kind == "vff") { slot.push_back({'v', vf.size()}); vf.push_back(Vff{m.p1, m.p2}); }
        else slot.push_back({'r', 0});
    }
    Trace T;
    T.excess.assign(M.size(), std::vector<float>(kN, 0));
    T.mem.assign(M.size(), std::vector<float>(kN, 0));
    T.sqerr.assign(M.size(), std::vector<float>(kN, 0));
    T.e2.assign(M.size(), std::vector<float>(kN, 0));
    std::vector<double> lp_rank(kArms), lp_true(kArms), rho(kArms);
    auto mixlog = [&](const std::vector<double>& w) {
        double s = 0; for (std::size_t k = 0; k < kArms; ++k) s += w[k] * std::exp(lp_true[k]); return std::log(s); };
    auto mixrho = [&](const std::vector<double>& w) {
        double s = 0; for (std::size_t k = 0; k < kArms; ++k) s += w[k] * rho[k]; return std::clamp(s, -1.0, 1.0); };
    auto mixmem = [&](const std::vector<double>& w) {
        double s = 0; for (std::size_t k = 0; k < kArms; ++k) s += w[k] * arm_mem()[k]; return s; };

    for (std::size_t t = 0; t < kN; ++t) {
        const auto s = src.next();
        const double u = px.transform(s.x), v = py.transform(s.y);  // operational (rank) PIT
        const double ut = Phi(s.x), vt = Phi(s.y);                   // evaluation (true) PIT
        const bool scorable = px.size() >= 30;
        const std::string& fam = (t < kChange) ? sc.before : sc.after;
        const double lc = true_logc(fam, s.rho, s.x, s.y);
        for (std::size_t k = 0; k < kArms; ++k) {
            if (g_xiao_arms) {
                lp_rank[k] = xarms[k].log_density_uv(u, v);
                lp_true[k] = xarms[k].log_density_uv(ut, vt);
                rho[k] = xarms[k].estimate();
            } else {
                lp_rank[k] = arms[k].logpdf(u, v);
                lp_true[k] = arms[k].logpdf(ut, vt);
                rho[k] = arms[k].rho();
            }
        }
        // Predictions made before seeing t, evaluated at the true PIT.
        for (std::size_t i = 0; i < M.size(); ++i) {
            double lg, r, me, e2 = 0;
            switch (slot[i].first) {
                case 'x': lg = lp_true[slot[i].second]; r = rho[slot[i].second]; me = arm_mem()[slot[i].second]; break;
                case 'f': { auto& b = fs[slot[i].second]; lg = mixlog(b.w); r = mixrho(b.w); me = mixmem(b.w); break; }
                case 'd': { auto& b = ds[slot[i].second]; lg = mixlog(b.w); r = mixrho(b.w); me = mixmem(b.w); break; }
                case 'v': { auto& b = vf[slot[i].second]; lg = b.h.logpdf(ut, vt); r = b.h.rho(); me = b.mem(); e2 = b.last_e2; break; }
                default: lg = mixlog(rf.w); r = mixrho(rf.w); me = mixmem(rf.w);
            }
            T.excess[i][t] = float(lc - lg);
            T.mem[i][t] = float(me);
            T.sqerr[i][t] = float((r - s.rho_s) * (r - s.rho_s));
            T.e2[i][t] = float(e2);
        }
        // Learn from the operational PIT only.
        const double point = 12 * (u - .5) * (v - .5);
        rf.learn(point, rho, mixrho(rf.w), scorable);
        for (auto& b : fs) b.learn(lp_rank, scorable);
        for (auto& b : ds) b.learn(lp_rank, scorable);
        for (auto& b : vf) b.observe(u, v, scorable);
        if (g_xiao_arms) for (auto& a : xarms) a.update(s.x, s.y);
        else for (auto& a : arms) a.update(u, v);
    }
    return T;
}

double window_mean(const std::vector<float>& x, std::size_t a, std::size_t b) {
    double s = 0; for (std::size_t t = a; t < b; ++t) s += x[t]; return s / double(b - a);
}

// ---- sanity checks --------------------------------------------------------------------------

void sanity(std::FILE* out) {
    // (1) The fixed-share loop structure reproduces EstimatorBank (rho score, Xiao arms).
    {
        Scenario sc{"jump", "jump", "gaussian", "gaussian", 0, true};
        auto dep = make_dep(sc); auto cop = make_cop(sc); CopulaSource src(*dep, *cop, 42);
        std::vector<std::unique_ptr<DependenceEstimator>> ref_m;
        std::vector<std::shared_ptr<XiaoSpearmanEstimator>> mine;
        for (double m : arm_mem()) {
            auto cfg = XiaoSpearmanEstimator::Config{.bins = 20, .window = 0, .lambda = lam_of(m), .calibration = 100};
            ref_m.push_back(std::make_unique<XiaoSpearmanEstimator>(cfg, "m"));
            mine.push_back(std::make_shared<XiaoSpearmanEstimator>(cfg, "m"));
        }
        EstimatorBank ref(std::move(ref_m), EstimatorBank::Config{}, "ref");
        RhoFixedShare rf; EwMarginal px{default_marginal_neff()}, py{default_marginal_neff()};
        double maxdiff = 0; std::size_t n = 0;
        for (std::size_t t = 0; t < kN; ++t) {
            const auto s = src.next();
            const double u = px.transform(s.x), v = py.transform(s.y);
            std::vector<double> r(kArms);
            for (std::size_t k = 0; k < kArms; ++k) r[k] = mine[k]->estimate();
            double mr = 0; for (std::size_t k = 0; k < kArms; ++k) mr += rf.w[k] * r[k];
            rf.learn(12 * (u - .5) * (v - .5), r, std::clamp(mr, -1.0, 1.0), px.size() >= 30);
            for (auto& m : mine) m->update(s.x, s.y);
            ref.update(s.x, s.y);
            double mine_est = 0; for (std::size_t k = 0; k < kArms; ++k) mine_est += rf.w[k] * mine[k]->estimate();
            if (++n >= 200) maxdiff = std::max(maxdiff, std::abs(ref.estimate() - std::clamp(mine_est, -1.0, 1.0)));
        }
        std::fprintf(out, "sanity 1: max |EstimatorBank - this loop| = %.3g (must be ~0)\n", maxdiff);
    }
    // (2) True densities integrate to 1: E_uniform[c(U,V)] by Monte Carlo.
    {
        std::mt19937 rng(7); std::uniform_real_distribution<double> U(1e-6, 1 - 1e-6);
        for (const char* f : {"gaussian", "clayton", "gumbel", "t"})
            for (double d : {0.2, 0.6, 0.8}) {
                double s = 0; const int n = 400000;
                for (int i = 0; i < n; ++i) {
                    const double u = U(rng), v = U(rng);
                    const double x = special::normal_ppf(u), y = special::normal_ppf(v);
                    s += std::exp(true_logc(f, d, x, y));
                }
                std::fprintf(out, "sanity 2: %-8s d=%.1f  E_unif[c] = %.4f (must be ~1)\n", f, d, s / n);
            }
    }
}

// Rosenblatt null calibration: mean e2 for fixed-memory histograms, rank vs true PIT.
void calibration(std::FILE* out) {
    std::fprintf(out, "\nRosenblatt e2 calibration (mean over ticks 1500-2500, 20 seeds; null = 1)\n");
    std::fprintf(out, "%-14s %-6s %10s %10s\n", "copula", "mem", "rank PIT", "true PIT");
    for (double d : {0.0, 0.6})
        for (const char* f : {"gaussian", "clayton"}) {
            if (d == 0.0 && std::string(f) == "clayton") continue;
            for (double m : {250.0, 1000.0, 4000.0}) {
                double er = 0, et = 0; int n = 0;
                for (unsigned seed = 900; seed < 920; ++seed) {
                    ConstantDependence dep(d); auto cop = dgp::make_copula(f); CopulaSource src(dep, *cop, seed);
                    EwMarginal px{default_marginal_neff()}, py{default_marginal_neff()}; Hist hr{lam_of(m)}, ht{lam_of(m)};
                    for (std::size_t t = 0; t < kN; ++t) {
                        const auto s = src.next();
                        const double u = px.transform(s.x), v = py.transform(s.y), ut = Phi(s.x), vt = Phi(s.y);
                        if (t >= 1500 && t < 2500) { er += hr.rosenblatt_e2(u, v); et += ht.rosenblatt_e2(ut, vt); ++n; }
                        hr.update(u, v); ht.update(ut, vt);
                    }
                }
                char lab[32]; std::snprintf(lab, sizeof lab, "%s d=%.1f", f, d);
                std::fprintf(out, "%-14s %-6.0f %10.3f %10.3f\n", lab, m, er / n, et / n);
            }
        }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string out_dir = argc > 1 ? argv[1] : "../notes/studies/memory_tradeoff";
    g_xiao_arms = argc > 2 && std::string(argv[2]) == "xiao";
    auto open = [&](const std::string& f) { return std::fopen((out_dir + "/" + f).c_str(), "w"); };
    std::FILE* chk = open("checks.txt");
    if (!chk) { std::fprintf(stderr, "cannot write to %s (create it first)\n", out_dir.c_str()); return 1; }
    sanity(chk);
    calibration(chk);
    std::fclose(chk);

    const auto M = all_methods();
    const auto& S = scenarios();

    // ---- 1. Tuning: one configuration per method family -----------------------------------
    // Objective, fixed in advance: mean over {jump, drift} of mean excess log loss on [500, 5000),
    // tuning seeds 42..71. (Same ranking as true-PIT log score; the true entropy is common.)
    std::vector<std::vector<double>> tune(S.size(), std::vector<double>(M.size(), 0));
    for (std::size_t si = 0; si < S.size(); ++si)
        for (int k = 0; k < kTuneSeeds; ++k) {
            const auto T = run(S[si], kTuneSeed + k, M);
            for (std::size_t i = 0; i < M.size(); ++i) tune[si][i] += window_mean(T.excess[i], kBurn, kN) / kTuneSeeds;
        }
    std::FILE* tf = open("tuning.csv");
    std::fprintf(tf, "method,kind");
    for (const auto& s : S) std::fprintf(tf, ",%s", s.name.c_str());
    std::fprintf(tf, ",objective\n");
    std::map<std::string, std::size_t> chosen;
    std::map<std::string, double> best;
    for (std::size_t i = 0; i < M.size(); ++i) {
        const double obj = 0.5 * (tune[0][i] + tune[1][i]);
        std::fprintf(tf, "%s,%s", M[i].label.c_str(), M[i].kind.c_str());
        for (std::size_t si = 0; si < S.size(); ++si) std::fprintf(tf, ",%.6f", tune[si][i]);
        std::fprintf(tf, ",%.6f\n", obj);
        if (!best.count(M[i].kind) || obj < best[M[i].kind]) { best[M[i].kind] = obj; chosen[M[i].kind] = i; }
    }
    std::fclose(tf);

    // Methods carried to evaluation: the chosen config of each family, plus the most aggressive
    // VFF (fastest rate, shortest base) as the "is the signal there at all?" diagnostic.
    std::size_t aggressive = 0;
    for (std::size_t i = 0; i < M.size(); ++i) if (M[i].label == "vff_r0.5_m250") aggressive = i;
    std::vector<std::size_t> E{chosen["fixed"], chosen["fs"], chosen["disc"], chosen["vff"], aggressive, chosen["rhofs"]};
    std::vector<Method> EM; for (auto i : E) EM.push_back(M[i]);
    const std::vector<std::string> role{"fixed", "fs", "disc", "vff", "vff_aggr", "rho_fs"};

    // ---- 2. Evaluation on fresh seeds -----------------------------------------------------
    struct Win { std::string name; std::size_t a, b; };
    auto windows = [&](const Scenario& s) {
        std::vector<Win> w{{"whole", kBurn, kN}};
        if (s.name == "jump") {
            w.push_back({"pre", kBurn, kChange});
            w.push_back({"rec_0_100", kChange, kChange + 100});
            w.push_back({"rec_100_500", kChange + 100, kChange + 500});
            w.push_back({"rec_500_1500", kChange + 500, kChange + 1500});
            w.push_back({"post", kChange, kN});
        }
        if (s.before != s.after) w.push_back({"post_0_500", kChange, kChange + 500});
        return w;
    };
    std::FILE* ew = open("eval_windows.csv");
    std::fprintf(ew, "scenario,window,seed,role,method,excess_x1000,rmse\n");
    std::FILE* tr = open("traces.csv");
    std::fprintf(tr, "scenario,t,role,excess,mem,vff_e2\n");
    for (const auto& sc : S) {
        std::vector<Trace> R;
        for (int k = 0; k < kEvalSeeds; ++k) R.push_back(run(sc, kEvalSeed + k, EM));
        for (const auto& w : windows(sc))
            for (int k = 0; k < kEvalSeeds; ++k)
                for (std::size_t j = 0; j < EM.size(); ++j)
                    std::fprintf(ew, "%s,%s,%u,%s,%s,%.5f,%.6f\n", sc.name.c_str(), w.name.c_str(), kEvalSeed + k,
                                 role[j].c_str(), EM[j].label.c_str(), 1000 * window_mean(R[k].excess[j], w.a, w.b),
                                 std::sqrt(window_mean(R[k].sqerr[j], w.a, w.b)));
        for (std::size_t t = 2000; t < 4000; ++t)
            for (std::size_t j = 0; j < EM.size(); ++j) {
                double ex = 0, me = 0, e2 = 0;
                for (const auto& r : R) { ex += r.excess[j][t]; me += r.mem[j][t]; e2 += r.e2[j][t]; }
                std::fprintf(tr, "%s,%zu,%s,%.6f,%.2f,%.4f\n", sc.name.c_str(), t, role[j].c_str(),
                             ex / kEvalSeeds, me / kEvalSeeds, e2 / kEvalSeeds);
            }
    }
    std::fclose(ew);
    std::fclose(tr);
    std::printf("chosen: fixed=%s fs=%s disc=%s vff=%s (aggressive diag: %s); outputs in %s\n",
                M[chosen["fixed"]].label.c_str(), M[chosen["fs"]].label.c_str(), M[chosen["disc"]].label.c_str(),
                M[chosen["vff"]].label.c_str(), M[aggressive].label.c_str(), out_dir.c_str());
    return 0;
}
