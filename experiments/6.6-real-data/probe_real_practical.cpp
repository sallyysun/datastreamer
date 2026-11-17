// Real-data rerun of the 32-pair MFF comparison on the Chapter 6.3 construction (probe_strategies.cpp): one cumulative
// DataSketches KLL k = 256 pseudo-observation sequence per leg (clipped to [1e-3, 1 - 1e-3]) feeding the five RMPL arms
// (gain 2/(N+1), N = 39 ... 1999) and two MFF banks: the log-loss / fixed-share rule (eta 1, delta 1e-4) scored by
// the Gaussian copula (MFF-G, the thesis MFF) or the t4 copula (MFF-t).
// Input: EWMA-vol-standardised daily returns, one CSV per pair (date,x,y). Output per day: (U, V), the seven rho_S
// forecasts held before the day, and each forecast's Gaussian and t4 copula log density at (U, V).
// Practical-application extension: adds VFF (probe_strategies.cpp Vff: RMPL, gain 2/(N+1) set from VffController,
// rate 0.02, start N = 199, p_n = 12(U-1/2)(V-1/2) observed once more than 30 margin observations are held) and MFF's
// predictive MIXTURE density log sum_k w_k c(U, V; rho_k) (Chapter 5), for each working family, with the weights held
// before the day.
// EST=rmpl|xiao|hermite (default rmpl): the arm and VFF estimator, configured as in probe_strategies.cpp; counting
// matrix and Hermite arms read the standardised returns (their own margins), RMPL reads probit of the KLL pseudo-obs.
// Usage: build/probe_real_practical in.csv out.csv     ROLL=1: + rolling windows   LIVE=1: inputs over TCP (REPLAY)
#include <chrono>
#include <cmath>
#include <deque>
#include <numeric>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "../../src/estimators/dependence/ds_margin.hpp"
#include "../../src/estimators/dependence/marginal_rank_pit.hpp"
#include "../../src/estimators/copula_family.hpp"
#include "../../src/estimators/dependence/vff.hpp"
#include "../../src/estimators/evaluation.hpp"
#include "../../src/estimators/registry.hpp"
#include "../common/live_source.hpp"

namespace {
using thesis::kK; using thesis::kMem;
constexpr double kDelta = thesis::kClip, kNu = 4.0;

std::string est_kind() { const char* e = std::getenv("EST"); return e ? e : "rmpl"; }
bool raw_inputs() { return est_kind() != "rmpl"; }
std::vector<std::unique_ptr<DependenceEstimator>> make_arms() {
    std::vector<std::unique_ptr<DependenceEstimator>> a;
    const std::string e = est_kind();
    for (double n : kMem) a.push_back(thesis::arm(e, n, "m"));
    return a;
}

double logc(copula::Family F, double u, double v, double rho) {
    return copula::log_density(F, u, v, copula::param_from_spearman(F, rho, kNu), kNu);
}

// MFF with log loss of working family F, the margins supplied from outside (probe_strategies' Mff
// with the working family as a parameter).
struct Mff {
    copula::Family F;
    double eta = 1.0, delta = 1e-4;
    std::vector<double> w = std::vector<double>(kK, 1.0 / kK), loss = std::vector<double>(kK), lw = std::vector<double>(kK);
    double estimate(const std::vector<std::unique_ptr<DependenceEstimator>>& arms) const {
        double r = 0;
        for (std::size_t k = 0; k < kK; ++k) r += w[k] * arms[k]->estimate();
        return std::clamp(r, -1.0, 1.0);
    }
    void score(const std::vector<std::unique_ptr<DependenceEstimator>>& arms, double u, double v, std::size_t n_seen) {
        bool scorable = n_seen >= 30;
        for (const auto& a : arms) scorable = scorable && a->ready();
        if (!scorable) return;
        for (std::size_t k = 0; k < kK; ++k) loss[k] = -logc(F, u, v, arms[k]->estimate());
        double worst = 0;
        for (double l : loss) if (std::isfinite(l)) worst = std::max(worst, l);
        for (double& l : loss) if (!std::isfinite(l)) l = worst;
        double top = -INFINITY;
        for (std::size_t k = 0; k < kK; ++k) {
            lw[k] = (w[k] > 0 ? std::log(w[k]) : -INFINITY) - eta * loss[k];
            top = std::max(top, lw[k]);
        }
        double tot = 0;
        for (std::size_t k = 0; k < kK; ++k) { w[k] = std::exp(lw[k] - top); tot += w[k]; }
        for (double& x : w) x = (1 - delta) * x / tot + delta / kK;
    }
};

struct Vff {
    std::unique_ptr<DependenceEstimator> inner;
    MemoryControlled* mc;
    VffController ctl;
    explicit Vff(double rate) : inner(thesis::arm(est_kind(), 199, "v")),   // probe_strategies' vff_inner
        mc(dynamic_cast<MemoryControlled*>(inner.get())), ctl(VffController::Config{.rate = rate}) {}
    void update(double x, double y, double u, double v, std::size_t n_after) {   // probe_strategies' Vff::update
        if (n_after > 30) ctl.observe(12.0 * (u - 0.5) * (v - 0.5));
        if (mc) mc->set_neff(ctl.neff());
        if (raw_inputs()) inner->update(x, y); else inner->update(RankPit::probit(u), RankPit::probit(v));
    }
};

double mixture(copula::Family F, const std::vector<double>& w, const std::vector<std::unique_ptr<DependenceEstimator>>& arms,
               double u, double v) {
    double top = -INFINITY; double l[kK];
    for (std::size_t k = 0; k < kK; ++k) { l[k] = std::log(w[k]) + logc(F, u, v, arms[k]->estimate()); top = std::max(top, l[k]); }
    double s = 0; for (double x : l) s += std::exp(x - top);
    return top + std::log(s);
}

// Rolling-window batch benchmark: exact sample Spearman correlation of the preceding W pairs (ranks within the window,
// ties averaged; Pearson correlation of the ranks). Before W pairs are held it uses all pairs so far.
double spearman_window(const std::deque<std::pair<double, double>>& w) {
    const std::size_t n = w.size();
    if (n < 3) return 0.0;
    auto ranks = [&](bool first) {
        std::vector<std::size_t> o(n); std::iota(o.begin(), o.end(), 0);
        auto key = [&](std::size_t i) { return first ? w[i].first : w[i].second; };
        std::sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return key(a) < key(b); });
        std::vector<double> r(n);
        for (std::size_t i = 0; i < n;) {
            std::size_t j = i; while (j + 1 < n && key(o[j + 1]) == key(o[i])) ++j;
            const double avg = 0.5 * (double(i) + double(j)) + 1.0;
            for (std::size_t k = i; k <= j; ++k) r[o[k]] = avg;
            i = j + 1;
        }
        return r;
    };
    const auto rx = ranks(true), ry = ranks(false);
    const double m = 0.5 * (double(n) + 1.0);
    double sxy = 0, sxx = 0, syy = 0;
    for (std::size_t i = 0; i < n; ++i) { sxy += (rx[i] - m) * (ry[i] - m); sxx += (rx[i] - m) * (rx[i] - m); syy += (ry[i] - m) * (ry[i] - m); }
    return (sxx > 0 && syy > 0) ? sxy / std::sqrt(sxx * syy) : 0.0;
}

// Exact Gaussian MPL on an unweighted set of normal scores: the root in [-1, 1] of -W r^3 + P r^2 + (W - S) r + P with
// W = count, S = sum (a^2 + b^2), P = sum ab (the cubic RecursiveMPL::step_discounted solves, solved the same way).
double mpl_root(double Wn, double Sn, double Pn, double r0) {
    const auto f = [&](double r) { return -Wn * r * r * r + Pn * r * r + (Wn - Sn) * r + Pn; };
    const auto df = [&](double r) { return -3.0 * Wn * r * r + 2.0 * Pn * r + (Wn - Sn); };
    double lo = -1.0, hi = 1.0, r = r0;
    for (int it = 0; it < 60; ++it) {
        const double fr = f(r);
        if (fr > 0.0) lo = r; else hi = r;
        const double d = df(r); double nr = (d != 0.0) ? r - fr / d : 0.5 * (lo + hi);
        if (!(nr > lo && nr < hi)) nr = 0.5 * (lo + hi);
        if (std::abs(nr - r) < 1e-12) { r = nr; break; }
        r = nr;
    }
    return std::clamp(r, -0.999, 0.999);
}
struct RollingMpl {   // exact MPL over the last W normal-score pairs; O(1) per day from running sums
    std::size_t W; std::deque<std::pair<double, double>> q; double S = 0, P = 0, r = 0;
    double estimate() const { return q.size() < 3 ? 0.0 : (6.0 / M_PI) * std::asin(r / 2.0); }
    void update(double a, double b) {
        q.emplace_back(a, b); S += a * a + b * b; P += a * b;
        if (q.size() > W) { auto [x, y] = q.front(); q.pop_front(); S -= x * x + y * y; P -= x * y; }
        r = mpl_root(double(q.size()), S, P, r);
    }
};

struct Data { std::vector<std::string> date; std::vector<double> x, y; };
Data read(const char* path) {
    Data d; std::ifstream f(path); std::string line; std::getline(f, line);
    while (std::getline(f, line)) {
        std::stringstream ss(line); std::string a, b, c;
        std::getline(ss, a, ','); std::getline(ss, b, ','); std::getline(ss, c, ',');
        d.date.push_back(a); d.x.push_back(std::stod(b)); d.y.push_back(std::stod(c));
    }
    return d;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: probe_real_pairs in.csv [out.csv]\n"); return 1; }
    Data d = read(argv[1]);
    if (live_mode()) {   // the same returns over the live pipeline: the server replays the file, as 0.01 x
        const LiveStream L = live_replay(argv[1], d.x.size());
        d.x = L.x; d.y = L.y;
    }
    ds_margin::seed(1);
    ds_margin::Kll kx(thesis::kKll), ky(thesis::kKll);
    auto arms = make_arms();
    Mff mg{copula::Family::Gaussian}, mt{copula::Family::StudentT};
    Vff vff(0.02);
    FILE* o = std::fopen(argv[2], "w");
    std::fprintf(o, "date,u,v");
    const char* nm[7] = {"fixed_39", "fixed_99", "fixed_199", "fixed_399", "fixed_1999", "mff_g", "mff_t"};
    for (auto s : nm) std::fprintf(o, ",rho_%s", s);
    for (auto s : nm) std::fprintf(o, ",lg_%s,lt_%s", s, s);
    for (std::size_t k = 0; k < kK; ++k) std::fprintf(o, ",wg_%g", kMem[k]);
    for (std::size_t k = 0; k < kK; ++k) std::fprintf(o, ",wt_%g", kMem[k]);
    std::fprintf(o, ",rho_vff,lg_vff,lt_vff,neff_vff,lg_mffmix_g,lt_mffmix_t,neff_mff_g,neff_mff_t");
    const bool roll = std::getenv("ROLL") != nullptr;
    const std::size_t kW[5] = {39, 99, 199, 399, 1999};
    std::deque<std::pair<double, double>> hist; double roll_secs = 0;
    if (roll) for (std::size_t W : kW) std::fprintf(o, ",rho_roll_%zu,lg_roll_%zu,lt_roll_%zu", W, W, W);
    // matched benchmarks on the RMPL normal scores: rolling-window exact MPL and exponentially weighted exact MPL
    std::vector<RollingMpl> rmpl_roll; std::vector<std::unique_ptr<RecursiveMPL>> rmpl_exp;
    if (roll) for (std::size_t i = 0; i < 5; ++i) {
        rmpl_roll.push_back(RollingMpl{kW[i]});
        RecursiveMPL::Config m{}; m.margin = RecursiveMPL::Margin::ExactNormal; m.update = RecursiveMPL::Update::Exact; m.gain = 2.0 / (kMem[i] + 1.0);
        rmpl_exp.push_back(std::make_unique<RecursiveMPL>(m, "e"));
        std::fprintf(o, ",rho_mplroll_%zu,lg_mplroll_%zu,lt_mplroll_%zu,rho_mplexp_%g,lg_mplexp_%g,lt_mplexp_%g", kW[i], kW[i], kW[i], kMem[i], kMem[i], kMem[i]);
    }
    using clk = std::chrono::steady_clock;
    auto since = [](clk::time_point t0) { return std::chrono::duration<double>(clk::now() - t0).count(); };
    double t_kll = 0, t_arms = 0, t_mff = 0, t_vff = 0, t_mplroll = 0, t_mplexp = 0; std::vector<double> t_roll(5, 0.0);
    std::fprintf(o, "\n");
    for (std::size_t t = 0; t < d.x.size(); ++t) {
        auto tk = clk::now();
        const double u = std::clamp(kx.cdf(d.x[t]), kDelta, 1 - kDelta), v = std::clamp(ky.cdf(d.y[t]), kDelta, 1 - kDelta);
        kx.insert(d.x[t]); ky.insert(d.y[t]);
        t_kll += since(tk);
        double f[7];
        for (std::size_t k = 0; k < kK; ++k) f[k] = arms[k]->estimate();
        f[5] = mg.estimate(arms); f[6] = mt.estimate(arms);
        std::fprintf(o, "%s,%.8f,%.8f", d.date[t].c_str(), u, v);
        for (double x : f) std::fprintf(o, ",%.8f", x);
        for (double x : f) std::fprintf(o, ",%.8f,%.8f", logc(copula::Family::Gaussian, u, v, x), logc(copula::Family::StudentT, u, v, x));
        for (double x : mg.w) std::fprintf(o, ",%.6g", x);   // weights held before the day
        for (double x : mt.w) std::fprintf(o, ",%.6g", x);
        const double fv = vff.inner->estimate();
        double ng = 0, nt = 0; for (std::size_t k = 0; k < kK; ++k) { ng += mg.w[k] * kMem[k]; nt += mt.w[k] * kMem[k]; }
        std::fprintf(o, ",%.8f,%.8f,%.8f,%.4f,%.8f,%.8f,%.3f,%.3f", fv, logc(copula::Family::Gaussian, u, v, fv),
                     logc(copula::Family::StudentT, u, v, fv), vff.ctl.neff(), mixture(copula::Family::Gaussian, mg.w, arms, u, v),
                     mixture(copula::Family::StudentT, mt.w, arms, u, v), ng, nt);
        if (roll) {
            for (std::size_t wi = 0; wi < 5; ++wi) {
                const std::size_t W = kW[wi];
                const auto t0 = std::chrono::steady_clock::now();
                std::deque<std::pair<double, double>> win(hist.end() - std::min(W, hist.size()), hist.end());
                const double r = spearman_window(win);
                const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); roll_secs += dt; t_roll[wi] += dt;
                std::fprintf(o, ",%.8f,%.8f,%.8f", r, logc(copula::Family::Gaussian, u, v, r), logc(copula::Family::StudentT, u, v, r));
            }
            for (std::size_t i = 0; i < 5; ++i) {
                const double rr = rmpl_roll[i].estimate(), re = rmpl_exp[i]->estimate();
                std::fprintf(o, ",%.8f,%.8f,%.8f,%.8f,%.8f,%.8f", rr, logc(copula::Family::Gaussian, u, v, rr), logc(copula::Family::StudentT, u, v, rr),
                             re, logc(copula::Family::Gaussian, u, v, re), logc(copula::Family::StudentT, u, v, re));
            }
            hist.emplace_back(d.x[t], d.y[t]);   // the day enters the window only after it has been scored
            if (hist.size() > 1999) hist.pop_front();
        }
        std::fprintf(o, "\n");
        auto tm = clk::now(); mg.score(arms, u, v, t); mt.score(arms, u, v, t); t_mff += since(tm);
        const double zx = RankPit::probit(u), zy = RankPit::probit(v);
        auto ta = clk::now(); for (auto& a : arms) { if (raw_inputs()) a->update(d.x[t], d.y[t]); else a->update(zx, zy); } t_arms += since(ta);
        auto tv = clk::now(); vff.update(d.x[t], d.y[t], u, v, t + 1); t_vff += since(tv);
        if (roll) {
            if (t >= 30) {   // the same warm-up as the RMPL arms (30 margin observations held)
                auto tr = clk::now(); for (auto& rmr : rmpl_roll) rmr.update(zx, zy); t_mplroll += since(tr);
            }
            auto te = clk::now(); for (auto& e : rmpl_exp) e->update(zx, zy); t_mplexp += since(te);
        }
    }
    std::fclose(o);
    if (roll) {
        const double n = double(d.x.size()), us = 1e6 / n;
        std::fprintf(stderr, "TIME %s days %zu | us/day: KLL margins %.3f | 5 %s arms %.3f | MFF scoring (2 banks) %.3f | VFF %.3f | rolling MPL x5 %.3f | exp exact MPL x5 %.3f | rolling Spearman W=39 %.2f 99 %.2f 199 %.2f 399 %.2f 1999 %.2f\n",
                     argv[1], d.x.size(), t_kll * us, est_kind().c_str(), t_arms * us, t_mff * us, t_vff * us, t_mplroll * us, t_mplexp * us,
                     t_roll[0] * us, t_roll[1] * us, t_roll[2] * us, t_roll[3] * us, t_roll[4] * us);
    }
    return 0;
}
