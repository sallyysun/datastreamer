// Compare forgetting strategies on paired daily returns; write daily forecasts, losses and weights.
// Usage: real_data_comparison in.csv out.csv
// EST selects the estimator; ROLL=1 adds rolling benchmarks; LIVE=1 replays over TCP.
// VERBOSE=1 prints timing with ROLL=1.
#include <algorithm>
#include <array>
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
#include <string_view>
#include <vector>

#include "../../src/estimators/dependence/datasketches_kll.hpp"
#include "../../src/special_functions.hpp"
#include "../../src/copula_family.hpp"
#include "../../src/estimators/dependence/vff.hpp"
#include "../../src/estimators/evaluation.hpp"
#include "../../src/estimators/registry.hpp"
#include "live_source.hpp"

namespace {
using thesis::kK; using thesis::kMem;
constexpr double kDelta = thesis::kClip, kNu = 4.0;

std::string est_kind() { const char* e = std::getenv("EST"); return e ? e : "rmpl"; }
std::vector<std::unique_ptr<DependenceEstimator>> make_arms(const std::string& estimator) {
    std::vector<std::unique_ptr<DependenceEstimator>> arms;
    for (double n : kMem) arms.push_back(thesis::arm(estimator, n, "m"));
    return arms;
}

double logc(copula::Family F, double u, double v, double rho) {
    return copula::log_density(F, u, v, copula::param_from_spearman(F, rho, kNu), kNu);
}

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

// Solve the Gaussian MPL score equation from rolling sums.
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
    if (argc < 3) {
        std::fprintf(stderr, "usage: real_data_comparison in.csv out.csv\n");
        return 1;
    }
    Data d = read(argv[1]);
    if (live_mode()) {   // the same returns over the live pipeline: the server replays the file, as 0.01 x
        const LiveStream L = live_replay(argv[1], d.x.size());
        d.x = L.x; d.y = L.y;
    }
    datasketches_kll::seed(1);
    datasketches_kll::Kll kx(thesis::kKll), ky(thesis::kKll);
    const std::string estimator = est_kind();
    const bool use_raw_inputs = estimator != "rmpl";
    auto arms = make_arms(estimator);
    thesis::Mff mg{copula::Family::Gaussian}, mt{copula::Family::StudentT};
    thesis::Vff vff(estimator, thesis::vff_rates().front());
    FILE* o = std::fopen(argv[2], "w");
    std::fprintf(o, "date,u,v");
    const auto& strategy_names = thesis::strategy_names();
    std::vector<std::string> forecast_names(strategy_names.begin(), strategy_names.begin() + kK);
    forecast_names.insert(forecast_names.end(), {"mff_g", "mff_t"});
    for (const auto& name : forecast_names) std::fprintf(o, ",rho_%s", name.c_str());
    for (const auto& name : forecast_names) std::fprintf(o, ",lg_%s,lt_%s", name.c_str(), name.c_str());
    for (std::size_t k = 0; k < kK; ++k) std::fprintf(o, ",wg_%g", kMem[k]);
    for (std::size_t k = 0; k < kK; ++k) std::fprintf(o, ",wt_%g", kMem[k]);
    std::fprintf(o, ",rho_vff,lg_vff,lt_vff,neff_vff,lg_mffmix_g,lt_mffmix_t,neff_mff_g,neff_mff_t");
    const bool roll = std::getenv("ROLL") != nullptr;
    std::array<std::size_t, kK> rolling_windows;
    std::copy_n(kMem, kK, rolling_windows.begin());
    const auto longest_window = *std::max_element(rolling_windows.begin(), rolling_windows.end());
    std::deque<std::pair<double, double>> hist;
    if (roll) for (std::size_t W : rolling_windows) std::fprintf(o, ",rho_roll_%zu,lg_roll_%zu,lt_roll_%zu", W, W, W);
    // matched benchmarks on the RMPL normal scores: rolling-window exact MPL and exponentially weighted exact MPL
    std::vector<RollingMpl> rmpl_roll; std::vector<std::unique_ptr<RecursiveMPL>> rmpl_exp;
    if (roll) for (std::size_t i = 0; i < kK; ++i) {
        rmpl_roll.push_back(RollingMpl{rolling_windows[i]});
        RecursiveMPL::Config m{}; m.margin = RecursiveMPL::Margin::ExactNormal; m.update = RecursiveMPL::Update::Exact; m.gain = 2.0 / (kMem[i] + 1.0);
        rmpl_exp.push_back(std::make_unique<RecursiveMPL>(m, "e"));
        std::fprintf(o, ",rho_mplroll_%zu,lg_mplroll_%zu,lt_mplroll_%zu,rho_mplexp_%g,lg_mplexp_%g,lt_mplexp_%g", rolling_windows[i], rolling_windows[i], rolling_windows[i], kMem[i], kMem[i], kMem[i]);
    }
    using clk = std::chrono::steady_clock;
    auto since = [](clk::time_point t0) { return std::chrono::duration<double>(clk::now() - t0).count(); };
    double t_kll = 0, t_arms = 0, t_mff = 0, t_vff = 0, t_mplroll = 0, t_mplexp = 0; std::vector<double> t_roll(kK, 0.0);
    std::fprintf(o, "\n");
    for (std::size_t t = 0; t < d.x.size(); ++t) {
        auto tk = clk::now();
        const double u = std::clamp(kx.cdf(d.x[t]), kDelta, 1 - kDelta), v = std::clamp(ky.cdf(d.y[t]), kDelta, 1 - kDelta);
        kx.insert(d.x[t]); ky.insert(d.y[t]);
        t_kll += since(tk);
        double f[kK + 2];
        for (std::size_t k = 0; k < kK; ++k) f[k] = arms[k]->estimate();
        f[kK] = mg.estimate(arms);
        f[kK + 1] = mt.estimate(arms);
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
            for (std::size_t wi = 0; wi < kK; ++wi) {
                const std::size_t W = rolling_windows[wi];
                const auto t0 = std::chrono::steady_clock::now();
                std::deque<std::pair<double, double>> win(hist.end() - std::min(W, hist.size()), hist.end());
                const double r = spearman_window(win);
                const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); t_roll[wi] += dt;
                std::fprintf(o, ",%.8f,%.8f,%.8f", r, logc(copula::Family::Gaussian, u, v, r), logc(copula::Family::StudentT, u, v, r));
            }
            for (std::size_t i = 0; i < kK; ++i) {
                const double rr = rmpl_roll[i].estimate(), re = rmpl_exp[i]->estimate();
                std::fprintf(o, ",%.8f,%.8f,%.8f,%.8f,%.8f,%.8f", rr, logc(copula::Family::Gaussian, u, v, rr), logc(copula::Family::StudentT, u, v, rr),
                             re, logc(copula::Family::Gaussian, u, v, re), logc(copula::Family::StudentT, u, v, re));
            }
            hist.emplace_back(d.x[t], d.y[t]);   // the day enters the window only after it has been scored
            if (hist.size() > longest_window) hist.pop_front();
        }
        std::fprintf(o, "\n");
        auto tm = clk::now();
        mg.score(arms, u, v, t);
        mt.score(arms, u, v, t);
        t_mff += since(tm);

        const double zx = special::probit(u), zy = special::probit(v);
        auto ta = clk::now();
        for (auto& arm : arms) {
            if (use_raw_inputs) arm->update(d.x[t], d.y[t]);
            else arm->update(zx, zy);
        }
        t_arms += since(ta);

        auto tv = clk::now();
        vff.update(d.x[t], d.y[t], u, v, t + 1, use_raw_inputs);
        t_vff += since(tv);

        if (roll) {
            if (t >= 30) {
                auto tr = clk::now();
                for (auto& model : rmpl_roll) model.update(zx, zy);
                t_mplroll += since(tr);
            }
            auto te = clk::now();
            for (auto& model : rmpl_exp) model->update(zx, zy);
            t_mplexp += since(te);
        }
    }
    std::fclose(o);
    const char* verbose = std::getenv("VERBOSE");
    if (roll && verbose && std::string_view(verbose) == "1") {
        const double n = double(d.x.size()), us = 1e6 / n;
        std::fprintf(stderr, "TIME %s days %zu | us/day: KLL margins %.3f | 5 %s arms %.3f | MFF scoring (2 banks) %.3f | VFF %.3f | rolling MPL x5 %.3f | exp exact MPL x5 %.3f | rolling Spearman W=39 %.2f 99 %.2f 199 %.2f 399 %.2f 1999 %.2f\n",
                     argv[1], d.x.size(), t_kll * us, estimator.c_str(), t_arms * us, t_mff * us, t_vff * us, t_mplroll * us, t_mplexp * us,
                     t_roll[0] * us, t_roll[1] * us, t_roll[2] * us, t_roll[3] * us, t_roll[4] * us);
    }
    return 0;
}
