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
#include <optional>
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
#include "../simulation/run_live_stream.hpp"

namespace {
using thesis::kK; using thesis::kMem;
constexpr double kNu = 4.0;
using Clock = std::chrono::steady_clock;

double elapsed_seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string est_kind() { const char* e = std::getenv("EST"); return e ? e : "rmpl"; }
std::vector<std::unique_ptr<DependenceEstimator>> make_arms(const std::string& estimator) {
    std::vector<std::unique_ptr<DependenceEstimator>> arms;
    for (double n : kMem) arms.push_back(thesis::arm(estimator, n, "m"));
    return arms;
}

double logc(copula::Family F, double u, double v, double rho) {
    return copula::log_density(F, u, v, copula::param_from_spearman(F, rho, kNu), kNu);
}

double mixture(copula::Family F, const thesis::Mff& mff, const std::vector<std::unique_ptr<DependenceEstimator>>& arms,
               double u, double v) {
    double log_density[kK];
    for (std::size_t k = 0; k < kK; ++k) log_density[k] = logc(F, u, v, arms[k]->estimate());
    return mff.mixture_log_density(log_density);
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

class RollingBenchmarks {
public:
    RollingBenchmarks() {
        std::copy_n(kMem, kK, windows_.begin());
        longest_window_ = *std::max_element(windows_.begin(), windows_.end());
        for (std::size_t i = 0; i < kK; ++i) {
            rolling_mpl_.push_back(RollingMpl{windows_[i]});
            RecursiveMPL::Config config{};
            config.margin = RecursiveMPL::Margin::ExactNormal;
            config.update = RecursiveMPL::Update::Exact;
            config.gain = 2.0 / (kMem[i] + 1.0);
            exponential_mpl_.push_back(std::make_unique<RecursiveMPL>(config, "e"));
        }
    }

    void write_header(FILE* out) const {
        for (auto width : windows_)
            std::fprintf(out, ",rho_roll_%zu,lg_roll_%zu,lt_roll_%zu", width, width, width);
        for (std::size_t i = 0; i < kK; ++i) {
            const auto width = windows_[i];
            const double memory = kMem[i];
            std::fprintf(out, ",rho_mplroll_%zu,lg_mplroll_%zu,lt_mplroll_%zu,rho_mplexp_%g,lg_mplexp_%g,lt_mplexp_%g",
                         width, width, width, memory, memory, memory);
        }
    }

    void write_scores(FILE* out, double u, double v) {
        auto write_score = [&](double rho) {
            std::fprintf(out, ",%.8f,%.8f,%.8f", rho, logc(copula::Family::Gaussian, u, v, rho),
                         logc(copula::Family::StudentT, u, v, rho));
        };
        for (std::size_t i = 0; i < kK; ++i) {
            const auto start = Clock::now();
            const auto first = history_.end() - std::min(windows_[i], history_.size());
            const std::deque<std::pair<double, double>> window(first, history_.end());
            const double rho = spearman_window(window);
            spearman_seconds_[i] += elapsed_seconds(start);
            write_score(rho);
        }
        for (std::size_t i = 0; i < kK; ++i) {
            write_score(rolling_mpl_[i].estimate());
            write_score(exponential_mpl_[i]->estimate());
        }
    }

    // The current day enters each benchmark after its forecast has been scored.
    void update(double x, double y, double zx, double zy, std::size_t t) {
        history_.emplace_back(x, y);
        if (history_.size() > longest_window_) history_.pop_front();
        if (t >= 30) {
            const auto start = Clock::now();
            for (auto& model : rolling_mpl_) model.update(zx, zy);
            rolling_mpl_seconds_ += elapsed_seconds(start);
        }
        const auto start = Clock::now();
        for (auto& model : exponential_mpl_) model->update(zx, zy);
        exponential_mpl_seconds_ += elapsed_seconds(start);
    }

    void print_timing(const char* path, std::size_t days, const std::string& estimator,
                      double kll_seconds, double arms_seconds, double mff_seconds, double vff_seconds) const {
        const double us = 1e6 / double(days);
        std::fprintf(stderr, "TIME %s days %zu | us/day: KLL margins %.3f | %zu %s arms %.3f | MFF scoring (2 banks) %.3f | VFF %.3f"
                     " | rolling MPL x%zu %.3f | exp exact MPL x%zu %.3f | rolling Spearman W=",
                     path, days, kll_seconds * us, windows_.size(), estimator.c_str(), arms_seconds * us,
                     mff_seconds * us, vff_seconds * us, rolling_mpl_.size(), rolling_mpl_seconds_ * us,
                     exponential_mpl_.size(), exponential_mpl_seconds_ * us);
        for (std::size_t i = 0; i < kK; ++i)
            std::fprintf(stderr, "%s%zu %.2f", i == 0 ? "" : " ", windows_[i], spearman_seconds_[i] * us);
        std::fprintf(stderr, "\n");
    }

private:
    std::array<std::size_t, kK> windows_;
    std::size_t longest_window_;
    std::deque<std::pair<double, double>> history_;
    std::vector<RollingMpl> rolling_mpl_;
    std::vector<std::unique_ptr<RecursiveMPL>> exponential_mpl_;
    std::array<double, kK> spearman_seconds_{};
    double rolling_mpl_seconds_ = 0, exponential_mpl_seconds_ = 0;
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
    datasketches_kll::PairedKll margins(thesis::kKll, thesis::kClip);
    const std::string estimator = est_kind();
    const bool use_raw_inputs = estimator != "rmpl";
    auto arms = make_arms(estimator);
    thesis::Mff mg{copula::Family::Gaussian}, mt{copula::Family::StudentT};
    thesis::Vff vff(estimator, thesis::vff_rates().front());
    FILE* o = std::fopen(argv[2], "w");
    if (!o) {
        std::perror(argv[2]);
        return 1;
    }
    std::fprintf(o, "date,u,v");
    const auto& strategy_names = thesis::strategy_names();
    std::vector<std::string> forecast_names(strategy_names.begin(), strategy_names.begin() + kK);
    forecast_names.insert(forecast_names.end(), {"mff_g", "mff_t"});
    for (const auto& name : forecast_names) std::fprintf(o, ",rho_%s", name.c_str());
    for (const auto& name : forecast_names) std::fprintf(o, ",lg_%s,lt_%s", name.c_str(), name.c_str());
    for (std::size_t k = 0; k < kK; ++k) std::fprintf(o, ",wg_%g", kMem[k]);
    for (std::size_t k = 0; k < kK; ++k) std::fprintf(o, ",wt_%g", kMem[k]);
    std::fprintf(o, ",rho_vff,lg_vff,lt_vff,neff_vff,lg_mffmix_g,lt_mffmix_t,neff_mff_g,neff_mff_t");
    std::optional<RollingBenchmarks> rolling;
    if (std::getenv("ROLL")) {
        rolling.emplace();
        rolling->write_header(o);
    }
    double t_kll = 0, t_arms = 0, t_mff = 0, t_vff = 0;
    std::fprintf(o, "\n");
    for (std::size_t t = 0; t < d.x.size(); ++t) {
        const auto tk = Clock::now();
        const auto [u, v] = margins.update(d.x[t], d.y[t]);
        t_kll += elapsed_seconds(tk);
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
                     logc(copula::Family::StudentT, u, v, fv), vff.ctl.neff(), mixture(copula::Family::Gaussian, mg, arms, u, v),
                     mixture(copula::Family::StudentT, mt, arms, u, v), ng, nt);
        if (rolling) rolling->write_scores(o, u, v);
        std::fprintf(o, "\n");
        const auto tm = Clock::now();
        mg.score_and_update_weights(arms, u, v, t);
        mt.score_and_update_weights(arms, u, v, t);
        t_mff += elapsed_seconds(tm);

        const double zx = special::probit(u), zy = special::probit(v);
        const auto ta = Clock::now();
        for (auto& arm : arms) {
            if (use_raw_inputs) arm->update(d.x[t], d.y[t]);
            else arm->update(zx, zy);
        }
        t_arms += elapsed_seconds(ta);

        const auto tv = Clock::now();
        vff.update(d.x[t], d.y[t], u, v, t + 1, use_raw_inputs);
        t_vff += elapsed_seconds(tv);

        if (rolling) rolling->update(d.x[t], d.y[t], zx, zy, t);
    }
    std::fclose(o);
    const char* verbose = std::getenv("VERBOSE");
    if (rolling && verbose && std::string_view(verbose) == "1")
        rolling->print_timing(argv[1], d.x.size(), estimator, t_kll, t_arms, t_mff, t_vff);
    return 0;
}
