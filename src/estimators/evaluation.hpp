#pragma once

// The Chapter 6 evaluation protocol, shared by the in-process probes (experiments/) and the live consumer
// (src/consumer/): the forgetting strategies MFF and VFF (Chapter 5) driven by the shared pseudo-observations (U, V),
// the evaluation windows, and the density metrics (excess log loss, squared Hellinger H^2 against the Gaussian working
// density).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numbers>
#include <memory>
#include <string>
#include <vector>

#include "copula_family.hpp"
#include "density_metrics.hpp"
#include "dependence/vff.hpp"
#include "registry.hpp"

namespace thesis {

constexpr std::size_t kK = 5;                       // MFF arms: mff_memories()
constexpr std::size_t kN = 5000, kBurn = 500, kChange = 2500;
constexpr double kClip = 1e-3;                      // pseudo-observations clipped to [kClip, 1 - kClip]
constexpr std::uint16_t kKll = 256;                 // capacity of the shared DataSketches KLL margins
inline const double kMem[kK] = {39, 99, 199, 399, 1999};
inline const std::vector<double>& vff_rates() { static const std::vector<double> v = {0.02, 0.1}; return v; }
// Strategy order of every output row: five fixed arms, MFF, VFF at each rate.
inline const std::vector<std::string>& strategy_names() {
    static const std::vector<std::string> v = {"fixed_39", "fixed_99", "fixed_199", "fixed_399", "fixed_1999", "mff", "vff_slow", "vff_fast"};
    return v;
}

inline copula::Family family_of(const std::string& f) {
    if (f == "clayton") return copula::Family::Clayton;
    if (f == "gumbel") return copula::Family::Gumbel;
    if (f == "t") return copula::Family::StudentT;
    return copula::Family::Gaussian;
}

// The true copula parameter at dependence knob d (rho_S = (6/pi) asin(d/2), clamped as the generator does), as the
// probes compute it; scored families only.
inline bool scorable(const std::string& f) { return f == "gaussian" || f == "t" || f == "clayton" || f == "gumbel"; }
inline double true_param(const std::string& family, double d) {
    const double rho = std::clamp(d, -0.999, 0.999);
    return copula::param_from_spearman(family_of(family), 6.0 / std::numbers::pi * std::asin(rho / 2.0));
}

struct Win { const char* name; std::size_t a, b; };
inline std::vector<Win> windows(const std::string& s) {
    std::vector<Win> w{{"whole", kBurn, kN}};
    if (s == "jump") {
        w.push_back({"pre", kBurn, kChange}); w.push_back({"rec_0_100", kChange, kChange + 100});
        w.push_back({"rec_100_500", kChange + 100, kChange + 500}); w.push_back({"post", kChange, kN});
    }
    return w;
}
// H^2 checkpoints: every 50 from kBurn, and every 10 in the 100 observations after the jump.
inline bool checkpoint(const std::string& sc, std::size_t t) {
    return t >= kBurn && ((t - kBurn) % 50 == 0 || (sc == "jump" && t >= kChange && t < kChange + 100 && (t - kChange) % 10 == 0));
}

// MFF: Hedge with fixed share over the fixed-memory arms, log loss of the Gaussian density at each arm's rho_S,
// eta = 1, delta = 1e-4; forecast = weighted mean of the arms (density: the weighted mixture).
struct Mff {
    double eta = 1.0, delta = 1e-4;
    std::vector<double> w = std::vector<double>(kK, 1.0 / kK), loss = std::vector<double>(kK), lw = std::vector<double>(kK);
    double estimate(const std::vector<std::unique_ptr<DependenceEstimator>>& arms) const {
        double r = 0;
        for (std::size_t k = 0; k < kK; ++k) r += w[k] * arms[k]->estimate();
        return std::clamp(r, -1.0, 1.0);
    }
    // n_seen: margin observations held before this tick (the bank's mx_.size() before transform)
    void score(const std::vector<std::unique_ptr<DependenceEstimator>>& arms, double u, double v, std::size_t n_seen) {
        bool scorable = n_seen >= 30;
        for (const auto& a : arms) scorable = scorable && a->ready();
        if (!scorable) return;
        for (std::size_t k = 0; k < kK; ++k) {
            const double par = copula::param_from_spearman(copula::Family::Gaussian, arms[k]->estimate());
            loss[k] = -copula::log_density(copula::Family::Gaussian, u, v, par);
        }
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

// VFF: one estimator whose N_eff follows VffController, fed p_n = 12 (U - 1/2)(V - 1/2) once 30 margin observations
// are held; Counting Matrix / Hermite read the raw observations, RMPL reads probit(U).
struct Vff {
    std::unique_ptr<DependenceEstimator> inner;
    MemoryControlled* mc;
    VffController ctl;
    Vff(const std::string& e, double rate) : inner(arm(e, 199, "m")), mc(dynamic_cast<MemoryControlled*>(inner.get())),
                                             ctl(VffController::Config{.rate = rate}) {}
    // n_after: margin observations held after absorbing this tick
    void update(double x, double y, double u, double v, std::size_t n_after, bool raw) {
        if (n_after > 30) ctl.observe(12.0 * (u - 0.5) * (v - 0.5));
        if (mc) mc->set_neff(ctl.neff());
        if (raw) inner->update(x, y); else inner->update(RankPit::probit(u), RankPit::probit(v));
    }
};

inline double gauss_logc(double rho, double u, double v) {
    rho = std::clamp(rho, -0.999, 0.999);
    return copula::Gaussian::log_density(u, v, copula::param_from_spearman(copula::Family::Gaussian, rho));
}

// H^2(c_true(par), c_G(rho)) tabulated over rho in [-0.99, 0.99] (step 0.0025) once per truth parameter.
struct H2Table {
    const dmetric::Grid g = dmetric::Grid::full(81, 8.0);
    copula::Family F;
    std::map<double, std::vector<double>> cache;
    static constexpr double lo = -0.99, step = 0.0025;
    static constexpr int n = 793;
    double direct(double par, double rho) const {
        const dmetric::Table truth(g, [&](double a, double b) { const double l = copula::log_density(F, a, b, par); return std::isfinite(l) ? std::exp(l) : 0.0; });
        const dmetric::Table f(g, [&](double a, double b) { return std::exp(gauss_logc(rho, a, b)); });
        return dmetric::hellinger2(g, truth, f);
    }
    // H^2(c_true(par), sum_k w_k c_G(rho_k)) on the same grid, computed directly (no rho interpolation).
    std::map<double, std::vector<double>> truth;
    double mix(double par, const double* rho, const double* w, std::size_t K) {
        auto it = truth.find(par);
        if (it == truth.end()) {
            const dmetric::Table t(g, [&](double a, double b) { const double l = copula::log_density(F, a, b, par); return std::isfinite(l) ? std::exp(l) : 0.0; });
            it = truth.emplace(par, t.v).first;
        }
        const int n = g.size();
        std::vector<double> r(K), c0(K), c1(K);
        for (std::size_t k = 0; k < K; ++k) {
            r[k] = copula::param_from_spearman(copula::Family::Gaussian, std::clamp(rho[k], -0.999, 0.999));
            c0[k] = 1 / std::sqrt(1 - r[k] * r[k]); c1[k] = 1 / (2 * (1 - r[k] * r[k]));
        }
        double s = 0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                const double x = g.z[i], y = g.z[j];
                double m = 0;
                for (std::size_t k = 0; k < K; ++k)
                    if (w[k] > 0) m += w[k] * c0[k] * std::exp(-(r[k] * r[k] * (x * x + y * y) - 2 * r[k] * x * y) * c1[k]);
                s += std::sqrt(std::max(it->second[std::size_t(i) * n + j], 0.0) * m) * g.w[i] * g.w[j];
            }
        return 1.0 - s;
    }
    // log sum_k w_k c_G(rho_k)(u, v), each component as gauss_logc
    static double mix_logc(const double* rho, const double* w, std::size_t K, double u, double v) {
        double top = -INFINITY;
        std::vector<double> l(K);
        for (std::size_t k = 0; k < K; ++k) { l[k] = w[k] > 0 ? std::log(w[k]) + gauss_logc(rho[k], u, v) : -INFINITY; top = std::max(top, l[k]); }
        double s = 0;
        for (double x : l) s += std::exp(x - top);
        return top + std::log(s);
    }
    double operator()(double par, double rho) {
        auto it = cache.find(par);
        if (it == cache.end()) {
            const dmetric::Table truth(g, [&](double a, double b) { const double l = copula::log_density(F, a, b, par); return std::isfinite(l) ? std::exp(l) : 0.0; });
            std::vector<double> tab(n);
            for (int i = 0; i < n; ++i) {
                const double r = lo + step * i;
                const dmetric::Table f(g, [&](double a, double b) { return std::exp(gauss_logc(r, a, b)); });
                tab[i] = dmetric::hellinger2(g, truth, f);
            }
            it = cache.emplace(par, std::move(tab)).first;
        }
        const double x = (std::clamp(rho, lo, lo + step * (n - 1)) - lo) / step;
        const int i = std::min(n - 2, int(x));
        return it->second[i] + (x - i) * (it->second[i + 1] - it->second[i]);
    }
};

}  // namespace thesis
