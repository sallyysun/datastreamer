#pragma once

// Run the Chapter 6 strategies on one stream with shared KLL margins.
// Score each forecast before updating. Counting Matrix and Hermite use raw inputs; RMPL uses probit(U).
// Scorer uses the weighted MFF density mixture.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "dependence/datasketches_kll.hpp"
#include "evaluation.hpp"
#include "../special_functions.hpp"

namespace thesis {

// One estimator's strategies: the fixed-memory arms, MFF over them, and VFF at each rate.
class EstimatorStrategies {
public:
    explicit EstimatorStrategies(const std::string& est) : est_(est), raw_(est != "rmpl") {
        for (double n : kMem) arms_.push_back(arm(est, n, "m"));
        vffs_.reserve(vff_rates().size());
        for (double r : vff_rates()) vffs_.emplace_back(est, r);
    }
    std::size_t size() const { return kK + 1 + vffs_.size(); }
    void forecasts(std::vector<double>& f) const {
        f.resize(size());
        for (std::size_t k = 0; k < kK; ++k) f[k] = arms_[k]->estimate();
        f[kK] = mff_.estimate(arms_);
        for (std::size_t j = 0; j < vffs_.size(); ++j) f[kK + 1 + j] = vffs_[j].inner->estimate();
    }
    const std::vector<double>& mff_weights() const { return mff_.w; }
    // t = observations already absorbed (the margins' size before this one).
    void update(double x, double y, double u, double v, std::size_t t) {
        mff_.score(arms_, u, v, t);
        const double zx = special::probit(u), zy = special::probit(v);
        for (auto& a : arms_) { if (raw_) a->update(x, y); else a->update(zx, zy); }
        for (auto& vf : vffs_) vf.update(x, y, u, v, t + 1, raw_);
    }
    const std::string& name() const { return est_; }

private:
    std::string est_;
    bool raw_;
    std::vector<std::unique_ptr<DependenceEstimator>> arms_;
    Mff mff_;
    std::vector<Vff> vffs_;
};

class Device {
public:
    Device() : sets_(make_sets()), seeded_(reseed()), margins_(kKll, kClip) {}
    // Forecasts held before the next observation: sets()[e].forecasts().
    const std::vector<EstimatorStrategies>& sets() const { return sets_; }
    // Absorb one observation; returns the shared pseudo-observations (U, V) it was scored at.
    std::pair<double, double> update(double x, double y) {
        const auto [u, v] = margins_.update(x, y);
        for (auto& s : sets_) s.update(x, y, u, v, t_);
        ++t_;
        return {u, v};
    }
    std::size_t count() const { return t_; }

private:
    static std::vector<EstimatorStrategies> make_sets() {
        std::vector<EstimatorStrategies> v;
        for (const auto& e : estimators()) v.emplace_back(e);
        return v;
    }
    // Seed KLL compaction to match the in-process stream.
    static bool reseed() { datasketches_kll::seed(1); return true; }
    std::vector<EstimatorStrategies> sets_;
    bool seeded_;
    datasketches_kll::PairedKll margins_;
    std::size_t t_ = 0;
};

// Per-tick squared error, excess log loss at the true PIT and H^2 at the checkpoints, then window means.
class Scorer {
public:
    Scorer(const std::string& family, const std::string& scenario)
        : fam_(family), sc_(scenario), F_(family_of(family)), h2_{.F = family_of(family)} {}
    // Call before dev.update for the observation with truth (rho_s, pit_x, pit_y, par).
    void observe(const Device& dev, std::size_t t, double rho_s, double pu, double pv, double par) {
        if (rows_.empty()) rows_.resize(dev.sets().size() * dev.sets()[0].size());
        const double lc = copula::log_density(F_, pu, pv, par);
        const bool cp = checkpoint(sc_, t);
        std::vector<double> f;
        std::size_t r = 0;
        for (const auto& s : dev.sets()) {
            s.forecasts(f);
            for (std::size_t i = 0; i < f.size(); ++i, ++r) {
                const double d = f[i] - rho_s;
                double ex, hh = NAN;
                if (i == kK) {
                    ex = lc - H2Table::mix_logc(f.data(), s.mff_weights().data(), kK, pu, pv);
                    if (cp) hh = h2_.mix(par, f.data(), s.mff_weights().data(), kK);
                } else {
                    ex = lc - gauss_logc(f[i], pu, pv);
                    if (cp) hh = h2_(par, f[i]);
                }
                rows_[r].push_back({d * d, ex, hh});
            }
        }
    }
    // The runner's strategies_seeds rows: copula,scenario,estimator,strategy,seed,window,rmse,excess_x1000,h2_x1000
    void write(std::FILE* out, const Device& dev, unsigned seed) const {
        const auto& names = strategy_names();
        std::size_t r0 = 0;
        for (const auto& s : dev.sets()) {
            for (const auto& w : windows(sc_)) {
                const bool dense = std::string(w.name) == "rec_0_100";
                for (std::size_t i = 0; i < s.size(); ++i) {
                    const auto& rs = rows_[r0 + i];
                    double a = 0, b = 0, h = 0; int nh = 0;
                    for (std::size_t t = w.a; t < w.b && t < rs.size(); ++t) {
                        a += rs[t].se; b += rs[t].ex;
                        if (!std::isnan(rs[t].h2) && (dense || (t - kBurn) % 50 == 0)) { h += rs[t].h2; ++nh; }
                    }
                    const double n = double(w.b - w.a);
                    std::fprintf(out, "%s,%s,%s,%s,%u,%s,%.8f,%.6f,%.6f\n", fam_.c_str(), sc_.c_str(), s.name().c_str(),
                                 names[i].c_str(), seed, w.name, std::sqrt(a / n), 1000 * b / n, nh ? 1000 * h / nh : NAN);
                }
            }
            r0 += s.size();
        }
    }

private:
    struct Tick { double se, ex, h2; };
    std::string fam_, sc_;
    copula::Family F_;
    H2Table h2_;
    std::vector<std::vector<Tick>> rows_;
};

}  // namespace thesis
