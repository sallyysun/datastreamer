#pragma once

// The streaming device: every Chapter 6 strategy for each estimator on one stream, with shared KLL margins.
// Counting Matrix and Hermite use raw inputs; RMPL uses probit(U). Needs no ground truth (scorer.hpp scores it).

#include <cmath>
#include <string>
#include <vector>

#include "dependence/datasketches_kll.hpp"
#include "evaluation.hpp"
#include "../special_functions.hpp"

namespace thesis {

// One estimator's strategies: the fixed-memory arms, MFF over them, and VFF at each rate.
class EstimatorStrategies {
public:
    explicit EstimatorStrategies(const std::string& est, const std::vector<double>& rates = vff_rates())
        : est_(est), raw_(est != "rmpl") {
        for (double n : kMem) arms_.push_back(arm(est, n, "m"));
        vffs_.reserve(rates.size());
        for (double r : rates) vffs_.emplace_back(est, r);
    }
    std::size_t size() const { return kK + 1 + vffs_.size(); }
    void forecasts(std::vector<double>& f) const {
        f.resize(size());
        for (std::size_t k = 0; k < kK; ++k) f[k] = arms_[k]->estimate();
        f[kK] = mff_.estimate(arms_);
        for (std::size_t j = 0; j < vffs_.size(); ++j) f[kK + 1 + j] = vffs_[j].inner->estimate();
    }
    const std::vector<double>& mff_weights() const { return mff_.w; }
    bool mff_ready(std::size_t t) const { return mff_.ready(arms_, t); }
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

}  // namespace thesis
