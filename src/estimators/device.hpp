#pragma once

// The streaming device: every Chapter 6 strategy for each estimator on one stream, with shared KLL margins.
// Counting Matrix and Hermite use raw inputs; RMPL uses probit(U). Needs no ground truth (scorer.hpp scores it).

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "dependence/datasketches_kll.hpp"
#include "evaluation.hpp"
#include "../special_functions.hpp"

namespace thesis {

// One estimator's fixed-memory arms. Counting Matrix and Hermite take raw inputs; RMPL takes probit(U).
class FixedMemoryArms {
public:
    explicit FixedMemoryArms(const std::string& est) : raw_(est != "rmpl") {
        arms_.reserve(kK);
        for (double n : kMem) arms_.push_back(arm(est, n, "m"));
    }
    double estimate(std::size_t k) const { return arms_[k]->estimate(); }
    void forecasts(double* out) const {
        for (std::size_t k = 0; k < arms_.size(); ++k) out[k] = estimate(k);
    }
    bool ready() const {
        return std::all_of(arms_.begin(), arms_.end(), [](const auto& a) { return a->ready(); });
    }
    bool raw() const { return raw_; }
    void update(double x, double y, double u, double v) {
        if (!raw_) { x = special::probit(u); y = special::probit(v); }
        for (auto& a : arms_) a->update(x, y);
    }
    const std::vector<std::unique_ptr<DependenceEstimator>>& estimators() const { return arms_; }

private:
    bool raw_;
    std::vector<std::unique_ptr<DependenceEstimator>> arms_;
};

// One estimator's strategies: the fixed-memory arms, MFF over them, and VFF at each rate.
class EstimatorStrategies {
public:
    explicit EstimatorStrategies(const std::string& est, const std::vector<double>& rates = vff_rates())
        : est_(est), arms_(est) {
        vffs_.reserve(rates.size());
        for (double r : rates) vffs_.emplace_back(est, r);
    }
    std::size_t size() const { return kK + 1 + vffs_.size(); }
    void forecasts(std::vector<double>& f) const {
        f.resize(size());
        arms_.forecasts(f.data());
        f[kK] = mff_.estimate(arms_.estimators());
        for (std::size_t j = 0; j < vffs_.size(); ++j) f[kK + 1 + j] = vffs_[j].inner->estimate();
    }
    const std::vector<double>& mff_weights() const { return mff_.w; }
    bool mff_ready(std::size_t t) const { return mff_.ready(arms_.estimators(), t); }
    // t = observations already absorbed (the margins' size before this one).
    void update(double x, double y, double u, double v, std::size_t t) {
        mff_.score_and_update_weights(arms_.estimators(), u, v, t);
        arms_.update(x, y, u, v);
        for (auto& vf : vffs_) vf.update(x, y, u, v, t + 1, arms_.raw());
    }
    const std::string& name() const { return est_; }

private:
    std::string est_;
    FixedMemoryArms arms_;
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
