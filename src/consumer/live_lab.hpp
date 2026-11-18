#pragma once

// Run a viewer-configured MFF over working copulas and memory lengths, with VFF alongside it.
// Score each candidate on shared pseudo-observations before updating the estimators.

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "../estimators/dependence/datasketches_kll.hpp"
#include "../estimators/evaluation.hpp"

class LiveLab {
public:
    LiveLab(const std::string& est, const std::vector<double>& memories, const std::vector<std::string>& families,
            double eta, double delta)
        : est_(est), raw_(est != "rmpl"), mem_(memories), fam_names_(families), eta_(eta), delta_(delta),
          vff_(est, thesis::vff_rates()[0]), margins_(thesis::kKll, thesis::kClip) {
        for (double n : mem_) arms_.push_back(thesis::arm(est, n, "m"));
        for (const auto& f : fam_names_) fam_.push_back(copula::family_of(f));
        const std::size_t M = fam_.size() * mem_.size();
        w_.assign(M, 1.0 / double(M));
        loss_.resize(M);
        log_weights_.resize(M);
    }

    // Forecasts held before the next observation.
    double mff() const {
        double r = 0;
        for (std::size_t f = 0; f < fam_.size(); ++f)
            for (std::size_t k = 0; k < mem_.size(); ++k) r += w_[f * mem_.size() + k] * arms_[k]->estimate();
        return std::clamp(r, -1.0, 1.0);
    }
    double vff() const { return vff_.inner->estimate(); }
    double vff_neff() const { return vff_.ctl.neff(); }

    void update(double x, double y) {
        const auto [u, v] = margins_.update(x, y);
        score(u, v);
        const double zx = special::probit(u), zy = special::probit(v);
        for (auto& a : arms_) { if (raw_) a->update(x, y); else a->update(zx, zy); }
        vff_.update(x, y, u, v, t_ + 1, raw_);
        ++t_;
    }

    nlohmann::json config_json() const {
        std::vector<std::string> models;
        for (const auto& f : fam_names_) for (double n : mem_) models.push_back(f + " " + std::to_string(int(n)));
        return {{"type", "config"}, {"estimator", est_}, {"memories", mem_}, {"families", fam_names_}, {"models", models},
                {"eta", eta_}, {"delta", delta_}};
    }
    nlohmann::json state_json(double rho_true, const std::string& family_true) const {
        std::vector<double> arm_rho;
        for (const auto& a : arms_) arm_rho.push_back(a->estimate());
        double nbar = 0;
        std::vector<double> w_mem(mem_.size(), 0.0), w_fam(fam_.size(), 0.0);
        for (std::size_t f = 0; f < fam_.size(); ++f)
            for (std::size_t k = 0; k < mem_.size(); ++k) {
                const double w = w_[f * mem_.size() + k];
                nbar += w * mem_[k]; w_mem[k] += w; w_fam[f] += w;
            }
        return {{"type", "state"}, {"t", t_}, {"rho", rho_true}, {"family", family_true}, {"mff", mff()}, {"vff", vff()},
                {"vff_neff", vff_neff()}, {"mff_neff", nbar}, {"arms", arm_rho}, {"w", w_}, {"w_mem", w_mem}, {"w_fam", w_fam}};
    }

private:
    void score(double u, double v) {
        bool ok = t_ >= 30;
        for (const auto& a : arms_) ok = ok && a->ready();
        if (!ok) return;
        const std::size_t K = mem_.size();
        for (std::size_t f = 0; f < fam_.size(); ++f) {
            const auto family = fam_[f];
            for (std::size_t k = 0; k < K; ++k) {
                const double rho = arms_[k]->estimate();
                // Preserve the lab's forecast range for non-Gaussian working copulas.
                const double target = family == copula::Family::Gaussian ? rho : std::clamp(rho, -0.98, 0.98);
                const double parameter = copula::param_from_spearman(family, target);
                loss_[f * K + k] = -copula::log_density(family, u, v, parameter);
            }
        }
        thesis::update_mff_weights(w_, loss_, log_weights_, eta_, delta_);
    }

    std::string est_;
    bool raw_;
    std::vector<double> mem_;
    std::vector<std::string> fam_names_;
    std::vector<copula::Family> fam_;
    double eta_, delta_;
    std::vector<std::unique_ptr<DependenceEstimator>> arms_;
    thesis::Vff vff_;
    datasketches_kll::PairedKll margins_;
    std::vector<double> w_, loss_, log_weights_;
    std::size_t t_ = 0;
};
