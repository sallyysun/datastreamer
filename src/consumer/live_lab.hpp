#pragma once

// The browser lab's device: a viewer-built MFF. Models = working families x memories; each memory is one arm of the
// chosen estimator (RMPL / Counting Matrix / Hermite), and each model scores that arm's rho_S as a copula of its working
// family at the shared pseudo-observations (U, V): loss -log c_W(theta_W(rho_k))(U, V), Hedge (eta) with fixed share
// (delta), forecast sum w rho_k. With the Gaussian family alone this is the thesis MFF (thesis::Mff); with several it is
// the family-augmented MFF of 6.5. VFF (rate 0.02) on the same estimator runs alongside. Margins, arms and update
// order are the Chapter 6 device's (thesis_device.hpp).

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "../estimators/dependence/ds_margin.hpp"
#include "../estimators/evaluation.hpp"

class LiveLab {
public:
    LiveLab(const std::string& est, const std::vector<double>& memories, const std::vector<std::string>& families,
            double eta, double delta)
        : est_(est), raw_(est != "rmpl"), mem_(memories), fam_names_(families), eta_(eta), delta_(delta),
          vff_(est, thesis::vff_rates()[0]), kx_(thesis::kKll), ky_(thesis::kKll) {
        for (double n : mem_) arms_.push_back(thesis::arm(est, n, "m"));
        for (const auto& f : fam_names_) { fam_.push_back(thesis::family_of(f)); tab_.push_back(Inverse(fam_.back())); }
        const std::size_t M = fam_.size() * mem_.size();
        w_.assign(M, 1.0 / double(M));
        loss_.assign(M, 0.0);
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
        const double u = std::clamp(kx_.cdf(x), thesis::kClip, 1 - thesis::kClip);
        const double v = std::clamp(ky_.cdf(y), thesis::kClip, 1 - thesis::kClip);
        kx_.insert(x); ky_.insert(y);
        score(u, v);
        const double zx = RankPit::probit(u), zy = RankPit::probit(v);
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
    // rho_S -> family parameter by linear interpolation on a fine grid (param_from_spearman per model per tick is too
    // slow for an interactive rate); Gaussian is closed form.
    struct Inverse {
        copula::Family F;
        std::vector<double> rho, par;
        explicit Inverse(copula::Family f) : F(f) {
            if (F == copula::Family::Gaussian) return;
            for (int i = 0; i <= 400; ++i) {
                const double r = -0.98 + 1.96 * i / 400.0;
                rho.push_back(r);
                par.push_back(copula::param_from_spearman(F, r));
            }
        }
        double operator()(double r) const {
            if (F == copula::Family::Gaussian) return copula::param_from_spearman(F, r);
            const double x = (std::clamp(r, rho.front(), rho.back()) - rho.front()) / (rho.back() - rho.front()) * 400.0;
            const std::size_t i = std::min<std::size_t>(399, std::size_t(x));
            return par[i] + (x - double(i)) * (par[i + 1] - par[i]);
        }
    };

    // thesis::Mff::score with working families: warm-up 30 margin observations and every arm ready; non-finite losses
    // replaced by the worst finite one; Hedge then fixed share.
    void score(double u, double v) {
        bool ok = t_ >= 30;
        for (const auto& a : arms_) ok = ok && a->ready();
        if (!ok) return;
        const std::size_t K = mem_.size(), M = w_.size();
        for (std::size_t f = 0; f < fam_.size(); ++f)
            for (std::size_t k = 0; k < K; ++k)
                loss_[f * K + k] = -copula::log_density(fam_[f], u, v, tab_[f](arms_[k]->estimate()));
        double worst = 0;
        for (double l : loss_) if (std::isfinite(l)) worst = std::max(worst, l);
        for (double& l : loss_) if (!std::isfinite(l)) l = worst;
        double top = -INFINITY;
        std::vector<double> lw(M);
        for (std::size_t m = 0; m < M; ++m) { lw[m] = (w_[m] > 0 ? std::log(w_[m]) : -INFINITY) - eta_ * loss_[m]; top = std::max(top, lw[m]); }
        double tot = 0;
        for (std::size_t m = 0; m < M; ++m) { w_[m] = std::exp(lw[m] - top); tot += w_[m]; }
        for (double& x : w_) x = (1 - delta_) * x / tot + delta_ / double(M);
    }

    std::string est_;
    bool raw_;
    std::vector<double> mem_;
    std::vector<std::string> fam_names_;
    std::vector<copula::Family> fam_;
    std::vector<Inverse> tab_;
    double eta_, delta_;
    std::vector<std::unique_ptr<DependenceEstimator>> arms_;
    thesis::Vff vff_;
    ds_margin::Kll kx_, ky_;
    std::vector<double> w_, loss_;
    std::size_t t_ = 0;
};
