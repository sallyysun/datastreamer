#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>

#include "estimator.hpp"
#include "memory_control.hpp"
#include "marginal_rank_pit.hpp"
#include "predictive_density.hpp"

// Online Hermite-series estimator of Spearman's rho.
//
// C++ port of the method of Stephanou & Varughese (2021), "Sequential estimation
// of Spearman rank correlation using Hermite series estimators"; reference
// implementation (R/Rcpp): external/hermiter. The orthonormal Hermite-function,
// integral, full-domain-integral, and online-standardization recurrences match
// that package exactly.
//
// Scope: this port covers only the online bivariate Spearman's-rho path. The
// reference package additionally does univariate pdf/cdf/quantile estimation,
// Kendall's tau, consistent merging of sub-estimators, and batch mode -- none
// ported (see archive/notes/code_changes). The Spearman coupling matrix W is built in
// closed form (see integral_matrix), matching the reference; there is no
// remaining quadrature/approximation step.
class HermiteEstimator : public DependenceEstimator, public PredictiveDensity, public MemoryControlled {
public:
    struct Config {
        int order = 20;
        double lambda = 0.0;
        std::size_t burn_in = 0;   // >0: report 0 until this many observations (output hold only)
        // Variable forgetting on the update weight. lambda here is the UPDATE
        // WEIGHT, so shortening memory means raising it — the mirror image of the
        // MLE/Xiao retention convention. See the registry.hpp header.
        double vff_rate = 0.0;
        // Divide the three EWMA accumulators by the weight actually accumulated,
        // 1 - (1-w)^n. Without it a long-memory Hermite is still shrunk toward
        // zero at t ~ ESS, which reads as a memory effect and is not one. Off by
        // default so no registered number moves. See archive/notes/studies/adaptive_memory.md.
        bool debias = false;
        // Coefficient weight on the m-th absorbed observation (EW mode):
        //   Normalised  : w_m = lambda / W_m, W_m = (1 - lambda) W_{m-1} + lambda, W_0 = 0. The thesis
        //                 5.1.2 weights (1 - lambda_thesis) lambda_thesis^{m-i} divided by their sum, so
        //                 every observation, the first included, carries its EW share. w_1 = 1 gives
        //                 a_1 = h(X_1) as in 5.1.2.
        //   Literal     : w_1 = 1, then lambda: 5.1.2 verbatim. The first observation keeps weight
        //                 (1 - lambda)^{m-1}, i.e. counts as 1/lambda observations.
        //   RunningMean : max(lambda, 1/m), the pre-2026-11-09 rule.
        // Cumulative mode (lambda = 0) is the running mean 1/m. archive/notes/studies/ew_marginals.md §3.
        enum class Start { Normalised, Literal, RunningMean };
        Start start = Start::Normalised;
        // Inputs are already normal scores z = probit(u) of shared pseudo-observations (the bank
        // protocol): no standardisation, every observation is absorbed from the first.
        bool normal_score_inputs = false;
        // Raw inputs: standardise X_n with the moments of X_1..X_{n-1} (predictive) instead of
        // X_1..X_n (the hermiter convention, the default). Not a scoring leak either way: the bank
        // scores before update(). Moments use the same Start weights as the coefficients.
        bool predictive_moments = false;
        // Reproduce the pre-2026-11-09 update exactly (moments updated before standardising, first
        // observation not absorbed, start-up max(lambda, 1/m), cumulative weight 1/n). Regression only.
        bool legacy = false;

        // Marginals by rank-PIT + probit instead of Welford standardization.
        // Standardization is exact only for Gaussian margins; ranks are exact for
        // any continuous margin and are invariant to monotone marginal change.
        std::size_t rank_pit_window = 0;   // 0 keeps standardization

        // Rate at which the surprise BASELINE (its mean and sd) is updated. This
        // must not be the coefficient weight: the baseline would then chase a
        // change at exactly the speed the coefficients absorb it, and the
        // standardized surprise would return to zero as fast as it rose. That is
        // the bug section 14.11 records.
        //   0 (default) : freeze after the calibration window -- a Phase-I
        //                 control chart, and the fix.
        //   > 0         : an EWMA at its own, slower rate.
        double surprise_baseline_rate = 0.0;

        // Which surprise statistic the detector monitors.
        //   0 : ||h_x h_y' - A||, the one-observation residual. Its norm is
        //       dominated by ||h_x|| ||h_y||, i.e. by where in the margins this
        //       single point landed, so the dependence signal is a small part of
        //       a large number. Section 14.11 measures how little power it has.
        //   > 0 : ||A_fast - A_slow||, this rate against the coefficient rate.
        //       Two smoothed copies of the same matrix, so the per-observation
        //       noise cancels and only a persistent drift survives. Costs one
        //       extra (order+1)^2 matrix.
        double surprise_fast_rate = 0.0;

        // What an alarm should do. The Hermite analogue of covariance inflation:
        // forget faster for a short burst instead of discarding A_. Section 14.11.
        // log_predictive: the series density is clipped at 0 and mixed with the independence
        // copula at this weight, so a truncated series that dips negative still scores finitely.
        double predictive_floor = 0.01;

        double reset_boost = -1.0;         // <0 hard reset; 0 no-op; >1 boost w
        std::size_t reset_boost_ticks = 20;
    };

    HermiteEstimator(Config config, const char* name)
        : config_(config),
        name_(name),
        ax_(Eigen::VectorXd::Zero(config.order + 1)),
        ay_(Eigen::VectorXd::Zero(config.order + 1)),
        A_(Eigen::MatrixXd::Zero(config.order + 1, config.order + 1)),
        A_fast_(config.surprise_fast_rate > 0.0 ? Eigen::MatrixXd::Zero(config.order + 1, config.order + 1)
                                                : Eigen::MatrixXd()),
        W_(integral_matrix(config.order)),
        z_(basis_integrals(config.order)) {}
    
    void update(double x, double y) override {
        if (!config_.legacy) { update_current(x, y); return; }
        ++n_;
        update_moments(x, y);
        if (n_ < 2) return; // Need at least 2 samples to compute covariance

        double sx, sy;
        if (config_.rank_pit_window > 0) {
            sx = RankPit::probit(pit_x_.transform(x));
            sy = RankPit::probit(pit_y_.transform(y));
            if (pit_x_.size() < 30) return;   // no usable rank scale yet
        } else {
            sx = standardize(x, mean_x_, var_x_);
            sy = standardize(y, mean_y_, var_y_);
        }
        const Eigen::VectorXd hx = hermite_functions(sx);
        const Eigen::VectorXd hy = hermite_functions(sy);

        ++absorbed_;
        // Cumulative mode: 1/absorbed (the mean of the absorbed terms). 1/n_ divided n-1 terms by n.
        double w = (config_.lambda > 0.0)
            ? lambda_
            : 1.0 / static_cast<double>(absorbed_);
        if (config_.lambda > 0.0)
            w = std::max(w, 1.0 / static_cast<double>(absorbed_));
        absorb(hx, hy, w);
    }

private:
    // Raw inputs are standardised with EW moments (predictive_moments: before this observation); a
    // scale needs two observations, so coefficients start once the moments hold two.
    void update_current(double x, double y) {
        ++n_;
        double sx = x, sy = y;
        if (config_.rank_pit_window > 0) {
            sx = RankPit::probit(pit_x_.transform(x));
            sy = RankPit::probit(pit_y_.transform(y));
            if (pit_x_.size() < 30) return;
        } else if (!config_.normal_score_inputs) {
            const bool before = config_.predictive_moments;
            if (!before) moments_step(x, y);
            const bool ready = moments_n_ >= 2;
            if (ready) {
                sx = standardize(x, mean_x_, var_x_);
                sy = standardize(y, mean_y_, var_y_);
            }
            if (before) moments_step(x, y);
            if (!ready) return;
        }
        ++absorbed_;
        double w;
        if (config_.lambda <= 0.0) {
            w = 1.0 / static_cast<double>(absorbed_);
        } else {
            switch (config_.start) {
                case Config::Start::Normalised:
                    wcoef_ = (1.0 - lambda_) * wcoef_ + lambda_;
                    w = lambda_ / wcoef_;
                    break;
                case Config::Start::Literal:
                    w = absorbed_ == 1 ? 1.0 : lambda_;
                    break;
                default:
                    w = std::max(lambda_, 1.0 / static_cast<double>(absorbed_));
            }
        }
        absorb(hermite_functions(sx), hermite_functions(sy), w);
    }

    // EW mean and variance with the coefficients' Start weights on the fixed lambda (Welford when lambda = 0).
    void moments_step(double x, double y) {
        ++moments_n_;
        double c;
        if (config_.lambda > 0.0) {
            const double l = lambda_;
            if (config_.start == Config::Start::Normalised) {
                wmom_ = (1.0 - l) * wmom_ + l;
                c = l / wmom_;
            } else if (config_.start == Config::Start::Literal) {
                c = moments_n_ == 1 ? 1.0 : l;
            } else {
                c = std::max(l, 1.0 / static_cast<double>(moments_n_));
            }
        } else {
            c = 1.0 / static_cast<double>(moments_n_);
        }
        const double dx = x - mean_x_, dy = y - mean_y_;
        mean_x_ += c * dx;
        mean_y_ += c * dy;
        if (moments_n_ == 1) { var_x_ = var_y_ = 0.0; return; }
        if (config_.lambda > 0.0) {
            var_x_ = (1.0 - c) * (var_x_ + c * dx * dx);
            var_y_ = (1.0 - c) * (var_y_ + c * dy * dy);
        } else {
            m2_x_ += dx * (x - mean_x_);
            m2_y_ += dy * (y - mean_y_);
            var_x_ = m2_x_ / static_cast<double>(moments_n_ - 1);
            var_y_ = m2_y_ / static_cast<double>(moments_n_ - 1);
        }
    }

    void absorb(const Eigen::VectorXd& hx, const Eigen::VectorXd& hy, double w) {
        if (boost_left_ > 0) {
            w = std::min(1.0, w * config_.reset_boost);
            --boost_left_;
        }
        
            wsum_ += w * (1.0 - wsum_);
            ax_ += w * (hx - ax_);
            ay_ += w * (hy - ay_);

            // The coefficient-matrix surprise, before it is absorbed. This is the
            // Hermite analogue of a Kalman innovation, and it carries shape change
            // that estimate()'s collapse to a scalar rho throws away.
            const Eigen::MatrixXd outer = hx * hy.transpose();
            double surprise = (outer - A_).norm();
            A_ += w * (outer - A_);
            if (config_.surprise_fast_rate > 0.0) {
                A_fast_ += config_.surprise_fast_rate * (outer - A_fast_);
                surprise = (A_fast_ - A_).norm();
            }

            // Baseline on its own clock. Inside the calibration window it is a
            // running mean/variance; after it, either frozen or a slow EWMA --
            // never the coefficient weight w.
            if (surprise_n_ < kSurpriseBurnIn) {
                const double c = 1.0 / static_cast<double>(surprise_n_ + 1);
                const double d = surprise - surprise_mean_;
                surprise_mean_ += c * d;
                surprise_var_ = (1.0 - c) * (surprise_var_ + c * d * d);
            } else if (config_.surprise_baseline_rate > 0.0) {
                const double c = config_.surprise_baseline_rate;
                const double d = surprise - surprise_mean_;
                surprise_mean_ += c * d;
                surprise_var_ = (1.0 - c) * (surprise_var_ + c * d * d);
            }
            ++surprise_n_;
            last_surprise_ = surprise;

            if (config_.vff_rate > 0.0 && config_.lambda > 0.0) {
                fast_ += 0.05 * (surprise - fast_);
                slow_ += 0.002 * (surprise - slow_);
                const double ratio = fast_ / std::max(slow_, 1e-12);
                lambda_ = std::clamp(config_.lambda * std::pow(ratio, config_.vff_rate),
                                     config_.lambda * 0.2, config_.lambda * 20.0);
            }
    }

public:
    // Standardized coefficient-matrix surprise, for a detector to monitor. Zero
    // until enough history exists to have a scale for it.
    double standardized_surprise() const {
        if (surprise_n_ <= kSurpriseBurnIn) return 0.0;
        const double sd = std::sqrt(std::max(surprise_var_, 1e-12));
        return (last_surprise_ - surprise_mean_) / sd;
    }

    const Eigen::MatrixXd& coefficients() const { return A_; }

    // Series density f(s_x, s_y) = h(s_x)' A h(s_y) of the standardised inputs, carried to the
    // unit square: inputs arrive as z = probit(u), s = (z - mean)/sd, so the density of (u, v) is
    // f(s_x, s_y) / (sd_x sd_y phi(z_x) phi(z_y)). Clipped and floored (predictive_floor).
    double log_predictive(double u, double v) const override {
        constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
        if (config_.rank_pit_window > 0) return kNaN;
        if (!(u > 0.0 && u < 1.0 && v > 0.0 && v < 1.0)) return kNaN;
        if (absorbed_ == 0 || n_ < config_.burn_in + 1) return 0.0;
        const double zx = RankPit::probit(u), zy = RankPit::probit(v);
        const double sdx = std::sqrt(std::max(var_x_, 1e-12)), sdy = std::sqrt(std::max(var_y_, 1e-12));
        const double c = (config_.debias && wsum_ > 0.0) ? 1.0 / wsum_ : 1.0;
        const double f = c * hermite_functions((zx - mean_x_) / sdx).dot(A_ * hermite_functions((zy - mean_y_) / sdy));
        constexpr double kLogTwoPi = 1.8378770664093453;
        const double log_phi = -kLogTwoPi - 0.5 * (zx * zx + zy * zy);
        const double dens = std::max(f, 0.0) * std::exp(-log_phi) / (sdx * sdy);
        const double e = config_.predictive_floor;
        return std::log((1.0 - e) * dens + e);
    }

    double estimate() const override {
        if (absorbed_ == 0 || n_ < config_.burn_in + 1) return 0.0;
        // Spearman's rho = 12 * E[(F_X - 1/2)(F_Y - 1/2)], expanded in the
        // Hermite basis. u = int F_X h dx, v = int F_Y h dy, z = int h dx, so
        // (u - z/2) and (v - z/2) are the centred marginal CDF projections and
        // rho = 12 (u - z/2)^T A (v - z/2).
        // rho is not homogeneous in the accumulators (its four terms are degree
        // 3, 2, 2 and 1 in them), so a common shrinkage does not cancel here the
        // way it does inside Xiao's ratio.
        const double c = (config_.debias && wsum_ > 0.0) ? 1.0 / wsum_ : 1.0;
        const Eigen::VectorXd u = (W_ * ax_) * c;
        const Eigen::VectorXd v = (W_ * ay_) * c;
        const Eigen::VectorXd Av = (A_ * v) * c;
        const Eigen::VectorXd Az = (A_ * z_) * c;
        const double rho = 12.0 * u.dot(Av) - 6.0 * u.dot(Az)
                         - 6.0 * z_.dot(Av) + 3.0 * z_.dot(Az);
        return std::clamp(rho, -1.0, 1.0);
    }

    void reset() override {
        if (config_.reset_boost == 0.0) return;                    // no-op control
        if (config_.reset_boost > 0.0) {                            // soft: forget faster
            boost_left_ = config_.reset_boost_ticks;                // A_ and the baseline kept
            return;
        }
        n_ = 0;
        boost_left_ = 0;
        mean_x_ = mean_y_ = 0.0;
        var_x_ = var_y_ = 1.0;
        m2_x_ = m2_y_ = 0.0;
        ax_.setZero();
        ay_.setZero();
        A_.setZero();
        A_fast_.setZero();
        last_surprise_ = 0.0;
        surprise_mean_ = 0.0;
        surprise_var_ = 0.0;
        surprise_n_ = 0;
        lambda_ = config_.lambda;
        wsum_ = 0.0;
        absorbed_ = 0;
        moments_n_ = 0;
        wmom_ = wcoef_ = 0.0;
        fast_ = slow_ = 0.0;
        pit_x_.reset();
        pit_y_.reset();
    }

    double kendall() const {
        if (n_ < config_.burn_in) return 0.0;
        const double tau = 4.0 * (W_.transpose() * A_.transpose() * W_).trace() - 1.0;
        return std::clamp(tau, -1.0, 1.0);
    }

    // Everything the object holds: itself, the coefficients a_x, a_y, A (and A_fast), and the fixed
    // coupling matrix W and integrals z that estimate() needs, plus the rank windows' heap.
    std::size_t store_bytes() const override {
        const auto n = [](const auto& m) { return static_cast<std::size_t>(m.size()); };
        return sizeof(*this)
             + (n(ax_) + n(ay_) + n(z_) + n(A_) + n(A_fast_) + n(W_)) * sizeof(double)
             + pit_x_.heap_bytes() + pit_y_.heap_bytes();
    }

    const char* name() const override {
        return name_;
    }
    bool ready() const override { return absorbed_ > 0 && n_ >= config_.burn_in + 1; }
    // Update weight 2/(N + 1) for the coefficients and the standardisation moments; needs EW mode.
    void set_neff(double neff) override { lambda_ = 2.0 / (neff + 1.0); }

private:
    Eigen::VectorXd hermite_functions(double x) const {
        const int dim = config_.order + 1;
        Eigen::VectorXd h(dim);
        const double gaussian = std::exp(-0.5 * x * x);
        h(0) = kQuarterRootPiInv * gaussian;
        if (dim > 1) h(1) = std::sqrt(2.0) * x * h(0);
        for (int k = 2; k < dim; ++k) {
            h(k) = x * std::sqrt(2.0 / k) * h(k - 1) - std::sqrt((k - 1.0) / k) * h(k - 2);
        }
        return h;
    }

    static double standardize(double value, double mean, double var) {
        const double sd = std::sqrt(std::max(var, 1e-12)); // Avoid division by zero
        return (value - mean) / sd;
    }

    void update_moments(double x, double y) {
        if (config_.lambda > 0.0) {
            if (n_ == 1) {
                mean_x_ = x;
                mean_y_ = y;
                var_x_ = 1.0;
                var_y_ = 1.0;
                return;
            }
            const double l = config_.lambda;
            mean_x_ = (1.0 - l) * mean_x_ + l * x;
            mean_y_ = (1.0 - l) * mean_y_ + l * y;
            var_x_ = (1.0 - l) * var_x_ + l * (x - mean_x_) * (x - mean_x_);
            var_y_ = (1.0 - l) * var_y_ + l * (y - mean_y_) * (y - mean_y_);
        } else {
            const double n = static_cast<double>(n_);
            const double delta_x = x - mean_x_;
            const double delta_y = y - mean_y_;
            mean_x_ += delta_x / n;
            mean_y_ += delta_y / n;
            m2_x_ += delta_x * (x - mean_x_);
            m2_y_ += delta_y * (y - mean_y_);
            if (n_ > 1) {
                var_x_ = m2_x_ / (n - 1.0);
                var_y_ = m2_y_ / (n - 1.0);
            }
        }
    }

    static Eigen::VectorXd basis_integrals(int order) {
        const int dim = order + 1;
        Eigen::VectorXd z = Eigen::VectorXd::Zero(dim);
        z(0) = std::sqrt(2.0) / kQuarterRootPiInv; // = sqrt(2) * pi^(1/4)
        for (int k = 2; k < dim; k += 2) {
            z(k) = std::sqrt((k-1.0) / k) * z(k - 2);
        }
        return z;
    }

    // Spearman coupling matrix W(k,j) = int h_k(x) I_j(x) dx, computed in CLOSED
    // FORM (no quadrature). Two exact facts:
    //   * integration by parts: W(k,j) + W(j,k) = z_k z_j   (z = full-domain integrals)
    //   * I_j = -sqrt(2/j) h_{j-1} + sqrt((j-1)/j) I_{j-2}, with int h_k h_{j-1} = delta,
    //     giving the column recurrence W(k,j) = -sqrt(2/j)[k==j-1] + sqrt((j-1)/j) W(k,j-2).
    // Bases: W(0,0)=z0^2/2, W(0,1)=-sqrt2, W(k,1)=-sqrt2 [k==0], W(k,0)=z_k z_0 - W(0,k).
    static Eigen::MatrixXd integral_matrix(int order) {
        const int dim = order + 1;
        const Eigen::VectorXd z = basis_integrals(order);
        const double sqrt2 = std::sqrt(2.0);

        // Row 0 across all columns: W(0,j).
        Eigen::VectorXd row0(dim);
        row0(0) = z(0) * z(0) / 2.0;
        if (dim > 1) row0(1) = -sqrt2;
        for (int j = 2; j < dim; ++j) {
            row0(j) = std::sqrt((j - 1.0) / j) * row0(j - 2);
        }

        Eigen::MatrixXd W(dim, dim);
        // Column bases j = 0 (IBP symmetry) and j = 1 (orthonormality).
        for (int k = 0; k < dim; ++k) {
            W(k, 0) = z(k) * z(0) - row0(k);
            if (dim > 1) W(k, 1) = (k == 0) ? -sqrt2 : 0.0;
        }
        // Columns j >= 2 via the recurrence.
        for (int j = 2; j < dim; ++j) {
            const double aj = std::sqrt(2.0 / j);
            const double bj = std::sqrt((j - 1.0) / j);
            for (int k = 0; k < dim; ++k) {
                W(k, j) = bj * W(k, j - 2) - (k == j - 1 ? aj : 0.0);
            }
        }
        return W;
    }

    // 1 / pi^(1/4): ground-state Hermite-function normaliser. std::sqrt is not
    // constexpr until C++23, so this is computed once at static-init (inline const).
    inline static const double kQuarterRootPiInv =
        1.0 / std::sqrt(std::sqrt(std::numbers::pi));

    Config config_;
    const char* name_;
    std::size_t n_ = 0;

    double mean_x_ = 0.0;
    double mean_y_ = 0.0;
    double var_x_ = 1.0;
    double var_y_ = 1.0;
    double m2_x_ = 0.0; // Welford accumulators (cumulative mode)
    double m2_y_ = 0.0;

    static constexpr std::size_t kSurpriseBurnIn = 200;

    RankPit pit_x_{config_.rank_pit_window ? config_.rank_pit_window : 1};
    RankPit pit_y_{config_.rank_pit_window ? config_.rank_pit_window : 1};
    double lambda_ = config_.lambda;
    double wsum_ = 0.0;
    std::size_t absorbed_ = 0;
    std::size_t moments_n_ = 0;      // observations in the standardisation moments
    double wmom_ = 0.0, wcoef_ = 0.0;   // accumulated EW weight of the moments and of the coefficients
    std::size_t boost_left_ = 0;
    double fast_ = 0.0;
    double slow_ = 0.0;

    double last_surprise_ = 0.0;
    double surprise_mean_ = 0.0;
    double surprise_var_ = 0.0;
    std::size_t surprise_n_ = 0;

    Eigen::VectorXd ax_;
    Eigen::VectorXd ay_;
    Eigen::MatrixXd A_;
    Eigen::MatrixXd A_fast_;
    Eigen::MatrixXd W_;
    Eigen::VectorXd z_;
};