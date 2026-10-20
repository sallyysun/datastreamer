#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>

#include "estimator.hpp"
#include "marginal_rank_pit.hpp"

// Online Hermite-series Spearman's rho: port of Stephanou & Varughese (2021),
// reference external/hermiter. Closed-form W: notes/hermiter_change.md.
class HermiteEstimator : public DependenceEstimator {
public:
    struct Config {
        int order = 20;
        double lambda = 0.0;
        std::size_t burn_in = 100;
        // ⚠️ lambda is the UPDATE WEIGHT (mirror of Xiao/MLE retention): shorter memory = larger.
        double vff_rate = 0.0;
        // Divide accumulators by 1 - (1-w)^n; off by default so no registered
        // number moves. notes/studies/adaptive_memory.md.
        bool debias = false;

        std::size_t rank_pit_window = 0;   // 0 keeps standardization

        // Never the coefficient weight (the §14.11 bug). 0 freezes after calibration.
        double surprise_baseline_rate = 0.0;

        // 0: ||h_x h_y' - A||; >0: ||A_fast - A|| at this rate (one extra matrix). §14.11.
        double surprise_fast_rate = 0.0;

        double reset_boost = -1.0;         // <0 hard reset; 0 no-op; >1 boost w
        std::size_t reset_boost_ticks = 20;
    };

    HermiteEstimator(Config config, const char* name)
        : config_(config),
        name_(name),
        ax_(Eigen::VectorXd::Zero(config.order + 1)),
        ay_(Eigen::VectorXd::Zero(config.order + 1)),
        A_(Eigen::MatrixXd::Zero(config.order + 1, config.order + 1)),
        A_fast_(Eigen::MatrixXd::Zero(config.order + 1, config.order + 1)),
        W_(integral_matrix(config.order)),
        z_(basis_integrals(config.order)) {}
    
    void update(double x, double y) override {
        ++n_;
        update_moments(x, y);
        if (n_ < 2) return;

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

        double w = (config_.lambda > 0.0)
            ? lambda_
            : 1.0 / static_cast<double>(n_);
        if (boost_left_ > 0) {
            w = std::min(1.0, w * config_.reset_boost);
            --boost_left_;
        }
        
            wsum_ += w * (1.0 - wsum_);
            ax_ += w * (hx - ax_);
            ay_ += w * (hy - ay_);

            // Surprise is taken before the observation is absorbed.
            const Eigen::MatrixXd outer = hx * hy.transpose();
            double surprise = (outer - A_).norm();
            A_ += w * (outer - A_);
            if (config_.surprise_fast_rate > 0.0) {
                A_fast_ += config_.surprise_fast_rate * (outer - A_fast_);
                surprise = (A_fast_ - A_).norm();
            }

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

    double standardized_surprise() const {
        if (surprise_n_ <= kSurpriseBurnIn) return 0.0;
        const double sd = std::sqrt(std::max(surprise_var_, 1e-12));
        return (last_surprise_ - surprise_mean_) / sd;
    }

    const Eigen::MatrixXd& coefficients() const { return A_; }

    double estimate() const override {
        if (n_ < config_.burn_in + 1) return 0.0;
        // rho = 12 (u - z/2)^T A (v - z/2). Not homogeneous in the accumulators, so
        // shrinkage does not cancel as in Xiao's ratio; debias scales every factor.
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
        if (config_.reset_boost == 0.0) return;
        if (config_.reset_boost > 0.0) {                            // soft: A_ and baseline kept
            boost_left_ = config_.reset_boost_ticks;
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
        fast_ = slow_ = 0.0;
        pit_x_.reset();
        pit_y_.reset();
    }

    double kendall() const {
        if (n_ < config_.burn_in) return 0.0;
        const double c = (config_.debias && wsum_ > 0.0) ? 1.0 / wsum_ : 1.0;
        const double tau = 4.0 * c * c * (A_.transpose() * W_ * A_ * W_.transpose()).trace() - 1.0;
        return std::clamp(tau, -1.0, 1.0);
    }

    std::size_t store_bytes() const override {
        const std::size_t dim = static_cast<std::size_t>(config_.order + 1);
        return (2 * dim + dim * dim * (config_.surprise_fast_rate > 0.0 ? 2 : 1) + 6) * sizeof(double)
             + (config_.rank_pit_window > 0 ? pit_x_.store_bytes() + pit_y_.store_bytes() : 0);
    }

    const char* name() const override {
        return name_;
    }

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
        const double sd = std::sqrt(std::max(var, 1e-12));
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

    // W(k,j) = int h_k I_j dx in closed form; derivation in notes/hermiter_change.md.
    static Eigen::MatrixXd integral_matrix(int order) {
        const int dim = order + 1;
        const Eigen::VectorXd z = basis_integrals(order);
        const double sqrt2 = std::sqrt(2.0);

        Eigen::VectorXd row0(dim);
        row0(0) = z(0) * z(0) / 2.0;
        if (dim > 1) row0(1) = -sqrt2;
        for (int j = 2; j < dim; ++j) {
            row0(j) = std::sqrt((j - 1.0) / j) * row0(j - 2);
        }

        Eigen::MatrixXd W(dim, dim);
        for (int k = 0; k < dim; ++k) {
            W(k, 0) = z(k) * z(0) - row0(k);
            if (dim > 1) W(k, 1) = (k == 0) ? -sqrt2 : 0.0;
        }
        for (int j = 2; j < dim; ++j) {
            const double aj = std::sqrt(2.0 / j);
            const double bj = std::sqrt((j - 1.0) / j);
            for (int k = 0; k < dim; ++k) {
                W(k, j) = bj * W(k, j - 2) - (k == j - 1 ? aj : 0.0);
            }
        }
        return W;
    }

    // 1 / pi^(1/4); not constexpr because std::sqrt isn't until C++23.
    inline static const double kQuarterRootPiInv =
        1.0 / std::sqrt(std::sqrt(std::numbers::pi));

    Config config_;
    const char* name_;
    std::size_t n_ = 0;

    double mean_x_ = 0.0;
    double mean_y_ = 0.0;
    double var_x_ = 1.0;
    double var_y_ = 1.0;
    double m2_x_ = 0.0;
    double m2_y_ = 0.0;

    static constexpr std::size_t kSurpriseBurnIn = 200;

    RankPit pit_x_{config_.rank_pit_window ? config_.rank_pit_window : 1};
    RankPit pit_y_{config_.rank_pit_window ? config_.rank_pit_window : 1};
    double lambda_ = config_.lambda;
    double wsum_ = 0.0;
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