#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>

#include "estimator.hpp"

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
// ported (see notes/code_changes). The Spearman coupling matrix W is built in
// closed form (see integral_matrix), matching the reference; there is no
// remaining quadrature/approximation step.
class HermiteEstimator : public DependenceEstimator {
public:
    struct Config {
        int order = 20;
        double lambda = 0.0;
        std::size_t burn_in = 100; // Number of initial samples to discard
    };

    HermiteEstimator(Config config, const char* name)
        : config_(config),
        name_(name),
        ax_(Eigen::VectorXd::Zero(config.order + 1)),
        ay_(Eigen::VectorXd::Zero(config.order + 1)),
        A_(Eigen::MatrixXd::Zero(config.order + 1, config.order + 1)),
        W_(integral_matrix(config.order)),
        z_(basis_integrals(config.order)) {}
    
    void update(double x, double y) override {
        ++n_;
        update_moments(x, y);
        if (n_ < 2) return; // Need at least 2 samples to compute covariance

        const Eigen::VectorXd hx = hermite_functions(standardize(x, mean_x_, var_x_));
        const Eigen::VectorXd hy = hermite_functions(standardize(y, mean_y_, var_y_));

        const double w = (config_.lambda > 0.0) 
            ? config_.lambda
            : 1.0 / static_cast<double>(n_);
        
            ax_ += w * (hx - ax_);
            ay_ += w * (hy - ay_);
            A_ += w * (hx * hy.transpose() - A_);
    }

    double estimate() const override {
        if (n_ < config_.burn_in + 1) return 0.0; // Not enough data to estimate
        // Spearman's rho = 12 * E[(F_X - 1/2)(F_Y - 1/2)], expanded in the
        // Hermite basis. u = int F_X h dx, v = int F_Y h dy, z = int h dx, so
        // (u - z/2) and (v - z/2) are the centred marginal CDF projections and
        // rho = 12 (u - z/2)^T A (v - z/2).
        const Eigen::VectorXd u = W_ * ax_;
        const Eigen::VectorXd v = W_ * ay_;
        const Eigen::VectorXd Av = A_ * v;
        const Eigen::VectorXd Az = A_ * z_;
        const double rho = 12.0 * u.dot(Av) - 6.0 * u.dot(Az)
                         - 6.0 * z_.dot(Av) + 3.0 * z_.dot(Az);
        return std::clamp(rho, -1.0, 1.0);
    }

    void reset() override {
        n_ = 0;
        mean_x_ = mean_y_ = 0.0;
        var_x_ = var_y_ = 1.0;
        m2_x_ = m2_y_ = 0.0;
        ax_.setZero();
        ay_.setZero();
        A_.setZero();
    }

    double kendall() const {
        if (n_ < config_.burn_in) return 0.0;
        const double tau = 4.0 * (W_.transpose() * A_.transpose() * W_).trace() - 1.0;
        return std::clamp(tau, -1.0, 1.0);
    }

    std::size_t store_bytes() const override {
        const std::size_t dim = static_cast<std::size_t>(config_.order + 1);
        return (2 * dim + dim * dim + 6) * sizeof(double);
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

    Eigen::VectorXd ax_;
    Eigen::VectorXd ay_;
    Eigen::MatrixXd A_;
    Eigen::MatrixXd W_;
    Eigen::VectorXd z_;
};