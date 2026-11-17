#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>

#include "estimator.hpp"
#include "memory_control.hpp"

// Online Hermite-series estimator of Spearman's rho (thesis 4.2): C++ port of Stephanou & Varughese (2021),
// "Sequential estimation of Spearman rank correlation using Hermite series estimators" (R package hermiter).
// The Hermite-function, integral and full-domain-integral recurrences match that package; the Spearman coupling
// matrix W is built in closed form (integral_matrix).
//
// Raw inputs are standardised with running moments of X_1..X_n (EW with the same weights as the coefficients,
// or Welford when cumulative); normal_score_inputs = true takes z = probit(u) as already standardised.
// Weight on the m-th absorbed observation: lambda / W_m, W_m = (1 - lambda) W_{m-1} + lambda (the thesis 5.1.2
// EW weights normalised to sum to one); lambda = 0 is the running mean 1/m. lambda = 2/(N_eff + 1).
class HermiteEstimator : public DependenceEstimator, public MemoryControlled {
public:
    struct Config {
        int order = 20;
        double lambda = 0.0;
        bool normal_score_inputs = false;
    };

    HermiteEstimator(Config config, const char* name)
        : config_(config), name_(name),
          ax_(Eigen::VectorXd::Zero(config.order + 1)), ay_(Eigen::VectorXd::Zero(config.order + 1)),
          A_(Eigen::MatrixXd::Zero(config.order + 1, config.order + 1)),
          W_(integral_matrix(config.order)), z_(basis_integrals(config.order)) {}

    void update(double x, double y) override {
        ++n_;
        double sx = x, sy = y;
        if (!config_.normal_score_inputs) {
            moments_step(x, y);
            if (moments_n_ < 2) return;   // a scale needs two observations
            sx = standardize(x, mean_x_, var_x_);
            sy = standardize(y, mean_y_, var_y_);
        }
        ++absorbed_;
        double w;
        if (config_.lambda <= 0.0) {
            w = 1.0 / static_cast<double>(absorbed_);
        } else {
            wcoef_ = (1.0 - lambda_) * wcoef_ + lambda_;
            w = lambda_ / wcoef_;
        }
        const Eigen::VectorXd hx = hermite_functions(sx), hy = hermite_functions(sy);
        ax_ += w * (hx - ax_);
        ay_ += w * (hy - ay_);
        A_ += w * (hx * hy.transpose() - A_);
    }

    // rho = 12 E[(F_X - 1/2)(F_Y - 1/2)] in the Hermite basis: with u = W a_x, v = W a_y, z = int h,
    // rho = 12 (u - z/2)' A (v - z/2).
    double estimate() const override {
        if (absorbed_ == 0) return 0.0;
        const Eigen::VectorXd u = W_ * ax_, v = W_ * ay_;
        const Eigen::VectorXd Av = A_ * v, Az = A_ * z_;
        const double rho = 12.0 * u.dot(Av) - 6.0 * u.dot(Az) - 6.0 * z_.dot(Av) + 3.0 * z_.dot(Az);
        return std::clamp(rho, -1.0, 1.0);
    }

    void reset() override {
        n_ = absorbed_ = moments_n_ = 0;
        mean_x_ = mean_y_ = 0.0;
        var_x_ = var_y_ = 1.0;
        m2_x_ = m2_y_ = 0.0;
        wmom_ = wcoef_ = 0.0;
        lambda_ = config_.lambda;
        ax_.setZero(); ay_.setZero(); A_.setZero();
    }

    // The object plus the coefficients a_x, a_y, A and the fixed W, z that estimate() needs.
    std::size_t store_bytes() const override {
        return sizeof(*this) + (ax_.size() + ay_.size() + A_.size() + W_.size() + z_.size()) * sizeof(double);
    }
    const char* name() const override { return name_; }
    bool ready() const override { return absorbed_ > 0; }
    void set_neff(double neff) override { lambda_ = 2.0 / (neff + 1.0); }

private:
    // EW mean and variance with the coefficients' weights (Welford when cumulative).
    void moments_step(double x, double y) {
        ++moments_n_;
        double c;
        if (config_.lambda > 0.0) {
            wmom_ = (1.0 - lambda_) * wmom_ + lambda_;
            c = lambda_ / wmom_;
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

    Eigen::VectorXd hermite_functions(double x) const {
        const int dim = config_.order + 1;
        Eigen::VectorXd h(dim);
        h(0) = kQuarterRootPiInv * std::exp(-0.5 * x * x);
        if (dim > 1) h(1) = std::sqrt(2.0) * x * h(0);
        for (int k = 2; k < dim; ++k) h(k) = x * std::sqrt(2.0 / k) * h(k - 1) - std::sqrt((k - 1.0) / k) * h(k - 2);
        return h;
    }

    static double standardize(double value, double mean, double var) {
        return (value - mean) / std::sqrt(std::max(var, 1e-12));
    }

    static Eigen::VectorXd basis_integrals(int order) {
        const int dim = order + 1;
        Eigen::VectorXd z = Eigen::VectorXd::Zero(dim);
        z(0) = std::sqrt(2.0) / kQuarterRootPiInv;   // sqrt(2) pi^(1/4)
        for (int k = 2; k < dim; k += 2) z(k) = std::sqrt((k - 1.0) / k) * z(k - 2);
        return z;
    }

    // W(k,j) = int h_k(x) I_j(x) dx in closed form, from W(k,j) + W(j,k) = z_k z_j and
    // I_j = -sqrt(2/j) h_{j-1} + sqrt((j-1)/j) I_{j-2}: W(k,j) = -sqrt(2/j)[k == j-1] + sqrt((j-1)/j) W(k,j-2).
    static Eigen::MatrixXd integral_matrix(int order) {
        const int dim = order + 1;
        const Eigen::VectorXd z = basis_integrals(order);
        const double sqrt2 = std::sqrt(2.0);
        Eigen::VectorXd row0(dim);
        row0(0) = z(0) * z(0) / 2.0;
        if (dim > 1) row0(1) = -sqrt2;
        for (int j = 2; j < dim; ++j) row0(j) = std::sqrt((j - 1.0) / j) * row0(j - 2);
        Eigen::MatrixXd W(dim, dim);
        for (int k = 0; k < dim; ++k) {
            W(k, 0) = z(k) * z(0) - row0(k);
            if (dim > 1) W(k, 1) = (k == 0) ? -sqrt2 : 0.0;
        }
        for (int j = 2; j < dim; ++j) {
            const double aj = std::sqrt(2.0 / j), bj = std::sqrt((j - 1.0) / j);
            for (int k = 0; k < dim; ++k) W(k, j) = bj * W(k, j - 2) - (k == j - 1 ? aj : 0.0);
        }
        return W;
    }

    inline static const double kQuarterRootPiInv = 1.0 / std::sqrt(std::sqrt(std::numbers::pi));

    Config config_;
    const char* name_;
    std::size_t n_ = 0, absorbed_ = 0, moments_n_ = 0;
    double mean_x_ = 0.0, mean_y_ = 0.0, var_x_ = 1.0, var_y_ = 1.0, m2_x_ = 0.0, m2_y_ = 0.0;
    double lambda_ = config_.lambda;
    double wmom_ = 0.0, wcoef_ = 0.0;   // accumulated EW weight of the moments and of the coefficients
    Eigen::VectorXd ax_, ay_;
    Eigen::MatrixXd A_, W_;
    Eigen::VectorXd z_;
};
