#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>

#include "estimator.hpp"

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
        const Eigen::VectorXd u = W_ * ax_;
        const Eigen::VectorXd v = W_ * ay_;
        const Eigen::VectorXd Av = A_ * v;
        const Eigen::VectorXd Az = A_ * z_;
        const double rho = 12.0 * u.dot(Av) - 12.0 * u.dot(Av) - 6.0 * u.dot(Az) - 6.0 * z_.dot(Av) + 3.0 * z_.dot(Az);
        return std::clamp(val:rho, lo:-1.0, hi:1.0);
    }

    double kendall() const {
        if (n_ < config_.burn_in) return 0.0;
        const double tau = 4.0 * (W_transpose() * A_.transpose() * W_).trace() - 1.0;
        return std::clamp(val:tau, lo:-1.0, hi:1.0);
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