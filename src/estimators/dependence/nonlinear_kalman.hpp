#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "estimator.hpp"

// EKF and UKF on the RAW block Spearman, with no Fisher transform.
//
// KalmanLocalLevel transforms the observation first (y = atanh(rho-hat)) so the
// model becomes linear with constant noise. That is what makes the exact filter
// available, and section 0.9 records what it costs. These two do the opposite:
// keep the state as z = atanh(rho) but observe rho-hat directly through
// h(z) = tanh(z), with the heteroscedastic variance the raw statistic actually
// has, Var ~ 1.06 (1 - rho^2)^2 / (m - 1).
//
// So the comparison against `kalman` is precisely: is the Fisher transform's
// variance stabilization worth more than handling the nonlinearity properly?
//
// See section 11 of notes/studies/kalman_changepoint_results.md.
class NonlinearKalman : public DependenceEstimator {
public:
    enum class Mode { Extended, Unscented };

    struct Config {
        std::size_t block = 20;
        double sigma_omega2 = 1e-3;
        Mode mode = Mode::Extended;
        std::size_t burn_in_blocks = 2;
        double alpha = 1e-3;      // UKF spread
        double beta = 2.0;        // UKF: 2 is optimal for a Gaussian prior
        double kappa = 0.0;
    };

    NonlinearKalman(Config config, const char* name)
        : config_(config), name_(name) {
        xs_.reserve(config_.block);
        ys_.reserve(config_.block);
    }

    void update(double x, double y) override {
        xs_.push_back(x);
        ys_.push_back(y);
        if (xs_.size() < config_.block) return;

        const double r_obs = block_spearman();
        const double m = static_cast<double>(config_.block);

        const double P_pred = P_ + config_.sigma_omega2;
        const double z_pred = z_;

        if (config_.mode == Mode::Extended) {
            const double h = std::tanh(z_pred);
            const double H = 1.0 - h * h;               // dh/dz = sech^2
            const double R = obs_var(h, m);
            const double S = H * H * P_pred + R;
            const double K = (S > 0.0) ? P_pred * H / S : 0.0;
            z_ = z_pred + K * (r_obs - h);
            P_ = (1.0 - K * H) * P_pred;
        } else {
            // Scalar UKF: three sigma points.
            const double L = 1.0;
            const double lam = config_.alpha * config_.alpha * (L + config_.kappa) - L;
            const double c = L + lam;
            const double sd = std::sqrt(std::max(c * P_pred, 0.0));
            const std::array<double, 3> sig = {z_pred, z_pred + sd, z_pred - sd};

            const double wm0 = lam / c;
            const double wc0 = wm0 + (1.0 - config_.alpha * config_.alpha + config_.beta);
            const double wi = 0.5 / c;
            const std::array<double, 3> wm = {wm0, wi, wi};
            const std::array<double, 3> wc = {wc0, wi, wi};

            std::array<double, 3> hs{};
            double y_hat = 0.0;
            for (std::size_t i = 0; i < 3; ++i) {
                hs[i] = std::tanh(sig[i]);
                y_hat += wm[i] * hs[i];
            }
            const double R = obs_var(y_hat, m);
            double Pyy = R, Pzy = 0.0;
            for (std::size_t i = 0; i < 3; ++i) {
                const double dy = hs[i] - y_hat;
                Pyy += wc[i] * dy * dy;
                Pzy += wc[i] * (sig[i] - z_pred) * dy;
            }
            const double K = (Pyy > 0.0) ? Pzy / Pyy : 0.0;
            z_ = z_pred + K * (r_obs - y_hat);
            P_ = P_pred - K * Pyy * K;
        }

        P_ = std::max(P_, 1e-12);
        ++blocks_;
        xs_.clear();
        ys_.clear();
    }

    double estimate() const override {
        if (blocks_ < config_.burn_in_blocks) return 0.0;
        return std::clamp(std::tanh(z_), -1.0, 1.0);
    }

    void reset() override {
        xs_.clear();
        ys_.clear();
        z_ = 0.0;
        P_ = kPriorVariance;
        blocks_ = 0;
    }

    std::size_t store_bytes() const override {
        return sizeof(*this) + 2 * config_.block * sizeof(double);
    }

    const char* name() const override { return name_; }

private:
    static constexpr double kPriorVariance = 1.0;

    // The variance the raw statistic actually has. This is the heteroscedasticity
    // the Fisher transform exists to remove; a floor keeps it finite as rho -> 1.
    static double obs_var(double rho, double m) {
        const double s = 1.0 - rho * rho;
        return std::max(1.06 * s * s / std::max(m - 1.0, 1.0), 1e-6);
    }

    static std::vector<double> average_ranks(const std::vector<double>& v) {
        const std::size_t n = v.size();
        std::vector<std::size_t> idx(n);
        for (std::size_t i = 0; i < n; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(),
                  [&](std::size_t a, std::size_t b) { return v[a] < v[b]; });
        std::vector<double> r(n);
        std::size_t i = 0;
        while (i < n) {
            std::size_t j = i;
            while (j + 1 < n && v[idx[j + 1]] == v[idx[i]]) ++j;
            const double avg = static_cast<double>(i + j) / 2.0 + 1.0;
            for (std::size_t k = i; k <= j; ++k) r[idx[k]] = avg;
            i = j + 1;
        }
        return r;
    }

    double block_spearman() const {
        const std::size_t n = xs_.size();
        const std::vector<double> rx = average_ranks(xs_);
        const std::vector<double> ry = average_ranks(ys_);
        const double mean = static_cast<double>(n + 1) / 2.0;
        double cov = 0.0, var_x = 0.0, var_y = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double dx = rx[i] - mean, dy = ry[i] - mean;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
        const double denom = std::sqrt(var_x * var_y);
        return (denom > 0.0) ? cov / denom : 0.0;
    }

    Config config_;
    const char* name_;
    double z_ = 0.0;
    double P_ = kPriorVariance;
    std::size_t blocks_ = 0;
    std::vector<double> xs_;
    std::vector<double> ys_;
};
