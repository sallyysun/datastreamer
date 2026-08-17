#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>

// A DependenceSchedule produces the true copula parameter rho_t for each step,
// advancing an internal clock on every call to next(). These drive the
// simulated data-generating process so estimators can be scored against a
// known ground truth (roadmap: "Dynamic latent-process model").
struct Schedule {
    virtual ~Schedule() = default;
    virtual double next() = 0;
    virtual const char* name() const = 0;
};

// rho_t = rho, constant.
class ConstantSchedule : public Schedule {
public:
    explicit ConstantSchedule(double rho) : rho_(rho) {}
    double next() override { return rho_; }
    const char* name() const override { return "constant"; }

private:
    double rho_;
};

// Linear ramp from r0 to r1 over T steps, then held at r1.
class DriftSchedule : public Schedule {
public:
    DriftSchedule(double r0, double r1, std::size_t T) : r0_(r0), r1_(r1), T_(T) {}
    double next() override {
        const double frac = (T_ == 0) ? 1.0 : std::min(1.0, static_cast<double>(t_) / static_cast<double>(T_));
        ++t_;
        return r0_ + (r1_ - r0_) * frac;
    }
    const char* name() const override { return "drift"; }

private:
    double r0_, r1_;
    std::size_t T_;
    std::size_t t_ = 0;
};

// Abrupt structural break: r0 for t < tau, r1 afterwards.
class JumpSchedule : public Schedule {
public:
    JumpSchedule(double r0, double r1, std::size_t tau) : r0_(r0), r1_(r1), tau_(tau) {}
    double next() override { return (t_++ < tau_) ? r0_ : r1_; }
    const char* name() const override { return "jump"; }

private:
    double r0_, r1_;
    std::size_t tau_;
    std::size_t t_ = 0;
};

// Stochastic latent dependence: z_t = z_{t-1} + sigma * omega_t, rho_t = tanh(z_t).
class StochasticSchedule : public Schedule {
public:
    StochasticSchedule(double sigma, double z0, unsigned seed)
        : sigma_(sigma), z_(z0), rng_(seed), normal_(0.0, 1.0) {}
    double next() override {
        z_ += sigma_ * normal_(rng_);
        return std::tanh(z_);
    }
    const char* name() const override { return "stochastic"; }

private:
    double sigma_;
    double z_;
    std::mt19937 rng_;
    std::normal_distribution<double> normal_;
};
