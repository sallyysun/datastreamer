#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <random>

// DGP dependence paths: next() advances the clock and returns the knob d_t. d is a Gaussian-copula
// correlation, NOT Spearman: copula.hpp maps it to rho_S = (6/pi) asin(d/2). Notes: notes/studies/roadmap.md.
struct Dependence {
    virtual ~Dependence() = default;
    virtual double next() = 0;
    virtual const char* name() const = 0;
};

class ConstantDependence : public Dependence {
public:
    explicit ConstantDependence(double rho) : rho_(rho) {}
    double next() override { return rho_; }
    const char* name() const override { return "constant"; }

private:
    double rho_;
};

// Never settles, so it suits the infinite live stream (a one-off jump would go flat).
class OscillatingDependence : public Dependence {
public:
    OscillatingDependence(double amplitude, double period, double offset = 0.0)
        : amplitude_(amplitude), period_(period), offset_(offset) {}
    double next() override {
        const double rho = offset_ + amplitude_ * std::sin(2.0 * std::numbers::pi * static_cast<double>(t_) / period_);
        ++t_;
        return rho;
    }
    const char* name() const override { return "oscillate"; }

private:
    double amplitude_, period_, offset_;
    std::size_t t_ = 0;
};

class DriftDependence : public Dependence {
public:
    DriftDependence(double r0, double r1, std::size_t T) : r0_(r0), r1_(r1), T_(T) {}
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

class JumpDependence : public Dependence {
public:
    JumpDependence(double r0, double r1, std::size_t tau) : r0_(r0), r1_(r1), tau_(tau) {}
    double next() override { return (t_++ < tau_) ? r0_ : r1_; }
    const char* name() const override { return "jump"; }

private:
    double r0_, r1_;
    std::size_t tau_;
    std::size_t t_ = 0;
};

class StochasticDependence : public Dependence {
public:
    StochasticDependence(double sigma, double z0, unsigned seed)
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
