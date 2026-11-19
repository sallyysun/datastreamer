#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numbers>
#include <string>

#include "study_config.hpp"

// A Dependence produces the target Spearman rho for each step, advancing an internal clock on every call to next();
// each copula family maps it to its own parameter.
struct Dependence {
    virtual ~Dependence() = default;
    virtual double next() = 0;
    virtual const char* name() const = 0;
};

// rho_t = rho, constant.
class ConstantDependence : public Dependence {
public:
    explicit ConstantDependence(double rho) : rho_(rho) {}
    double next() override { return rho_; }
    const char* name() const override { return "constant"; }

private:
    double rho_;
};

// Linear ramp from r0 to r1 over T steps, then held at r1.
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

// Abrupt structural break: r0 for t < tau, r1 afterwards.
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

// Steered from outside (the browser lab): holds rho_S until told to jump to a new value or drift to one linearly
// over a number of steps. Not thread-safe: the owner serialises next() and the commands.
class ControlledDependence : public Dependence {
public:
    explicit ControlledDependence(double rho_s) : rho_(clamp(rho_s)) {}
    double next() override {
        if (left_ > 0) { rho_ += (target_ - rho_) / static_cast<double>(left_); --left_; }
        return rho_;
    }
    const char* name() const override { return "controlled"; }
    double current() const { return rho_; }
    void jump(double rho_s) { rho_ = clamp(rho_s); left_ = 0; }
    void drift(double rho_s, std::size_t steps) { target_ = clamp(rho_s); left_ = std::max<std::size_t>(steps, 1); }
    static double clamp(double rho_s) { return std::clamp(rho_s, -0.95, 0.95); }

private:
    double rho_, target_ = 0.0;
    std::size_t left_ = 0;
};

// Use the study change point at the default horizon, or the midpoint for other horizons.
inline std::unique_ptr<Dependence> make_scenario(const std::string& name, std::size_t n) {
    if (name == "constant") return std::make_unique<ConstantDependence>(thesis::kConstantRhoS);
    if (name == "drift") return std::make_unique<DriftDependence>(thesis::kDriftFrom, thesis::kDriftTo, n - 1);
    if (name == "jump") {
        const auto change = n == thesis::kN ? thesis::kChange : n / 2;
        return std::make_unique<JumpDependence>(thesis::kJumpFrom, thesis::kJumpTo, change);
    }
    return nullptr;
}
