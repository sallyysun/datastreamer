#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numbers>
#include <string>

// A Dependence produces the true copula parameter rho_t for each step,
// advancing an internal clock on every call to next(). These drive the
// simulated data-generating process so estimators can be scored against a
// known ground truth (roadmap: "Dynamic latent-process model").
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
// over a number of steps. Values are in Spearman's rho; the knob is d = 2 sin(pi rho_S / 6). Not thread-safe: the
// owner serialises next() and the commands.
class ControlledDependence : public Dependence {
public:
    explicit ControlledDependence(double rho_s) : d_(knob(rho_s)) {}
    double next() override {
        if (left_ > 0) { d_ += (target_ - d_) / static_cast<double>(left_); --left_; }
        return d_;
    }
    const char* name() const override { return "controlled"; }
    void jump(double rho_s) { d_ = knob(rho_s); left_ = 0; }
    void drift(double rho_s, std::size_t steps) { target_ = knob(rho_s); left_ = std::max<std::size_t>(steps, 1); }
    static double knob(double rho_s) { return 2.0 * std::sin(std::numbers::pi * std::clamp(rho_s, -0.95, 0.95) / 6.0); }

private:
    double d_, target_ = 0.0;
    std::size_t left_ = 0;
};

// The thesis scenarios (Table 6.2), over n observations: constant 0.6; drift 0.2 -> 0.8 linearly, reaching 0.8 at
// the last observation; jump 0.2 -> 0.8 at n / 2. nullptr for an unknown name.
inline std::unique_ptr<Dependence> make_scenario(const std::string& name, std::size_t n) {
    if (name == "constant") return std::make_unique<ConstantDependence>(0.6);
    if (name == "drift") return std::make_unique<DriftDependence>(0.2, 0.8, n - 1);
    if (name == "jump") return std::make_unique<JumpDependence>(0.2, 0.8, n / 2);
    return nullptr;
}
