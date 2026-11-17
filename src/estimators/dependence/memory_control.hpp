#pragma once

// An estimator whose exponential-forgetting memory can be set from outside, in effective sample size
// N_eff = (1 + lambda) / (1 - lambda), so one controller can drive Xiao, Hermite and RMPL alike
// (the VFF in evaluation.hpp). Each maps N_eff to its own knob.
class MemoryControlled {
public:
    virtual ~MemoryControlled() = default;
    virtual void set_neff(double neff) = 0;
};
