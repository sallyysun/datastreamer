#include "synthetic_source.hpp"

#include <chrono>
#include <cmath>
#include <numbers>
#include <utility>

SyntheticSource::SyntheticSource(std::unique_ptr<Dependence> dependence,
                                 std::unique_ptr<dgp::Copula> copula)
    : price_x_(100.0),
      price_y_(100.0),
      volume_x_(1000.0),
      volume_y_(1000.0),
      drift_x_(0.0001),
      drift_y_(0.0001),
      vol_x_(0.01),
      vol_y_(0.01),
      dependence_(std::move(dependence)),
      copula_(std::move(copula)),
      rng_(std::random_device{}()),
      normal_(0.0, 1.0)
{
}

TickPair SyntheticSource::next() {
    // Clamp away from +-1 so the copula constructions stay well defined.
    double d = dependence_->next();
    if (d > 0.999) d = 0.999;
    if (d < -0.999) d = -0.999;

    const dgp::CopulaDraw draw = copula_->sample(d, rng_);
    const double z1 = draw.z1;
    const double z2 = draw.z2;

    const double r_x = drift_x_ + vol_x_ * z1;
    const double r_y = drift_y_ + vol_y_ * z2;

    price_x_ *= std::exp(r_x);
    price_y_ *= std::exp(r_y);

    auto now = std::chrono::system_clock::now();
    auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch())
                  .count();

    const double rho_s = draw.rho_s;

    const SymbolTick x{"", price_x_, volume_x_, ts, rho_s};
    const SymbolTick y{"", price_y_, volume_y_, ts, rho_s};
    return TickPair{x, y};
}
