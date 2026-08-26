#include "synthetic_source.hpp"

#include <chrono>
#include <cmath>
#include <numbers>
#include <utility>

SyntheticSource::SyntheticSource(std::unique_ptr<Dependence> dependence)
    : price_x_(100.0),
      price_y_(100.0),
      volume_x_(1000.0),
      volume_y_(1000.0),
      drift_x_(0.0001),
      drift_y_(0.0001),
      vol_x_(0.01),
      vol_y_(0.01),
      dependence_(std::move(dependence)),
      rng_(std::random_device{}()),
      normal_(0.0, 1.0)
{
}

TickPair SyntheticSource::next() {
    // Current true dependence for this step (clamped away from +-1 so the
    // Gaussian-copula construction stays well defined).
    double rho = dependence_->next();
    if (rho > 0.999) rho = 0.999;
    if (rho < -0.999) rho = -0.999;

    // Two standard normals correlated at rho: a Gaussian copula.
    const double z1 = normal_(rng_);
    const double z2 = rho * z1 + std::sqrt(1.0 - rho * rho) * normal_(rng_);

    const double r_x = drift_x_ + vol_x_ * z1;
    const double r_y = drift_y_ + vol_y_ * z2;

    price_x_ *= std::exp(r_x);
    price_y_ *= std::exp(r_y);

    auto now = std::chrono::system_clock::now();
    auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch())
                  .count();

    // Gaussian copula: Spearman's rho = (6/pi) * asin(rho/2). r_x/r_y were used
    // to advance the prices above; the wire carries price + volume, and each
    // device re-derives the return from consecutive prices.
    const double rho_s = 6.0 / std::numbers::pi * std::asin(rho / 2.0);

    // Symbols are left empty here (the DGP is instrument-agnostic); the server
    // names the two legs before publishing.
    const SymbolTick x{"", price_x_, volume_x_, ts, rho_s};
    const SymbolTick y{"", price_y_, volume_y_, ts, rho_s};
    return TickPair{x, y};
}
