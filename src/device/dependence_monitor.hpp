#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../estimators/dependence/estimator.hpp"
#include "../estimators/dependence/hermiter.hpp"
#include "../estimators/dependence/xiao_spearman.hpp"
#include "../streamgenerator/event/market_event.hpp"

// Server-side online dependence analytics over the shared event stream.
//
// Unlike the per-client Consumer/StateManager (whose results never leave the
// server), this runs the nonparametric copula estimators once on the live
// stream and its output is broadcast to every client. It is updated exactly
// once per event from the producer thread, so access is single-threaded and
// needs no locking.
//
// Configs here are tuned for a live demo (short calibration / burn-in) so
// estimates appear within seconds rather than the longer windows the offline
// harness uses for statistical accuracy.
class DependenceMonitor {
public:
    // NOTE: these configs are kept IDENTICAL to the offline registry
    // (src/estimators/registry.hpp make_estimator) so live and offline results
    // are directly comparable. Change both together.
    DependenceMonitor() {
        estimators_.emplace_back(std::make_unique<XiaoSpearmanEstimator>(
            XiaoSpearmanEstimator::Config{.bins = 20, .window = 0, .lambda = 0.0, .calibration = 100},
            "xiao"));
        estimators_.emplace_back(std::make_unique<XiaoSpearmanEstimator>(
            XiaoSpearmanEstimator::Config{.bins = 20, .window = 0, .lambda = 0.99, .calibration = 100},
            "xiao_ew"));
        estimators_.emplace_back(std::make_unique<HermiteEstimator>(
            HermiteEstimator::Config{.order = 10, .lambda = 0.0, .burn_in = 50}, "hermite"));
        estimators_.emplace_back(std::make_unique<HermiteEstimator>(
            HermiteEstimator::Config{.order = 10, .lambda = 0.01, .burn_in = 50}, "hermite_ew"));
    }

    void update(const MarketEvent& event) {
        for (auto& est : estimators_) est->update(event.return_x, event.return_y);
    }

    // Current estimate (Spearman's rho) for each estimator, in registration order.
    std::vector<std::pair<std::string, double>> snapshot() const {
        std::vector<std::pair<std::string, double>> out;
        out.reserve(estimators_.size());
        for (const auto& est : estimators_) out.emplace_back(est->name(), est->estimate());
        return out;
    }

private:
    std::vector<std::unique_ptr<DependenceEstimator>> estimators_;
};
