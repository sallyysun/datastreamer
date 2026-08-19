#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../streamgenerator/event/market_event.hpp"
#include "dependence_monitor.hpp"
#include "ring_buffer.hpp"
#include "state_manager.hpp"

class ConsumerLagException : public std::runtime_error {
public:
    ConsumerLagException()
        : std::runtime_error("consumer fell behind ring buffer") {}
};

enum class ConsumerPolicy {
    ProcessAll,      // strict: never miss data (throws if overwritten)
    Snapshot,        // batch/periodic processing, catch up to oldest available
    SkipToLatest     // real-time only, always jump to newest
};

// One connected client. Each Consumer owns its own ring buffer, its own lag
// policy, and its own StateManager, so clients are fully independent: a slow
// client lagging or skipping events doesn't affect anyone else.
//
// on_event() is invoked for each event the device receives (from the broker /
// consumer read loop). It appends to this consumer's buffer and folds the
// newly-available events into the running state and inference, honouring the lag
// policy.
class Consumer {
public:
    explicit Consumer(
        std::size_t capacity = 1024,
        ConsumerPolicy policy = ConsumerPolicy::SkipToLatest)
        : buffer_(std::make_shared<RingBuffer>(capacity)),
          cursor_(0),
          policy_(policy) {}

    // ----------- ingestion (called per received event) -----------

    void on_event(const MarketEvent& event) {
        buffer_->push(event);
        process_available();
    }

    // Drain everything newly available, applying the lag policy, and fold each
    // event into the running state and this device's inference layer.
    void process_available() {
        handle_policy();

        while (cursor_ < buffer_->head()) {
            // could fall behind mid-drain if the buffer wrapped under us
            if (cursor_ < buffer_->oldest_available()) {
                handle_policy();
                continue;
            }
            const MarketEvent e = buffer_->get(cursor_);
            state_.update(e);
            monitor_.update(e);   // per-device inference layer (thesis: inference on device state)
            ++cursor_;
        }
    }

    // ----------- inference layer (the device's statistical objective) -----------

    // Current dependence estimate (Spearman rho) from each estimator this device
    // runs. Each device owns its own estimators, independent of every other.
    std::vector<std::pair<std::string, double>> dependence() const {
        return monitor_.snapshot();
    }

    // ----------- statistics -----------

    StateManager::Snapshot stats() const { return state_.snapshot(); }

    double rolling_correlation(std::size_t window) const {
        return StateManager::rolling_correlation(*buffer_, window);
    }

    // ----------- introspection -----------

    bool has_next() const { return cursor_ < buffer_->head(); }
    size_t cursor() const { return cursor_; }
    bool is_lagging() const { return cursor_ < buffer_->oldest_available(); }
    ConsumerPolicy policy() const { return policy_; }
    const RingBuffer& buffer() const { return *buffer_; }

    // ----------- explicit controls -----------

    void seek_to_beginning() { cursor_ = buffer_->oldest_available(); }
    void seek_latest() { cursor_ = buffer_->head(); }

private:
    // Reconcile the cursor with the buffer according to the lag policy.
    void handle_policy() {
        const size_t oldest = buffer_->oldest_available();
        const size_t head = buffer_->head();

        switch (policy_) {
        // 1. ProcessAll: never skip, fail loudly if we missed data
        case ConsumerPolicy::ProcessAll:
            if (cursor_ < oldest) {
                throw ConsumerLagException();
            }
            return;

        // 2. Snapshot: catch up to the oldest still-valid event, never fail
        case ConsumerPolicy::Snapshot:
            if (cursor_ < oldest) {
                cursor_ = oldest;
            }
            return;

        // 3. SkipToLatest: drop any backlog, keep only the newest event so the
        //    state always reflects the live edge
        case ConsumerPolicy::SkipToLatest:
            if (head > 0 && cursor_ < head - 1) {
                cursor_ = head - 1;
            }
            return;
        }
    }

    std::shared_ptr<RingBuffer> buffer_;
    size_t cursor_;
    ConsumerPolicy policy_;
    StateManager state_;
    DependenceMonitor monitor_;   // this device's inference layer
};
