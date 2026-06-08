#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>

#include "../streamgenerator/event/market_event.hpp"
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
// on_event() is invoked by the EventBus on the producer thread. It appends to
// this consumer's buffer and folds the newly-available events into the running
// state, honouring the lag policy.
class Consumer {
public:
    explicit Consumer(
        std::size_t capacity = 1024,
        ConsumerPolicy policy = ConsumerPolicy::SkipToLatest)
        : buffer_(std::make_shared<RingBuffer>(capacity)),
          cursor_(0),
          policy_(policy) {}

    // ----------- ingestion (called by the EventBus) -----------

    void on_event(const MarketEvent& event) {
        buffer_->push(event);
        process_available();
    }
hah
    // Drain everything newly available, applying the lag policy, and fold each
    // event into the running state.
    void process_available() {
        handle_policy();

        while (cursor_ < buffer_->head()) {
            // could fall behind mid-drain if the buffer wrapped under us
            if (cursor_ < buffer_->oldest_available()) {
                handle_policy();
                continue;
            }
            state_.update(buffer_->get(cursor_));
            ++cursor_;
        }
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

    // ----------- bus subscription bookkeeping -----------

    void set_subscription(std::uint64_t id) { sub_id_ = id; }
    std::uint64_t subscription() const { return sub_id_; }

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
    std::uint64_t sub_id_ = 0;
};
