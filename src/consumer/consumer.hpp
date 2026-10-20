#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../protocol/symbol_tick.hpp"
#include "device_state.hpp"
#include "ring_buffer.hpp"

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

// One device: its own ring buffer, lag policy and state, so a slow client affects no one else.
class Consumer {
public:
    explicit Consumer(
        std::size_t capacity = 1024,
        ConsumerPolicy policy = ConsumerPolicy::SkipToLatest)
        : buffer_(std::make_shared<RingBuffer>(capacity)),
          cursor_(0),
          policy_(policy) {}

    void on_event(const TickPair& event) {
        buffer_->push(event);
        process_available();
    }

    void process_available() {
        handle_policy();

        while (cursor_ < buffer_->head()) {
            // could fall behind mid-drain if the buffer wrapped under us
            if (cursor_ < buffer_->oldest_available()) {
                handle_policy();
                continue;
            }
            const TickPair e = buffer_->get(cursor_);
            state_.update(e);
            ++cursor_;
        }
    }

    std::vector<std::pair<std::string, double>> dependence() const {
        return state_.dependence();
    }

    DeviceState::Moments moments() const { return state_.moments(); }

    bool has_next() const { return cursor_ < buffer_->head(); }
    size_t cursor() const { return cursor_; }
    bool is_lagging() const { return cursor_ < buffer_->oldest_available(); }
    ConsumerPolicy policy() const { return policy_; }
    const RingBuffer& buffer() const { return *buffer_; }

    void seek_to_beginning() { cursor_ = buffer_->oldest_available(); }
    void seek_latest() { cursor_ = buffer_->head(); }

private:
    void handle_policy() {
        const size_t oldest = buffer_->oldest_available();
        const size_t head = buffer_->head();

        switch (policy_) {
        case ConsumerPolicy::ProcessAll:
            if (cursor_ < oldest) {
                throw ConsumerLagException();
            }
            return;

        case ConsumerPolicy::Snapshot:
            if (cursor_ < oldest) {
                cursor_ = oldest;
            }
            return;

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
    DeviceState state_;
};
