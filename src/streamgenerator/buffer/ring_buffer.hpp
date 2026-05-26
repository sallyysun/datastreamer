#pragma once

#include <vector>
#include <atomic>
#include <mutex>

#include "../event/market_event.hpp"

class RingBuffer {
public:
    explicit RingBuffer(size_t capacity)
        : capacity_(capacity),
          data_(capacity),
          head_(0) {}

    // Producer writes latest event
    void push(const MarketEvent& event) {
        std::lock_guard<std::mutex> lock(write_mutex_);

        size_t idx = head_ % capacity_;
        data_[idx] = event;

        head_.fetch_add(1, std::memory_order_release);
    }

    size_t head() const {
        return head_.load(std::memory_order_acquire);
    }

    MarketEvent get(size_t index) const {
        size_t idx = index % capacity_;
        return data_[idx];
    }

    size_t capacity() const {
        return capacity_;
    }

private:
    size_t capacity_;
    std::vector<MarketEvent> data_;

    std::atomic<size_t> head_;
    mutable std::mutex write_mutex_;
};