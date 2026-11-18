#pragma once

#include <atomic>
#include <mutex>
#include <vector>

#include "../protocol.hpp"

class RingBuffer {
public:
    explicit RingBuffer(size_t capacity)
        : capacity_(capacity),
          data_(capacity),
          head_(0) {}

    void push(const TickPair& event) {
        std::lock_guard<std::mutex> lock(write_mutex_);

        const size_t idx = head_ % capacity_;
        data_[idx] = event;

        head_.fetch_add(1, std::memory_order_release);
    }

    size_t head() const {
        return head_.load(std::memory_order_acquire);
    }

    size_t capacity() const {
        return capacity_;
    }

    // oldest still valid index
    size_t oldest_available() const {
        const size_t h = head();

        if (h <= capacity_) {
            return 0;
        }

        return h - capacity_;
    }

    TickPair get(size_t index) const {
        return data_[index % capacity_];
    }

private:
    size_t capacity_;
    std::vector<TickPair> data_;

    std::atomic<size_t> head_;
    mutable std::mutex write_mutex_;
};