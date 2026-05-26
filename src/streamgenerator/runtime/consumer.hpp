#pragma once

#include <memory>

#include "../buffer/ring_buffer.hpp"
#include "../event/market_event.hpp"

class Consumer {
public:
    explicit Consumer(std::shared_ptr<RingBuffer> buffer)
        : buffer_(buffer), cursor_(0) {}

    // check if more data exists
    bool has_next() const {
        // return cursor_ < buffer_->size();
        return true;
    }

    // get next event
    MarketEvent next() {
        return buffer_->get(cursor_++);
    }

    // optional: fast forward
    void set_cursor(size_t cursor) {
        cursor_ = cursor;
    }

    size_t cursor() const {
        return cursor_;
    }

private:
    std::shared_ptr<RingBuffer> buffer_;
    size_t cursor_;
};