#pragma once

#include <memory>

#include "../buffer/ring_buffer.hpp"

class ConsumerState {
public:
    explicit ConsumerState(std::shared_ptr<RingBuffer> buffer)
        : buffer_(buffer),
          cursor_(buffer->head()) {} // start from "live"

    bool has_next() const {
        return cursor_ < buffer_->head();
    }

    MarketEvent next() {
        return buffer_->get(cursor_++);
    }

    void seek_to_beginning() {
        cursor_ = 0;
    }

    void seek_latest() {
        cursor_ = buffer_->head();
    }

private:
    std::shared_ptr<RingBuffer> buffer_;
    size_t cursor_;
};