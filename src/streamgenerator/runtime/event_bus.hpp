#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

#include "../event/market_event.hpp"

// Fan-out dispatcher between the single producer and the many consumers.
//
// The producer publishes events; each consumer subscribes a handler keyed by
// the event type it cares about (or EventType::All for everything). On publish
// the bus delivers the event to every matching handler, which is how each
// consumer's own ring buffer gets fed.
//
// Thread-safety: publish() is called from the producer thread while
// subscribe()/unsubscribe() are called from the websocket event-loop thread on
// connect/disconnect, so all access is guarded by a single mutex. Handlers run
// while the lock is held, so a handler must NOT call back into the bus.
class EventBus {
public:
    using Handler = std::function<void(const MarketEvent&)>;
    using SubId = std::uint64_t;

    SubId subscribe(EventType type, Handler handler) {
        std::lock_guard<std::mutex> lock(mutex_);
        const SubId id = next_id_++;
        subs_.push_back(Subscription{id, type, std::move(handler)});
        return id;
    }

    void unsubscribe(SubId id) {
        std::lock_guard<std::mutex> lock(mutex_);
        subs_.erase(
            std::remove_if(
                subs_.begin(), subs_.end(),
                [id](const Subscription& s) { return s.id == id; }),
            subs_.end());
    }

    void publish(const MarketEvent& event) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& sub : subs_) {
            if (sub.type == EventType::All || sub.type == event.type) {
                sub.handler(event);
            }
        }
    }

    std::size_t subscriber_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return subs_.size();
    }

private:
    struct Subscription {
        SubId id;
        EventType type;
        Handler handler;
    };

    mutable std::mutex mutex_;
    std::vector<Subscription> subs_;
    SubId next_id_ = 1;
};
