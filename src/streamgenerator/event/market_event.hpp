#pragma once

#include <chrono>

// Different kinds of events can flow through the same stream. The EventBus
// dispatches on this; `All` is a wildcard used by subscribers, not a real
// event payload type.
enum class EventType {
    Trade,
    Quote,
    All
};

struct MarketEvent {
    double price_x;
    double price_y;

    double volume_x;
    double volume_y;

    double return_x;
    double return_y;

    long long timestamp;

    // kept last so existing positional aggregate-init keeps working
    EventType type = EventType::Trade;
};