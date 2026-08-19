#pragma once

#include <chrono>

// Different kinds of events can flow through the same stream. The broker routes
// on this (via its topic); `All` is a wildcard used by subscribers, not a real
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

    // True Spearman's rho of the data-generating process at this event, when the
    // source knows it (synthetic DGP). 0 for real/file sources. Set by member
    // assignment, never positionally, so it stays out of the aggregate-init list.
    double true_rho_s = 0.0;
};