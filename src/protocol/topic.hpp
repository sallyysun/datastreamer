#pragma once

#include <string>

// Hierarchical, dot-separated topic matching shared by both transports (the TCP
// Broker and the uWS WebSocket bridge) so they agree on semantics.
//
// A subscription `sub` matches a published topic `pub` if it equals it, is the
// literal "all", or is a dot-boundary prefix of it. So "trades" matches
// "trades.equity.AAPL", "trades.equity" matches it too, "trades.equity.AAPL"
// selects just that one, and "all" matches everything.
inline bool topic_matches(const std::string& sub, const std::string& pub) {
    if (sub == "all" || sub == pub) return true;
    return pub.size() > sub.size() && pub[sub.size()] == '.' &&
           pub.compare(0, sub.size(), sub) == 0;
}
