#pragma once

#include <string>

// Topic matching shared by the TCP Broker and the uWS bridge so both agree: exact, "all", or a
// dot-boundary prefix ("trades" matches "trades.equity.AAPL"; "trades.eq" does not).
inline bool topic_matches(const std::string& sub, const std::string& pub) {
    if (sub == "all" || sub == pub) return true;
    return pub.size() > sub.size() && pub[sub.size()] == '.' &&
           pub.compare(0, sub.size(), sub) == 0;
}
