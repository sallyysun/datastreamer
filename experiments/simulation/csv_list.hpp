#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

// Comma-separated values (an environment variable's value), skipping empty entries; empty for a null string.
inline std::vector<std::string> csv_list(const char* text) {
    std::vector<std::string> items;
    if (!text) return items;
    const std::string_view csv(text);
    for (std::size_t first = 0; first <= csv.size();) {
        std::size_t comma = csv.find(',', first);
        if (comma == std::string_view::npos) comma = csv.size();
        if (comma > first) items.emplace_back(csv.substr(first, comma - first));
        first = comma + 1;
    }
    return items;
}
