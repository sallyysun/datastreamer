#include "file_source.hpp"

#include <sstream>
#include <stdexcept>

FileSource::FileSource(const std::string& filepath)
    : file_(filepath),
      header_skipped_(false)
{
    if (!file_.is_open()) {
        throw std::runtime_error(
            "Failed to open file: " + filepath
        );
    }
}

std::vector<std::string> FileSource::split(
    const std::string& line,
    char delimiter
) {
    std::vector<std::string> tokens;

    std::stringstream ss(line);

    std::string token;

    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}

TickPair FileSource::next() {
    std::string line;

    if (!header_skipped_) {
        std::getline(file_, line);
        header_skipped_ = true;
    }

    if (!std::getline(file_, line)) {
        throw std::runtime_error(
            "End of file reached"
        );
    }

    auto fields = split(line, ',');

    if (fields.size() < 7) {
        throw std::runtime_error(
            "Invalid CSV format"
        );
    }

    // Columns: price_x, price_y, volume_x, volume_y, return_x, return_y, timestamp. Returns are
    // ignored (the device re-derives them from price); real data has no ground truth, so rho_s = 0.
    const long long ts = std::stoll(fields[6]);
    const SymbolTick x{"", std::stod(fields[0]), std::stod(fields[2]), ts, 0.0};
    const SymbolTick y{"", std::stod(fields[1]), std::stod(fields[3]), ts, 0.0};
    return TickPair{x, y};
}