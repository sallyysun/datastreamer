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

MarketEvent FileSource::next() {
    std::string line;

    // skip CSV header once
    if (!header_skipped_) {
        std::getline(file_, line);
        header_skipped_ = true;
    }

    // EOF
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

    return MarketEvent{
        std::stod(fields[0]), // price_x
        std::stod(fields[1]), // price_y
        std::stod(fields[2]), // volume_x
        std::stod(fields[3]), // volume_y
        std::stod(fields[4]), // return_x
        std::stod(fields[5]), // return_y
        std::stoll(fields[6]) // timestamp
    };
}