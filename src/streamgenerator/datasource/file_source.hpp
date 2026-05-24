#pragma once

#include "datasource.hpp"

#include <fstream>
#include <string>
#include <vector>

class FileSource : public DataSource {
public:
    explicit FileSource(const std::string& filepath);

    MarketEvent next() override;

private:
    std::ifstream file_;

    bool header_skipped_;

    std::vector<std::string> split(
        const std::string& line,
        char delimiter
    );
};