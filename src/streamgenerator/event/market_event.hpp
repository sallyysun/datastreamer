#pragma once

#include <chrono>

struct MarketEvent {
    double price_x;
    double price_y;

    double volume_x;
    double volume_y;

    double return_x;
    double return_y;

    long long timestamp;
};