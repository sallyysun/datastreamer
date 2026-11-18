#pragma once

#include <functional>
#include <chrono>
#include <thread>
#include <random>

using DelayStrategy = std::function<void()>;

inline DelayStrategy zero_delay() {
    return []() {
    };
}

inline DelayStrategy fixed_delay(double rate) {
    return [rate]() {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(
                static_cast<int>(1000.0 / rate)
            )
        );
    };
}

// Exponential inter-arrival times.
inline DelayStrategy poisson_delay(double rate) {
    return [rate]() {
        static thread_local std::mt19937 gen(std::random_device{}());
        std::exponential_distribution<double> dist(rate);

        double interval_sec = dist(gen);
        std::this_thread::sleep_for(
            std::chrono::milliseconds(
                static_cast<int>(interval_sec * 1000.0)
            )
        );
    };
}
