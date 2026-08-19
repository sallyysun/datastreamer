#pragma once

#include <functional>
#include <chrono>
#include <thread>
#include <random>

using DelayStrategy = std::function<void()>;

// -------------------------
// no delay
// -------------------------
inline DelayStrategy zero_delay() {
    return []() {
        // do nothing
    };
}

// -------------------------
// fixed interval (deterministic rate)
// -------------------------
inline DelayStrategy fixed_delay(double rate) {
    return [rate]() {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(
                static_cast<int>(1000.0 / rate)
            )
        );
    };
}

// -------------------------
// Poisson process (exponential inter-arrival)
// -------------------------
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

// -------------------------
// bursty process (market clustering)
// -------------------------
inline DelayStrategy burst_delay(
    double rate,
    int burst_min = 2,
    int burst_max = 8,
    double pause_min = 0.5,
    double pause_max = 2.0
) {
    return [=]() mutable {
        static thread_local std::mt19937 gen(std::random_device{}());

        static int remaining = 0;

        std::uniform_int_distribution<int> burst_dist(burst_min, burst_max);
        std::uniform_real_distribution<double> pause_dist(pause_min, pause_max);
        std::exponential_distribution<double> exp_dist(rate);

        if (remaining <= 0) {
            remaining = burst_dist(gen);
            double pause = pause_dist(gen);

            std::this_thread::sleep_for(
                std::chrono::milliseconds(
                    static_cast<int>(pause * 1000.0)
                )
            );
        } else {
            remaining--;
            double interval = exp_dist(gen);

            std::this_thread::sleep_for(
                std::chrono::milliseconds(
                    static_cast<int>(interval * 1000.0)
                )
            );
        }
    };
}