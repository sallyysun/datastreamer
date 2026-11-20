#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

struct GuidedEvents {
    std::size_t jump_at = 0, drift_at = 0, drift_steps = 0;
    double jump_to = 0, drift_to = 0;
};

struct DemoConfig {
    std::size_t warmup_observations = 0;
    double warmup_seconds = 1;
    std::vector<double> rates;
    double default_rate = 0;
    double eta = 0, delta = 0;
    nlohmann::json lab;
    GuidedEvents guided;
    double market_start_rho = 0;
    std::vector<double> market_memories;
    std::size_t market_first_event = 0, market_event_interval = 1;
    bool market_loop = true;
    std::vector<std::array<double, 2>> market_events;

    double warmup_rate() const { return double(warmup_observations) / warmup_seconds; }

    double clamp_rate(double rate) const {
        return std::isfinite(rate) ? std::clamp(rate, rates.front(), rates.back()) : default_rate;
    }

    nlohmann::json to_json() const {
        return {{"rates", rates}, {"default", default_rate}, {"preroll", warmup_observations},
                {"warmupSeconds", warmup_seconds}, {"eta", eta}, {"delta", delta}, {"guided", lab.at("guided")}, {"lab", lab}};
    }

    static DemoConfig load(const std::string& path = "config.json") {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("cannot open " + path);
        const auto root = nlohmann::json::parse(in, nullptr, false);
        if (root.is_discarded()) throw std::runtime_error(path + " is not valid JSON");
        DemoConfig c;
        try {
            const auto& demo = root.at("demo");
            c.warmup_observations = demo.at("warmup").at("observations").get<std::size_t>();
            c.warmup_seconds = demo.at("warmup").at("seconds").get<double>();
            c.rates = demo.at("playback").at("rates").get<std::vector<double>>();
            c.default_rate = demo.at("playback").at("default").get<double>();
            c.eta = demo.at("mff").at("eta").get<double>();
            c.delta = demo.at("mff").at("delta").get<double>();
            const auto start_rho = demo.at("startRho").get<double>();
            const auto memories = demo.at("memories").get<std::vector<double>>();
            c.lab = demo.at("lab");
            c.lab["rho"] = c.lab.value("rho", start_rho);
            c.lab["memories"] = c.lab.value("memories", memories);
            const auto& g = c.lab.at("guided");
            c.guided = {g.at("jumpAt").get<std::size_t>(), g.at("driftAt").get<std::size_t>(),
                        g.at("driftSteps").get<std::size_t>(), g.at("jumpTo").get<double>(), g.at("driftTo").get<double>()};
            const auto& market = demo.at("market");
            c.market_start_rho = market.value("startRho", start_rho);
            c.market_memories = market.value("memories", memories);
            c.market_first_event = market.at("firstEvent").get<std::size_t>();
            c.market_event_interval = market.at("interval").get<std::size_t>();
            c.market_loop = market.value("loop", true);
            c.market_events = market.at("events").get<std::vector<std::array<double, 2>>>();
        } catch (const nlohmann::json::exception& error) {
            throw std::runtime_error(path + ": " + error.what());
        }
        if (c.rates.empty() || c.market_events.empty() || c.market_memories.empty() || c.warmup_seconds <= 0 || c.market_event_interval == 0)
            throw std::runtime_error(path + ": demo needs rates, market events and memories, warmup seconds > 0 and an interval > 0");
        std::sort(c.rates.begin(), c.rates.end());
        return c;
    }
};
