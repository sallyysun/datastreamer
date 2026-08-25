#pragma once

#include <functional>
#include <memory>

#include "../datasource/datasource.hpp"
#include "../../protocol/symbol_tick.hpp"
#include "delay_strategies.hpp"

class Producer {
public:
    using Sink = std::function<void(const TickPair&)>;

    Producer(
        std::shared_ptr<DataSource> source,
        DelayStrategy delay
    );

    // infinite stream
    void run(Sink sink);

    // bounded stream
    void run(Sink sink, int n_events);

private:
    std::shared_ptr<DataSource> source_;
    DelayStrategy delay_;
};