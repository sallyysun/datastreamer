#include "producer.hpp"

Producer::Producer(
    std::shared_ptr<DataSource> source,
    DelayStrategy delay
)
    : source_(std::move(source)),
      delay_(std::move(delay))
{}

void Producer::run(Sink sink) {
    while (true) {
        MarketEvent event = source_->next();
        delay_();
        sink(event);
    }
}

void Producer::run(Sink sink, int n_events) {
    for (int i = 0; i < n_events; i++) {
        MarketEvent event = source_->next();
        delay_();
        sink(event);
    }
}