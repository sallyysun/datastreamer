// Standalone C++ consumer for the datastreamer mini-Kafka.
//
// Subscribes to TWO per-symbol equity topics, FIFO-pairs the two single-symbol
// streams into bivariate samples, and runs its own device: a RingBuffer
// (buffer-type device), a StateManager, and an inference layer (the dependence
// estimators). Each consumer is an independent process -- its buffer is its only
// memory, so a slow consumer just falls behind.
//
// Usage:
//   ./consumer [--host 127.0.0.1] [--port 9003] [--symbols AAPL,MSFT]
//              [--policy all|snapshot|latest] [--print 20] [--rate 0]
//
// The consumer subscribes to "trades.equity.<X>" and "trades.equity.<Y>" and
// pairs by arrival order. This assumes the producers publish synchronously (one
// tick per symbol per step); asynchronous / multi-rate legs are a later concern
// (and --rate can desync the legs, so leave it 0 for estimation).

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <utility>

#include "consumer.hpp"
#include "../protocol/serializer.hpp"

namespace {

ConsumerPolicy parse_policy(const std::string& s) {
    if (s == "all") return ConsumerPolicy::ProcessAll;
    if (s == "snapshot") return ConsumerPolicy::Snapshot;
    return ConsumerPolicy::SkipToLatest;  // "latest"
}

const char* arg(int argc, char** argv, const char* key, const char* fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    }
    return fallback;
}

std::pair<std::string, std::string> parse_symbols(const std::string& s) {
    const std::size_t comma = s.find(',');
    if (comma == std::string::npos) return {s, s};
    return {s.substr(0, comma), s.substr(comma + 1)};
}

int connect_broker(const std::string& host, int port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) return -1;
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) return -1;
    return fd;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string host = arg(argc, argv, "--host", "127.0.0.1");
    const int port = std::atoi(arg(argc, argv, "--port", "9003"));
    const auto [sym_x, sym_y] = parse_symbols(arg(argc, argv, "--symbols", "AAPL,MSFT"));
    const ConsumerPolicy policy = parse_policy(arg(argc, argv, "--policy", "all"));
    const int print_every = std::atoi(arg(argc, argv, "--print", "20"));
    const double rate = std::atof(arg(argc, argv, "--rate", "0"));  // msg/s, 0 = unlimited

    const std::string topic_x = "trades.equity." + sym_x;
    const std::string topic_y = "trades.equity." + sym_y;

    const int fd = connect_broker(host, port);
    if (fd < 0) {
        std::fprintf(stderr, "could not connect to broker %s:%d (is ./server running?)\n",
                     host.c_str(), port);
        return 1;
    }
    for (const std::string& topic : {topic_x, topic_y}) {
        std::string sub = "SUBSCRIBE " + topic;
        if (rate > 0.0) sub += " " + std::to_string(rate);
        sub += "\n";
        ::send(fd, sub.data(), sub.size(), 0);
    }
    std::printf("consumer: connected to %s:%d, pairing '%s' x '%s'%s\n",
                host.c_str(), port, sym_x.c_str(), sym_y.c_str(),
                rate > 0.0 ? " (throttled)" : "");

    Consumer device(1024, policy);

    // One FIFO per leg; when both are non-empty we can emit a paired sample.
    std::deque<SymbolTick> queue_x, queue_y;

    std::string buf;
    char chunk[4096];
    std::size_t n_events = 0;
    while (true) {
        const ssize_t r = ::recv(fd, chunk, sizeof(chunk), 0);
        if (r <= 0) { std::printf("consumer: broker closed connection\n"); break; }
        buf.append(chunk, static_cast<std::size_t>(r));

        std::size_t pos;
        while ((pos = buf.find('\n')) != std::string::npos) {
            const std::string line = buf.substr(0, pos);
            buf.erase(0, pos + 1);
            if (line.empty()) continue;

            SymbolTick tick;
            try {
                tick = Serializer::symbol_from_json(line);
            } catch (...) {
                continue;  // skip malformed
            }
            if (tick.symbol == sym_x) queue_x.push_back(tick);
            else if (tick.symbol == sym_y) queue_y.push_back(tick);
            else continue;

            // Drain any complete pairs (synchronous producers keep the legs in
            // lockstep, so front-to-front pairing is exact).
            while (!queue_x.empty() && !queue_y.empty()) {
                const TickPair event{queue_x.front(), queue_y.front()};
                queue_x.pop_front();
                queue_y.pop_front();
                device.on_event(event);

                if (++n_events % static_cast<std::size_t>(print_every) == 0) {
                    std::printf("n=%zu true_rho_s=%+.3f | ", n_events, event.x.true_rho_s);
                    for (const auto& [name, rho] : device.dependence()) {
                        std::printf("%s=%+.3f ", name.c_str(), rho);
                    }
                    std::printf("\n");
                    std::fflush(stdout);
                }
            }
        }
    }

    ::close(fd);
    return 0;
}
