// Standalone consumer: subscribes to two per-symbol equity topics on the broker, pairs the legs into bivariate
// observations and runs its own device (ring buffer + state: moments and the Chapter 6 strategies). Each consumer is
// an independent process; its buffer is its only memory, so a slow consumer falls behind (or skips, by policy).
//
// Usage:
//   ./consumer [--host 127.0.0.1] [--port 9003] [--symbols AAPL,MSFT] [--policy all|snapshot|latest] [--print 20]
//              [--rate 0] [--score <scenario> --seed <s> --out <file.csv>] [--n <observations>]
// --score: score every strategy against the ground truth on the ticks and append the probe's strategies_seeds rows
// (copula,scenario,estimator,strategy,seed,window,rmse,excess_x1000,h2_x1000) to --out when the stream ends.
// Leave --rate 0 for estimation: throttling can desync the two legs.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

#include "broker_client.hpp"
#include "consumer.hpp"

namespace {

ConsumerPolicy parse_policy(const std::string& s) {
    if (s == "all") return ConsumerPolicy::ProcessAll;
    if (s == "snapshot") return ConsumerPolicy::Snapshot;
    return ConsumerPolicy::SkipToLatest;  // "latest"
}

const char* arg(int argc, char** argv, const char* key, const char* fallback) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    return fallback;
}

std::pair<std::string, std::string> parse_symbols(const std::string& s) {
    const std::size_t comma = s.find(',');
    if (comma == std::string::npos) return {s, s};
    return {s.substr(0, comma), s.substr(comma + 1)};
}

}  // namespace

int main(int argc, char** argv) {
    const std::string host = arg(argc, argv, "--host", "127.0.0.1");
    const int port = std::atoi(arg(argc, argv, "--port", "9003"));
    const auto [sym_x, sym_y] = parse_symbols(arg(argc, argv, "--symbols", "AAPL,MSFT"));
    const ConsumerPolicy policy = parse_policy(arg(argc, argv, "--policy", "all"));
    const int print_every = std::atoi(arg(argc, argv, "--print", "20"));
    const double rate = std::atof(arg(argc, argv, "--rate", "0"));
    const std::string score = arg(argc, argv, "--score", "");
    const unsigned seed = unsigned(std::atol(arg(argc, argv, "--seed", "0")));
    const std::string out = arg(argc, argv, "--out", "");
    const std::size_t n_max = std::strtoul(arg(argc, argv, "--n", "0"), nullptr, 10);

    BrokerClient client;
    if (!client.connect(host, port, sym_x, sym_y, rate)) {
        std::fprintf(stderr, "could not connect to broker %s:%d (is ./server running?)\n", host.c_str(), port);
        return 1;
    }
    std::printf("consumer: connected to %s:%d, pairing '%s' x '%s'%s\n", host.c_str(), port, sym_x.c_str(),
                sym_y.c_str(), rate > 0.0 ? " (throttled)" : "");

    Consumer device(1024, policy);
    if (!score.empty()) device.enable_scoring(score);
    TickPair event;
    std::size_t n_events = 0;
    while (client.next(event)) {
        device.on_event(event);
        ++n_events;
        if (print_every > 0 && n_events % static_cast<std::size_t>(print_every) == 0) {
            std::printf("n=%zu true_rho_s=%+.3f | ", device.observations(), event.x.true_rho_s);
            for (const auto& [name, rho] : device.dependence())
                if (name.ends_with("_mff") || name.ends_with("_vff_slow")) std::printf("%s=%+.3f ", name.c_str(), rho);
            std::printf("\n");
            std::fflush(stdout);
        }
        if (n_max && device.observations() >= n_max) break;
    }
    std::printf("consumer: end of stream after %zu observations\n", device.observations());
    client.close();

    if (!score.empty() && !out.empty()) {
        std::FILE* f = std::fopen(out.c_str(), "a");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", out.c_str()); return 1; }
        device.write_scores(f, seed);
        std::fclose(f);
    }
    return 0;
}
