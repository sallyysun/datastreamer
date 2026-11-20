// See README.md for consumer usage. Keep --rate at 0 for estimation to avoid desynchronizing the tick pair.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>
#include <string>
#include <string_view>
#include <utility>

#include "broker_client.hpp"
#include "consumer.hpp"
#include "live_lab.hpp"
#include "paired_returns.hpp"

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

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    for (std::string x; std::getline(ss, x, ',');) if (!x.empty()) v.push_back(x);
    return v;
}

int run_lab(BrokerClient& client, int argc, char** argv, const char* id) {
    if (const char* seed = arg(argc, argv, "--seed", nullptr))
        datasketches_kll::seed(std::strtoul(seed, nullptr, 10));
    std::vector<double> mem;
    for (const auto& m : split(arg(argc, argv, "--memories", thesis::kMemoryCsv))) mem.push_back(std::atof(m.c_str()));
    const auto fams = split(arg(argc, argv, "--families", "gaussian"));
    const bool fast_warmup = std::string_view(arg(argc, argv, "--fast-warmup", "0")) == "1";
    const char* eta = arg(argc, argv, "--eta", nullptr);
    const char* delta = arg(argc, argv, "--delta", nullptr);
    LiveLab lab(arg(argc, argv, "--estimator", "rmpl"), mem, fams, eta ? std::atof(eta) : thesis::Mff{}.eta,
                delta ? std::atof(delta) : thesis::Mff{}.delta, fast_warmup ? LiveLab::kDemoWarmup : thesis::kMffWarmup);
    const char* publish_every = arg(argc, argv, "--publish-every", nullptr);
    const std::size_t every = publish_every ? std::max<std::size_t>(1, std::strtoul(publish_every, nullptr, 10)) : 0;
    using Clock = std::chrono::steady_clock;
    auto next_publish = Clock::now();
    const std::string topic = std::string("device.") + id;
    client.publish(topic, lab.config_json().dump());
    TickPair e;
    PairedReturns returns;
    std::size_t n = 0;
    while (client.next(e)) {
        returns.on_tick(e, [&](double x, double y) {
            lab.update(x, y);
            ++n;
            const auto now = Clock::now();
            if (every ? n % every == 0 : now >= next_publish) {
                auto state = lab.state_json(e.x.true_rho_s, e.x.true_family);
                state["prices"] = {e.x.price, e.y.price};
                client.publish(topic, state.dump());
                next_publish = now + std::chrono::milliseconds(40);
            }
        });
    }
    return 0;
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

    if (const char* id = arg(argc, argv, "--live", nullptr)) return run_lab(client, argc, argv, id);

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
