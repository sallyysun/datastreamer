#include <App.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <random>
#include <thread>

#include <nlohmann/json.hpp>

#include <vector>

#include "datasource/synthetic_source.hpp"
#include "runtime/producer.hpp"
#include "runtime/broker.hpp"
#include "../protocol/serializer.hpp"
#include "../protocol/topic.hpp"
#include "runtime/delay_strategies.hpp"

using json = nlohmann::json;

// =========================================================
// CONFIG LOADER
// =========================================================
struct Config {
    int port = 9002;        // uWS HTTP/WebSocket (browser display)
    int broker_port = 9003; // raw-TCP broker (C++ consumers)
};

static Config load_config() {
    std::ifstream f("config.json");
    if (!f.is_open()) return {};
    json j;
    f >> j;
    Config c;
    c.port = j["server"]["port"];
    if (j["server"].contains("broker_port")) c.broker_port = j["server"]["broker_port"];
    return c;
}

// Experiment controls (all optional): SEED (DGP seed; default random), N (stop after N observations and exit; 0 =
// endless), RATE (observations per second; 0 = as fast as delivery allows; default 10), WAIT_CONSUMERS (hold the first
// observation until this many consumers have subscribed to both legs; delivery is then lossless), PORT / BROKER_PORT.
static long env_long(const char* k, long d) { const char* e = std::getenv(k); return e && *e ? std::atol(e) : d; }
static double env_double(const char* k, double d) { const char* e = std::getenv(k); return e && *e ? std::atof(e) : d;
}

// =========================================================
// MIME + topic helpers
// =========================================================
static std::string mime_type(const std::string &path) {
    if (path.ends_with(".html")) return "text/html";
    if (path.ends_with(".js")) return "application/javascript";
    if (path.ends_with(".css")) return "text/css";
    if (path.ends_with(".json")) return "application/json";
    if (path.ends_with(".svg")) return "image/svg+xml";
    return "text/plain";
}

// The two equity symbols this server streams (the two legs of the DGP's pair).
// Overridable via SYMBOLS=<X>,<Y>. Equity-only for now.
static std::pair<std::string, std::string> live_symbols() {
    const char *env = std::getenv("SYMBOLS");
    const std::string s = env && *env ? env : "AAPL,MSFT";
    const std::size_t comma = s.find(',');
    if (comma == std::string::npos) return {s, s};
    return {s.substr(0, comma), s.substr(comma + 1)};
}

// Per-symbol topic: "trades.equity.<symbol>". Consumers subscribe by prefix --
// "trades" / "trades.equity" get every symbol, "trades.equity.AAPL" gets one
// (see Broker::Sub::matches).
static std::string equity_topic(const std::string &symbol) {
    return "trades.equity." + symbol;
}

// Live regime from DEPENDENCE=constant|drift|jump (default jump).
// The thesis scenarios (dependence.hpp) over kLiveN observations, as offline; the clock keeps running after.
constexpr std::size_t kLiveN = 5000;
static std::unique_ptr<Dependence> make_live_dependence() {
    const char* env = std::getenv("DEPENDENCE");
    const std::string name = env ? env : "jump";
    if (auto d = make_scenario(name, kLiveN)) return d;
    std::cout << "unknown DEPENDENCE='" << name << "', using jump\n";
    return make_scenario("jump", kLiveN);
}

// Pick the live copula family from the COPULA env var (gaussian|clayton|gumbel|t,
// default gaussian). The Dependence sets rho_S; the copula sets its SHAPE.
static std::unique_ptr<dgp::Copula> make_live_copula() {
    const char* env = std::getenv("COPULA");
    return dgp::make_copula(env ? env : "gaussian");
}

// =========================================================
// SERVER = stream generator (producer) + broker.
// The server does NOT estimate: it publishes raw events. Estimation lives in
// each consumer's device (see src/consumer). The browser is a display client.
// =========================================================
int main() {
    Config cfg = load_config();
    cfg.port = int(env_long("PORT", cfg.port));
    cfg.broker_port = int(env_long("BROKER_PORT", cfg.broker_port));
    const long n_obs = env_long("N", 0), wait_consumers = env_long("WAIT_CONSUMERS", 0);
    const double rate = env_double("RATE", 10.0);
    const unsigned seed = std::getenv("SEED") ? unsigned(env_long("SEED", 0)) : std::random_device{}();
    const char* cop_env = std::getenv("COPULA");
    const std::string cop = cop_env ? cop_env : "gaussian";
    const bool truth = cop == "gaussian" || cop == "clayton" || cop == "gumbel" || cop == "t";

    uWS::App app;
    Serializer serializer;

    auto dependence = make_live_dependence();
    auto copula = make_live_copula();
    const auto symbols = live_symbols();
    const std::string sym_x = symbols.first;
    const std::string sym_y = symbols.second;
    std::cout << "live dependence: " << dependence->name() << " | copula: " << copula->name()
              << " | symbols: " << sym_x << ", " << sym_y << " (topics " << equity_topic(sym_x)
              << " / " << equity_topic(sym_y) << ")\n";
    std::cout << "seed " << seed << (n_obs ? " | n " + std::to_string(n_obs) : std::string()) << " | rate "
              << (rate > 0 ? std::to_string(rate) : std::string("max")) << "\n";
    auto source = std::make_shared<SyntheticSource>(std::move(dependence), std::move(copula), seed, truth ? cop : "");

    // Raw-TCP broker for C++ consumers.
    Broker broker(cfg.broker_port);
    if (broker.start()) {
        std::cout << "broker listening on tcp/" << cfg.broker_port << "\n";
    } else {
        std::cout << "failed to start broker on tcp/" << cfg.broker_port << "\n";
    }

    Producer producer(source, rate > 0 ? fixed_delay(rate) : zero_delay());
    if (wait_consumers > 0) broker.set_lossless(true);

    // Serve the React build (display client).
    app.get("/*", [](auto *res, auto *req) {
        std::string path = std::string(req->getUrl());
        if (path == "/") path = "/index.html";
        std::string filePath = "client/dist" + path;
        std::ifstream file(filePath, std::ios::binary);
        if (!file.is_open()) {
            std::ifstream fallback("client/dist/index.html");
            std::stringstream buffer;
            buffer << fallback.rdbuf();
            res->writeHeader("Content-Type", "text/html");
            res->end(buffer.str());
            return;
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        res->writeHeader("Content-Type", mime_type(filePath));
        res->end(buffer.str());
    });

    // The concrete per-symbol topics the browser can subscribe to, and the
    // catalog message announcing them on connect.
    const std::vector<std::string> catalog_topics = {equity_topic(sym_x), equity_topic(sym_y)};
    const std::string catalog_msg =
        json{{"type", "catalog"}, {"symbols", json::array({sym_x, sym_y})}}.dump();

    // Browser display over WebSocket. Topic-aware, symmetric with the TCP broker:
    // the browser sends {"action":"subscribe","topics":[...]} and we expand each
    // (possibly a dot-prefix like "trades") against the catalog, so it receives
    // only the concrete symbol topics it asked for. No server-side estimation.
    app.ws<int>("/*", {
        .open = [&catalog_msg](auto *ws) {
            ws->send(catalog_msg, uWS::OpCode::TEXT);  // announce available symbols
        },
        .message = [&catalog_topics](auto *ws, std::string_view message, uWS::OpCode) {
            json j = json::parse(message, nullptr, /*allow_exceptions=*/false);
            if (j.is_discarded() || !j.contains("action") || !j.contains("topics")) return;
            const std::string action = j["action"].get<std::string>();
            for (const auto &t : j["topics"]) {
                const std::string req = t.get<std::string>();
                for (const auto &concrete : catalog_topics) {
                    if (!topic_matches(req, concrete)) continue;
                    if (action == "subscribe") ws->subscribe(concrete);
                    else if (action == "unsubscribe") ws->unsubscribe(concrete);
                }
            }
        }
    });

    // Producer thread: pull each bivariate DGP step, split it into two per-symbol
    // ticks, and publish each to its own topic on BOTH transports (TCP broker and
    // WS). A subscriber -- C++ consumer or browser -- pairs the symbols it wants.
    // No estimation here.
    std::thread producer_thread([&]() {
        while (wait_consumers > 0 && broker.ready_count(2) < std::size_t(wait_consumers))
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::size_t n = 0;
        auto sink = [&](const TickPair &event) {
            // The DGP leaves the legs unnamed; the server names the instruments.
            SymbolTick tick_x = event.x; tick_x.symbol = sym_x;
            SymbolTick tick_y = event.y; tick_y.symbol = sym_y;
            const std::string msg_x = serializer.to_json(tick_x);
            const std::string msg_y = serializer.to_json(tick_y);
            broker.publish(equity_topic(sym_x), msg_x);
            broker.publish(equity_topic(sym_y), msg_y);
            app.publish(equity_topic(sym_x), msg_x, uWS::OpCode::TEXT);
            app.publish(equity_topic(sym_y), msg_y, uWS::OpCode::TEXT);
            if (++n % 1000 == 0) {
                std::cout << "events=" << n
                          << " tcp_consumers=" << broker.subscriber_count() << "\n";
            }
        };
        if (n_obs > 0) {
            producer.run(sink, int(n_obs) + 1);   // + the opening tick
            std::cout << "done: " << n - 1 << " observations" << std::endl;
            broker.stop();       // closes the consumers' sockets after the queued bytes: they see end of stream
            std::_Exit(0);
        }
        producer.run(sink);
    });

    app.listen(cfg.port, [&](auto *token) {
        std::cout << (token ? "server running on port " : "failed to bind port ")
                  << cfg.port << "\n";
    });

    app.run();
    producer_thread.join();
    broker.stop();
}
