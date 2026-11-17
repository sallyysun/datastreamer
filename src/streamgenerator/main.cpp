#include <App.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <random>
#include <map>
#include <thread>

#include <nlohmann/json.hpp>

#include <vector>

#include "datasource/replay_source.hpp"
#include "datasource/synthetic_source.hpp"
#include "runtime/producer.hpp"
#include "runtime/session.hpp"
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

// Experiment controls (all optional): REPLAY=<file.csv> (stream recorded returns date,x,y instead of the DGP; N defaults
// to the file's length), SEED (DGP seed; default random), N (stop after N observations and exit; 0 =
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
    std::shared_ptr<DataSource> source;
    long n_run = n_obs;
    if (const char* rp = std::getenv("REPLAY")) {
        auto replay = std::make_shared<ReplaySource>(rp);
        if (n_run <= 0) n_run = long(replay->size());
        std::cout << "replaying " << rp << " (" << replay->size() << " observations)\n";
        source = replay;
    } else {
        source = std::make_shared<SyntheticSource>(std::move(dependence), std::move(copula), seed, cop);
    }

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

    // Browser over WebSocket. Two uses:
    //   market view: {"action":"subscribe","topics":[...]} -> the live symbol topics (expanded against the catalog)
    //   MFF lab:     {"action":"start", ...LabConfig} starts a private session (its own steerable generator and its own
    //                consumer process); {"action":"jump","rho":r}, {"action":"drift","rho":r,"steps":n},
    //                {"action":"family","family":f}, {"action":"auto","on":b}, {"action":"rate","rate":x},
    //                {"action":"pause","on":b}, {"action":"stop"} steer it. The consumer's state arrives on the broker as
    //                "device.<id>" and is forwarded to that browser. The server estimates nothing.
    struct PerSocket { int session = 0; };
    using WS = uWS::WebSocket<false, true, PerSocket>;
    uWS::Loop* loop = uWS::Loop::get();
    std::map<int, std::unique_ptr<LabSession>> sessions;   // loop thread only
    std::map<int, WS*> session_ws;                         // loop thread only
    int next_session = 1;
    const int max_sessions = int(env_long("MAX_SESSIONS", 8));
    const char* consumer_env = std::getenv("CONSUMER");
    const std::string consumer_path = consumer_env ? consumer_env : "./consumer";
    broker.on_publish([&](const std::string& topic, const std::string& payload) {
        if (topic.rfind("device.", 0) != 0) return;
        const int id = std::atoi(topic.c_str() + 7);
        loop->defer([&session_ws, id, payload] {
            auto it = session_ws.find(id);
            if (it != session_ws.end()) it->second->send(payload, uWS::OpCode::TEXT);
        });
    });
    auto end_session = [&](PerSocket* d) {
        if (!d->session) return;
        session_ws.erase(d->session);
        sessions.erase(d->session);
        d->session = 0;
    };

    app.ws<PerSocket>("/*", {
        .open = [&catalog_msg](auto *ws) {
            ws->send(catalog_msg, uWS::OpCode::TEXT);  // announce available symbols
        },
        .message = [&](auto *ws, std::string_view message, uWS::OpCode) {
            json j = json::parse(message, nullptr, /*allow_exceptions=*/false);
            if (j.is_discarded() || !j.contains("action")) return;
            const std::string action = j["action"].get<std::string>();
            PerSocket* d = ws->getUserData();
            if (action == "subscribe" || action == "unsubscribe") {
                if (!j.contains("topics")) return;
                for (const auto &t : j["topics"]) {
                    const std::string req = t.get<std::string>();
                    for (const auto &concrete : catalog_topics) {
                        if (!topic_matches(req, concrete)) continue;
                        if (action == "subscribe") ws->subscribe(concrete);
                        else ws->unsubscribe(concrete);
                    }
                }
                return;
            }
            if (action == "start") {
                end_session(d);
                if (int(sessions.size()) >= max_sessions) {
                    ws->send(json{{"type", "error"}, {"message", "too many lab sessions; try again later"}}.dump(), uWS::OpCode::TEXT);
                    return;
                }
                LabConfig c;
                c.copula = j.value("copula", c.copula); c.rho = j.value("rho", c.rho); c.rate = j.value("rate", c.rate);
                c.estimator = j.value("estimator", c.estimator); c.memories = j.value("memories", c.memories);
                c.families = j.value("families", c.families); c.eta = j.value("eta", c.eta); c.delta = j.value("delta", c.delta);
                const int id = next_session++;
                sessions[id] = std::make_unique<LabSession>(id, c, broker, cfg.broker_port, consumer_path,
                    [loop, &session_ws, id](std::string msg) {
                        loop->defer([&session_ws, id, msg] {
                            auto w = session_ws.find(id);
                            if (w != session_ws.end()) w->second->send(msg, uWS::OpCode::TEXT);
                        });
                    });
                session_ws[id] = ws;
                d->session = id;
                ws->send(json{{"type", "started"}, {"session", id}}.dump(), uWS::OpCode::TEXT);
                return;
            }
            auto it = sessions.find(d->session);
            if (it == sessions.end()) return;
            LabSession& s = *it->second;
            if (action == "jump") s.jump(j.value("rho", 0.5));
            else if (action == "drift") s.drift(j.value("rho", 0.5), std::size_t(j.value("steps", 1000)));
            else if (action == "family") s.family(j.value("family", std::string("gaussian")));
            else if (action == "auto") s.set_auto(j.value("on", false));
            else if (action == "rate") s.set_rate(j.value("rate", 100.0));
            else if (action == "pause") s.set_paused(j.value("on", false));
            else if (action == "stop") end_session(d);
        },
        .close = [&](auto *ws, int, std::string_view) { end_session(ws->getUserData()); }
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
            loop->defer([&app, tx = equity_topic(sym_x), ty = equity_topic(sym_y), msg_x, msg_y] {   // uWS is single-threaded
                app.publish(tx, msg_x, uWS::OpCode::TEXT);
                app.publish(ty, msg_y, uWS::OpCode::TEXT);
            });
            if (++n % 1000 == 0) {
                std::cout << "events=" << n
                          << " tcp_consumers=" << broker.subscriber_count() << "\n";
            }
        };
        if (n_run > 0) {
            producer.run(sink, int(n_run) + 1);   // + the opening tick
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
