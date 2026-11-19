#include <App.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <chrono>
#include <random>
#include <map>
#include <thread>

#include <nlohmann/json.hpp>

#include <vector>

#include "datasource/file_source.hpp"
#include "datasource/synthetic_source.hpp"
#include "runtime/session.hpp"
#include "runtime/broker.hpp"
#include "runtime/market.hpp"
#include "../protocol.hpp"

using json = nlohmann::json;

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

static long env_long(const char* k, long d) { const char* e = std::getenv(k); return e && *e ? std::atol(e) : d; }
static double env_double(const char* k, double d) { const char* e = std::getenv(k); return e && *e ? std::atof(e) : d;
}

static std::string mime_type(const std::string &path) {
    if (path.ends_with(".html")) return "text/html";
    if (path.ends_with(".js")) return "application/javascript";
    if (path.ends_with(".css")) return "text/css";
    if (path.ends_with(".json")) return "application/json";
    if (path.ends_with(".svg")) return "image/svg+xml";
    return "text/plain";
}

static std::pair<std::string, std::string> live_symbols() {
    const char *env = std::getenv("SYMBOLS");
    const std::string s = env && *env ? env : "AAPL,MSFT";
    const std::size_t comma = s.find(',');
    if (comma == std::string::npos) return {s, s};
    return {s.substr(0, comma), s.substr(comma + 1)};
}

// N controls delivery length independently of the scenario horizon.
static std::unique_ptr<Dependence> make_live_dependence() {
    const char* env = std::getenv("DEPENDENCE");
    const std::string name = env ? env : "jump";
    if (auto d = make_scenario(name, thesis::kN)) return d;
    std::cout << "unknown DEPENDENCE='" << name << "', using jump\n";
    return make_scenario("jump", thesis::kN);
}

static std::unique_ptr<dgp::Copula> make_live_copula() {
    const char* env = std::getenv("COPULA");
    return dgp::make_copula(env ? env : "gaussian");
}

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
        auto replay = std::make_shared<FileSource>(rp);
        if (n_run <= 0) n_run = long(replay->size());
        std::cout << "replaying " << rp << " (" << replay->size() << " observations)\n";
        source = replay;
    } else {
        source = std::make_shared<SyntheticSource>(std::move(dependence), std::move(copula), seed, cop);
    }

    Broker broker(cfg.broker_port);
    if (broker.start()) {
        std::cout << "broker listening on tcp/" << cfg.broker_port << "\n";
    } else {
        std::cout << "failed to start broker on tcp/" << cfg.broker_port << "\n";
    }

    // Emit n events (n < 0: forever), one every 1000 / rate ms (as fast as possible when rate <= 0).
    auto produce = [&source, rate](auto&& sink, long n) {
        for (long i = 0; n < 0 || i < n; ++i) {
            const TickPair event = source->next();
            if (rate > 0) std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(1000.0 / rate)));
            sink(event);
        }
    };
    if (wait_consumers > 0) broker.set_lossless(true);

    const bool market_enabled = n_run <= 0 && rate > 0 && wait_consumers <= 0 && !std::getenv("REPLAY");
    struct PerSocket { int session = 0; };
    using WS = uWS::WebSocket<false, true, PerSocket>;
    struct MarketFeed {
        MarketFeed(unsigned seed, const char* prices, const char* tickers, const char* events)
            : broadcast(seed, prices, tickers, events) {}
        MarketFeed(const std::filesystem::path& file, std::vector<std::string> symbols)
            : broadcast(file, std::move(symbols)) {}

        market::Broadcast broadcast;
        std::map<WS*, market::Subscription> subscriptions;
    };
    using MarketFeeds = std::map<std::string, MarketFeed, std::less<>>;
    MarketFeeds feeds;
    if (market_enabled) {
        feeds.try_emplace("simulation", seed, nullptr, nullptr, std::getenv("MARKET_EVENTS"));
        if (const char* prices = std::getenv("MARKET_PRICES_DIR"); prices && *prices) {
            try {
                feeds.try_emplace("replay", seed, prices, std::getenv("MARKET_TICKERS"), nullptr);
            } catch (const std::exception& error) {
                std::cerr << "Market replay unavailable: " << error.what() << '\n';
            }
        }
        if (const char* directory = std::getenv("MARKET_THESIS_DIR"); directory && *directory) {
            namespace fs = std::filesystem;
            std::error_code ec;
            fs::directory_iterator file(directory, ec);
            for (; !ec && file != fs::directory_iterator(); file.increment(ec)) {
                std::error_code status_error;
                if (!file->is_regular_file(status_error) || file->path().extension() != ".csv") continue;
                const auto id = file->path().stem().string();
                const auto split = id.find('_');
                if (split == 0 || split == std::string::npos || split + 1 == id.size()) continue;
                try {
                    feeds.try_emplace("thesis/" + id, file->path(),
                                      std::vector<std::string>{id.substr(0, split), id.substr(split + 1)});
                } catch (const std::exception& error) {
                    std::cerr << "Market thesis pair unavailable: " << error.what() << '\n';
                }
            }
            if (ec) std::cerr << "Market thesis directory unavailable: " << ec.message() << '\n';
        }
    }
    auto unsubscribe_market = [&](WS* ws) {
        for (auto& [name, feed] : feeds) feed.subscriptions.erase(ws);
    };
    app.get("/api/market/*", [](auto* res, auto*) { res->writeStatus("404 Not Found")->end(); });

    app.get("/*", [](auto *res, auto *req) {
        std::string path = std::string(req->getUrl());
        if (path == "/") path = "/index.html";
        namespace fs = std::filesystem;
        std::error_code ec;
        const auto root = fs::weakly_canonical(fs::absolute("client/dist"), ec);
        const auto filePath = fs::weakly_canonical(root / path.substr(1), ec);
        if (ec || !filePath.string().starts_with(root.string() + "/")) {
            res->writeStatus("404 Not Found")->end();
            return;
        }
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
        res->writeHeader("Content-Type", mime_type(filePath.extension().string()));
        res->end(buffer.str());
    });

    const std::vector<std::string> catalog_topics = {equity_topic(sym_x), equity_topic(sym_y)};
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

    auto sources = json::object();
    for (const auto& [name, feed] : feeds) {
        auto metadata = feed.broadcast.catalog();
        if (name.starts_with("thesis/")) {
            metadata["id"] = name.substr(7);
            sources["thesis"]["pairs"].push_back(std::move(metadata));
        } else {
            sources[name] = std::move(metadata);
        }
    }
    const auto catalog = json{{"type", "catalog"}, {"symbols", {sym_x, sym_y}}, {"market", std::move(sources)}}.dump();
    app.ws<PerSocket>("/*", {
        .open = [&catalog](auto* ws) {
            ws->send(catalog, uWS::OpCode::TEXT);
        },
        .message = [&](auto *ws, std::string_view message, uWS::OpCode) {
            json j = json::parse(message, nullptr, /*allow_exceptions=*/false);
            if (!j.contains("action") || !j["action"].is_string()) return;
            const std::string action = j["action"].get<std::string>();
            PerSocket* d = ws->getUserData();
            if (action == "market_unsubscribe") {
                unsubscribe_market(ws);
                return;
            }
            if (action == "market_subscribe") {
                const auto source = j.find("source");
                const auto request = j.find("requestId");
                if (source == j.end() || !source->is_string() || request == j.end() || !request->is_number_unsigned()) return;
                auto fail = [&](const char* message) {
                    ws->send(json{{"type", "market_error"}, {"requestId", *request}, {"message", message}}.dump(), uWS::OpCode::TEXT);
                };
                auto key = source->get<std::string>();
                if (key == "thesis") {
                    const auto id = j.find("pairId");
                    if (id == j.end() || !id->is_string()) {
                        fail("Choose a thesis pair.");
                        return;
                    }
                    key += "/" + id->get_ref<const std::string&>();
                } else if (key != "simulation" && key != "replay") {
                    fail("The selected source is unavailable.");
                    return;
                }
                const auto selected = feeds.find(key);
                if (selected == feeds.end()) {
                    fail("The selected source is unavailable.");
                    return;
                }
                const auto pair = j.find("pair");
                if (pair == j.end() || !pair->is_array() || pair->size() != 2
                    || !(*pair)[0].is_number_unsigned() || !(*pair)[1].is_number_unsigned()) {
                    fail("Choose two distinct assets.");
                    return;
                }
                const auto x = (*pair)[0].get<std::size_t>(), y = (*pair)[1].get<std::size_t>();
                auto& feed = selected->second;
                if (x == y || x >= feed.broadcast.symbol_count() || y >= feed.broadcast.symbol_count()) {
                    fail("Choose two distinct assets.");
                    return;
                }
                unsubscribe_market(ws);
                feed.subscriptions.try_emplace(ws, x, y, feed.broadcast.memories(), feed.broadcast.warmup());
                ws->send(json{{"type", "market_subscribed"}, {"requestId", *request}, {"pair", {x, y}}}.dump(), uWS::OpCode::TEXT);
                return;
            }
            if (action == "subscribe" || action == "unsubscribe") {
                if (!j.contains("topics") || !j["topics"].is_array()) return;
                for (const auto &t : j["topics"]) {
                    if (!t.is_string()) continue;
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
                c.quick_start = j.value("quick_start", c.quick_start);
                c.auto_events = j.value("auto", c.auto_events);
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
            else if (action == "rate") s.set_rate(j.value("rate", LabConfig::kDefaultRate));
            else if (action == "pause") s.set_paused(j.value("on", false));
            else if (action == "stop") end_session(d);
        },
        .close = [&](auto* ws, int, std::string_view) {
            unsubscribe_market(ws);
            end_session(ws->getUserData());
        }
    });

    bool listening = false;
    app.listen(cfg.port, [&](auto *token) {
        listening = token != nullptr;
        std::cout << (listening ? "server running on port " : "failed to bind port ")
                  << cfg.port << "\n";
    });
    if (!listening) return 1;

    std::thread producer_thread([&]() {
        while (wait_consumers > 0 && broker.ready_count(2) < std::size_t(wait_consumers))
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::size_t n = 0;
        auto sink = [&](const TickPair &event) {
            SymbolTick tick_x = event.x; tick_x.symbol = sym_x;
            SymbolTick tick_y = event.y; tick_y.symbol = sym_y;
            const std::string msg_x = serializer.to_json(tick_x);
            const std::string msg_y = serializer.to_json(tick_y);
            broker.publish(equity_topic(sym_x), msg_x);
            broker.publish(equity_topic(sym_y), msg_y);
            loop->defer([&app, tx = equity_topic(sym_x), ty = equity_topic(sym_y), msg_x, msg_y] {
                app.publish(tx, msg_x, uWS::OpCode::TEXT);
                app.publish(ty, msg_y, uWS::OpCode::TEXT);
            });
            if (++n % 1000 == 0) {
                std::cout << "events=" << n
                          << " tcp_consumers=" << broker.subscriber_count() << "\n";
            }
        };
        if (n_run > 0) {
            produce(sink, n_run + 1);   // + the opening tick
            std::cout << "done: " << n - 1 << " observations" << std::endl;
            broker.stop();       // closes the consumers' sockets after the queued bytes: they see end of stream
            std::_Exit(0);
        }
        produce(sink, -1);
    });

    if (!feeds.empty()) {
        const double market_rate = env_double("MARKET_RATE", market::kDefaultRate);
        const int interval = int(1000 / (std::isfinite(market_rate) ? std::clamp(market_rate, 0.5, kMaxBrowserRate) : market::kDefaultRate));
        auto* timer = us_create_timer(reinterpret_cast<us_loop_t*>(loop), 0, sizeof(MarketFeeds*));
        if (!timer) throw std::runtime_error("could not create market timer");
        *static_cast<MarketFeeds**>(us_timer_ext(timer)) = &feeds;
        us_timer_set(timer, [](us_timer_t* t) {
            auto& feeds = **static_cast<MarketFeeds**>(us_timer_ext(t));
            for (auto& [name, feed] : feeds) {
                const auto frame = feed.broadcast.next();
                for (auto it = feed.subscriptions.begin(); it != feed.subscriptions.end();) {
                    auto& [ws, subscription] = *it++;
                    ws->send(subscription.observe(frame).dump(), uWS::OpCode::TEXT);
                }
            }
        }, interval, interval);
    }
    app.run();
    producer_thread.join();
    broker.stop();
}
