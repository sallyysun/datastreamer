#pragma once

// Each browser lab session owns a steerable stream and a consumer process.
// The consumer returns estimates on its device.<id> topic.

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <cstdio>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "study_config.hpp"
#include "../../protocol.hpp"
#include "../datasource/dependence.hpp"
#include "../datasource/synthetic_source.hpp"
#include "broker.hpp"
#include "demo_config.hpp"

struct LabConfig {
    std::string copula = "gaussian";
    double rho = 0.5;                  // starting rho_S
    double rate = 0;
    bool quick_start = false;
    bool auto_events = false;
    std::string estimator = "rmpl";    // MFF arms: rmpl | xiao | hermite
    std::string memories = thesis::kMemoryCsv;
    std::string families = "gaussian"; // working families of the MFF models
    double eta = 0, delta = 0;
    GuidedEvents guided;
    DemoConfig demo;
};

class LabSession {
public:
    // notify: called with a one-line JSON event ({"type":"event","t":obs,"label":...}) when the dependence changes.
    LabSession(int id, LabConfig cfg, Broker& broker, int broker_port, std::string consumer,
               std::function<void(std::string)> notify)
        : notify_(std::move(notify)), id_(id), cfg_(std::move(cfg)), broker_(broker), sx_("S" + std::to_string(id) + "X"),
          sy_("S" + std::to_string(id) + "Y"), auto_(cfg_.auto_events) {
        auto dep = std::make_unique<ControlledDependence>(cfg_.rho);
        dep_ = dep.get();
        const unsigned seed = cfg_.auto_events ? kGuidedSeed : std::random_device{}();
        source_ = std::make_unique<SyntheticSource>(std::move(dep), dgp::make_copula(cfg_.copula), seed, cfg_.copula);
        cfg_.rate = clamp_rate(cfg_.rate);
        rate_ = cfg_.rate;
        spawn(broker_port, consumer);
        thread_ = std::thread([this] { run(); });
    }
    ~LabSession() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            running_ = false;
        }
        pacing_.notify_one();
        if (thread_.joinable()) thread_.join();
        if (pid_ > 0) { ::kill(pid_, SIGTERM); int st = 0; ::waitpid(pid_, &st, 0); }
    }

    void jump(double rho) { std::lock_guard<std::mutex> l(mu_); do_jump(rho); }
    void drift(double rho, std::size_t steps) { std::lock_guard<std::mutex> l(mu_); do_drift(rho, steps); }
    // Family switches only for a family MFF (several working families): a single-family MFF has no family weights to move.
    bool family_mff() const { return cfg_.families.find(',') != std::string::npos; }
    void family(const std::string& f) { if (!family_mff()) return; std::lock_guard<std::mutex> l(mu_); do_family(f); }
    void set_rate(double r) {
        std::lock_guard<std::mutex> lock(mu_);
        rate_ = clamp_rate(r);
        pacing_.notify_one();
    }
    void set_auto(bool on) { auto_ = on; }
    void set_paused(bool p) {
        std::lock_guard<std::mutex> lock(mu_);
        paused_ = p;
        pacing_.notify_one();
    }

private:
    static constexpr unsigned kGuidedSeed = 42;

    double clamp_rate(double rate) const { return cfg_.demo.clamp_rate(rate); }

    bool bursting() const { return cfg_.quick_start && n_ <= cfg_.demo.warmup_observations; }

    // n_ includes the opening tick, so it is also the one-based index of the next return.
    void event(const std::string& label) {
        if (notify_) notify_(nlohmann::json{{"type", "event"}, {"t", std::max<std::size_t>(n_, 1)}, {"label", label}}.dump());
    }
    static std::string number(double x) { char b[32]; std::snprintf(b, sizeof b, "%.10g", x); return b; }
    static std::string fmt(double x) { char b[16]; std::snprintf(b, sizeof b, "%.2f", x); return b; }
    void do_jump(double rho) {
        const double from = dep_->current();
        dep_->jump(rho);
        event("Jump from " + fmt(from) + " to " + fmt(dep_->current()));
    }
    void do_drift(double rho, std::size_t steps) {
        const auto duration = std::max<std::size_t>(steps, 1);
        dep_->drift(rho, duration);
        event("Drift to " + fmt(ControlledDependence::clamp(rho)) + " over " + std::to_string(duration) + " observations");
    }
    void do_family(const std::string& f) {
        source_->set_family(f);
        const std::string label = f == "t" ? "Student t₄" : f;
        event("Family switch to " + label);
    }

    void spawn(int broker_port, const std::string& consumer) {
        std::vector<std::string> a = {consumer, "--port", std::to_string(broker_port), "--symbols", sx_ + "," + sy_,
                                      "--print", "0", "--live", std::to_string(id_), "--estimator", cfg_.estimator,
                                      "--memories", cfg_.memories, "--families", cfg_.families,
                                      "--eta", number(cfg_.eta), "--delta", number(cfg_.delta)};
        if (cfg_.quick_start && cfg_.estimator == "rmpl") a.insert(a.end(), {"--fast-warmup", "1"});
        if (cfg_.auto_events) a.insert(a.end(), {"--seed", std::to_string(kGuidedSeed)});
        pid_ = ::fork();
        if (pid_ == 0) {
            std::vector<char*> argv;
            for (auto& s : a) argv.push_back(s.data());
            argv.push_back(nullptr);
            if (std::FILE* f = std::freopen("/dev/null", "w", stdout)) (void)f;
            ::execv(consumer.c_str(), argv.data());
            std::_Exit(127);
        }
    }

    void guided_event() {
        if (n_ == cfg_.guided.jump_at) do_jump(cfg_.guided.jump_to);
        if (n_ == cfg_.guided.drift_at) do_drift(cfg_.guided.drift_to, cfg_.guided.drift_steps);
    }

    void run() {
        const std::string tx = equity_topic(sx_), ty = equity_topic(sy_);
        while (running_ && !(broker_.subscribed(tx) && broker_.subscribed(ty)))   // wait for the session's consumer
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        while (running_) {
            TickPair e;
            double rate;
            {
                std::unique_lock<std::mutex> lock(mu_);
                pacing_.wait(lock, [&] { return !paused_ || !running_; });
                if (!running_) break;
                if (auto_) guided_event();
                e = source_->next();
                ++n_;
                rate = rate_;
            }
            e.x.symbol = sx_; e.y.symbol = sy_;
            broker_.publish(tx, Serializer::to_json(e.x));
            broker_.publish(ty, Serializer::to_json(e.y));
            std::unique_lock<std::mutex> lock(mu_);
            const auto sent = std::chrono::steady_clock::now();
            const auto gap = bursting() ? std::chrono::microseconds(static_cast<long>(1e6 / cfg_.demo.warmup_rate()))
                                        : std::chrono::microseconds(static_cast<long>(1e6 / rate));
            pacing_.wait_until(lock, sent + gap, [&] { return !running_ || paused_ || (!bursting() && rate_ != rate); });
        }
    }

    std::function<void(std::string)> notify_;
    int id_;
    LabConfig cfg_;
    Broker& broker_;
    std::string sx_, sy_;
    std::mutex mu_;
    std::condition_variable pacing_;
    ControlledDependence* dep_ = nullptr;
    std::unique_ptr<SyntheticSource> source_;
    std::atomic<bool> running_{true}, auto_{false};
    bool paused_ = false;
    double rate_ = 0;
    std::size_t n_ = 0;
    pid_t pid_ = -1;
    std::thread thread_;
};
