#pragma once

// Each browser lab session owns a steerable stream and a consumer process.
// The consumer returns estimates on its device.<id> topic.

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <cstdio>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "study_config.hpp"
#include "../../protocol/serializer.hpp"
#include "../datasource/dependence.hpp"
#include "../datasource/synthetic_source.hpp"
#include "broker.hpp"

struct LabConfig {
    std::string copula = "gaussian";
    double rho = 0.5;                  // starting rho_S
    double rate = 100.0;               // observations per second
    std::string estimator = "rmpl";    // MFF arms: rmpl | xiao | hermite
    std::string memories = thesis::kMemoryCsv;
    std::string families = "gaussian"; // working families of the MFF models
    double eta = 1.0, delta = 1e-4;
};

class LabSession {
public:
    // notify: called with a one-line JSON event ({"type":"event","t":obs,"label":...}) when the dependence changes.
    LabSession(int id, LabConfig cfg, Broker& broker, int broker_port, std::string consumer,
               std::function<void(std::string)> notify)
        : notify_(std::move(notify)), id_(id), cfg_(std::move(cfg)), broker_(broker), sx_("S" + std::to_string(id) + "X"),
          sy_("S" + std::to_string(id) + "Y"), rng_(std::random_device{}()) {
        auto dep = std::make_unique<ControlledDependence>(cfg_.rho);
        dep_ = dep.get();
        source_ = std::make_unique<SyntheticSource>(std::move(dep), dgp::make_copula(cfg_.copula), rng_(), cfg_.copula);
        rate_ = cfg_.rate;
        spawn(broker_port, consumer);
        thread_ = std::thread([this] { run(); });
    }
    ~LabSession() {
        running_ = false;
        if (thread_.joinable()) thread_.join();
        if (pid_ > 0) { ::kill(pid_, SIGTERM); int st = 0; ::waitpid(pid_, &st, 0); }
    }

    void jump(double rho) { std::lock_guard<std::mutex> l(mu_); do_jump(rho); }
    void drift(double rho, std::size_t steps) { std::lock_guard<std::mutex> l(mu_); do_drift(rho, steps); }
    // Family switches only for a family MFF (several working families): a single-family MFF has no family weights to move.
    bool family_mff() const { return cfg_.families.find(',') != std::string::npos; }
    void family(const std::string& f) { if (!family_mff()) return; std::lock_guard<std::mutex> l(mu_); do_family(f); }
    void set_rate(double r) { rate_ = std::clamp(r, 1.0, 2000.0); }
    void set_auto(bool on) { auto_ = on; }
    void set_paused(bool p) { paused_ = p; }

private:
    // Under mu_. The event takes effect from the next observation, whose index (from 0) is n_ - 1: the opening tick
    // carries no observation.
    void event(const std::string& label) {
        if (notify_) notify_("{\"type\":\"event\",\"t\":" + std::to_string(n_ > 0 ? n_ - 1 : 0) + ",\"label\":\"" + label + "\"}");
    }
    static std::string fmt(double x) { char b[16]; std::snprintf(b, sizeof b, "%.2f", x); return b; }
    void do_jump(double rho) { dep_->jump(rho); event("jump " + fmt(rho)); }
    void do_drift(double rho, std::size_t steps) { dep_->drift(rho, steps); event("drift to " + fmt(rho)); }
    void do_family(const std::string& f) { source_->set_family(f); event(f); }

    void spawn(int broker_port, const std::string& consumer) {
        const std::size_t every = std::max<std::size_t>(1, static_cast<std::size_t>(cfg_.rate / 25.0));
        std::vector<std::string> a = {consumer, "--port", std::to_string(broker_port), "--symbols", sx_ + "," + sy_,
                                      "--print", "0", "--live", std::to_string(id_), "--estimator", cfg_.estimator,
                                      "--memories", cfg_.memories, "--families", cfg_.families,
                                      "--eta", std::to_string(cfg_.eta), "--delta", std::to_string(cfg_.delta),
                                      "--publish-every", std::to_string(every)};
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

    // Automatic events: every 400-1500 observations, a jump, a drift or (family MFF only) a family switch, at random.
    void auto_event() {
        std::uniform_real_distribution<double> r(0.1, 0.85), u(0, 1);
        std::uniform_int_distribution<int> gap(400, 1500), len(300, 1500);
        const double e = u(rng_);
        const double cut = family_mff() ? 0.75 : 1.0;   // family switches only for a family MFF
        if (e < 0.4 * cut) do_jump(r(rng_));
        else if (e < cut) do_drift(r(rng_), std::size_t(len(rng_)));
        else {
            static const char* fams[4] = {"gaussian", "t", "clayton", "gumbel"};
            do_family(fams[std::uniform_int_distribution<int>(0, 3)(rng_)]);
        }
        next_auto_ = n_ + std::size_t(gap(rng_));
    }

    void run() {
        const std::string tx = "trades.equity." + sx_, ty = "trades.equity." + sy_;
        while (running_ && !(broker_.subscribed(tx) && broker_.subscribed(ty)))   // wait for the session's consumer
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        auto next_time = std::chrono::steady_clock::now();
        while (running_) {
            if (paused_) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); next_time = std::chrono::steady_clock::now(); continue; }
            TickPair e;
            {
                std::lock_guard<std::mutex> l(mu_);
                if (auto_ && n_ >= next_auto_) auto_event();
                e = source_->next();
                ++n_;
            }
            e.x.symbol = sx_; e.y.symbol = sy_;
            broker_.publish(tx, Serializer::to_json(e.x));
            broker_.publish(ty, Serializer::to_json(e.y));
            next_time += std::chrono::microseconds(static_cast<long>(1e6 / rate_.load()));
            std::this_thread::sleep_until(next_time);
        }
    }

    std::function<void(std::string)> notify_;
    int id_;
    LabConfig cfg_;
    Broker& broker_;
    std::string sx_, sy_;
    std::mt19937 rng_;
    std::mutex mu_;
    ControlledDependence* dep_ = nullptr;
    std::unique_ptr<SyntheticSource> source_;
    std::atomic<bool> running_{true}, auto_{false}, paused_{false};
    std::atomic<double> rate_{100.0};
    std::size_t n_ = 0, next_auto_ = 0;
    pid_t pid_ = -1;
    std::thread thread_;
};
