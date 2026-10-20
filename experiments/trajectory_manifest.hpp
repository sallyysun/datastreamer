#pragma once

// Opt-in bit-exact refactor gate for probes that build estimators outside registry.hpp.
// TRAJ_MANIFEST=<file>: recorded double series are FNV-1a hashed and written sorted by tag
// at exit as "<tag> n=<count> h=<hex>". Unset, every entry point is a single branch.

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace traj {

struct Fnv {
    std::uint64_t h = 1469598103934665603ull;
    std::size_t n = 0;
    void add(double v) {
        const auto* b = reinterpret_cast<const unsigned char*>(&v);
        for (std::size_t i = 0; i < sizeof v; ++i) { h ^= b[i]; h *= 1099511628211ull; }
        ++n;
    }
};

class Manifest {
public:
    static Manifest& get() {
        static Manifest m;
        return m;
    }
    bool on() const { return !path_.empty(); }
    void put(const std::string& tag, const Fnv& f) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!rows_.emplace(tag, f).second) {
            // abort, not exit: exit would run ~Manifest on this thread while mu_ is held.
            std::fprintf(stderr, "trajectory_manifest: duplicate tag '%s'\n", tag.c_str());
            std::abort();
        }
    }
    void flush() {
        std::lock_guard<std::mutex> lk(mu_);
        if (!on()) return;
        std::FILE* f = std::fopen(path_.c_str(), "w");
        if (!f) {
            std::fprintf(stderr, "trajectory_manifest: cannot write '%s'\n", path_.c_str());
            return;
        }
        for (const auto& [tag, v] : rows_)
            std::fprintf(f, "%s n=%zu h=%016llx\n", tag.c_str(), v.n, static_cast<unsigned long long>(v.h));
        std::fclose(f);
    }
    ~Manifest() { flush(); }

private:
    Manifest() {
        if (const char* p = std::getenv("TRAJ_MANIFEST")) path_ = p;
    }
    std::string path_;
    std::mutex mu_;
    std::map<std::string, Fnv> rows_;
};

inline bool on() { return Manifest::get().on(); }
inline void flush() { Manifest::get().flush(); }

// Incremental: hash as values are produced, commit once under one tag.
class Series {
public:
    Series() : on_(on()) {}
    void add(double v) {
        if (on_) f_.add(v);
    }
    void add(const std::vector<double>& v) {
        if (on_)
            for (double x : v) f_.add(x);
    }
    void commit(const std::string& tag) {
        if (on_) Manifest::get().put(tag, f_);
    }

private:
    bool on_;
    Fnv f_;
};

inline void record(const std::string& tag, const double* p, std::size_t n) {
    if (!on()) return;
    Fnv f;
    for (std::size_t i = 0; i < n; ++i) f.add(p[i]);
    Manifest::get().put(tag, f);
}
inline void record(const std::string& tag, const std::vector<double>& v) { record(tag, v.data(), v.size()); }

inline std::string tag(const char* fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    return buf;
}

}  // namespace traj
