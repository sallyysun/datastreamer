#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <kll_sketch.hpp>

// One marginal rank summary backed by Apache DataSketches' KLL (external/datasketches-cpp). Same interface as
// KllSketch. See archive/notes/studies/budget_mpl.md §10.
namespace ds_margin {

// Counts the heap the sketch actually holds, so store_bytes() is allocation, not serialised size.
struct Counter { std::size_t live = 0; };

template <typename T>
struct CountingAlloc {
    using value_type = T;
    std::shared_ptr<Counter> c;
    explicit CountingAlloc(std::shared_ptr<Counter> cc = std::make_shared<Counter>()) : c(std::move(cc)) {}
    template <typename U> CountingAlloc(const CountingAlloc<U>& o) : c(o.c) {}
    T* allocate(std::size_t n) { c->live += n * sizeof(T); return std::allocator<T>{}.allocate(n); }
    void deallocate(T* p, std::size_t n) { c->live -= n * sizeof(T); std::allocator<T>{}.deallocate(p, n); }
    template <typename U> bool operator==(const CountingAlloc<U>& o) const { return c == o.c; }
    template <typename U> bool operator!=(const CountingAlloc<U>& o) const { return c != o.c; }
};

// DataSketches flips its compaction coins with a thread_local generator seeded from the clock, and
// override_seed() does not reach it. Reseeding it here makes a run reproducible, because each
// estimator runs start to finish on one thread.
inline void seed(std::uint64_t s) { datasketches::random_utils::random_bit.seed(static_cast<std::uint32_t>(s)); }

template <typename Sketch>
class Margin {
public:
    template <typename... Args>
    explicit Margin(std::uint16_t k, Args... args)
        : counter_(std::make_shared<Counter>()),
          sk_(make(k, CountingAlloc<float>(counter_), args...)) {}

    // Weighted share of retained items strictly below x. get_rank() would rebuild a sorted view
    // after every update; one linear pass over the retained items is cheaper at one query per tick.
    double cdf(double x) const {
        const std::uint64_t n = sk_.get_n();
        if (n == 0) return 0.5;
        const float xf = static_cast<float>(x);
        std::uint64_t w = 0;
        for (auto it = sk_.begin(); it != sk_.end(); ++it) {
            const auto pr = *it;
            if (pr.first < xf) w += pr.second;
        }
        return static_cast<double>(w) / static_cast<double>(n);
    }
    void insert(double x) { sk_.update(static_cast<float>(x)); }
    std::size_t n() const { return sk_.get_n(); }
    std::size_t store_bytes() const { return sizeof(*this) + sizeof(Counter) + counter_->live; }

private:
    static Sketch make(std::uint16_t k, const CountingAlloc<float>& a) {
        return Sketch(k, std::less<float>(), a);
    }
    std::shared_ptr<Counter> counter_;
    Sketch sk_;
};

using Kll = Margin<datasketches::kll_sketch<float, std::less<float>, CountingAlloc<float>>>;

}  // namespace ds_margin
