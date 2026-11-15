// Diagnostic only (Section 6.2 audit, 2026-11-15): decomposes store_bytes() of the E1 estimators after 5000 Gaussian draws.
// Build (from experiments/): g++ -std=c++20 -O2 -w -I../src -I/usr/include/eigen3 $(DS_INC) diag_mem_breakdown.cpp -o diag_mem_breakdown
// Run: ./diag_mem_breakdown > results/budget_mem_breakdown.csv
#include <cstdio>
#include <random>
#include <vector>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <memory>
#include <deque>
#include <functional>
#include <Eigen/Dense>
#include <kll_sketch.hpp>
#include <req_sketch.hpp>
#define private public
#define protected public
#include "estimators/dependence/recursive_mpl.hpp"
#include "estimators/dependence/xiao_spearman.hpp"
#include "estimators/dependence/hermiter.hpp"
#undef private
#undef protected
int main() {
    std::mt19937_64 g(42); std::normal_distribution<double> n;
    std::vector<double> x(5000), y(5000);
    for (int t = 0; t < 5000; ++t) { double a = n(g), b = n(g); x[t] = a; y[t] = 0.6 * a + 0.8 * b; }
    std::printf("est,approx,gain,total_peak,object,margin_x,margin_y,unused_heap,other_heap\n");
    for (double gain : {0.0, 0.005}) for (int k : {8, 16, 32, 64, 128, 256, 512, 1024}) {
        RecursiveMPL::Config c{}; c.gain = gain > 0 ? gain : 1e-12; c.kll_k = k; c.seed = 42; c.margin = RecursiveMPL::Margin::DsKll;
        RecursiveMPL m(c, "mpl"); std::size_t peak = 0, mx = 0, my = 0;
        for (int t = 0; t < 5000; ++t) { m.update(x[t], y[t]); if (m.store_bytes() > peak) { peak = m.store_bytes(); mx = m.dx_->store_bytes(); my = m.dy_->store_bytes(); } }
        std::size_t unused = m.kx_.heap_bytes() + m.ky_.heap_bytes() + m.px_.heap_bytes() + m.py_.heap_bytes() + m.ex_.heap_bytes() + m.ey_.heap_bytes();
        std::printf("mpl_dskll,%d,%g,%zu,%zu,%zu,%zu,%zu,%zu\n", k, gain, peak, sizeof(m), mx, my, unused, peak - sizeof(m) - mx - my - unused);
    }
    for (double gain : {0.0, 0.005}) for (int b : {10, 20, 30, 45, 65, 90, 130, 180}) {
        XiaoSpearmanEstimator::Config c{}; c.bins = b; c.lambda = gain > 0 ? 1 - gain : 0.0;
        XiaoSpearmanEstimator e(c, "xiao"); std::size_t peak = 0;
        for (int t = 0; t < 5000; ++t) { e.update(x[t], y[t]); peak = std::max(peak, e.store_bytes()); }
        std::size_t cnt = (e.counts_.capacity() + e.row_counts_.capacity() + e.col_counts_.capacity()) * 8, cuts = (e.cuts_x_.capacity() + e.cuts_y_.capacity()) * 8,
                    cal = e.calibration_.capacity() * 16, pits = e.pit_x_.heap_bytes() + e.pit_y_.heap_bytes() + e.vff_m_x_.heap_bytes() + e.vff_m_y_.heap_bytes();
        std::printf("xiao,%d,%g,%zu,%zu,counts=%zu,cuts=%zu,calib=%zu,pits=%zu\n", b, gain, peak, sizeof(e), cnt, cuts, cal, pits);
    }
    for (int N : {5, 10, 15, 20, 30, 43, 60, 89, 128}) {
        HermiteEstimator::Config c{}; c.order = N; c.lambda = 0.0;
        HermiteEstimator h(c, "hermite"); std::size_t peak = 0;
        for (int t = 0; t < 5000; ++t) { h.update(x[t], y[t]); peak = std::max(peak, h.store_bytes()); }
        const auto s = [](const auto& m) { return static_cast<std::size_t>(m.size()) * 8; };
        std::printf("hermite,%d,0,%zu,%zu,A=%zu,W=%zu,ax+ay+z=%zu,Afast=%zu,pits=%zu\n", N, peak, sizeof(h), s(h.A_), s(h.W_), s(h.ax_) + s(h.ay_) + s(h.z_), s(h.A_fast_), h.pit_x_.heap_bytes() + h.pit_y_.heap_bytes());
    }
}
