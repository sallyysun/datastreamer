// reference_comparison harness -- drives the ACTUAL C++ estimator classes on a
// shared (x,y) CSV and prints per-step Spearman-rho estimates at full
// precision, so the output can be compared line-for-line against the reference
// implementations (Python onlineNPCORR, R hermiter).
//
// `#define private public` is a deliberate test-only hack so the harness can
// read the Xiao estimator's auto-calibrated cutpoints and hand the *identical*
// doubles to the Python reference -- removing cutpoint drift as a source of
// mismatch so only the estimator arithmetic is under test. It does not modify
// the shipped headers.
//
// Build (from repo root):
//   clang++ experiments/reference_comparison/harness.cpp -std=c++20 -O2 \
//     -Isrc -I$(brew --prefix eigen)/include/eigen3 \
//     -o experiments/reference_comparison/harness
//
// Usage:
//   harness xiao    <csv> [bins]  [calib] [window]   # window=0 => cumulative
//   harness hermite <csv> [order] [_]     [lambda]   # lambda=0 => cumulative
#define private public
#include "estimators/dependence/xiao_spearman.hpp"
#include "estimators/dependence/hermiter.hpp"
#undef private

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

static std::vector<std::pair<double, double>> read_csv(const std::string& path) {
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);  // header
    std::vector<std::pair<double, double>> rows;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string a, b;
        std::getline(ss, a, ',');
        std::getline(ss, b, ',');
        rows.emplace_back(std::stod(a), std::stod(b));
    }
    return rows;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <xiao|hermite> <csv> [order|bins] [calib] [window|lambda]\n",
                     argv[0]);
        return 1;
    }
    const std::string which = argv[1];
    const auto rows = read_csv(argv[2]);

    if (which == "xiao") {
        const std::size_t bins = (argc > 3) ? std::stoul(argv[3]) : 20;
        const std::size_t calib = (argc > 4) ? std::stoul(argv[4]) : 100;
        const std::size_t window = (argc > 5) ? std::stoul(argv[5]) : 0;
        XiaoSpearmanEstimator est(
            XiaoSpearmanEstimator::Config{.bins = bins, .window = window, .lambda = 0.0,
                                          .calibration = calib},
            "xiao");
        std::vector<double> out;
        out.reserve(rows.size());
        for (const auto& r : rows) {
            est.update(r.first, r.second);
            out.push_back(est.estimate());
        }
        // Dump the calibrated cutpoints to stderr for the Python reference.
        std::fprintf(stderr, "CUTS_X");
        for (double c : est.cuts_x_) std::fprintf(stderr, " %.17g", c);
        std::fprintf(stderr, "\nCUTS_Y");
        for (double c : est.cuts_y_) std::fprintf(stderr, " %.17g", c);
        std::fprintf(stderr, "\nCALIB %zu\n", calib);
        for (double v : out) std::printf("%.17g\n", v);
    } else if (which == "hermite") {
        const int order = (argc > 3) ? std::stoi(argv[3]) : 10;
        const double lambda = (argc > 5) ? std::stod(argv[5]) : 0.0;
        // burn_in=1 so estimate() emits from n>=2, matching R hermiter's range.
        HermiteEstimator est(
            HermiteEstimator::Config{.order = order, .lambda = lambda, .burn_in = 1,
                                     .init_first_obs = false},
            "hermite");
        for (const auto& r : rows) {
            est.update(r.first, r.second);
            std::printf("%.17g\n", est.estimate());
        }
    } else {
        std::fprintf(stderr, "unknown estimator '%s'\n", which.c_str());
        return 1;
    }
    return 0;
}
