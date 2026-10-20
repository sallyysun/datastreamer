// Focused ARL0 sweep for the tail detector only -- the full calibrate_arl0 run
// re-measures every other detector and takes ten minutes.
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "../src/estimators/dependence/change_detector.hpp"
#include "../src/estimators/dependence/tail_cusum.hpp"
#include "../src/estimators/registry.hpp"

int main() {
    const long ticks = 60000;
    const int seeds = 4;
    const std::vector<double> rhos = {0.0, 0.3, 0.6, 0.8};
    const std::vector<double> hs = {2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0};
    for (double rate : {0.002, 0.0005}) {
        std::printf("=== tail_cusum, count_rate = %.4f ===\n", rate);
        for (double h : hs) {
            double worst = 1e18; double worst_rho = 0;
            for (double rho : rhos) {
                long alarms = 0;
                for (int s = 0; s < seeds; ++s) {
                    TailCusumDetector det(make_estimator("xiao_ew"),
                        TailCusumDetector::Config{.count_rate = rate, .L = h}, "p");
                    std::mt19937 rng(500 + s);
                    std::normal_distribution<double> z(0, 1);
                    for (long t = 0; t < ticks; ++t) {
                        const double zx = z(rng);
                        const double zy = rho * zx + std::sqrt(1 - rho * rho) * z(rng);
                        det.update(zx, zy);
                        if (det.alarmed_this_block()) ++alarms;
                    }
                }
                const double blocks = (double(ticks) / 20.0) * seeds;
                const double arl = alarms ? blocks / alarms : blocks;
                if (arl < worst) { worst = arl; worst_rho = rho; }
            }
            std::printf("  L=%6.1f  worst ARL0 = %8.0f blocks (rho=%.1f)%s\n",
                        h, worst, worst_rho, worst >= 1000 ? "  <- meets target" : "");
        }
    }
    return 0;
}
