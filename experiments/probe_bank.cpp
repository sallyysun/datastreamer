// What does the bank's posterior over sigma_omega2 actually do at a jump?
#include <cstdio>
#include <memory>
#include "../src/estimators/dependence/kalman_bank.hpp"
#include "../src/estimators/registry.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"

int main() {
    for (const char* k : {"kalman_fast", "kalman_vff", "kalman_bank_sticky",
                          "kalman_bocpd", "xiao_ew"}) {
        auto e = make_estimator(k);
        for (int i = 0; i < 3000; ++i) e->update(0.1 * i, 0.07 * i);
        std::printf("%-20s %6zu bytes\n", k, e->store_bytes());
    }
    std::printf("\nposterior-mean sigma_omega2 around a jump at t=2500 (seed 42)\n");
    JumpDependence dep(0.2, 0.8, 2500);
    auto cop = dgp::make_copula("gaussian");
    CopulaSource src(dep, *cop, 42);
    KalmanBank bank(KalmanBank::Config{.switch_prob = 0.002}, "b");
    for (std::size_t t = 0; t < 4000; ++t) {
        const auto s = src.next();
        bank.update(s.x, s.y);
        if (t % 200 == 0 || (t > 2400 && t < 2800 && t % 40 == 0)) {
            std::printf("  t=%4zu  q_hat=%9.5f   rho_hat=%6.3f  (true %5.3f)\n",
                        t, bank.effective_sigma_omega2(), bank.estimate(), s.rho_s);
        }
    }
    return 0;
}
