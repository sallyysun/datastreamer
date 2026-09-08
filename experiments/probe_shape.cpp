// Validity check for the `shape` scenario: is Spearman's rho really flat across
// the switch, and does the tail dependence really change?
//
// If rho_S steps at the switch the experiment is void -- a rho-monitoring
// detector could see it, and the whole point is that it cannot.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numeric>
#include <vector>

#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"

namespace {
double spearman(const std::vector<double>& x, const std::vector<double>& y) {
    const std::size_t n = x.size();
    std::vector<std::size_t> ix(n), iy(n);
    std::iota(ix.begin(), ix.end(), 0);
    std::iota(iy.begin(), iy.end(), 0);
    std::sort(ix.begin(), ix.end(), [&](std::size_t a, std::size_t b){ return x[a] < x[b]; });
    std::sort(iy.begin(), iy.end(), [&](std::size_t a, std::size_t b){ return y[a] < y[b]; });
    std::vector<double> rx(n), ry(n);
    for (std::size_t i = 0; i < n; ++i) { rx[ix[i]] = double(i); ry[iy[i]] = double(i); }
    const double m = (n - 1) / 2.0;
    double c = 0, vx = 0, vy = 0;
    for (std::size_t i = 0; i < n; ++i) {
        c += (rx[i]-m)*(ry[i]-m); vx += (rx[i]-m)*(rx[i]-m); vy += (ry[i]-m)*(ry[i]-m);
    }
    return c / std::sqrt(vx*vy);
}
// P(Y below its q-quantile | X below its q-quantile) -- the empirical lower tail.
double tail(const std::vector<double>& x, const std::vector<double>& y, double q) {
    std::vector<double> sx = x, sy = y;
    std::sort(sx.begin(), sx.end()); std::sort(sy.begin(), sy.end());
    const double cx = sx[std::size_t(q * x.size())], cy = sy[std::size_t(q * y.size())];
    std::size_t both = 0, one = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x[i] < cx) { ++one; if (y[i] < cy) ++both; }
    }
    return one ? double(both) / double(one) : 0.0;
}
}  // namespace

int main() {
    // Equal windows either side, so the two tail estimates carry the same noise.
    const std::size_t N = 200000, sw = N / 2;
    ConstantDependence dep(0.6);
    auto cop = std::unique_ptr<dgp::Copula>(new dgp::SwitchingCopula(
        std::make_unique<dgp::GaussianCopula>(), std::make_unique<dgp::StudentTCopula>(4.0),
        sw, "gauss2t"));
    CopulaSource src(dep, *cop, 11);

    std::vector<double> bx, by, ax, ay;
    double truth_before = 0, truth_after = 0;
    std::size_t nb = 0, na = 0;
    for (std::size_t t = 0; t < N; ++t) {
        const CopulaSource::Sample s = src.next();
        if (t < sw) { bx.push_back(s.x); by.push_back(s.y); truth_before += s.rho_s; ++nb; }
        else        { ax.push_back(s.x); ay.push_back(s.y); truth_after  += s.rho_s; ++na; }
    }
    std::printf("                       before switch   after switch\n");
    std::printf("reported true rho_s    %13.5f  %13.5f   <- must match\n",
                truth_before/nb, truth_after/na);
    std::printf("empirical Spearman     %13.4f  %13.4f\n", spearman(bx,by), spearman(ax,ay));
    std::printf("lower tail P(Y|X) q=.05%13.3f  %13.3f   <- must differ\n",
                tail(bx,by,0.05), tail(ax,ay,0.05));
    std::printf("lower tail P(Y|X) q=.01%13.3f  %13.3f\n", tail(bx,by,0.01), tail(ax,ay,0.01));
    return 0;
}
