// Does each DGP family actually achieve the Spearman rho it reports? A family
// whose reported truth is wrong makes every RMSE against it meaningless.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"

namespace {
double spearman(const std::vector<double>& x, const std::vector<double>& y) {
    const std::size_t n = x.size();
    std::vector<std::size_t> ix(n), iy(n);
    std::iota(ix.begin(), ix.end(), 0); std::iota(iy.begin(), iy.end(), 0);
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
}  // namespace

int main() {
    const std::vector<std::string> fams =
        {"gaussian", "clayton", "gumbel", "t", "frechet", "mixgauss"};
    const std::vector<double> ds = {0.2, 0.6, 0.9};
    const std::size_t N = 200000;
    std::printf("%-10s", "family");
    for (double d : ds) std::printf("   d=%.1f: reported  achieved   err", d);
    std::printf("\n");
    for (const std::string& f : fams) {
        std::printf("%-10s", f.c_str());
        for (double d : ds) {
            ConstantDependence dep(d);
            auto cop = dgp::make_copula(f);
            CopulaSource src(dep, *cop, 5);
            std::vector<double> xs, ys; double rep = 0;
            for (std::size_t t = 0; t < N; ++t) {
                auto s = src.next(); xs.push_back(s.x); ys.push_back(s.y); rep += s.rho_s;
            }
            rep /= N;
            const double ach = spearman(xs, ys);
            std::printf("   %8.4f  %8.4f %+6.4f", rep, ach, ach - rep);
        }
        std::printf("\n");
    }
    return 0;
}
