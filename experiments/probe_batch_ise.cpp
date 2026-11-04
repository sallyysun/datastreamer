// Copula ISE and offline batch comparators for the Chapter 6 banks, on the same streams as
// probe_family_bank (same seeds, scenarios and DGP, so results pair by seed).
//
// ISE: e_t = int_{[0,1]^2} (C_hat_t - C_t)^2 du dv, Gauss-Legendre 12 x 12, every kEvery ticks, for
//   family_bank        its mixture CDF sum_k w_k C_k (exact per arm: closed form for Clayton/Gumbel and their
//                      rotations, a tabulated 1-D quadrature for Gaussian/t), and its 24 fixed arms;
//   *_bank_log         the Gaussian copula at the bank's rho (its method-of-moments working model);
//   *_bank_own         its predictive density (mixture of the arms' own densities) integrated to a CDF on a
//                      32 x 32 midpoint grid and normalised to total mass 1.
// Offline batch estimators (use the whole sample, future included; margins by global ranks):
//   batch_full         sample Spearman / empirical copula of all T ticks;
//   batch_segment      the same over the true regime segment containing t (oracle: knows the change point);
//   batch_kernel_<h>   two-sided Epanechnikov-weighted Spearman / empirical copula centred on t, bandwidth h.
//                      The table picks the best h in hindsight per cell.
// rho_S RMSE is scored at every tick (online estimators: the estimate held before the tick); ISE on the
// evaluation ticks. Note: notes/studies/thesis_bank_ise_batch.md.
//
// Usage (from experiments/):  ./probe_batch_ise <first_seed> <seeds>   (COPULA as probe_family_bank)
//   -> results/batch_ise_<first_seed>[_<family>].csv   copula,scenario,seed,window,method,metric,value

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "../src/estimators/registry.hpp"
#include "../src/streamgenerator/datasource/copula.hpp"
#include "../src/streamgenerator/datasource/dependence.hpp"
#include "copula_source.hpp"

namespace {

constexpr std::size_t kN = 5000, kBurn = 500, kChange = 2500, kEvery = 50;
constexpr int kG = 12, kD = 32, kRho = 400;
constexpr double kNu = 4.0;
const std::vector<std::string> kBanks{"xiao_bank_log", "mle_bank_log", "xiao_bank_own", "hermite_bank_own"};
const std::vector<int> kH{50, 100, 200, 400, 800, 1600};
const char* kShapeName[] = {"gaussian", "t", "clayton", "clayton90", "gumbel", "gumbel90", "clayton270", "gumbel270"};

std::unique_ptr<Dependence> make_dep(const std::string& s) {
    if (s == "constant" || s == "shape") return std::make_unique<ConstantDependence>(0.6);
    if (s == "drift") return std::make_unique<DriftDependence>(-0.5, 0.8, kN);
    if (s == "jump") return std::make_unique<JumpDependence>(0.2, 0.8, kChange);
    return std::make_unique<OscillatingDependence>(0.6, 400.0, 0.0);
}

copula::Shape shape_of(const std::string& f) {
    if (f == "clayton") return copula::Shape::Clayton;
    if (f == "gumbel") return copula::Shape::Gumbel;
    if (f == "t") return copula::Shape::StudentT;
    return copula::Shape::Gaussian;
}

struct Win { const char* name; std::size_t a, b; };
std::vector<Win> windows(const std::string& s) {
    std::vector<Win> w{{"whole", kBurn, kN}};
    if (s == "jump" || s == "shape") {
        w.push_back({"pre", kBurn, kChange});
        w.push_back({"rec_0_100", kChange, kChange + 100});
        w.push_back({"rec_100_500", kChange + 100, kChange + 500});
        w.push_back({"rec_500_1500", kChange + 500, kChange + 1500});
        w.push_back({"post", kChange, kN});
    }
    return w;
}

// The ISE grid and the copula CDFs on it.
struct Grid {
    std::vector<double> x, w;   // nodes and weights on [0,1]
    // Gaussian / t: C(x_i, x_j; r) on a grid of r, built once from the conditional-CDF quadrature with the
    // quantiles of the fixed inner nodes cached (copula::detail::elliptical_cdf would redo them per call).
    std::array<std::vector<double>, 2> tab;   // [gauss, t][(i*kG + j)*(kRho+1) + k]
    Grid() {
        copula::GaussLegendre gl(kG), in(64);
        x = gl.x;
        w = gl.w;
        for (int e = 0; e < 2; ++e) {
            const bool t = e == 1;
            auto ppf = [&](double p) { return t ? special::t_ppf(p, kNu) : special::normal_ppf(p); };
            std::vector<double> yq(kG);
            std::vector<std::vector<double>> xq(kG, std::vector<double>(in.x.size()));
            for (int i = 0; i < kG; ++i) {
                yq[i] = ppf(x[i]);
                for (std::size_t k = 0; k < in.x.size(); ++k) xq[i][k] = ppf(x[i] * in.x[k]);
            }
            tab[e].assign(kG * kG * (kRho + 1), 0.0);
            for (int i = 0; i < kG; ++i)
                for (int j = 0; j < kG; ++j)
                    for (int k = 0; k <= kRho; ++k) {
                        const double r = -0.999 + 1.998 * k / kRho, s = std::sqrt(1 - r * r);
                        double tot = 0;
                        for (std::size_t m = 0; m < in.x.size(); ++m) {
                            const double xx = xq[i][m];
                            tot += in.w[m] * (t ? special::t_cdf((yq[j] - r * xx) / (std::sqrt((kNu + xx * xx) / (kNu + 1)) * s), kNu + 1)
                                                : special::normal_cdf((yq[j] - r * xx) / s));
                        }
                        tab[e][(i * kG + j) * (kRho + 1) + k] = x[i] * tot;
                    }
        }
    }
    double ellip(bool t, int i, int j, double r) const {
        const double z = (std::clamp(r, -0.999, 0.999) + 0.999) / 1.998 * kRho;
        const int k = std::min(int(z), kRho - 1);
        const double* p = &tab[t][(i * kG + j) * (kRho + 1)];
        return p[k] + (z - k) * (p[k + 1] - p[k]);
    }
    // C of a shape at parameter par, at node (i, j).
    double C(copula::Shape s, double par, int i, int j) const {
        using copula::Shape;
        const double a = x[i], b = x[j];
        switch (s) {
            case Shape::Gaussian: return ellip(false, i, j, par);
            case Shape::StudentT: return ellip(true, i, j, par);
            case Shape::Clayton: return copula::Clayton::cdf(a, b, par);
            case Shape::Gumbel: return copula::Gumbel::cdf(a, b, par);
            case Shape::Clayton90: return b - copula::Clayton::cdf(1 - a, b, par);
            case Shape::Gumbel90: return b - copula::Gumbel::cdf(1 - a, b, par);
            case Shape::Clayton270: return a - copula::Clayton::cdf(a, 1 - b, par);
            case Shape::Gumbel270: default: return a - copula::Gumbel::cdf(a, 1 - b, par);
        }
    }
    double ise(const std::vector<double>& c1, const std::vector<double>& c2) const {
        double s = 0;
        for (int i = 0; i < kG; ++i)
            for (int j = 0; j < kG; ++j) s += w[i] * w[j] * std::pow(c1[i * kG + j] - c2[i * kG + j], 2);
        return s;
    }
};

// CDF on the grid from a density on the unit square: midpoint grid kD x kD, cumulative sums at the cell
// boundaries, normalised to total mass 1, bilinear interpolation at the nodes.
template <typename F>
std::vector<double> cdf_from_density(const Grid& g, F&& f) {
    std::vector<double> S((kD + 1) * (kD + 1), 0.0);
    for (int i = 1; i <= kD; ++i)
        for (int j = 1; j <= kD; ++j) {
            const double d = f((i - 0.5) / kD, (j - 0.5) / kD) / (kD * kD);
            S[i * (kD + 1) + j] = d + S[(i - 1) * (kD + 1) + j] + S[i * (kD + 1) + j - 1] - S[(i - 1) * (kD + 1) + j - 1];
        }
    const double tot = S[kD * (kD + 1) + kD];
    std::vector<double> c(kG * kG);
    for (int i = 0; i < kG; ++i)
        for (int j = 0; j < kG; ++j) {
            const double xa = g.x[i] * kD, xb = g.x[j] * kD;
            const int ia = std::min(int(xa), kD - 1), ib = std::min(int(xb), kD - 1);
            const double fa = xa - ia, fb = xb - ib;
            auto at = [&](int p, int q) { return S[p * (kD + 1) + q]; };
            c[i * kG + j] = ((1 - fa) * (1 - fb) * at(ia, ib) + fa * (1 - fb) * at(ia + 1, ib)
                             + (1 - fa) * fb * at(ia, ib + 1) + fa * fb * at(ia + 1, ib + 1)) / tot;
        }
    return c;
}

// Weighted empirical copula on the grid and weighted Spearman, from global-rank pseudo-observations.
struct Batch {
    const std::vector<double>& u;
    const std::vector<double>& v;
    void eval(std::size_t a, std::size_t b, double t0, double h, std::vector<double>* c, double* rho, const Grid& g) const {
        std::vector<double> cell((kG + 1) * (kG + 1), 0.0);
        double sw = 0, mu = 0, mv = 0, suv = 0, suu = 0, svv = 0;
        for (std::size_t s = a; s < b; ++s) {
            const double wgt = h > 0 ? std::max(0.0, 1 - std::pow((double(s) - t0) / h, 2)) : 1.0;
            if (wgt <= 0) continue;
            sw += wgt;
            mu += wgt * u[s];
            mv += wgt * v[s];
            suv += wgt * u[s] * v[s];
            suu += wgt * u[s] * u[s];
            svv += wgt * v[s] * v[s];
            if (c) {
                const int ia = int(std::lower_bound(g.x.begin(), g.x.end(), u[s]) - g.x.begin());
                const int ib = int(std::lower_bound(g.x.begin(), g.x.end(), v[s]) - g.x.begin());
                cell[ia * (kG + 1) + ib] += wgt;   // counts toward every node (i >= ia, j >= ib)
            }
        }
        mu /= sw; mv /= sw;
        const double cov = suv / sw - mu * mv, vu = suu / sw - mu * mu, vv = svv / sw - mv * mv;
        *rho = cov / std::sqrt(std::max(vu * vv, 1e-18));
        if (c) {
            c->assign(kG * kG, 0.0);
            for (int i = 0; i < kG; ++i)
                for (int j = 0; j < kG; ++j) {
                    double acc = 0;
                    for (int p = 0; p <= i; ++p)
                        for (int q = 0; q <= j; ++q) acc += cell[p * (kG + 1) + q];
                    (*c)[i * kG + j] = acc / sw;
                }
        }
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: probe_batch_ise <first_seed> <seeds>\n"); return 1; }
    const unsigned first = static_cast<unsigned>(std::atoi(argv[1]));
    const int seeds = std::atoi(argv[2]);
    const char* env = std::getenv("COPULA");
    const std::string cop_name = env ? env : "gaussian";
    const bool shape = cop_name == "gauss2t" || cop_name == "clayton2gumbel" || cop_name == "gauss2clayton";
    const copula::Shape S0 = cop_name == "gauss2t" || cop_name == "gauss2clayton" ? copula::Shape::Gaussian
                           : cop_name == "clayton2gumbel" ? copula::Shape::Clayton : shape_of(cop_name);
    const copula::Shape S1 = cop_name == "gauss2t" ? copula::Shape::StudentT
                           : cop_name == "clayton2gumbel" ? copula::Shape::Gumbel
                           : cop_name == "gauss2clayton" ? copula::Shape::Clayton : S0;
    const std::vector<std::string> scenarios = shape ? std::vector<std::string>{"shape"}
        : std::vector<std::string>{"constant", "drift", "jump", "oscillate"};
    const std::string sfx = (cop_name == "gaussian" ? "" : "_" + cop_name);
    std::FILE* out = std::fopen(("results/batch_ise_" + std::to_string(first) + sfx + ".csv").c_str(), "w");
    if (!out) { std::fprintf(stderr, "cannot write results/ (run from experiments/)\n"); return 1; }
    std::fprintf(out, "copula,scenario,seed,window,method,metric,value\n");
    const Grid g;

    for (const std::string& sc : scenarios) {
        for (int s = 0; s < seeds; ++s) {
            auto dep = make_dep(sc);
            auto cop = dgp::make_copula(cop_name);
            CopulaSource src(*dep, *cop, first + s);
            auto e_fb = make_estimator("family_bank");
            auto* fb = dynamic_cast<FamilyBank*>(e_fb.get());
            std::vector<std::unique_ptr<DependenceEstimator>> banks;
            for (const auto& k : kBanks) banks.push_back(make_estimator(k));

            std::vector<std::string> meth{"family_bank"};
            for (const auto& k : kBanks) meth.push_back(k);
            for (std::size_t k = 0; k < fb->size(); ++k)
                meth.push_back(std::string("arm_") + kShapeName[int(fb->arm(k).shape)] + "_" + std::to_string(fb->arm(k).mem));
            const std::size_t nOn = meth.size();
            meth.push_back("batch_full");
            meth.push_back("batch_segment");
            for (int h : kH) meth.push_back("batch_kernel_" + std::to_string(h));
            const std::size_t M = meth.size();
            std::vector<std::vector<float>> se(M, std::vector<float>(kN, NAN)), ise(M, std::vector<float>(kN, NAN));

            std::vector<double> xs(kN), ys(kN), truth(kN);
            std::vector<std::vector<double>> trueC(kN);
            for (std::size_t t = 0; t < kN; ++t) {
                const auto smp = src.next();
                xs[t] = smp.x;
                ys[t] = smp.y;
                truth[t] = smp.rho_s;
                const bool ev = t >= kBurn && (t - kBurn) % kEvery == 0;
                std::vector<double> ct;
                if (ev) {
                    const copula::Shape St = t < kChange ? S0 : S1;
                    const double par = copula::param_from_spearman(copula::base_family(St),
                                                                   dgp::detail::target_spearman(smp.rho), kNu);
                    ct.resize(kG * kG);
                    for (int i = 0; i < kG; ++i)
                        for (int j = 0; j < kG; ++j) ct[i * kG + j] = g.C(St, par, i, j);
                    trueC[t] = ct;
                }
                // Online estimators: the state held before this tick.
                se[0][t] = float(std::pow(fb->estimate() - smp.rho_s, 2));
                for (std::size_t b = 0; b < kBanks.size(); ++b) se[1 + b][t] = float(std::pow(banks[b]->estimate() - smp.rho_s, 2));
                for (std::size_t k = 0; k < fb->size(); ++k) se[1 + kBanks.size() + k][t] = float(std::pow(fb->arm_rho(k) - smp.rho_s, 2));
                if (ev) {
                    std::vector<std::vector<double>> armC(fb->size(), std::vector<double>(kG * kG));
                    std::vector<double> mix(kG * kG, 0.0);
                    for (std::size_t k = 0; k < fb->size(); ++k) {
                        const auto& a = fb->arm(k);
                        for (int i = 0; i < kG; ++i)
                            for (int j = 0; j < kG; ++j) armC[k][i * kG + j] = g.C(a.shape, a.fit.param(), i, j);
                        for (int p = 0; p < kG * kG; ++p) mix[p] += fb->weight(k) * armC[k][p];
                        ise[1 + kBanks.size() + k][t] = float(g.ise(armC[k], ct));
                    }
                    ise[0][t] = float(g.ise(mix, ct));
                    for (std::size_t b = 0; b < kBanks.size(); ++b) {
                        auto* fbk = dynamic_cast<ForgettingBank*>(banks[b].get());
                        std::vector<double> c(kG * kG);
                        if (kBanks[b].find("_own") != std::string::npos) {
                            c = cdf_from_density(g, [&](double a, double bb) { return std::exp(fbk->log_density(a, bb)); });
                        } else {
                            const double r = copula::param_from_spearman(copula::Family::Gaussian, fbk->estimate());
                            for (int i = 0; i < kG; ++i)
                                for (int j = 0; j < kG; ++j) c[i * kG + j] = g.ellip(false, i, j, r);
                        }
                        ise[1 + b][t] = float(g.ise(c, ct));
                    }
                }
                fb->update(smp.x, smp.y);
                for (auto& b : banks) b->update(smp.x, smp.y);
            }

            // Offline batch estimators: global ranks of the whole sample as pseudo-observations.
            std::vector<double> u(kN), v(kN);
            {
                std::vector<std::size_t> idx(kN);
                std::iota(idx.begin(), idx.end(), 0);
                std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return xs[a] < xs[b]; });
                for (std::size_t r = 0; r < kN; ++r) u[idx[r]] = (r + 1.0) / (kN + 1.0);
                std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return ys[a] < ys[b]; });
                for (std::size_t r = 0; r < kN; ++r) v[idx[r]] = (r + 1.0) / (kN + 1.0);
            }
            const Batch bt{u, v};
            const bool split = sc == "jump" || sc == "shape";
            std::vector<double> cFull, cSeg0, cSeg1;
            double rFull, rSeg0, rSeg1;
            bt.eval(0, kN, 0, 0, &cFull, &rFull, g);
            bt.eval(0, split ? kChange : kN, 0, 0, &cSeg0, &rSeg0, g);
            bt.eval(split ? kChange : 0, kN, 0, 0, &cSeg1, &rSeg1, g);
            for (std::size_t t = 0; t < kN; ++t) {
                const bool ev = !trueC[t].empty();
                const bool seg1 = split && t >= kChange;
                se[nOn][t] = float(std::pow(rFull - truth[t], 2));
                se[nOn + 1][t] = float(std::pow((seg1 ? rSeg1 : rSeg0) - truth[t], 2));
                if (ev) {
                    ise[nOn][t] = float(g.ise(cFull, trueC[t]));
                    ise[nOn + 1][t] = float(g.ise(seg1 ? cSeg1 : cSeg0, trueC[t]));
                }
                for (std::size_t hi = 0; hi < kH.size(); ++hi) {
                    if (t < kBurn) continue;
                    const double h = kH[hi];
                    const std::size_t a = std::size_t(std::max(0.0, double(t) - h)), b = std::min(kN, std::size_t(double(t) + h + 1));
                    std::vector<double> c;
                    double r;
                    bt.eval(a, b, double(t), h, ev ? &c : nullptr, &r, g);
                    se[nOn + 2 + hi][t] = float(std::pow(r - truth[t], 2));
                    if (ev) ise[nOn + 2 + hi][t] = float(g.ise(c, trueC[t]));
                }
            }
            for (const auto& w : windows(sc))
                for (std::size_t m = 0; m < M; ++m) {
                    double a = 0, b = 0;
                    std::size_t na = 0, nb = 0;
                    for (std::size_t t = w.a; t < w.b; ++t) {
                        if (!std::isnan(se[m][t])) { a += se[m][t]; ++na; }
                        if (!std::isnan(ise[m][t])) { b += ise[m][t]; ++nb; }
                    }
                    std::fprintf(out, "%s,%s,%u,%s,%s,rmse,%.6f\n", cop_name.c_str(), sc.c_str(), first + s, w.name,
                                 meth[m].c_str(), std::sqrt(a / na));
                    if (nb) std::fprintf(out, "%s,%s,%u,%s,%s,ise_x1e4,%.5f\n", cop_name.c_str(), sc.c_str(), first + s,
                                         w.name, meth[m].c_str(), 1e4 * b / nb);
                }
        }
        std::fprintf(stderr, "%s %s done\n", cop_name.c_str(), sc.c_str());
    }
    std::fclose(out);
    return 0;
}
