#pragma once

#include <cmath>
#include <numbers>

// Normal and Student-t density/CDF/quantile support for the copula estimators.
namespace special {

inline double normal_cdf(double z) {
    return 0.5 * std::erfc(-z * std::numbers::sqrt2 / 2.0);
}

inline double normal_ppf(double p) {
    if (p <= 0.0) return -1.0e300;
    if (p >= 1.0) return 1.0e300;
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
                               -2.759285104469687e+02, 1.383577518672690e+02,
                               -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
                               -1.556989798598866e+02, 6.680131188771972e+01,
                               -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                               -2.400758277161838e+00, -2.549732539343734e+00,
                               4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
                               2.445134137142996e+00, 3.754408661907416e+00};
    const double plow = 0.02425, phigh = 1.0 - 0.02425;
    if (p < plow) {
        const double q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (p > phigh) {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    const double q = p - 0.5;
    const double r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
}

inline double betacf(double a, double b, double x) {
    constexpr int kMaxIt = 300;
    constexpr double kEps = 3.0e-14;
    constexpr double kTiny = 1.0e-300;

    const double qab = a + b;
    const double qap = a + 1.0;
    const double qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::fabs(d) < kTiny) d = kTiny;
    d = 1.0 / d;
    double h = d;

    for (int m = 1; m <= kMaxIt; ++m) {
        const double m2 = 2.0 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::fabs(d) < kTiny) d = kTiny;
        c = 1.0 + aa / c;
        if (std::fabs(c) < kTiny) c = kTiny;
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::fabs(d) < kTiny) d = kTiny;
        c = 1.0 + aa / c;
        if (std::fabs(c) < kTiny) c = kTiny;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < kEps) break;
    }
    return h;
}

inline double betai(double a, double b, double x) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    const double lbeta = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b);
    const double front = std::exp(lbeta + a * std::log(x) + b * std::log(1.0 - x));
    if (x < (a + 1.0) / (a + b + 2.0)) {
        return front * betacf(a, b, x) / a;
    }
    return 1.0 - front * betacf(b, a, 1.0 - x) / b;
}

inline double t_pdf(double t, double nu) {
    const double logc = std::lgamma((nu + 1.0) / 2.0) - std::lgamma(nu / 2.0)
                        - 0.5 * std::log(nu * std::numbers::pi);
    return std::exp(logc - (nu + 1.0) / 2.0 * std::log1p(t * t / nu));
}

inline double t_cdf(double t, double nu) {
    const double x = nu / (nu + t * t);
    const double ib = 0.5 * betai(nu / 2.0, 0.5, x);
    return (t >= 0.0) ? 1.0 - ib : ib;
}

inline double t_ppf(double p, double nu) {
    if (p <= 0.0) return -1.0e300;
    if (p >= 1.0) return 1.0e300;
    if (p == 0.5) return 0.0;

    double lo = -1.0e4, hi = 1.0e4;
    for (int it = 0; it < 80; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (t_cdf(mid, nu) < p) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi);
}

}  // namespace special
