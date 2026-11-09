#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "estimator.hpp"
#include "marginal_rank_pit.hpp"
#include "predictive_density.hpp"
#include "../special_functions.hpp"

// Streaming empirical-copula estimator of Spearman's rho (the "Xiao" branch).
//
// C++ port of Xiao (2019), "Novel Online Algorithms for Nonparametric
// Correlations with Application to Analyze Sensor Data" (IEEE Big Data),
// reference implementation: external/onlineNPCORR (corr.py online_corr /
// utility.py spearmanr_by_matrix, get_index). This estimate() reproduces
// spearmanr_by_matrix exactly; we add two extensions beyond the paper:
// auto-calibrated cutpoints (below) and exponential forgetting (lambda).
//
// Marginals are discretised into `bins` equiprobable cells whose cutpoints are
// fixed from an initial calibration window. A bins x bins count grid then
// approximates the joint distribution, and Spearman's rho is read off as a
// rank correlation over the grid.
//
// Three memory regimes share the same estimate() formula (it is scale
// invariant, so the total weight need not be an integer):
//   * lambda > 0  : exponential forgetting  (counts *= lambda each step)
//   * window > 0  : hard sliding window      (oldest cell decremented)
//   * otherwise   : cumulative               (all history)
//
// rank_pit_window > 0 replaces the calibrated cutpoints with rank-PIT: each
// marginal goes through RankPit, so the cutpoints are exactly k/bins and no
// calibration window is needed. Section 14.19 of
// notes/studies/kalman_changepoint_results.md.
class XiaoSpearmanEstimator : public DependenceEstimator, public PredictiveDensity {
public:
    struct Config {
        std::size_t bins = 30;
        std::size_t window = 0;       // >0: hard sliding window
        double lambda = 0.0;          // in (0,1): retention factor; overrides window
        std::size_t calibration = 300;
        std::size_t rank_pit_window = 0;  // >0: bin on rank-PIT pseudo-observations
        double density_prior = 0.5;       // pseudo-count per cell, for log_density_uv
        // log_predictive: counts pooled into density_bins^2 cells (density_bins must divide bins),
        // with predictive_prior pseudo-observations per pooled cell spread by cell area.
        std::size_t density_bins = 5;
        double predictive_prior = 0.5;

        // Variable forgetting, driven by the section 14.14 rank residual
        //   s = 12 (u - 1/2)(v - 1/2) - rho_hat,
        // over its KNOWN null variance (see vff_step). C_ then tracks E[e~^2],
        // which is 1 when the memory is right -- the same statistic and the same
        // multiplicative rule as KalmanLocalLevel's vff_rate. Mirror image: here
        // it drives alpha = 1 - lambda, so C_ > 1 LOWERS lambda.
        // See notes/studies/adaptive_memory.md.
        double vff_rate = 0.0;             // 0 disables
        std::size_t vff_block = 20;        // ticks per C_ update, as the Kalman's
        double vff_null_var = 1.0;         // section 14.14 measured 0.92-1.00
        std::size_t vff_pit_window = 500;
        double vff_mem_min = 25.0;         // clamps, in ticks of effective memory
        double vff_mem_max = 1000.0;
    };

    XiaoSpearmanEstimator(Config config, const char* name)
        : config_(config),
          name_(name),
          counts_(config.bins * config.bins, 0.0),
          row_counts_(config.bins, 0.0),
          col_counts_(config.bins, 0.0),
          pit_x_(config.rank_pit_window),
          pit_y_(config.rank_pit_window) {
        if (config_.window > 0 && !(config_.lambda > 0.0)) ring_.resize(config_.window);
        if (config_.rank_pit_window > 0) {
            use_uniform_cutpoints();
        } else {
            calibration_.reserve(config_.calibration);
        }
    }

    void update(double x, double y) override {
        if (config_.vff_rate > 0.0) vff_step(x, y);
        if (config_.rank_pit_window > 0) {
            insert(pit_x_.transform(x), pit_y_.transform(y));
            return;
        }
        if (!calibrated_) {
            calibration_.emplace_back(x, y);
            if (calibration_.size() >= config_.calibration) {
                build_cutpoints();
                for (const auto& p : calibration_) insert(p.first, p.second);
                calibration_.clear();
                calibration_.shrink_to_fit();
                calibrated_ = true;
            }
            return;
        }
        insert(x, y);
    }

    double estimate() const override {
        if (!calibrated_ || total_ <= 0.0) return 0.0;
        const std::size_t m = config_.bins;

        // Mid-rank of each marginal cell, centred on the grand mean (total_/2).
        std::vector<double> rx(m), ry(m);
        double cum = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            rx[i] = cum + row_counts_[i] / 2.0 - total_ / 2.0;
            cum += row_counts_[i];
        }
        cum = 0.0;
        for (std::size_t j = 0; j < m; ++j) {
            ry[j] = cum + col_counts_[j] / 2.0 - total_ / 2.0;
            cum += col_counts_[j];
        }

        double var_x = 0.0, var_y = 0.0;
        for (std::size_t i = 0; i < m; ++i) var_x += row_counts_[i] * rx[i] * rx[i];
        for (std::size_t j = 0; j < m; ++j) var_y += col_counts_[j] * ry[j] * ry[j];

        const double denom = std::sqrt(var_x * var_y);
        if (denom <= 0.0) return 0.0;

        double cov = 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < m; ++j) {
                const double c = counts_[i * m + j];
                if (c != 0.0) cov += c * rx[i] * ry[j];
            }
        }
        return std::clamp(cov / denom, -1.0, 1.0);
    }

    void reset() override {
        std::fill(counts_.begin(), counts_.end(), 0.0);
        std::fill(row_counts_.begin(), row_counts_.end(), 0.0);
        std::fill(col_counts_.begin(), col_counts_.end(), 0.0);
        ring_head_ = ring_n_ = 0;
        total_ = 0.0;
        calibrated_ = false;
        calibration_.clear();
        lambda_ = config_.lambda;
        vff_pit_x_.reset();
        vff_pit_y_.reset();
        vff_c_ = 1.0;
        vff_sum_ = 0.0;
        vff_raw_sum_ = vff_den_sum_ = 0.0;
        vff_in_block_ = 0;
        vff_blocks_ = 0;
        if (config_.rank_pit_window > 0) {
            pit_x_.reset();
            pit_y_.reset();
            use_uniform_cutpoints();
        } else {
            calibration_.reserve(config_.calibration);
        }
    }

    // Everything the object holds: itself plus every heap buffer, by capacity (allocation, not use).
    std::size_t store_bytes() const override {
        return sizeof(*this)
             + (counts_.capacity() + row_counts_.capacity() + col_counts_.capacity()) * sizeof(double)
             + (cuts_x_.capacity() + cuts_y_.capacity()) * sizeof(double)
             + ring_.capacity() * sizeof(BinPair)
             + calibration_.capacity() * sizeof(std::pair<double, double>)
             + pit_x_.heap_bytes() + pit_y_.heap_bytes() + vff_pit_x_.heap_bytes() + vff_pit_y_.heap_bytes();
    }

    const char* name() const override { return name_; }

    // Histogram log-density at pseudo-observations (u, v) on the equal-area bins x bins grid of
    // rank-PIT mode, smoothed by density_prior per cell. With unforgotten counts this is a
    // Dirichlet posterior predictive; with fixed forgetting it is discounted-count smoothing. It
    // integrates to one but its margins are uniform only approximately, so it is a density on the
    // unit square, not a copula density. NaN outside rank-PIT mode (raw-scale calibrated cells are
    // not equal-area in u) and under VFF (the stored counts mix observations weighted by different
    // lambdas). notes/studies/memory_tradeoff.md.
    double log_density_uv(double u, double v) const {
        constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
        if (config_.rank_pit_window == 0 || config_.vff_rate > 0.0) return kNaN;
        if (!std::isfinite(config_.density_prior) || !(config_.density_prior > 0.0)) return kNaN;
        if (!(u >= 0.0 && u <= 1.0 && v >= 0.0 && v <= 1.0)) return kNaN;
        if (total_ <= 0.0) return 0.0;
        const std::size_t m = config_.bins;
        const auto cell = [m](double p) {
            return std::min(static_cast<std::size_t>(p * static_cast<double>(m)), m - 1);
        };
        // Fixed forgetting stores each observation with weight 1 - lambda, so total_ tends to 1;
        // rescale to observation units or the prior (density_prior * bins^2) swamps the data.
        const double unit = (config_.lambda > 0.0) ? 1.0 / (1.0 - config_.lambda) : 1.0;
        const double k2 = static_cast<double>(m * m);
        const double n = counts_[cell(u) * m + cell(v)] * unit;
        return std::log((n + config_.density_prior) / (total_ * unit + config_.density_prior * k2) * k2);
    }

    // Histogram predictive density on the calibrated grid, for inputs fed as (probit(u), probit(v)).
    // Counts are pooled into density_bins^2 cells; the prior is uniform on [0,1]^2 (pseudo-counts
    // proportional to cell area in u), so a cell with no data predicts the independence copula.
    double log_predictive(double u, double v) const override {
        constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
        if (config_.rank_pit_window > 0 || config_.vff_rate > 0.0) return kNaN;
        if (!(u > 0.0 && u < 1.0 && v > 0.0 && v < 1.0)) return kNaN;
        const std::size_t m = config_.bins;
        const std::size_t g = (config_.density_bins > 0 && m % config_.density_bins == 0) ? m / config_.density_bins : 1;
        const std::size_t D = m / g;
        if (!calibrated_ || total_ <= 0.0) return 0.0;
        const std::size_t I = bin_of(cuts_x_, RankPit::probit(u)) / g;
        const std::size_t J = bin_of(cuts_y_, RankPit::probit(v)) / g;
        const auto edge = [&](const std::vector<double>& cuts, std::size_t b) {
            if (b == 0) return 0.0;
            if (b >= D) return 1.0;
            return special::normal_cdf(cuts[b * g - 1]);
        };
        const double du = edge(cuts_x_, I + 1) - edge(cuts_x_, I);
        const double dv = edge(cuts_y_, J + 1) - edge(cuts_y_, J);
        if (!(du > 0.0 && dv > 0.0)) return kNaN;
        double n = 0.0;
        for (std::size_t i = I * g; i < (I + 1) * g; ++i)
            for (std::size_t j = J * g; j < (J + 1) * g; ++j) n += counts_[i * m + j];
        // Counts carry weight 1 - lambda each under forgetting; rescale to observations.
        const double unit = (lambda_ > 0.0) ? 1.0 / (1.0 - lambda_) : 1.0;
        const double a = config_.predictive_prior * static_cast<double>(D * D);
        return std::log((n * unit + a * du * dv) / (total_ * unit + a) / (du * dv));
    }

    // Ablation hooks (probe_vff_norm.cpp): swap the VFF's null scale per tick and
    // read back the controller's per-block state.
    void set_vff_null_var(double v) { config_.vff_null_var = v; }
    double vff_c() const { return vff_c_; }
    double vff_last_ms() const { return vff_last_ms_; }
    double vff_last_denom() const { return vff_last_denom_; }

    double effective_memory_ticks() const {
        const double alpha = 1.0 - lambda_;
        return (alpha > 0.0) ? (2.0 - alpha) / alpha : 0.0;
    }

private:
    struct BinPair {
        std::uint16_t i;
        std::uint16_t j;
    };

    void use_uniform_cutpoints() {
        cuts_x_.resize(config_.bins - 1);
        for (std::size_t k = 1; k < config_.bins; ++k) {
            cuts_x_[k - 1] = static_cast<double>(k) / static_cast<double>(config_.bins);
        }
        cuts_y_ = cuts_x_;
        calibrated_ = true;
    }

    void build_cutpoints() {
        std::vector<double> xs, ys;
        xs.reserve(calibration_.size());
        ys.reserve(calibration_.size());
        for (const auto& p : calibration_) {
            xs.push_back(p.first);
            ys.push_back(p.second);
        }
        std::sort(xs.begin(), xs.end());
        std::sort(ys.begin(), ys.end());

        cuts_x_.resize(config_.bins - 1);
        cuts_y_.resize(config_.bins - 1);
        for (std::size_t k = 1; k < config_.bins; ++k) {
            const std::size_t idx = k * xs.size() / config_.bins;
            cuts_x_[k - 1] = xs[std::min(idx, xs.size() - 1)];
            cuts_y_[k - 1] = ys[std::min(idx, ys.size() - 1)];
        }
    }

    void vff_step(double x, double y) {
        // Residual against the estimate held BEFORE this observation is absorbed.
        const double rho_hat = estimate();
        const double u = vff_pit_x_.transform(x);
        const double v = vff_pit_y_.transform(y);
        if (vff_pit_x_.size() < 30) return;
        const double p = 12.0 * (u - 0.5) * (v - 0.5);
        const double s = p - rho_hat;

        // Fixed null scale, not an estimated one. An EWMA of s^2 chases the very
        // lag the VFF exists to remove (14.11's trap); E[p^2] - rho_hat^2 biases
        // the other way, since rho_hat is the lagging quantity. Section 14.14
        // measured Var(s) at 0.92-1.00 across the whole rho range, so 1 is a
        // KNOWN scale to within 8% and nothing has to be estimated. This is the
        // nonparametric stand-in for the Kalman's F.
        vff_sum_ += s * s / config_.vff_null_var;
        vff_raw_sum_ += s * s;
        vff_den_sum_ += config_.vff_null_var;
        if (++vff_in_block_ < config_.vff_block) return;
        const double e2 = vff_sum_ / static_cast<double>(config_.vff_block);
        vff_last_ms_ = vff_raw_sum_ / static_cast<double>(config_.vff_block);
        vff_last_denom_ = vff_den_sum_ / static_cast<double>(config_.vff_block);
        vff_sum_ = 0.0;
        vff_raw_sum_ = 0.0;
        vff_den_sum_ = 0.0;
        vff_in_block_ = 0;
        if (++vff_blocks_ <= kVffBurnInBlocks) return;

        vff_c_ += config_.vff_rate * (e2 - vff_c_);
        double alpha = (1.0 - lambda_) * std::pow(std::max(vff_c_, 1e-6), config_.vff_rate);
        alpha = std::clamp(alpha, 2.0 / (config_.vff_mem_max + 1.0),
                                  2.0 / (config_.vff_mem_min + 1.0));
        lambda_ = 1.0 - alpha;
    }

    std::size_t bin_of(const std::vector<double>& cuts, double value) const {
        return static_cast<std::size_t>(
            std::upper_bound(cuts.begin(), cuts.end(), value) - cuts.begin());
    }

    void insert(double x, double y) {
        const std::size_t i = bin_of(cuts_x_, x);
        const std::size_t j = bin_of(cuts_y_, y);
        const std::size_t m = config_.bins;

        if (config_.lambda > 0.0) {
            const double l = lambda_;
            for (auto& c : counts_) c *= l;
            for (auto& c : row_counts_) c *= l;
            for (auto& c : col_counts_) c *= l;
            total_ *= l;
            counts_[i * m + j] += (1.0 - l);
            row_counts_[i] += (1.0 - l);
            col_counts_[j] += (1.0 - l);
            total_ += (1.0 - l);
            return;
        }

        counts_[i * m + j] += 1.0;
        row_counts_[i] += 1.0;
        col_counts_[j] += 1.0;
        total_ += 1.0;

        if (config_.window > 0) {
            const BinPair now{static_cast<std::uint16_t>(i), static_cast<std::uint16_t>(j)};
            if (ring_n_ < config_.window) {
                ring_[(ring_head_ + ring_n_++) % config_.window] = now;
                return;
            }
            const BinPair old = ring_[ring_head_];
            ring_[ring_head_] = now;
            ring_head_ = (ring_head_ + 1) % config_.window;
            counts_[old.i * m + old.j] -= 1.0;
            row_counts_[old.i] -= 1.0;
            col_counts_[old.j] -= 1.0;
            total_ -= 1.0;
        }
    }

    Config config_;
    const char* name_;
    bool calibrated_ = false;
    double total_ = 0.0;

    std::vector<double> counts_;
    std::vector<double> row_counts_;
    std::vector<double> col_counts_;
    std::vector<double> cuts_x_;
    std::vector<double> cuts_y_;
    std::vector<BinPair> ring_;   // hard-window ring buffer, allocated only in window mode
    std::size_t ring_head_ = 0, ring_n_ = 0;
    std::vector<std::pair<double, double>> calibration_;

    // Declared AFTER config_: a RankPit placed before it reads an uninitialised
    // window size (the parametric_copula.hpp bug of 2026-09).
    RankPit pit_x_;
    RankPit pit_y_;
    // Sized only when the VFF runs: an unused RankPit(500) still reserves 16 KB.
    RankPit vff_pit_x_{config_.vff_rate > 0.0 ? config_.vff_pit_window : 1};
    RankPit vff_pit_y_{config_.vff_rate > 0.0 ? config_.vff_pit_window : 1};

    static constexpr std::size_t kVffBurnInBlocks = 5;
    double lambda_ = config_.lambda;
    double vff_c_ = 1.0;
    double vff_sum_ = 0.0;
    double vff_raw_sum_ = 0.0, vff_den_sum_ = 0.0;
    double vff_last_ms_ = 0.0, vff_last_denom_ = 0.0;
    std::size_t vff_in_block_ = 0;
    std::size_t vff_blocks_ = 0;
};
