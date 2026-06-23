# Test hermiter's online bivariate Spearman estimator against batch baselines.
# No C++ server needed — generates synthetic data in pure R.
#
# Usage: Rscript test_hermiter.R

library(hermiter)

set.seed(42)

N <- 1000L

# ── generate correlated data ─────────────────────────────────────────────

rho <- 0.6
z1  <- rnorm(N)
z2  <- rnorm(N)
x   <- z1
y   <- rho * z1 + sqrt(1 - rho^2) * z2

grade_corr <- (6 / pi) * asin(rho / 2)
cat(sprintf("Generated %d observations with rho = %.2f\n", N, rho))
cat(sprintf("Grade correlation Cor(F(X),G(Y)) = %.8f\n\n", grade_corr))

# ══════════════════════════════════════════════════════════════════════════
# TEST 1: Stationary bivariate — Spearman
# ══════════════════════════════════════════════════════════════════════════

cat("── Test 1: hermiter Spearman (stationary) ──\n")

est <- hermite_estimator(N = 30, standardize = TRUE, est_type = "bivariate")
for (i in seq_len(N)) {
  est <- update_sequential(est, c(x[i], y[i]))
}

herm_sp <- spearmans(est)
batch_sp <- cor(x, y, method = "spearman")

cat(sprintf("  hermiter:    %.8f\n", herm_sp))
cat(sprintf("  batch:       %.8f\n", batch_sp))
cat(sprintf("  grade corr:  %.8f\n", grade_corr))
cat(sprintf("  diff (vs batch):  %.4e\n\n", abs(herm_sp - batch_sp)))

# ══════════════════════════════════════════════════════════════════════════
# TEST 2: Stationary bivariate — density and CDF spot check
# ══════════════════════════════════════════════════════════════════════════

cat("── Test 2: hermiter bivariate density + CDF ──\n")

d_val <- dens(est, c(0, 0))
c_val <- cum_prob(est, c(0, 0))

cat(sprintf("  density  at (0,0):  %.6f  (theoretical ~0.092 for rho=0.6)\n", d_val))
cat(sprintf("  cum_prob at (0,0):  %.6f  (should be near 0.25)\n\n", c_val))

# ══════════════════════════════════════════════════════════════════════════
# TEST 3: Non-stationary (exponentially weighted) — Spearman
# ══════════════════════════════════════════════════════════════════════════

cat("── Test 3: hermiter Spearman (exp-weighted, lambda=0.01) ──\n")

est_ew <- hermite_estimator(N = 30, standardize = TRUE, est_type = "bivariate",
                            exp_weight_lambda = 0.01)
for (i in seq_len(N)) {
  est_ew <- update_sequential(est_ew, c(x[i], y[i]))
}

herm_sp_ew <- spearmans(est_ew)
cat(sprintf("  hermiter (weighted):   %.8f\n", herm_sp_ew))
cat(sprintf("  hermiter (stationary): %.8f\n", herm_sp))
cat("  (weighted should track recent data more closely)\n\n")

# ══════════════════════════════════════════════════════════════════════════
# TEST 4: Non-stationary — detect correlation regime change
# ══════════════════════════════════════════════════════════════════════════

cat("── Test 4: hermiter detects correlation regime change ──\n")

N2 <- 500L
x2 <- c(rnorm(N2), rnorm(N2))
y2 <- c(0.8 * rnorm(N2) + 0.6 * x2[1:N2],
        -0.8 * rnorm(N2) - 0.6 * x2[(N2+1):(2*N2)])

est_shift <- hermite_estimator(N = 30, standardize = TRUE, est_type = "bivariate",
                               exp_weight_lambda = 0.02)

sp_at_mid <- NA
sp_at_end <- NA
for (i in seq_along(x2)) {
  est_shift <- update_sequential(est_shift, c(x2[i], y2[i]))
  if (i == N2) sp_at_mid <- spearmans(est_shift)
  if (i == 2 * N2) sp_at_end <- spearmans(est_shift)
}

cat(sprintf("  Spearman at t=%d (rho~+0.6):  %+.4f\n", N2, sp_at_mid))
cat(sprintf("  Spearman at t=%d (rho~-0.6):  %+.4f\n", 2*N2, sp_at_end))
cat(sprintf("  sign flipped: %s\n\n",
            if (sp_at_mid > 0 && sp_at_end < 0) "PASS" else "CHECK"))

# ══════════════════════════════════════════════════════════════════════════
# TEST 5: Merge estimators (parallelism check)
# ══════════════════════════════════════════════════════════════════════════

cat("── Test 5: hermiter merge ──\n")

est_a <- hermite_estimator(N = 30, standardize = TRUE, est_type = "bivariate")
est_b <- hermite_estimator(N = 30, standardize = TRUE, est_type = "bivariate")

half <- N %/% 2
for (i in 1:half) est_a <- update_sequential(est_a, c(x[i], y[i]))
for (i in (half+1):N) est_b <- update_sequential(est_b, c(x[i], y[i]))

est_merged <- merge_hermite(list(est_a, est_b))

sp_merged <- spearmans(est_merged)
cat(sprintf("  merged Spearman:  %.8f\n", sp_merged))
cat(sprintf("  full   Spearman:  %.8f\n", herm_sp))
cat(sprintf("  diff:             %.4e\n\n", abs(sp_merged - herm_sp)))

cat("Done.\n")
