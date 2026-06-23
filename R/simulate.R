# Simulation: online Spearman estimators vs batch Spearman vs grade correlation
#
# Cor(F(X), G(Y)) = (6/pi) * arcsin(rho/2)
#
# Two self-contained functions:
#   run_stationary(N, rho, ...)       — constant rho, convergence test
#   run_nonstationary(N, regimes, ...) — regime-switching, tracking test
#
# Wrapper:
#   simulate("stationary", ...)
#   simulate("nonstationary", ...)
#
# Outputs: live RStudio plot, stderr status table, CSV + PDF
#
# Usage:
#   simulate("stationary", N = 2000, rho = 0.6)
#   simulate("nonstationary", N = 2000)
#   rstudioapi::jobRunScript("R/simulate.R")       # background job

library(hermiter)

# resolve script directory
SCRIPT_DIR <- local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- grep("^--file=", a, value = TRUE)
  if (length(f) > 0) return(normalizePath(dirname(sub("^--file=", "", f[1]))))
  for (i in seq_len(sys.nframe())) {
    of <- tryCatch(sys.frame(i)$ofile, error = function(e) NULL)
    if (!is.null(of)) return(normalizePath(dirname(of)))
  }
  if (dir.exists(file.path(getwd(), "R"))) return(normalizePath(file.path(getwd(), "R")))
  normalizePath(getwd())
})

source(file.path(SCRIPT_DIR, "online_npcorr.R"))
source(file.path(SCRIPT_DIR, "ring_buffer.R"))

PROJ_ROOT <- normalizePath(file.path(SCRIPT_DIR, ".."))

# ── shared helpers ──────────────────────────────────────────────────────

grade_correlation <- function(rho) (6 / pi) * asin(rho / 2)

gen_one <- function(rho) {
  z1 <- rnorm(1)
  c(z1, rho * z1 + sqrt(1 - rho^2) * rnorm(1))
}

log_msg <- function(...) cat(..., file = stderr())

# ═══════════════════════════════════════════════════════════════════════
# Stationary simulation
# ═══════════════════════════════════════════════════════════════════════

run_stationary <- function(N = 2000, rho = 0.6, warmup = 50L,
                           report_every = 20L, delay_ms = 10,
                           hermiter_N = 30L, npcorr_bins = 10L,
                           seed = as.integer(Sys.time())) {
  set.seed(seed)

  title <- sprintf("Stationary: Convergence to Grade Correlation (rho = %.2f)", rho)
  gc_val <- grade_correlation(rho)

  # ── output paths ──
  csv_dir <- file.path(PROJ_ROOT, "simulated-data")
  dir.create(csv_dir, showWarnings = FALSE, recursive = TRUE)
  ts   <- format(Sys.time(), "%Y%m%d_%H%M%S")
  csv  <- file.path(csv_dir, sprintf("sim_stationary_%s.csv", ts))

  cat(paste("obs", "rho", "grade_corr", "batch_spearman",
            "npcorr_spearman", "hermiter_spearman",
            sep = ","),
      "\n", sep = "", file = csv)

  # ── estimators ──
  herm <- hermite_estimator(N = hermiter_N, standardize = TRUE, est_type = "bivariate")
  npc  <- NULL
  warmup_vals <- matrix(NA_real_, nrow = warmup, ncol = 2)

  # ── storage ──
  all_x   <- numeric(N)
  all_y   <- numeric(N)
  v_gc    <- numeric(N)
  v_batch <- rep(NA_real_, N)
  v_npc   <- rep(NA_real_, N)
  v_herm  <- rep(NA_real_, N)

  # ── plot ──
  # viridis-based colorblind-friendly palette (end=0.9 avoids pale yellow)
  pal <- viridisLite::viridis(4, begin = 0.0, end = 0.9)

  draw <- function(up_to) {
    idx  <- 1:up_to
    ylim <- range(c(v_gc[idx], v_batch[idx], v_npc[idx], v_herm[idx]), na.rm = TRUE)
    ylim <- c(max(ylim[1], -1), min(ylim[2], 1))

    plot(idx, v_gc[idx], type = "l", col = pal[1], lwd = 2.5,
         xlim = c(1, N), ylim = ylim,
         xlab = "Observation", ylab = "Spearman", main = title)
    lines(idx, v_batch[idx], col = pal[2], lwd = 2, lty = 2)
    lines(idx, v_npc[idx],   col = pal[3], lwd = 1.5)
    lines(idx, v_herm[idx],  col = pal[4], lwd = 1.5, lty = 4)
    legend("bottomright", cex = 0.75, bty = "n",
           legend = c("grade corr", "batch Spearman",
                      "npcorr (online)", "hermiter (online)"),
           col = pal,
           lwd = c(2.5, 2, 1.5, 1.5), lty = c(1, 2, 1, 4))
  }

  # ── status ──
  status <- function(i) {
    na_fmt <- function(x) if (is.na(x)) sprintf("%10s", "NA") else sprintf("%10.6f", x)
    log_msg(sprintf("── Observation %d / %d  (rho = %.2f) ──\n", i, N, rho))
    log_msg(sprintf("  %-28s %s\n", "grade corr Cor(F(X),G(Y))", na_fmt(v_gc[i])))
    log_msg(sprintf("  %-28s %s\n", "batch Spearman",             na_fmt(v_batch[i])))
    log_msg(sprintf("  %-28s %s\n", "npcorr (online)",            na_fmt(v_npc[i])))
    log_msg(sprintf("  %-28s %s\n", "hermiter (online)",          na_fmt(v_herm[i])))
    log_msg(sprintf("  csv: %s\n", basename(csv)))
  }

  log_msg(sprintf("Stationary  |  N: %d  |  rho: %.2f\n", N, rho))
  log_msg(sprintf("CSV: %s\n\n", csv))

  # ── loop ──
  for (i in seq_len(N)) {
    obs <- gen_one(rho)
    all_x[i] <- obs[1]
    all_y[i] <- obs[2]

    herm <- update_sequential(herm, obs)

    if (i <= warmup) {
      warmup_vals[i, ] <- obs
      if (i == warmup) {
        cp_probs <- seq(1, npcorr_bins - 1L) / npcorr_bins
        cpx <- quantile(warmup_vals[, 1], probs = cp_probs)
        cpy <- quantile(warmup_vals[, 2], probs = cp_probs)
        npc <- new_online_npcorr(cpx, cpy, "spearmanr")
        for (k in seq_len(warmup)) npcorr_update(npc, warmup_vals[k, 1], warmup_vals[k, 2])
      }
    } else {
      npcorr_update(npc, obs[1], obs[2])
    }

    v_gc[i]    <- gc_val
    v_npc[i]   <- if (!is.null(npc)) npcorr_value(npc) else NA
    v_herm[i]  <- tryCatch(spearmans(herm), error = function(e) NA)
    v_batch[i] <- if (i >= 3) cor(all_x[1:i], all_y[1:i], method = "spearman") else NA

    cat(paste(i, rho, v_gc[i], v_batch[i], v_npc[i], v_herm[i], sep = ","),
        "\n", sep = "", file = csv, append = TRUE)

    if (i %% report_every == 0) { draw(i); status(i) }
    if (delay_ms > 0) Sys.sleep(delay_ms / 1000)
  }

  draw(N); status(N)
  log_msg(sprintf("\nDone. %d observations written to %s\n", N, csv))

  pdf_path <- sub("\\.csv$", ".pdf", csv)
  pdf(pdf_path, width = 10, height = 6); draw(N); dev.off()
  log_msg(sprintf("Plot saved to %s\n", pdf_path))

  invisible(list(csv = csv, pdf = pdf_path))
}

# ═══════════════════════════════════════════════════════════════════════
# Non-stationary simulation
# ═══════════════════════════════════════════════════════════════════════

run_nonstationary <- function(N = 2000, regimes = c(0.7, -0.3, 0.9, -0.6, 0.2),
                              window = 100L, warmup = 50L,
                              report_every = 20L, delay_ms = 10,
                              hermiter_N = 30L, exp_weight_lambda = 0.01,
                              npcorr_bins = 10L,
                              seed = as.integer(Sys.time())) {
  set.seed(seed)

  title <- "Non-Stationary: Tracking Regime-Switching Grade Correlation"
  regime_len <- N %/% length(regimes)
  rho_at <- function(i) regimes[min(((i - 1) %/% regime_len) + 1, length(regimes))]

  # ── output paths ──
  csv_dir <- file.path(PROJ_ROOT, "simulated-data")
  dir.create(csv_dir, showWarnings = FALSE, recursive = TRUE)
  ts   <- format(Sys.time(), "%Y%m%d_%H%M%S")
  csv  <- file.path(csv_dir, sprintf("sim_nonstationary_%s.csv", ts))

  cat(paste("obs", "rho", "grade_corr", "batch_spearman",
            "npcorr_spearman", "npcorr_spearman_win",
            "hermiter_spearman", "hermiter_spearman_ew",
            sep = ","),
      "\n", sep = "", file = csv)

  # ── estimators ──
  herm_stat <- hermite_estimator(N = hermiter_N, standardize = TRUE, est_type = "bivariate")
  herm_ew   <- hermite_estimator(N = hermiter_N, standardize = TRUE, est_type = "bivariate",
                                 exp_weight_lambda = exp_weight_lambda)
  npc_stat <- NULL
  npc_win  <- NULL
  warmup_vals <- matrix(NA_real_, nrow = warmup, ncol = 2)
  buf <- new_ring_buffer(max(window, 512L), ncol = 2L)

  # ── storage ──
  all_x      <- numeric(N)
  all_y      <- numeric(N)
  v_gc       <- numeric(N)
  v_batch    <- rep(NA_real_, N)
  v_npc      <- rep(NA_real_, N)
  v_npc_win  <- rep(NA_real_, N)
  v_herm     <- rep(NA_real_, N)
  v_herm_ew  <- rep(NA_real_, N)

  # viridis-based colorblind-friendly palette (end=0.9 avoids pale yellow)
  pal <- viridisLite::viridis(6, begin = 0.0, end = 0.9)

  # ── plot ──
  draw <- function(up_to) {
    idx  <- 1:up_to
    ylim <- range(c(v_gc[idx], v_npc_win[idx], v_herm_ew[idx], v_batch[idx]),
                  na.rm = TRUE)
    ylim <- c(max(ylim[1], -1), min(ylim[2], 1))

    plot(idx, v_gc[idx], type = "l", col = pal[1], lwd = 2.5,
         xlim = c(1, N), ylim = ylim,
         xlab = "Observation", ylab = "Spearman", main = title)
    lines(idx, v_batch[idx],    col = pal[2], lwd = 1.5, lty = 2)
    lines(idx, v_npc[idx],      col = pal[3], lwd = 1,   lty = 3)
    lines(idx, v_npc_win[idx],  col = pal[4], lwd = 1.5)
    lines(idx, v_herm[idx],     col = pal[5], lwd = 1,   lty = 4)
    lines(idx, v_herm_ew[idx],  col = pal[6], lwd = 1.5, lty = 5)
    legend("bottomright", cex = 0.7, bty = "n",
           legend = c("grade corr", "batch Spearman",
                      "npcorr (cumulative)", sprintf("npcorr (win=%d)", window),
                      "hermiter (cumulative)", "hermiter (exp-wt)"),
           col = pal,
           lwd = c(2.5, 1.5, 1, 1.5, 1, 1.5),
           lty = c(1, 2, 3, 1, 4, 5))
  }

  # ── status ──
  status <- function(i) {
    rho <- rho_at(i)
    na_fmt <- function(x) if (is.na(x)) sprintf("%10s", "NA") else sprintf("%10.6f", x)
    log_msg(sprintf("── Observation %d / %d  (rho = %.2f) ──\n", i, N, rho))
    log_msg(sprintf("  %-30s %s\n", "grade corr Cor(F(X),G(Y))",  na_fmt(v_gc[i])))
    log_msg(sprintf("  %-30s %s\n", "batch Spearman",              na_fmt(v_batch[i])))
    log_msg(sprintf("  %-30s %s\n", "npcorr (cumulative)",         na_fmt(v_npc[i])))
    log_msg(sprintf("  %-30s %s\n", sprintf("npcorr (win=%d)", window), na_fmt(v_npc_win[i])))
    log_msg(sprintf("  %-30s %s\n", "hermiter (cumulative)",       na_fmt(v_herm[i])))
    log_msg(sprintf("  %-30s %s\n", "hermiter (exp-weighted)",     na_fmt(v_herm_ew[i])))
    log_msg(sprintf("  csv: %s\n", basename(csv)))
  }

  log_msg(sprintf("Non-stationary  |  N: %d  |  window: %d  |  regimes: %s\n",
                  N, window, paste(regimes, collapse = ", ")))
  log_msg(sprintf("CSV: %s\n\n", csv))

  # ── loop ──
  for (i in seq_len(N)) {
    rho_now <- rho_at(i)
    obs <- gen_one(rho_now)
    all_x[i] <- obs[1]
    all_y[i] <- obs[2]
    ring_push(buf, obs)

    herm_stat <- update_sequential(herm_stat, obs)
    herm_ew   <- update_sequential(herm_ew,   obs)

    if (i <= warmup) {
      warmup_vals[i, ] <- obs
      if (i == warmup) {
        cp_probs <- seq(1, npcorr_bins - 1L) / npcorr_bins
        cpx <- quantile(warmup_vals[, 1], probs = cp_probs)
        cpy <- quantile(warmup_vals[, 2], probs = cp_probs)
        npc_stat <- new_online_npcorr(cpx, cpy, "spearmanr")
        npc_win  <- new_online_npcorr(cpx, cpy, "spearmanr")
        for (k in seq_len(warmup)) {
          npcorr_update(npc_stat, warmup_vals[k, 1], warmup_vals[k, 2])
          npcorr_update(npc_win,  warmup_vals[k, 1], warmup_vals[k, 2])
        }
      }
    } else {
      npcorr_update(npc_stat, obs[1], obs[2])
      npcorr_update(npc_win,  obs[1], obs[2])
      if (npc_win$n > window) {
        old <- ring_get(buf, buf$head - window)
        npcorr_remove(npc_win, old[1], old[2])
      }
    }

    v_gc[i]      <- grade_correlation(rho_now)
    v_npc[i]     <- if (!is.null(npc_stat)) npcorr_value(npc_stat) else NA
    v_npc_win[i] <- if (!is.null(npc_win))  npcorr_value(npc_win)  else NA
    v_herm[i]    <- tryCatch(spearmans(herm_stat), error = function(e) NA)
    v_herm_ew[i] <- tryCatch(spearmans(herm_ew),   error = function(e) NA)
    v_batch[i]   <- if (i >= 3) cor(all_x[1:i], all_y[1:i], method = "spearman") else NA

    cat(paste(i, rho_now, v_gc[i], v_batch[i], v_npc[i], v_npc_win[i],
              v_herm[i], v_herm_ew[i], sep = ","),
        "\n", sep = "", file = csv, append = TRUE)

    if (i %% report_every == 0) { draw(i); status(i) }
    if (delay_ms > 0) Sys.sleep(delay_ms / 1000)
  }

  draw(N); status(N)
  log_msg(sprintf("\nDone. %d observations written to %s\n", N, csv))

  pdf_path <- sub("\\.csv$", ".pdf", csv)
  pdf(pdf_path, width = 10, height = 6); draw(N); dev.off()
  log_msg(sprintf("Plot saved to %s\n", pdf_path))

  invisible(list(csv = csv, pdf = pdf_path))
}

# ═══════════════════════════════════════════════════════════════════════
# Wrapper
# ═══════════════════════════════════════════════════════════════════════

simulate <- function(mode = c("stationary", "nonstationary"), ...) {
  mode <- match.arg(mode)
  if (mode == "stationary") run_stationary(...) else run_nonstationary(...)
}

# ── CLI entry point ─────────────────────────────────────────────────────

if (!interactive() || length(commandArgs(trailingOnly = TRUE)) > 0) {
  args <- commandArgs(trailingOnly = TRUE)
  flag <- function(name, default) {
    idx <- match(name, args)
    if (is.na(idx)) return(default)
    if (is.logical(default)) return(TRUE)
    val <- args[idx + 1]
    if (is.numeric(default)) return(as.numeric(val))
    val
  }

  if (flag("--nonstationary", FALSE)) {
    run_nonstationary(
      N          = as.integer(flag("--N", 2000)),
      delay_ms   = flag("--delay", 10),
      report_every = as.integer(flag("--report", 20))
    )
  } else {
    run_stationary(
      N          = as.integer(flag("--N", 2000)),
      rho        = flag("--rho", 0.6),
      delay_ms   = flag("--delay", 10),
      report_every = as.integer(flag("--report", 20))
    )
  }
}

# ═══════════════════════════════════════════════════════════════════════
# Testing out
# ═══════════════════════════════════════════════════════════════════════

simulate("stationary", N = 2000, rho = 0.6)
simulate("nonstationary", N = 2000)
