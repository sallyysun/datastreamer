# Simulation: online Spearman estimators vs batch Spearman vs grade correlation
#
# Generates correlated bivariate normal data one observation at a time and
# compares four estimators against the theoretical grade correlation
# Cor(F(X), G(Y)) = (6/pi) * arcsin(rho/2).
#
# Outputs:
#   1. Live plot in RStudio Plots pane (updates every REPORT_EVERY obs)
#   2. Console table of current estimates
#   3. CSV + PDF in simulated-data/
#
# Usage (source in RStudio or run via Rscript):
#   source("R/simulate.R")
#   Rscript R/simulate.R --N 2000 --rho 0.8
#   Rscript R/simulate.R --nonstationary

library(hermiter)

# resolve script directory (Rscript, source(), or interactive)
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

# ── parse args ───────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)
flag <- function(name, default) {
  idx <- match(name, args)
  if (is.na(idx)) return(default)
  if (is.logical(default)) return(TRUE)
  val <- args[idx + 1]
  if (is.numeric(default)) return(as.numeric(val))
  val
}

N              <- as.integer(flag("--N", 2000))
RHO            <- flag("--rho", 0.6)
NONSTATIONARY  <- flag("--nonstationary", FALSE)
WARMUP         <- 50L
DELAY_MS       <- flag("--delay", 10)
REPORT_EVERY   <- as.integer(flag("--report", 20))

PROJ_ROOT <- normalizePath(file.path(SCRIPT_DIR, ".."))
CSV_DIR   <- file.path(PROJ_ROOT, "simulated-data")
dir.create(CSV_DIR, showWarnings = FALSE, recursive = TRUE)

timestamp_str <- format(Sys.time(), "%Y%m%d_%H%M%S")
mode_str      <- if (NONSTATIONARY) "nonstationary" else "stationary"
CSV_PATH      <- file.path(CSV_DIR, sprintf("sim_%s_%s.csv", mode_str, timestamp_str))

# ── grade correlation: Cor(F(X), G(Y)) ──────────────────────────────────

grade_correlation <- function(rho) (6 / pi) * asin(rho / 2)

# ── regime schedule (non-stationary) ─────────────────────────────────────

rho_at <- if (NONSTATIONARY) {
  regimes <- c(0.7, -0.3, 0.9, -0.6, 0.2)
  regime_len <- N %/% length(regimes)
  function(i) regimes[min(((i - 1) %/% regime_len) + 1, length(regimes))]
} else {
  function(i) RHO
}

# ── data generator ──────────────────────────────────────────────────────

gen_one <- function(rho) {
  z1 <- rnorm(1)
  z2 <- rnorm(1)
  c(z1, rho * z1 + sqrt(1 - rho^2) * z2)
}

# ── estimators ──────────────────────────────────────────────────────────

herm_stat <- hermite_estimator(N = 30, standardize = TRUE, est_type = "bivariate")
herm_ew   <- hermite_estimator(N = 30, standardize = TRUE, est_type = "bivariate",
                               exp_weight_lambda = 0.01)
est_spear <- NULL

warmup_vals <- matrix(NA_real_, nrow = WARMUP, ncol = 2)

init_cutpoint_estimators <- function() {
  cpx <- quantile(warmup_vals[, 1], probs = seq(0.1, 0.9, by = 0.1))
  cpy <- quantile(warmup_vals[, 2], probs = seq(0.1, 0.9, by = 0.1))
  est_spear <<- new_online_npcorr(cpx, cpy, "spearmanr")
  for (k in seq_len(WARMUP)) {
    npcorr_update(est_spear, warmup_vals[k, 1], warmup_vals[k, 2])
  }
}

# ── storage vectors (for batch Spearman + plotting) ─────────────────────

all_x  <- numeric(N)
all_y  <- numeric(N)

v_gc   <- numeric(N)
v_ns   <- rep(NA_real_, N)
v_hs   <- rep(NA_real_, N)
v_hse  <- rep(NA_real_, N)
v_batch <- rep(NA_real_, N)

# ── CSV setup ────────────────────────────────────────────────────────────

cat(paste("obs", "rho", "grade_corr", "batch_spearman",
          "npcorr_spearman", "hermiter_spearman", "hermiter_spearman_ew",
          sep = ","),
    "\n", sep = "", file = CSV_PATH)

csv_append <- function(...) {
  cat(paste(..., sep = ","), "\n", sep = "", file = CSV_PATH, append = TRUE)
}

# ── console output ──────────────────────────────────────────────────────

chart_title <- if (NONSTATIONARY) {
  "Non-Stationary: Tracking Regime-Switching Grade Correlation"
} else {
  sprintf("Stationary: Convergence to Grade Correlation (rho = %.2f)", RHO)
}

print_status <- function(i, rho) {
  na_fmt <- function(x) if (is.na(x)) sprintf("%10s", "NA") else sprintf("%10.6f", x)
  cat(sprintf("── Observation %d / %d  (rho = %.2f) ──\n", i, N, rho))
  cat(sprintf("  %-28s %s\n", "grade corr Cor(F(X),G(Y))", na_fmt(v_gc[i])))
  cat(sprintf("  %-28s %s\n", "batch Spearman",             na_fmt(v_batch[i])))
  cat(sprintf("  %-28s %s\n", "npcorr (online)",            na_fmt(v_ns[i])))
  cat(sprintf("  %-28s %s\n", "hermiter (stationary)",      na_fmt(v_hs[i])))
  cat(sprintf("  %-28s %s\n", "hermiter (exp-weighted)",    na_fmt(v_hse[i])))
  cat(sprintf("  csv: %s\n", basename(CSV_PATH)))
}

# ── live plot helper ────────────────────────────────────────────────────

draw_plot <- function(up_to) {
  idx <- 1:up_to
  ylim <- range(c(v_gc[idx], v_batch[idx], v_ns[idx], v_hs[idx]), na.rm = TRUE)
  ylim <- c(max(ylim[1], -1), min(ylim[2], 1))

  plot(idx, v_gc[idx], type = "l", col = "#E41A1C", lwd = 2.5,
       xlim = c(1, N), ylim = ylim,
       xlab = "Observation", ylab = "Spearman",
       main = chart_title)
  lines(idx, v_batch[idx], col = "#984EA3", lwd = 2, lty = 2)
  lines(idx, v_ns[idx],    col = "#377EB8", lwd = 1.5)
  lines(idx, v_hs[idx],    col = "#4DAF4A", lwd = 1.5)
  legend("bottomright", cex = 0.75, bty = "n",
         legend = c("grade corr", "batch Spearman",
                    "npcorr (online)", "hermiter (online)"),
         col = c("#E41A1C", "#984EA3", "#377EB8", "#4DAF4A"),
         lwd = c(2.5, 2, 1.5, 1.5), lty = c(1, 2, 1, 1))
}

# ── main loop ────────────────────────────────────────────────────────────

set.seed(as.integer(Sys.time()))

cat(sprintf("Mode: %s  |  N: %d  |  rho: %.2f\n", mode_str, N, RHO))
cat(sprintf("CSV:  %s\n\n", CSV_PATH))

for (i in seq_len(N)) {
  rho_now <- rho_at(i)
  obs     <- gen_one(rho_now)
  xi      <- obs[1]
  yi      <- obs[2]

  all_x[i] <- xi
  all_y[i] <- yi

  # ── hermiter ──
  herm_stat <- update_sequential(herm_stat, c(xi, yi))
  herm_ew   <- update_sequential(herm_ew,   c(xi, yi))

  # ── npcorr (after warmup) ──
  if (i <= WARMUP) {
    warmup_vals[i, ] <- c(xi, yi)
    if (i == WARMUP) init_cutpoint_estimators()
  } else {
    npcorr_update(est_spear, xi, yi)
  }

  # ── collect estimates ──
  v_gc[i]    <- grade_correlation(rho_now)
  v_ns[i]    <- if (!is.null(est_spear)) npcorr_value(est_spear) else NA
  v_hs[i]    <- tryCatch(spearmans(herm_stat), error = function(e) NA)
  v_hse[i]   <- tryCatch(spearmans(herm_ew),   error = function(e) NA)
  v_batch[i] <- if (i >= 3) cor(all_x[1:i], all_y[1:i], method = "spearman") else NA

  # ── CSV ──
  csv_append(i, rho_now, v_gc[i], v_batch[i], v_ns[i], v_hs[i], v_hse[i])

  # ── live plot + console (every REPORT_EVERY obs) ──
  if (i %% REPORT_EVERY == 0) {
    draw_plot(i)
    print_status(i, rho_now)
  }

  if (DELAY_MS > 0) Sys.sleep(DELAY_MS / 1000)
}

# ── final plot + console ──
draw_plot(N)
print_status(N, rho_at(N))

cat(sprintf("\nDone. %d observations written to %s\n", N, CSV_PATH))

# ── save PDF ────────────────────────────────────────────────────────────

pdf_path <- sub("\\.csv$", ".pdf", CSV_PATH)
pdf(pdf_path, width = 10, height = 6)
draw_plot(N)
dev.off()
cat(sprintf("Plot saved to %s\n", pdf_path))
