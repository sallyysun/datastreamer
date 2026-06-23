# Test the onlineNPCORR Spearman port following Xiao's example style.
# Uses integer data with known cutpoints, matching the doctest in corr.py:
#   x = np.random.randint(0, 10, 1000)
#   y = np.random.randint(0, 13, 1000)
#   cutpoints_x = range(1,10), cutpoints_y = range(1,13)
#
# No C++ server needed — generates synthetic data in pure R.
#
# Usage: Rscript test_npcorr.R

# resolve script directory (works from Rscript, source(), and interactive)
.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
if (length(.file_arg) > 0) {
  SCRIPT_DIR <- normalizePath(dirname(sub("^--file=", "", .file_arg[1])))
} else {
  SCRIPT_DIR <- tryCatch(
    normalizePath(dirname(sys.frame(1)$ofile)),
    error = function(e) normalizePath(file.path(getwd(), "R"))
  )
}

source(file.path(SCRIPT_DIR, "online_npcorr.R"))
source(file.path(SCRIPT_DIR, "ring_buffer.R"))

set.seed(42)

N    <- 1000L
NWIN <- 100L
NGAP <- 1L

# ── generate integer data (matching Xiao's doctest) ─────────────────────

x <- sample(0:9,  N, replace = TRUE)
y <- sample(0:12, N, replace = TRUE)

cpx <- 1:9
cpy <- 1:12

cat(sprintf("Generated %d observations: x in {0..9}, y in {0..12}\n", N))
cat(sprintf("Cutpoints: x = 1:9 (%d bins), y = 1:12 (%d bins)\n\n",
            length(cpx) + 1L, length(cpy) + 1L))

# ══════════════════════════════════════════════════════════════════════════
# TEST 1: Online Spearman (stationary) — record at every ngap observation,
#         compare against batch cor() at the same checkpoints.
#         Matches: online_corr(x, y, method="spearmanr", cutpoints_x=range(1,10),
#                               cutpoints_y=range(1,13))
# ══════════════════════════════════════════════════════════════════════════

cat("── Test 1: Online Spearman (stationary) ──\n")

est <- new_online_npcorr(cpx, cpy, "spearmanr")

online_corrs <- numeric(N %/% NGAP)
batch_corrs  <- numeric(N %/% NGAP)

for (i in seq_len(N)) {
  npcorr_update(est, x[i], y[i])
  if (i %% NGAP == 0) {
    idx <- i %/% NGAP
    online_corrs[idx] <- npcorr_value(est)
    batch_corrs[idx]  <- cor(x[1:i], y[1:i], method = "spearman")
  }
}

max_err <- max(abs(online_corrs - batch_corrs)[3:length(online_corrs)])
cat(sprintf("  final online:  %.8f\n", online_corrs[length(online_corrs)]))
cat(sprintf("  final batch:   %.8f\n", batch_corrs[length(batch_corrs)]))
cat(sprintf("  max |online - batch| (from obs 3 onward): %.2e  %s\n\n",
            max_err, if (max_err < 1e-8) "PASS" else "FAIL"))

# ══════════════════════════════════════════════════════════════════════════
# TEST 2: Online Spearman (moving window) — add newest, remove oldest,
#         compare against batch cor() on the same window.
#         Matches: online_mv_corr(x, y, nwin=100, method="spearmanr", ...)
# ══════════════════════════════════════════════════════════════════════════

cat(sprintf("── Test 2: Online Spearman (moving window, nwin=%d) ──\n", NWIN))

est_mv <- new_online_npcorr(cpx, cpy, "spearmanr")

n_checks <- (N - NWIN) %/% NGAP + 1L
online_mv <- numeric(n_checks)
batch_mv  <- numeric(n_checks)

for (i in seq_len(NWIN)) {
  npcorr_update(est_mv, x[i], y[i])
}
online_mv[1] <- npcorr_value(est_mv)
batch_mv[1]  <- cor(x[1:NWIN], y[1:NWIN], method = "spearman")

check_idx <- 2L
for (i in (NWIN + 1L):N) {
  npcorr_update(est_mv, x[i], y[i])
  npcorr_remove(est_mv, x[i - NWIN], y[i - NWIN])
  if ((i - NWIN + 1L) %% NGAP == 0) {
    online_mv[check_idx] <- npcorr_value(est_mv)
    batch_mv[check_idx]  <- cor(x[(i - NWIN + 1L):i], y[(i - NWIN + 1L):i], method = "spearman")
    check_idx <- check_idx + 1L
  }
}

max_err_mv <- max(abs(online_mv - batch_mv)[3:length(online_mv)])
cat(sprintf("  final online:  %.8f\n", online_mv[length(online_mv)]))
cat(sprintf("  final batch:   %.8f\n", batch_mv[length(batch_mv)]))
cat(sprintf("  max |online - batch| (from check 3 onward): %.2e  %s\n\n",
            max_err_mv, if (max_err_mv < 1e-8) "PASS" else "FAIL"))

# ══════════════════════════════════════════════════════════════════════════
# TEST 3: Online Spearman with ngap > 1 (matching Xiao's ngap=10 doctest)
# ══════════════════════════════════════════════════════════════════════════

NGAP10 <- 10L
cat(sprintf("── Test 3: Online Spearman (stationary, ngap=%d) ──\n", NGAP10))

est_gap <- new_online_npcorr(cpx, cpy, "spearmanr")

n_out <- N %/% NGAP10
online_gap <- numeric(n_out)
batch_gap  <- numeric(n_out)

for (i in seq_len(N)) {
  npcorr_update(est_gap, x[i], y[i])
  if (i %% NGAP10 == 0) {
    idx <- i %/% NGAP10
    online_gap[idx] <- npcorr_value(est_gap)
    batch_gap[idx]  <- cor(x[1:i], y[1:i], method = "spearman")
  }
}

max_err_gap <- max(abs(online_gap - batch_gap))
cat(sprintf("  max |online - batch| over %d checkpoints: %.2e  %s\n\n",
            n_out, max_err_gap, if (max_err_gap < 1e-8) "PASS" else "FAIL"))

cat("Done.\n")
