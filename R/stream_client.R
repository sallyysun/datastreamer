# R streaming client for datastreamer.
#
# Connects to the C++ server via WebSocket, receives market ticks, and runs
# two online Spearman estimators side-by-side:
#   1. hermiter  — Hermite-series bivariate estimator
#   2. npcorr    — cutpoint-matrix approach (ported from onlineNPCORR)
#
# Raw ticks are also logged to CSV for offline replay.
#
# Usage:
#   Rscript stream_client.R                           # defaults
#   Rscript stream_client.R ws://localhost:9002 500    # custom url + buffer size

library(websocket)
library(jsonlite)
library(hermiter)

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

source(file.path(SCRIPT_DIR, "ring_buffer.R"))
source(file.path(SCRIPT_DIR, "online_npcorr.R"))

# ── config ───────────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)

WS_URL      <- if (length(args) >= 1) args[1] else "ws://localhost:9002"
BUF_CAP     <- if (length(args) >= 2) as.integer(args[2]) else 512L
WINDOW      <- 100L
REPORT_EVERY <- 20L
CSV_PATH    <- file.path(SCRIPT_DIR, "stream_log.csv")

WARMUP      <- 30L

# ── state ────────────────────────────────────────────────────────────────

buf <- new_ring_buffer(BUF_CAP, ncol = 6L)

csv_con <- file(CSV_PATH, open = "wt")
writeLines("timestamp,price_x,price_y,volume_x,volume_y,return_x,return_y", csv_con)

herm_stat <- hermite_estimator(N = 30, standardize = TRUE, est_type = "bivariate")
herm_nonstat <- hermite_estimator(N = 30, standardize = TRUE, est_type = "bivariate",
                                  exp_weight_lambda = 0.01)

npc_spearman_stat <- NULL
npc_spearman_win  <- NULL

warmup_x <- numeric(WARMUP)
warmup_y <- numeric(WARMUP)

tick_count <- 0L

# ── helpers ──────────────────────────────────────────────────────────────

init_npcorr <- function() {
  qx <- quantile(warmup_x, probs = seq(0.1, 0.9, by = 0.1))
  qy <- quantile(warmup_y, probs = seq(0.1, 0.9, by = 0.1))

  npc_spearman_stat <<- new_online_npcorr(qx, qy, "spearmanr")
  npc_spearman_win  <<- new_online_npcorr(qx, qy, "spearmanr")

  for (k in seq_len(WARMUP)) {
    npcorr_update(npc_spearman_stat, warmup_x[k], warmup_y[k])
    npcorr_update(npc_spearman_win,  warmup_x[k], warmup_y[k])
  }
}

report <- function() {
  cat(sprintf("\n── tick %d ──\n", tick_count))

  sp_herm  <- tryCatch(spearmans(herm_stat),    error = function(e) NA)
  sp_herm_w <- tryCatch(spearmans(herm_nonstat), error = function(e) NA)

  sp_npc  <- if (!is.null(npc_spearman_stat)) npcorr_value(npc_spearman_stat) else NA
  sp_npc_w <- if (!is.null(npc_spearman_win)) npcorr_value(npc_spearman_win)  else NA

  cat(sprintf("  %-22s  %8s\n", "", "Spearman"))
  cat(sprintf("  %-22s  %8.4f\n", "hermiter (stationary)", sp_herm))
  cat(sprintf("  %-22s  %8.4f\n", "hermiter (weighted)",   sp_herm_w))
  cat(sprintf("  %-22s  %8.4f\n", "npcorr   (stationary)", sp_npc))
  cat(sprintf("  %-22s  %8.4f\n", "npcorr   (windowed)",   sp_npc_w))
  cat(sprintf("  buffer: %d / %d\n", ring_size(buf), BUF_CAP))
}

# ── websocket ────────────────────────────────────────────────────────────

cat(sprintf("connecting to %s ...\n", WS_URL))

ws <- WebSocket$new(WS_URL)

ws$onOpen(function(event) {
  cat("connected\n")
})

ws$onClose(function(event) {
  cat("disconnected\n")
  close(csv_con)
})

ws$onError(function(event) {
  cat("websocket error\n")
})

ws$onMessage(function(event) {
  msg <- tryCatch(fromJSON(event$data), error = function(e) NULL)
  if (is.null(msg)) return()

  px <- msg$price_x
  py <- msg$price_y
  vx <- if (!is.null(msg$volume_x)) msg$volume_x else 0
  vy <- if (!is.null(msg$volume_y)) msg$volume_y else 0
  rx <- if (!is.null(msg$return_x)) msg$return_x else 0
  ry <- if (!is.null(msg$return_y)) msg$return_y else 0
  ts <- if (!is.null(msg$timestamp)) msg$timestamp else as.numeric(Sys.time()) * 1000

  tick_count <<- tick_count + 1L

  # ── CSV log ──
  writeLines(paste(ts, px, py, vx, vy, rx, ry, sep = ","), csv_con)

  # ── ring buffer ──
  ring_push(buf, c(px, py, vx, vy, rx, ry))

  # ── hermiter update (bivariate: return_x, return_y) ──
  herm_stat     <<- update_sequential(herm_stat,     c(rx, ry))
  herm_nonstat  <<- update_sequential(herm_nonstat,  c(rx, ry))

  # ── npcorr: warmup then update ──
  if (tick_count <= WARMUP) {
    warmup_x[tick_count] <<- rx
    warmup_y[tick_count] <<- ry
    if (tick_count == WARMUP) init_npcorr()
  } else {
    npcorr_update(npc_spearman_stat, rx, ry)
    npcorr_update(npc_spearman_win,  rx, ry)
    if (npc_spearman_win$n > WINDOW) {
      old <- ring_get(buf, buf$head - WINDOW)
      npcorr_remove(npc_spearman_win, old[5], old[6])
    }
  }

  # ── periodic report ──
  if (tick_count %% REPORT_EVERY == 0) report()
})

cat("listening for ticks (Ctrl+C to stop) ...\n")

while (TRUE) {
  later::run_now(timeoutSecs = 1)
}
