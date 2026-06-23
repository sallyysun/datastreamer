# Port of wxiao0421/onlineNPCORR to R.
# Maintains a cutpoint-binned contingency matrix M that is updated O(m1*m2)
# per observation (m1, m2 = number of bins, typically small).
# Supports both cumulative (stationary) and sliding-window (non-stationary).

# ── helpers ──────────────────────────────────────────────────────────────

get_bin_index <- function(x, cutpoints) {
  # Binary-search bin assignment: returns 1-based index into (len(cutpoints)+1) bins.
  k <- length(cutpoints)
  if (x < cutpoints[1])   return(1L)
  if (x >= cutpoints[k])  return(k + 1L)
  # R's findInterval does exactly this
  findInterval(x, cutpoints) + 1L
}

spearmanr_from_matrix <- function(M, miplus, mplusj, n) {
  m1 <- length(miplus)
  m2 <- length(mplusj)

  rank_row <- numeric(m1)
  cum <- 0
  for (i in seq_len(m1)) {
    r <- miplus[i]
    if (r == 0) {
      rank_row[i] <- cum
    } else {
      rank_row[i] <- (cum + 1 + cum + r) / 2
      cum <- cum + r
    }
  }

  rank_col <- numeric(m2)
  cum <- 0
  for (j in seq_len(m2)) {
    r <- mplusj[j]
    if (r == 0) {
      rank_col[j] <- cum
    } else {
      rank_col[j] <- (cum + 1 + cum + r) / 2
      cum <- cum + r
    }
  }

  rbar <- (1 + n) / 2
  rd_row <- rank_row - rbar
  rd_col <- rank_col - rbar

  d1 <- sqrt(sum(rd_row^2 * miplus))
  d2 <- sqrt(sum(rd_col^2 * mplusj))
  if (d1 == 0 || d2 == 0) return(0)

  sum(diag(rd_row / d1) %*% M %*% diag(rd_col / d2))
}

# ── online Pearson (running sufficient statistics) ───────────────────────
# From Xiao's online_corr: maintains Sxy, Sxx, Syy, Sx, Sy.
# O(1) per update, O(1) memory.

new_online_pearson <- function() {
  env <- new.env(parent = emptyenv())
  env$Sxy <- 0
  env$Sxx <- 0
  env$Syy <- 0
  env$Sx  <- 0
  env$Sy  <- 0
  env$n   <- 0L
  env
}

pearson_update <- function(est, x, y) {
  est$Sxy <- est$Sxy + x * y
  est$Sxx <- est$Sxx + x^2
  est$Syy <- est$Syy + y^2
  est$Sx  <- est$Sx + x
  est$Sy  <- est$Sy + y
  est$n   <- est$n + 1L
  invisible(est)
}

pearson_remove <- function(est, x, y) {
  est$Sxy <- est$Sxy - x * y
  est$Sxx <- est$Sxx - x^2
  est$Syy <- est$Syy - y^2
  est$Sx  <- est$Sx - x
  est$Sy  <- est$Sy - y
  est$n   <- est$n - 1L
  invisible(est)
}

pearson_value <- function(est) {
  if (est$n < 3L) return(NA_real_)
  n  <- est$n
  Vx <- est$Sxx - (est$Sx^2) / n
  Vy <- est$Syy - (est$Sy^2) / n
  if (Vx <= 0 || Vy <= 0) return(NA_real_)
  (est$Sxy - est$Sx * est$Sy / n) / sqrt(Vx) / sqrt(Vy)
}

# ── estimator object ─────────────────────────────────────────────────────

new_online_npcorr <- function(cutpoints_x, cutpoints_y, method = "spearmanr") {
  m1 <- length(cutpoints_x) + 1L
  m2 <- length(cutpoints_y) + 1L

  env <- new.env(parent = emptyenv())
  env$method      <- method
  env$cutpoints_x <- cutpoints_x
  env$cutpoints_y <- cutpoints_y
  env$M           <- matrix(0, m1, m2)
  env$miplus      <- numeric(m1)
  env$mplusj      <- numeric(m2)
  env$n           <- 0L
  env
}

npcorr_update <- function(est, x, y) {
  i <- get_bin_index(x, est$cutpoints_x)
  j <- get_bin_index(y, est$cutpoints_y)
  est$M[i, j]  <- est$M[i, j] + 1
  est$miplus[i] <- est$miplus[i] + 1
  est$mplusj[j] <- est$mplusj[j] + 1
  est$n <- est$n + 1L
  invisible(est)
}

npcorr_remove <- function(est, x, y) {
  i <- get_bin_index(x, est$cutpoints_x)
  j <- get_bin_index(y, est$cutpoints_y)
  est$M[i, j]  <- est$M[i, j] - 1
  est$miplus[i] <- est$miplus[i] - 1
  est$mplusj[j] <- est$mplusj[j] - 1
  est$n <- est$n - 1L
  invisible(est)
}

npcorr_value <- function(est) {
  if (est$n < 3L) return(NA_real_)
  spearmanr_from_matrix(est$M, est$miplus, est$mplusj, est$n)
}
