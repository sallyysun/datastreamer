#!/usr/bin/env Rscript
# Hermite reference runner using the installed `hermiter` package (the R/Rcpp
# reference implementation). Feeds a shared (x,y) CSV one observation at a time
# via update_sequential and records spearmans() at every step, so the output
# lines up with the C++ harness step-for-step.
#
# Config mirrors the C++ HermiteEstimator: N=order, standardize=TRUE. lambda>0
# selects exponential weighting (exp_weight_lambda), matching the C++ EW path;
# lambda=0 (default) is cumulative (exp_weight_lambda=NA). spearmans() is called
# WITHOUT clipped so we compare the raw estimator arithmetic (the C++ side
# clamps to [-1,1], a no-op for the interior values seen here).
#
# Usage: hermite_reference.R <data.csv> [order] [lambda]
suppressMessages(library(hermiter))

args <- commandArgs(trailingOnly = TRUE)
csv <- args[[1]]
order <- if (length(args) >= 2) as.integer(args[[2]]) else 10L
lambda <- if (length(args) >= 3) as.numeric(args[[3]]) else 0
exp_w <- if (lambda > 0) lambda else NA

d <- read.csv(csv)
n <- nrow(d)

h <- hermite_estimator(N = order, standardize = TRUE,
                       exp_weight_lambda = exp_w, est_type = "bivariate")

out <- numeric(n)
for (i in seq_len(n)) {
  h <- update_sequential(h, c(d$x[i], d$y[i]))
  if (i < 2) {
    out[i] <- 0        # C++ harness (burn_in=1) emits 0 before n>=2
  } else {
    out[i] <- spearmans(h)
  }
}
writeLines(sprintf("%.17g", out))
