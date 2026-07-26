# Monte Carlo MAE study for the stationary case, replicating Stephanou &
# Varughese (2021), "Sequential estimation of Spearman rank correlation using
# Hermite series estimators", Journal of Multivariate Analysis 186, §6.1.
#
# Compares hermiter's online bivariate Spearman estimator (Hermite series, N)
# against the count-matrix estimator (online_npcorr.R, a port of
# onlineNPCORR), at one or more cut-point counts, over many i.i.d. bivariate
# normal replications per (n, rho).
#
# Two error metrics per replicate:
#   error2batch  |estimate - cor(x, y, method = "spearman")|
#     The finite-sample batch Spearman statistic. This is what the paper
#     calls "the exact Spearman correlation coefficient... for the sample"
#     (§6.1, steps 3-4) and is what its own MAE tables are computed against.
#     It is NOT the population value -- calling it "exact" only means
#     "computed non-sequentially over the whole sample," not "ground truth."
#   error2grade  |estimate - grade_correlation(rho)|
#     The population grade correlation. Arguably the more legitimate "true"
#     target for large n, but not what the paper's figures use -- tracked
#     here purely as an extra diagnostic.
#
# Usage:
#   source("R/mae_simulation_stationary.R")
#   run_mae_study(n_values = 1000L, m = 1000L)                    # quick draft
#   run_mae_study(n_values = c(1e4, 5e4, 1e5), m = 1000L)         # paper scale
#
# CLI:
#   Rscript R/mae_simulation_stationary.R
#   Rscript R/mae_simulation_stationary.R --n 10000,50000,100000 --m 1000
#
# Output: one summary CSV per replicated paper table (mean/SD only, no raw
# per-replicate data), written to R/mae_results/. Filenames carry a run_tag
# (default: derived from n and m) so different runs never overwrite each
# other -- see run_mae_study()'s `run_tag` argument.
#
# For running this at full paper scale on Katana (UNSW's PBS cluster) rather
# than locally, see R/katana/.

suppressPackageStartupMessages({
  library(tidyverse)
  library(future)
  library(furrr)
  library(hermiter)
})

# resolve script directory (same pattern as simulate.R)
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

# Only online_npcorr.R is sourced (not simulate.R): simulate.R runs full demo
# simulations unconditionally at the bottom of the file as a side effect of
# being source()'d, which we don't want triggered here.
source(file.path(SCRIPT_DIR, "online_npcorr.R"))

PROJ_ROOT <- normalizePath(file.path(SCRIPT_DIR, ".."))

# ── shared helpers ──────────────────────────────────────────────────────

grade_correlation <- function(rho) (6 / pi) * asin(rho / 2)

log_msg <- function(...) cat(..., file = stderr())

# ── cut-point construction ──────────────────────────────────────────────
# Two selectable ways to pick the count-matrix's cut-points:
#   theoretical - fixed quantiles of the standard normal distribution, per
#                 the paper's own definition of the count-matrix algorithm.
#   sample      - data-driven quantiles from a warmup subsample, the same
#                 approach R/simulate.R already uses for non-standard-normal
#                 streams.

theoretical_cutpoints <- function(n_cutpoints) qnorm((1:(n_cutpoints - 1)) / n_cutpoints)

sample_cutpoints <- function(warmup_x, warmup_y, n_cutpoints) {
  probs <- (1:(n_cutpoints - 1)) / n_cutpoints
  list(x = unname(quantile(warmup_x, probs = probs)),
       y = unname(quantile(warmup_y, probs = probs)))
}

build_cutpoints <- function(x, y, n_cutpoints, method = c("theoretical", "sample"), warmup_n = 50L) {
  method <- match.arg(method)
  if (method == "theoretical") {
    cp <- theoretical_cutpoints(n_cutpoints)
    list(x = cp, y = cp)
  } else {
    sample_cutpoints(x[seq_len(warmup_n)], y[seq_len(warmup_n)], n_cutpoints)
  }
}

# ── per-replicate simulation ────────────────────────────────────────────

simulate_stationary_rep <- function(n, rho,
                                     transform = c("identity", "exp"),
                                     hermiter_N = 20L,
                                     cutpoint_choices = c(20L, 30L),
                                     cutpoint_method = c("theoretical", "sample"),
                                     warmup_n = 50L) {
  transform <- match.arg(transform)
  cutpoint_method <- match.arg(cutpoint_method)
  transform_fn <- if (transform == "exp") exp else identity

  # Vectorized draw for the whole replicate at once (paper's bivariate normal
  # construction, same as simulate.R's gen_one(), vectorized over n). The
  # paper's own wording ("the exact same steps... are repeated for the count
  # matrix algorithm") doesn't say explicitly whether Hermite and the
  # count-matrix estimator see the same draw or independent ones each
  # replicate; here they share this one draw (paired comparison), which
  # lowers the variance of the head-to-head MAE difference vs. redrawing
  # per estimator.
  z1 <- rnorm(n)
  z2 <- rnorm(n)
  x <- transform_fn(z1)
  y <- transform_fn(rho * z1 + sqrt(1 - rho^2) * z2)

  batch_R <- cor(x, y, method = "spearman")
  grade_R <- grade_correlation(rho)

  herm <- hermite_estimator(N = hermiter_N, standardize = TRUE, est_type = "bivariate")

  npcs <- purrr::map(cutpoint_choices, function(cpts) {
    cp <- build_cutpoints(x, y, cpts, method = cutpoint_method, warmup_n = warmup_n)
    new_online_npcorr(cp$x, cp$y, "spearmanr")
  })
  names(npcs) <- paste0("matrix_c", cutpoint_choices)

  # Sequential online update: inherently non-vectorizable, since each
  # estimator's state depends on every prior observation. Vectorization
  # happened one level up in the data generation above; this loop is the only
  # part of a replicate that must run observation-by-observation.
  for (i in seq_len(n)) {
    herm <- update_sequential(herm, c(x[i], y[i]))
    for (j in seq_along(npcs)) npcorr_update(npcs[[j]], x[i], y[i])
  }

  estimates <- c(hermite = spearmans(herm),
                 vapply(npcs, npcorr_value, numeric(1)))

  tibble::tibble(
    estimator   = names(estimates),
    error2batch = abs(estimates - batch_R),
    error2grade = abs(estimates - grade_R)
  )
}

# ── study driver ─────────────────────────────────────────────────────────

run_mae_study <- function(n_values = 1000L,
                           rho_values = c(-0.75, -0.5, -0.25, 0.25, 0.5, 0.75),
                           m = 1000L,
                           hermiter_N = 20L,
                           cutpoint_choices = c(20L, 30L),
                           cutpoint_method = c("theoretical", "sample"),
                           transforms = c("identity", "exp"),
                           seed = 42,
                           parallel = TRUE,
                           workers = NULL,
                           run_tag = NULL,
                           out_dir = file.path(PROJ_ROOT, "R", "mae_results")) {
  cutpoint_method <- match.arg(cutpoint_method)

  # Every output filename is suffixed with this tag. Defaults to the n_values
  # and m actually used, so two separate calls/jobs with different scale
  # settings (e.g. a local n=1000 smoke test vs. a Katana n=100000 job, or
  # two Katana array tasks each covering a different n) never overwrite each
  # other's output -- pass an explicit run_tag to disambiguate further (e.g.
  # by cutpoint_method or seed) if you need that too.
  if (is.null(run_tag)) run_tag <- sprintf("n%s_m%d", paste(n_values, collapse = "-"), m)

  # ── parallel backend ──
  # Local laptop/desktop: `multisession` spins up background R processes on
  # this machine; `workers` defaults to future::availableCores() - 1.
  #
  # On Katana (PBS, via qsub -- see R/katana/), `multisession` still works
  # fine for a single-node job: request `#PBS -l select=1:ncpus=N:mem=...`
  # and pass `workers = N` (or `--workers N` from the CLI) explicitly to
  # match it -- don't rely on future::availableCores() on a shared cluster
  # node, it can report the whole node's core count rather than your
  # allocation. See R/katana/run_mae_study.pbs for a working example.
  #
  # For the heaviest combinations (n = 1e5), prefer splitting n_values across
  # a Katana job array (`#PBS -J`, one task per n) over looping through all
  # of them serially inside a single job -- a single n=1e5, m=1000
  # combination already dominates wall-clock time on its own. `run_tag`
  # (above) keeps each array task's output from colliding with the others;
  # see R/katana/run_mae_study_array.pbs.
  #
  # `furrr_options(seed = ...)` below is required for statistically
  # independent-but-reproducible RNG streams across parallel workers -- keep
  # it when adapting this code.
  if (parallel) {
    if (is.null(workers)) workers <- max(1L, future::availableCores() - 1L)
    future::plan(future::multisession, workers = workers)
  } else {
    future::plan(future::sequential)
  }
  on.exit(future::plan(future::sequential), add = TRUE)

  grid <- tidyr::expand_grid(n = n_values, transform = transforms, rho = rho_values)

  log_msg(sprintf(
    "MAE study | %d n-value(s) x %d transform(s) x %d rho-value(s), m=%d reps each | cutpoints=%s (%s) | hermiter_N=%d\n",
    length(n_values), length(transforms), length(rho_values), m,
    paste(cutpoint_choices, collapse = ","), cutpoint_method, hermiter_N
  ))

  # Each grid row's m replications run in parallel: every replicate builds
  # its own fresh estimator objects, so there's no shared mutable state
  # across workers (safe to parallelize at the replicate level). Memory per
  # replicate is O(n) and is discarded once it finishes, so this scales fine
  # in memory even at n = 1e5 -- the actual constraint at scale is
  # wall-clock, not memory.
  results <- purrr::pmap_dfr(grid, function(n, transform, rho) {
    log_msg(sprintf("  n=%-7d transform=%-8s rho=%+.2f ...\n", n, transform, rho))
    reps <- furrr::future_map_dfr(
      seq_len(m),
      function(rep_idx) simulate_stationary_rep(
        n = n, rho = rho, transform = transform,
        hermiter_N = hermiter_N, cutpoint_choices = cutpoint_choices,
        cutpoint_method = cutpoint_method
      ),
      .options = furrr::furrr_options(seed = seed)
    )
    dplyr::mutate(reps, n = .env$n, transform = .env$transform, rho = .env$rho, .before = 1)
  })

  # ── summarize (mean/SD, averaged across rho) ──
  summary_tbl <- results |>
    dplyr::group_by(n, transform, estimator) |>
    dplyr::summarise(
      dplyr::across(c(error2batch, error2grade), list(mean = mean, sd = sd), .names = "{.col}_{.fn}"),
      .groups = "drop"
    )

  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  # One CSV per (transform, cutpoints) combination actually present in this
  # study: Hermite vs. that single count-matrix cut-point choice. The four
  # canonical combinations Stephanou & Varughese (2021) tabulate get their
  # paper-table names; any other combination (e.g. a custom --cutpoints
  # value) still gets written, with a descriptive name instead -- otherwise
  # changing cutpoint_choices away from c(20, 30) would silently write zero
  # files despite the study having run correctly. Every filename also carries
  # `run_tag` (see above) so re-running at a different scale, or splitting
  # n_values across several Katana jobs, produces distinct files instead of
  # clobbering each other -- concatenate the matching files afterward (same
  # column schema, so `bind_rows()`/`rbind()` over all
  # mae_table1_normal_c20_*.csv is enough to reconstruct one combined table).
  paper_table_stems <- c(
    "identity.20" = "mae_table1_normal_c20",
    "identity.30" = "mae_table2_normal_c30",
    "exp.20"      = "mae_table3_lognormal_c20",
    "exp.30"      = "mae_table4_lognormal_c30"
  )

  combos <- tidyr::expand_grid(transform = transforms, cutpoints = cutpoint_choices)
  for (i in seq_len(nrow(combos))) {
    tr <- combos$transform[i]
    cp <- combos$cutpoints[i]
    key <- paste(tr, cp, sep = ".")
    stem <- if (key %in% names(paper_table_stems)) {
      unname(paper_table_stems[key])
    } else {
      sprintf("mae_summary_%s_c%d", tr, cp)
    }
    fname <- sprintf("%s_%s.csv", stem, run_tag)
    tbl <- summary_tbl |>
      dplyr::filter(transform == tr, estimator %in% c("hermite", paste0("matrix_c", cp))) |>
      dplyr::arrange(n, estimator)
    out_path <- file.path(out_dir, fname)
    readr::write_csv(tbl, out_path)
    log_msg(sprintf("  wrote %s\n", out_path))
  }

  log_msg("Done.\n")
  invisible(list(raw = results, summary = summary_tbl))
}

# ── CLI entry point ─────────────────────────────────────────────────────
# Only auto-runs when launched directly as `Rscript mae_simulation_stationary.R`
# (detected via the --file= flag Rscript sets). Unlike a plain !interactive()
# check, this correctly stays quiet when the file is source()'d from another
# non-interactive script (e.g. `Rscript -e 'source(...); run_mae_study(...)'`),
# which would otherwise trigger an unwanted default run before your own call.

if (any(grepl("^--file=", commandArgs(trailingOnly = FALSE)))) {
  args <- commandArgs(trailingOnly = TRUE)

  flag <- function(name, default) {
    idx <- match(name, args)
    if (is.na(idx) || idx == length(args)) return(default)
    args[idx + 1]
  }

  num_list <- function(name, default) {
    val <- flag(name, NULL)
    if (is.null(val)) return(default)
    as.numeric(strsplit(val, ",")[[1]])
  }

  # options
  #   --n <int,...>              sample size(s), comma-separated (default 1000)
  #   --m <int>                  Monte Carlo replications per (n, rho) (default 1000)
  #   --rho <float,...>          correlation value(s) (default -0.75,-0.5,-0.25,0.25,0.5,0.75)
  #   --hermiter_N <int>         Hermite series order (default 20)
  #   --cutpoints <int,...>      count-matrix cut-point count(s) (default 20,30)
  #   --cutpoint_method <str>    "theoretical" or "sample" (default theoretical)
  #   --transform <str,...>      "identity", "exp", or both comma-separated (default identity,exp)
  #   --seed <int>               RNG seed (default 42)
  #   --parallel <TRUE|FALSE>    use future/furrr multisession (default TRUE)
  #   --workers <int>            parallel worker count (default: available cores - 1).
  #                              On a shared HPC node, set this explicitly to match the
  #                              cores actually requested from the scheduler -- see
  #                              R/katana/ for PBS job scripts that do this.
  #   --run_tag <str>            output filename suffix (default: derived from --n and --m)
  run_mae_study(
    n_values         = as.integer(num_list("--n", 1000L)),
    m                = as.integer(flag("--m", 1000L)),
    rho_values       = num_list("--rho", c(-0.75, -0.5, -0.25, 0.25, 0.5, 0.75)),
    hermiter_N       = as.integer(flag("--hermiter_N", 20L)),
    cutpoint_choices = as.integer(num_list("--cutpoints", c(20L, 30L))),
    cutpoint_method  = flag("--cutpoint_method", "theoretical"),
    transforms       = strsplit(flag("--transform", "identity,exp"), ",")[[1]],
    seed             = as.numeric(flag("--seed", 42)),
    parallel         = as.logical(flag("--parallel", TRUE)),
    workers          = { w <- flag("--workers", NA); if (is.na(w)) NULL else as.integer(w) },
    run_tag          = flag("--run_tag", NULL)
  )
}
