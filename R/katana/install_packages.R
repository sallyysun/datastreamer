# One-time setup: install the R packages mae_simulation_stationary.R needs
# into your personal Katana library. Run this once, interactively, on the
# login node (or via Katana OnDemand RStudio) -- NOT inside a PBS batch job,
# since compute nodes may not have outbound internet access to reach CRAN.
#
#   module load r/<version>   # see `module avail r` for the exact name
#   Rscript R/katana/install_packages.R
#
# hermiter compiles C++ (Rcpp) code, so this can take a few minutes; if it
# fails to compile, you likely also need a compiler module loaded alongside
# R -- try `module avail gcc`.

pkgs <- c("tidyverse", "future", "furrr", "hermiter")
install.packages(pkgs, repos = "https://cloud.r-project.org")
