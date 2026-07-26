# Running mae_simulation_stationary.R on Katana

Katana is UNSW's PBS-based HPC cluster (`qsub`/`qstat`), documented at
https://docs.restech.unsw.edu.au/restech-docs-2/using_katana/running_jobs.html.
This folder has job scripts for running
[`R/mae_simulation_stationary.R`](../mae_simulation_stationary.R) at the
paper's full scale (n = 10<sup>4</sup>, 5×10<sup>4</sup>, 10<sup>5</sup>)
there instead of on a laptop.

## One-time setup

1. Get onto Katana (`ssh zXXXXXXX@katana.unsw.edu.au`) and clone or copy this
   repo there.
2. Find the R module: `module avail r` (the exact version string isn't
   pinned in these scripts — fill it into the `module load r/...` line in
   both `.pbs` files below once you know it).
3. Install the R packages this script needs, once, on the login node (NOT
   inside a batch job — compute nodes may not have outbound internet access
   to reach CRAN):
   ```bash
   module load r/<version>
   Rscript R/katana/install_packages.R
   ```
   `hermiter` compiles C++ (Rcpp) code and can take a few minutes; if it
   fails to compile, you likely also need a compiler module loaded alongside
   R (`module avail gcc`).

## Submitting a job

Two options, both submitted from the **repository root**:

- **`run_mae_study.pbs`** — one job, one multi-core node, runs all three `n`
  values (1e4, 5e4, 1e5) sequentially inside a single `Rscript` call.
  Simplest option; needs a longer walltime (~4.5h+ measured locally on 10
  cores — recalibrate once you've run it once on Katana).
  ```bash
  qsub R/katana/run_mae_study.pbs
  ```

- **`run_mae_study_array.pbs`** — a 3-task PBS job array, one task per `n`
  value, each on its own node. Shorter walltime per task, and tasks can run
  concurrently if the scheduler has capacity.
  ```bash
  qsub R/katana/run_mae_study_array.pbs
  ```

Fill in your email address (`#PBS -M`) in whichever script you use, or
delete the `-M`/`-m ae` lines if you don't want job-completion emails.

## Monitoring

```bash
qstat -u $USER
```

Output/error logs land in the submission directory as `<job name>.o<job id>`
(merged, since both scripts set `#PBS -j oe`).

## Output naming and merging

Every `run_mae_study()` call tags its output filenames with `run_tag`
(defaults to the `n`/`m` values used, or set explicitly by these job
scripts) specifically so separate runs — a local smoke test, one big Katana
job, or several array tasks — never overwrite each other's files. If you
used the array job, merge the per-`n` files for each table before comparing
against the paper, e.g. from R:

```r
library(tidyverse)
for (tbl in c("table1_normal_c20", "table2_normal_c30",
              "table3_lognormal_c20", "table4_lognormal_c30")) {
  list.files("R/mae_results", sprintf("^mae_%s_katana_n", tbl), full.names = TRUE) |>
    map_dfr(read_csv) |>
    write_csv(sprintf("R/mae_results/mae_%s_katana_combined.csv", tbl))
}
```

## Getting results back

From your local machine:
```bash
rsync -av zXXXXXXX@katana.unsw.edu.au:~/path/to/datastreamer/R/mae_results/ ./R/mae_results/
```

## What's *not* yet confirmed

- The exact R module name/version available on Katana (`module load r/...`
  is a placeholder in both scripts — fill it in after `module avail r`).
- Whether compute nodes have outbound internet access (relevant for
  `install.packages()` — assume login-node-only until confirmed).
- Real Katana walltime/memory usage at full scale — the numbers in these
  scripts are extrapolated from a 10-core laptop benchmark, not measured on
  Katana itself. Treat the first submission of either script as a
  calibration run, not a guaranteed-to-finish one.
