# Section 6.6 data and outputs

- `rerun/in/`, `rerun/vix.csv`: the thesis inputs (32 EWMA-standardised pair return series and VIX, from the
  Yahoo download of 2026-11-15, last day 2026-11-11).
- `outputs/*.tar.xz`: the daily estimator outputs (`practical_rmpl`, `practical_xiao`, `practical_hermite`,
  `practical_roll2`) from `experiments/6.6-real-data/run.sh` on these inputs, with the exact Student t rho_S
  (reruns branch, commit 449169a). Unpack into one folder next to `rerun/` to rebuild the figures without SIM=1.
- `results/`: the 6.6 tables and summaries from those outputs.

Rebuild everything: copy `rerun/` into a folder S, then `SIM=1 experiments/6.6-real-data/run.sh S` (Linux).
