#!/usr/bin/env python3
"""Heatmap of multi-seed RMSE: estimators x scenarios, one panel per DGP family.

Run from experiments/ after ./run_multiseed_grid.sh. Reads
results/multiseed_<scenario>[_<family>].csv and writes
results/multiseed_heatmap.png.

Colour is normalised WITHIN each column, not globally: absolute RMSE spans
0.012 (constant) to 0.44 (oscillate), so one shared scale would render every
constant cell identically green and show nothing. Printed values are raw.
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

DATA_DIR = "results"
OUT = os.path.join(DATA_DIR, "multiseed_heatmap.png")

SCENARIOS = ["constant", "drift", "jump", "stochastic", "oscillate"]
FAMILIES = [("gaussian", ""), ("clayton", "_clayton"),
            ("gumbel", "_gumbel"), ("student_t", "_student_t")]

# Clayton and Gumbel are positive-dependence only, so on paths that cross zero
# the Archimedean parameter floors at independence and the realised ground truth
# is a different, easier signal. Those cells are readable down a column but not
# across families -- see kalman_changepoint_results.md 10.1.
CLIPPED = {"drift", "oscillate"}
CLIPPED_FAMILIES = {"clayton", "gumbel"}


def load(scenario, suffix):
    path = os.path.join(DATA_DIR, f"multiseed_{scenario}{suffix}.csv")
    with open(path) as fh:
        return {r["estimator"]: float(r["rmse_mean"]) for r in csv.DictReader(fh)}


def main():
    data = {(fam, s): load(s, suf) for fam, suf in FAMILIES for s in SCENARIOS}
    keys = list(data[("gaussian", "jump")].keys())

    fig, axes = plt.subplots(1, 4, figsize=(19, 9), sharey=True)
    for ax, (fam, _) in zip(axes, FAMILIES):
        M = np.array([[data[(fam, s)][k] for s in SCENARIOS] for k in keys])

        # per-column min-max normalisation for colour only
        col_min, col_max = M.min(axis=0), M.max(axis=0)
        shaded = (M - col_min) / np.where(col_max > col_min, col_max - col_min, 1.0)

        ax.imshow(shaded, cmap="RdYlGn_r", aspect="auto", norm=Normalize(0, 1))
        ax.set_xticks(range(len(SCENARIOS)))
        ax.set_xticklabels(
            [s + ("\n(clipped)" if s in CLIPPED and fam in CLIPPED_FAMILIES else "")
             for s in SCENARIOS], rotation=45, ha="right", fontsize=9)

        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                best = np.isclose(M[i, j], col_min[j])
                # RdYlGn_r is dark at both ends, so pick text colour by luminance
                dark_cell = shaded[i, j] > 0.62 or shaded[i, j] < 0.08
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7.5,
                        fontweight="bold" if best else "normal",
                        color="white" if dark_cell else "#222")

        for j, s in enumerate(SCENARIOS):
            if s in CLIPPED and fam in CLIPPED_FAMILIES:
                ax.add_patch(plt.Rectangle((j - .5, -.5), 1, len(keys),
                                           fill=False, ec="#1f77b4", lw=2, ls="--"))
        ax.set_title(f"{fam} DGP", fontsize=12, fontweight="bold")

    axes[0].set_yticks(range(len(keys)))
    axes[0].set_yticklabels(keys, fontsize=9)

    fig.suptitle(
        "Mean RMSE vs true Spearman rho -- 20 seeds, N=5000, burn-in 500\n"
        "bold = best in column; colour normalised within each column; "
        "dashed = ground-truth path clipped at independence, not comparable across families",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
