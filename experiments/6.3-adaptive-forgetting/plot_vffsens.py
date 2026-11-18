"""VFF learning-rate sensitivity figures (appendix to 6.3.2) from results/vffsens_summary.csv: per scenario and estimator,
the whole-run family-mean metric of VFF against alpha (± 2 SE), MFF as a dashed line with its ± 2 SE band, and the
development choice alpha = 0.02. Writes images/ch6_vffsens_rmse.png and ch6_vffsens_h2.png."""
import os
import sys
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

sys.path.insert(0, "../common")
import config as C  # noqa: E402
import plot_style as ce  # noqa: E402

plt = ce.plt
EST = [(e, C.EST_LABEL[e], C.EST_COLOUR_KEY[e]) for e in C.ESTIMATORS]
SC = [(s, C.SCENARIO_LABEL[s]) for s in C.SCENARIOS]
RATES = C.VFF_RATES
T = ("_" + os.environ["DENS_TAG"]) if os.environ.get("DENS_TAG") else ""


def load():
    S = pd.read_csv(ce.R + f"vffsens_summary{T}.csv")
    return S[(S.window == "whole") & (S.copula == "mean4")]


def panel(ax, g, ck):
    c = np.array(to_rgb(ce.COL[ck]))
    dark = tuple(c * 0.55)
    m, s = g.loc["mff", "mean"], g.loc["mff", "se"]
    ax.axhspan(m - 2 * s, m + 2 * s, color=dark, alpha=0.15, lw=0, zorder=1)
    ax.axhline(m, color=dark, lw=1.1, ls=(0, (4, 2)), zorder=2)
    ax.axvline(0.02, color="#9a9a9a", lw=0.7, ls=(0, (1, 1.5)), zorder=1)
    y = [g.loc[f"vff_{a:g}", "mean"] for a in RATES]
    e2 = [2 * g.loc[f"vff_{a:g}", "se"] for a in RATES]
    ax.errorbar(RATES, y, yerr=e2, color=tuple(c), lw=1.2, marker="o", ms=3, capsize=0, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([0.005, 0.02, 0.1])
    ax.set_xticklabels(["0.005", "0.02", "0.1"])
    ax.minorticks_off()
    ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def fig(S, metric, ylab, out):
    out = "images/" + out.replace(".png", T + ".png")
    with plt.rc_context(ce.RC):
        f, axs = plt.subplots(3, 3, figsize=(ce.W, 5.2), sharex=True, sharey="row")
        for (sc, scl), row in zip(SC, axs):
            for ax, (e, el, ck) in zip(row, EST):
                panel(ax, S[(S.metric == metric) & (S.scenario == sc) & (S.estimator == e)].set_index("strategy"), ck)
            row[0].set_ylabel(f"{scl}\n{ylab}")
        for ax, (_, el, _) in zip(axs[0], EST):
            ax.set_title(el)
        h = [plt.Line2D([], [], color="#6b6b6b", lw=1.2, marker="o", ms=3), plt.Line2D([], [], color="#3a3a3a", lw=1.1, ls=(0, (4, 2))),
             plt.Line2D([], [], color="#9a9a9a", lw=0.7, ls=(0, (1, 1.5)))]
        f.legend(h, ["VFF (± 2 SE)", "MFF (± 2 SE band)", r"$\alpha=0.02$ (development)"], loc="upper center", ncol=3,
                 frameon=False, fontsize=7, bbox_to_anchor=(0.53, 1.0))
        f.text(0.53, 0.01, r"VFF learning rate, $\alpha$ (log scale)", ha="center", fontsize=8)
        f.subplots_adjust(left=0.12, right=0.985, top=0.9, bottom=0.08, hspace=0.35, wspace=0.1)
        f.savefig(out)
        plt.close(f)
    print("wrote " + out)


def main():
    S = load()
    fig(S, "rmse", r"RMSE ($\times10^{-2}$)", "ch6_vffsens_rmse.png")
    fig(S, "h2_x1000", r"$H^2$ ($\times10^{-3}$)", "ch6_vffsens_h2.png")


if __name__ == "__main__":
    main()
