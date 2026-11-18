"""Plot Spearman RMSE around a Gaussian dependence jump."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402
import plot_style as ce  # noqa: E402

plt = ce.plt
EST = [(e, C.EST_LABEL[e]) for e in C.ESTIMATORS]
CMAP = plt.get_cmap("plasma")
MEM = [(f"{m}.000000", str(m), CMAP(C.MEMORY_CMAP_POS[m]), "-") for m in C.MEMORIES] + [
    ("nocum", "No forgetting", "#0b0b0b", (0, (3, 1.6)))]
JUMP, BEFORE, AFTER = C.CHANGE, 300, 500


def main():
    d = pd.read_csv(ce.R + "fixed_ess_jumppath.csv", dtype={"neff": str})
    d = d[(d.t >= JUMP - BEFORE) & (d.t <= JUMP + AFTER)]
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(1, 3, figsize=(ce.W, 2.6), sharey=True)
        for ax, (e, title) in zip(axs, EST):
            for k, lab, col, ls in MEM:
                g = d[(d.estimator == e) & (d.neff == k)].sort_values("t")
                ax.plot(g.t - JUMP, 100 * np.sqrt(g.mse), color=col, ls=ls, lw=0.9, label=lab, zorder=3)
            ax.axvline(0, color="#52514e", lw=0.8, ls=(0, (4, 3)), zorder=2)
            ax.set_title(title)
            ax.set_xlim(-BEFORE, AFTER)
            ax.set_ylim(0, 65)
            ax.set_xticks([-250, 0, 250, 500])
            ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0)
            ax.set_axisbelow(True)
        axs[0].set_ylabel(r"Spearman RMSE ($\times10^{-2}$)")
        fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=6, frameon=False, fontsize=7,
                   bbox_to_anchor=(0.53, 1.0), handlelength=2.0, columnspacing=1.1, title=r"$N_{\mathrm{eff}}$",
                   title_fontsize=7)
        fig.text(0.53, 0.015, "Observations relative to the jump", ha="center", fontsize=8)
        fig.subplots_adjust(left=0.08, right=0.975, top=0.78, bottom=0.18, wspace=0.12)
        out = "images/ch6_jump_ess_rmse.png"
        fig.savefig(out)


if __name__ == "__main__":
    main()
