"""Figure: 90% recovery time after the jump against N_eff (log-log) for each estimator, with the exponential-forgetting
prediction ln 0.1 / ln((N-1)/(N+1)) -> images/ch6_jump_recovery_time.png."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402
import plot_style as ce  # noqa: E402

plt = ce.plt
EST = [(e, C.EST_LABEL[e], C.EST_COLOUR_KEY[e], C.EST_MARKER[e]) for e in C.ESTIMATORS]


def main():
    r = pd.read_csv(ce.R + "jump_recovery.csv")
    r = r[(r.copula == "gaussian") & (r.neff != "No forgetting")].copy()
    r["N"] = r.neff.astype(float)
    with plt.rc_context(ce.RC):
        fig, ax = plt.subplots(figsize=(ce.W * 0.62, 2.6))
        nn = np.geomspace(30, 2600, 200)
        ax.plot(nn, np.log(0.1) / np.log(1 - 2 / (nn + 1)), color="#9a9a9a", lw=0.9, ls=(0, (4, 2)), zorder=2,
                label="Theoretical")
        for (e, lab, ckey, mk), off in zip(EST, [0.94, 1.0, 1.06]):   # offsets keep overlapping markers visible
            g = r[r.estimator == e].sort_values("N")
            ax.plot(g.N * off, g.recovery, ls="none", marker=mk, ms=4.5, color=ce.COL[ckey], label=lab, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(C.MEMORIES)
        ax.set_xticklabels([str(m) for m in C.MEMORIES])
        ax.set_yticks([50, 100, 200, 500, 1000, 2000])
        ax.set_yticklabels(["50", "100", "200", "500", "1000", "2000"])
        ax.minorticks_off()
        ax.set_xlim(30, 2700)
        ax.set_ylim(35, 3000)
        ax.set_xlabel(r"Effective sample size, $N_{\mathrm{eff}}$")
        ax.set_ylabel("90% recovery time (observations)")
        ax.grid(True, color=ce.GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False, fontsize=7, handlelength=1.6, borderaxespad=0.3, labelspacing=0.35)
        fig.subplots_adjust(left=0.16, right=0.97, top=0.96, bottom=0.17)
        out = "images/ch6_jump_recovery_time.png"
        fig.savefig(out)
        plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
