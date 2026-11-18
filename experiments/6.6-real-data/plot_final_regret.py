"""Compare real-data loss against static and one-switch oracles."""
import sys

import pandas as pd

from real_common import EST, FIX, GRID, RC, W, eval_days, image, outputs, plt, read, result, verbose_print

SWITCH_LABEL = {"RMPL": "RMPL", "Counting matrix": "Counting matrix", "Hermite series": "Hermite"}
C_OTHER, C_BEST, C_VFF, C_MFF = "#cfcfcf", "#333333", "#8E6BB8", "#C2185B"
PLOT_FIX = [99, 199, 399]
BARS = [f"fixed{k}" for k in PLOT_FIX] + ["VFF", "MFF"]
XMAX = 7.5


def regrets(S):
    Sw = pd.read_csv(result("switching_per_pair.csv"))
    Sw = Sw[Sw.copula == "t4"]
    rows, worst = [], 0.0
    for e, d in EST:
        for p, f in outputs(S, d).items():
            x = eval_days(read(f))
            L = {k: -1e3 * x[f"lt_fixed_{k}"].mean() for k in FIX}
            Lm = -1e3 * x.lt_mffmix_t.mean()
            for m in (0, 1):
                g = Sw[(Sw.estimator == SWITCH_LABEL[e]) & (Sw.pair == p) & (Sw.m == m)].set_index("method").regret
                orc = Lm - g["MFF"]
                if m == 0:
                    worst = max(worst, abs(orc - min(L.values())))
                rows += [dict(estimator=e, m=m, method=f"fixed{k}", pair=p, regret=L[k] - orc) for k in FIX]
                rows += [dict(estimator=e, m=m, method=k, pair=p, regret=g[k]) for k in ("VFF", "MFF")]
    print(f"CHECK static oracle from switching_per_pair.csv vs min fixed loss: max |diff| = {worst:.2e}")
    return pd.DataFrame(rows).groupby(["estimator", "m", "method"]).regret.mean()


def panel(ax, R, m, title):
    bw = 0.13
    for ei, (e, _) in enumerate(EST):
        best = min((f"fixed{k}" for k in PLOT_FIX), key=lambda k: R[(e, m, k)])
        for i, k in enumerate(BARS):
            v = R[(e, m, k)]
            y = ei + (i - (len(BARS) - 1) / 2) * bw
            col = C_VFF if k == "VFF" else C_MFF if k == "MFF" else C_BEST if k == best else C_OTHER
            ax.barh(y, min(v, XMAX), height=bw * 0.62, color=col, edgecolor="#555555", lw=0.6, zorder=3)
            if v > XMAX:
                ax.plot([XMAX * 0.955, XMAX * 0.975], [y + bw * 0.3, y - bw * 0.3], color="white", lw=1.2, zorder=5)
                ax.text(XMAX * 1.01, y, f"{v:.1f}", ha="left", va="center", fontsize=5.4, color="#52514e", clip_on=False)
            if k == best:
                ax.text(max(v, 0) + 0.12, y, k[5:], ha="left", va="center", fontsize=5.4, color=C_BEST)
    ax.axvline(0, color="#52514e", lw=0.8, zorder=4)
    ax.set_xlim(-3.0, XMAX)
    ax.set_ylim(len(EST) - 0.5, -0.5)
    ax.set_title(title)
    ax.set_yticks(range(len(EST)))
    ax.set_yticklabels([e for e, _ in EST])
    ax.tick_params(axis="y", length=0)
    ax.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def figure(R):
    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, 2, figsize=(W, 3.1), sharey=True, sharex=True)
        for ax, (m, title) in zip(axs, ((0, "Static regret ($m=0$)"), (1, "One-switch regret ($m=1$)"))):
            panel(ax, R, m, title)
        fig.supxlabel(r"Average regret ($\times10^{-3}$ nats per day per pair)", fontsize=8, y=0.02)

        def sq(c):
            return plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="#555555", lw=0.6)
        fig.legend([sq(C_OTHER), sq(C_BEST), sq(C_VFF), sq(C_MFF)],
                   [r"Fixed $N_{\mathrm{eff}}$ = 99, 199, 399 (top to bottom)", "Fixed, lowest average regret (labelled)",
                    "VFF", "MFF"], loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.55, 1.0),
                   handlelength=1.2, columnspacing=1.5)
        fig.subplots_adjust(left=0.19, right=0.965, top=0.8, bottom=0.15, wspace=0.12)
        fig.savefig(image("ch6_practical_final_regret.png"))
        plt.close(fig)


def main():
    R = regrets(sys.argv[1])
    R.to_csv(result("final_regret_values.csv"))
    figure(R)
    verbose_print(R.unstack("method")[BARS].round(2).to_string())


if __name__ == "__main__":
    main()
