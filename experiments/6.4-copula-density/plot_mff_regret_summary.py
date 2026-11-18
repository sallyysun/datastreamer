"""Section 6.4 regret figures from results/regret_seeds.csv and regret_paths.csv (scenarios REGRET=1), mixture
forecaster: images/ch6_mff_regret_summary.png (final static and one-switch regret of fixed share, mean ± 2 SE) and
images/ch6_mff_regret_jump.png (cumulative regret through the jump, Hedge vs fixed share, ± 2 SE bands)."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402
import plot_style as ce  # noqa: E402

plt = ce.plt
OUT = "images/"
EST = [(e, C.EST_LABEL2[e]) for e in C.ESTIMATORS]
SC = [(s, C.SCENARIO_LABEL[s]) for s in C.SCENARIOS]
C_STAT, C_SW = "#3b4f7a", "#9fb0cf"
C_FS, C_H = "#0072B2", "#E69F00"
N = len(C.SEEDS)


def y_grid(ax):
    ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def summary(fs):
    with plt.rc_context(ce.RC):
        fig = plt.figure(figsize=(ce.W * 0.9, 2.45))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.32)
        axs = [fig.add_subplot(gs[0, 0])]
        axs.append(fig.add_subplot(gs[0, 1], sharey=axs[0]))
        axs.append(fig.add_subplot(gs[0, 2]))
        bw = 0.34
        for ax, (sc, scl) in zip(axs, SC):
            for ei, (e, el) in enumerate(EST):
                g = fs[(fs.scenario == sc) & (fs.estimator == e)]
                for i, (x, col) in enumerate([(g.loss_mix - g.best0, C_STAT), (g.loss_mix - g.best1, C_SW)]):
                    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
                    xp = ei + (i - 0.5) * bw
                    ax.bar(xp, m, width=bw * 0.92, color=col, zorder=3)
                    ax.errorbar(xp, m, yerr=2 * se, color="#3a3a3a", lw=0.8, capsize=0, zorder=4)
            ax.axhline(0, color="#222222", lw=0.8, zorder=2)
            ax.set_xticks(range(3))
            ax.set_xticklabels([el for _, el in EST], fontsize=6.8)
            ax.tick_params(axis="x", length=0)
            ax.set_xlim(-0.6, 2.6)
            y_grid(ax)
            ax.set_title(scl, fontsize=8, pad=3)
        axs[0].set_ylim(0, 5)
        axs[2].set_ylim(-19, 3)
        plt.setp(axs[1].get_yticklabels(), visible=False)
        axs[0].set_ylabel("Final cumulative regret (nats)", fontsize=7)
        h = [plt.Rectangle((0, 0), 1, 1, color=C_STAT), plt.Rectangle((0, 0), 1, 1, color=C_SW)]
        fig.legend(h, ["Static regret", "Switching regret"],
                   loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.54, 1.0), handlelength=1.2, columnspacing=1.6)
        fig.subplots_adjust(left=0.1, right=0.99, top=0.78, bottom=0.17)
        fig.savefig(OUT + "ch6_mff_regret_summary.png")
        plt.close(fig)
    print("wrote", OUT + "ch6_mff_regret_summary.png")


def jump(P, t0=2000):
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(2, 3, figsize=(ce.W * 0.9, 3.3), sharex=True, sharey="row")
        for ci, (e, _) in enumerate(EST):
            for ri, m in enumerate([0, 1]):
                ax = axs[ri, ci]
                for d, col in [(0.0, C_H), (1e-4, C_FS)]:
                    g = P[(P.scenario == "jump") & (P.estimator == e) & np.isclose(P.delta, d) & (P.m == m) & (P.t >= t0)].sort_values("t")
                    band = 2 * g.sd / np.sqrt(N)
                    ax.fill_between(g.t, g["mean"] - band, g["mean"] + band, color=col, alpha=0.15, lw=0, zorder=2)
                    ax.plot(g.t, g["mean"], color=col, ls="-", lw=1.3, zorder=3)
                ax.axhline(0, color="#b5b5b5", lw=0.5, zorder=1)
                ax.axvline(2500, color="#111111", lw=0.8, ls=(0, (4, 3)), zorder=4)
                y_grid(ax)
                ax.set_xlim(t0, 5000)
                ax.set_xticks([2500, 3500, 4500])
                if ri == 0:
                    ax.set_title(C.EST_LABEL[e], fontsize=8, pad=3)
        axs[0, 0].set_ylim(-18.5, 5)
        axs[1, 0].set_ylim(-4, 24)
        axs[0, 0].set_ylabel("Static regret\n(nats)", fontsize=7)
        axs[1, 0].set_ylabel("Switching regret,\n$\\leq$1 switch (nats)", fontsize=7)
        h = [plt.Line2D([], [], color=C_H, lw=1.3), plt.Line2D([], [], color=C_FS, lw=1.3)]
        fig.legend(h, [r"Ordinary Hedge ($\delta=0$)", r"Fixed share ($\delta=10^{-4}$)"], loc="upper center", ncol=2, frameon=False,
                   fontsize=7, bbox_to_anchor=(0.55, 1.0), handlelength=2.4, columnspacing=1.6)
        fig.text(0.55, 0.01, "Observation number, $n$", ha="center", fontsize=7.5)
        fig.subplots_adjust(left=0.13, right=0.99, top=0.86, bottom=0.12, hspace=0.12, wspace=0.08)
        fig.savefig(OUT + "ch6_mff_regret_jump.png")
        plt.close(fig)
    print("wrote", OUT + "ch6_mff_regret_jump.png")


def report(fs):
    for sc, _ in SC:
        for e, _ in EST:
            g = fs[(fs.scenario == sc) & (fs.estimator == e)]
            a, b = g.loss_mix - g.best0, g.loss_mix - g.best1
            print(f"{sc:9s} {e:8s} static {a.mean():7.2f} ± {2*a.std(ddof=1)/np.sqrt(len(a)):.2f}   "
                  f"switch1 {b.mean():6.2f} ± {2*b.std(ddof=1)/np.sqrt(len(b)):.2f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    S = pd.read_csv(ce.R + "regret_seeds.csv")
    P = pd.read_csv(ce.R + "regret_paths.csv")
    fs = S[np.isclose(S.delta, 1e-4)]
    summary(fs)
    jump(P[P.loss == "mix"])
    report(fs)


if __name__ == "__main__":
    main()
