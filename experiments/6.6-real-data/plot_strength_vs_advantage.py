"""Relate mean rolling dependence to the loss difference between MFF and VFF."""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from real_common import ECOL, EST, GRID, RC, W, eval_days, image, outputs, plt, read, result, verbose_print

EMK = {"RMPL": "^", "Counting matrix": "o", "Hermite series": "s"}
LABEL = {"XOM_CVX": "XOM/CVX", "SPY_IVV": "SPY/IVV"}
WIN = int(os.environ.get("WIN", "99"))


def rolling_spearman_before(x, y, w):
    """Exact Spearman of the previous w pairs, held before each day."""
    out = np.full(len(x), np.nan)
    for t in range(w, len(x)):
        out[t] = np.corrcoef(pd.Series(x[t - w:t]).rank().values, pd.Series(y[t - w:t]).rank().values)[0, 1]
    return out


def points(S):
    rows = []
    for p, f in outputs(S, "practical_roll2").items():
        xr = read(f)
        if WIN == 99:
            series = xr.rho_roll_99.values
        else:
            inp = pd.read_csv(f"{S}/rerun/in/{p}.csv")
            series = rolling_spearman_before(inp.x.values, inp.y.values, WIN)
        rho = eval_days(xr.assign(r=series)).r.mean()
        for e, d in EST:
            x = eval_days(read(f"{S}/{d}/out/{p}.csv"))
            rows.append(dict(pair=p, estimator=e, mean_rho=rho, mff_minus_vff=1e3 * ((-x.lt_mffmix_t).mean() - (-x.lt_vff).mean())))
    return pd.DataFrame(rows)


def panel(ax, g, e):
    s = spearmanr(g.mean_rho, g.mff_minus_vff)
    b1, b0 = np.polyfit(g.mean_rho, g.mff_minus_vff, 1)
    xs = np.linspace(g.mean_rho.min(), g.mean_rho.max(), 50)
    ax.axhline(0, color="#52514e", lw=0.8, zorder=1)
    ax.plot(xs, b0 + b1 * xs, color=ECOL[e], lw=0.9, ls=(0, (4, 2)), alpha=0.6, zorder=2)
    ax.text(0.04, 0.05, rf"$r_s = {s.correlation:.2f}$", transform=ax.transAxes, fontsize=7, color="#52514e", ha="left", va="bottom")
    gf, sp = g[g.pair != "SPY_IVV"], g[g.pair == "SPY_IVV"]
    ax.scatter(gf.mean_rho, gf.mff_minus_vff, s=16, marker=EMK[e], color=ECOL[e], edgecolor="white", lw=0.4, alpha=0.9, zorder=3)
    ax.scatter(sp.mean_rho, sp.mff_minus_vff, s=16, marker=EMK[e], facecolor="white", edgecolor=ECOL[e], lw=0.9, zorder=3)
    for _, r in g[g.pair.isin(LABEL)].iterrows():
        spy = r.pair == "SPY_IVV"
        ax.annotate(LABEL[r.pair], (r.mean_rho, r.mff_minus_vff), xytext=(-4, -4) if spy else (5, -3), textcoords="offset points",
                    fontsize=6.5, color="#52514e", ha="right" if spy else "left", va="top" if spy else "baseline")
    ax.set_title(e)
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(-28, 3)
    ax.set_xlim(0.42, 1.02)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
    return s


def figure(Df):
    stats = {}
    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, 3, figsize=(W, 2.35), sharex=True, sharey=True)
        for ax, (e, _) in zip(axs, EST):
            stats[e] = panel(ax, Df[Df.estimator == e], e)
        fig.supylabel("Mean prediction-loss difference:\n" + r"MFF $-$ VFF ($\times10^{-3}$ nats/day)", fontsize=8, x=0.012)
        fig.supxlabel(f"Mean rolling Spearman correlation ({WIN}-day window)", fontsize=8, y=0.015)
        fig.subplots_adjust(left=0.115, right=0.985, top=0.88, bottom=0.2, wspace=0.08)
        fig.savefig(image("ch6_practical_strength_vs_advantage.png"))
        plt.close(fig)
    return stats


def main():
    Df = points(sys.argv[1])
    Df.to_csv(result(f"strength_vs_advantage_w{WIN}.csv"), index=False)
    for e, s in figure(Df).items():
        verbose_print(f"{e:16s} Spearman rank correlation {s.correlation:+.3f} (p = {s.pvalue:.4f}); pairs below zero: {int((Df[Df.estimator == e].mff_minus_vff < 0).sum())}/32")
    r = Df.drop_duplicates("pair").set_index("pair").mean_rho
    verbose_print("mean rho range:", round(r.min(), 2), "-", round(r.max(), 2), "| SPY/IVV:", round(r["SPY_IVV"], 3))
    ex = Df[Df.pair != "SPY_IVV"]
    for e, _ in EST:
        g = ex[ex.estimator == e]
        s = spearmanr(g.mean_rho, g.mff_minus_vff)
        verbose_print(f"   excluding SPY/IVV: {e:16s} {s.correlation:+.2f} (p = {s.pvalue:.4f})")


if __name__ == "__main__":
    main()
