"""Dependence strength versus MFF's advantage over VFF (t4 copula; Chapter 6 style). For each of the 32 pairs:
x = mean of the 99-day rolling Spearman estimate over the evaluation days; y = mean log loss of MFF minus VFF over the same
days (x1e-3 nats/day; negative = MFF lower). Panels: RMPL, counting matrix, Hermite. Least-squares line over all 32 pairs, for illustration only;
Spearman rank correlation and permutation-free p-value (scipy) printed. Writes strength_vs_advantage.csv and
images/ch6_practical_strength_vs_advantage.png. Usage: python3 plot_strength_vs_advantage.py <scratch dir>"""
import os, sys, glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); S = sys.argv[1]
src = open(os.path.join(HERE, "plot_ch6style.py")).read()
exec(src[src.index("INK, GRID"):src.index("W = 6.1") + len("W = 6.1")])
EST = [("RMPL", "practical_rmpl"), ("Counting matrix", "practical_xiao"), ("Hermite series", "practical_hermite")]
ECOL = {"RMPL": "#1baf7a", "Counting matrix": "#2a78d6", "Hermite series": "#eb6834"}   # current Chapter 6 estimator colours (COL in experiments/plot_constant_ess.py)
EMK = {"RMPL": "^", "Counting matrix": "o", "Hermite series": "s"}
LABEL = {"XOM_CVX": "XOM/CVX", "SPY_IVV": "SPY/IVV"}
def ev(x): x = x.iloc[560:]; return x[x.date >= "2008-01-01"]
WIN = int(os.environ.get("WIN", "99"))   # rolling window (trading days) for the typical-dependence measure
def rolling_spearman_before(x, y, w):
    """Exact sample Spearman of the previous w pairs (held before each day): ranks within each window, then correlation."""
    out = np.full(len(x), np.nan)
    for t in range(w, len(x)):
        rx = pd.Series(x[t - w:t]).rank().values; ry = pd.Series(y[t - w:t]).rank().values
        out[t] = np.corrcoef(rx, ry)[0, 1]
    return out
rows = []
for f in sorted(glob.glob(f"{S}/practical_roll2/out/*.csv")):
    p = os.path.basename(f)[:-4]; xr = pd.read_csv(f, parse_dates=["date"])
    inp = pd.read_csv(f"{S}/rerun/in/{p}.csv")
    if WIN == 99: rho_series = xr.rho_roll_99.values
    else: rho_series = rolling_spearman_before(inp.x.values, inp.y.values, WIN)
    rho = ev(xr.assign(r=rho_series)).r.mean()
    for e, d in EST:
        x = ev(pd.read_csv(f"{S}/{d}/out/{p}.csv", parse_dates=["date"]))
        rows.append(dict(pair=p, estimator=e, mean_rho=rho, mff_minus_vff=1e3 * ((-x.lt_mffmix_t).mean() - (-x.lt_vff).mean())))
Df = pd.DataFrame(rows); Df.to_csv(os.path.join(HERE, "results", f"strength_vs_advantage_w{WIN}.csv"), index=False)
stats = {}
with plt.rc_context(RC):
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.35), sharex=True, sharey=True)
    for ax, (e, _) in zip(axs, EST):
        g = Df[Df.estimator == e]; s = spearmanr(g.mean_rho, g.mff_minus_vff); stats[e] = s
        gf = g[g.pair != "SPY_IVV"]   # drawn filled; SPY/IVV (t4 mapping capped near rho_S = 1) drawn hollow but kept in the fit and correlation
        b1, b0 = np.polyfit(g.mean_rho, g.mff_minus_vff, 1); xs = np.linspace(g.mean_rho.min(), g.mean_rho.max(), 50)
        ax.axhline(0, color="#52514e", lw=0.8, zorder=1)
        ax.plot(xs, b0 + b1 * xs, color=ECOL[e], lw=0.9, ls=(0, (4, 2)), alpha=0.6, zorder=2)
        ax.text(0.04, 0.05, rf"$r_s = {s.correlation:.2f}$", transform=ax.transAxes, fontsize=7, color="#52514e", ha="left", va="bottom")
        ax.scatter(gf.mean_rho, gf.mff_minus_vff, s=16, marker=EMK[e], color=ECOL[e], edgecolor="white", lw=0.4, alpha=0.9, zorder=3)
        sp = g[g.pair == "SPY_IVV"]; ax.scatter(sp.mean_rho, sp.mff_minus_vff, s=16, marker=EMK[e], facecolor="white", edgecolor=ECOL[e], lw=0.9, zorder=3)
        for _, r in g[g.pair.isin(LABEL)].iterrows():
            spy = r.pair == "SPY_IVV"
            ax.annotate(LABEL[r.pair], (r.mean_rho, r.mff_minus_vff), xytext=(-4, -4) if spy else (5, -3), textcoords="offset points",
                        fontsize=6.5, color="#52514e", ha="right" if spy else "left", va="top" if spy else "baseline")
        ax.set_title(e); ax.grid(True, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True); ax.set_ylim(-28, 3)
        ax.set_xlim(0.42, 1.02); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); ax.set_xticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
    fig.supylabel("Mean prediction-loss difference:\n" + r"MFF $-$ VFF ($\times10^{-3}$ nats/day)", fontsize=8, x=0.012)
    fig.supxlabel(f"Mean rolling Spearman correlation ({WIN}-day window)", fontsize=8, y=0.015)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.88, bottom=0.2, wspace=0.08)
    fig.savefig(os.path.join(HERE, "images", "ch6_practical_strength_vs_advantage.png"))
    plt.close(fig)
for e, s in stats.items(): print(f"{e:16s} Spearman rank correlation {s.correlation:+.3f} (p = {s.pvalue:.4f}); pairs below zero: {int((Df[Df.estimator == e].mff_minus_vff < 0).sum())}/32")
g = Df.pivot(index="pair", columns="estimator", values="mff_minus_vff"); r = Df.drop_duplicates("pair").set_index("pair").mean_rho
print("mean rho range:", round(r.min(), 2), "-", round(r.max(), 2), "| SPY/IVV:", round(r["SPY_IVV"], 3))
ex = Df[Df.pair != "SPY_IVV"]
for e, _ in EST:
    gg = ex[ex.estimator == e]; s = spearmanr(gg.mean_rho, gg.mff_minus_vff); print(f"   excluding SPY/IVV: {e:16s} {s.correlation:+.2f} (p = {s.pvalue:.4f})")
