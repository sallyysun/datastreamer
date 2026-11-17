"""Main-text figure: MFF weight allocation and effective sample size around large estimated dependence-change episodes
(t4-scored MFF; practical-application style). Revision of plot_change_weights_memory.py.
Change days: each pair's top 10% of |rho99[t] - rho99[t-21]| (saved 99-day rolling Spearman, held before day t; pair's own
trading days; evaluation days only). Episode start (day 0): a change day with no change day in the preceding 21 rows.
Window -63..+126; pair-episodes without the full window are dropped.
  top row    mean MFF weight on each ESS arm (weights held before the day), stacked, mean over pair-episodes
  bottom row median over pair-episodes of the weight-averaged component ESS sum_k w_k N_eff,k,
             and of VFF's controller N_eff (the controller is shared by all three inner estimators, so that line is the same
             in every column; checked below)
Also writes weights_ess_values.csv and prints the full-period ESS summaries used in the text. From saved outputs only.
Usage: python3 plot_main_weights_ess.py <scratch dir>"""
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
HERE = os.path.dirname(os.path.abspath(__file__)); S = sys.argv[1]
src = open(os.path.join(HERE, "plot_ch6style.py")).read()
exec(src[src.index("INK, GRID"):src.index("W = 6.1") + len("W = 6.1")])
FIX = [39, 99, 199, 399, 1999]; FA = np.array(FIX, float)
CMAP = plt.get_cmap("plasma"); MEMC = [CMAP(v) for v in (0.82, 0.64, 0.47, 0.30, 0.10)]; MFFC, VFFC = "#C2185B", "#8E6BB8"   # Chapter 6 MEM colours (plot_mff_weights.py); strategy colours of plot_strategies.py
LAGS = np.arange(-63, 127)
EST = [("RMPL", "practical_rmpl"), ("Counting matrix", "practical_xiao"), ("Hermite series", "practical_hermite")]
def evaldays(x): x = x.iloc[560:]; return x[x.date >= "2008-01-01"].reset_index(drop=True)
roll = {os.path.basename(f)[:-4]: pd.read_csv(f, parse_dates=["date"]) for f in sorted(glob.glob(f"{S}/practical_roll2/out/*.csv"))}
chg = {}
for p, x in roll.items():
    D = evaldays(x.assign(D=(x.rho_roll_99 - x.rho_roll_99.shift(21)).abs())).D.values
    chg[p] = D >= np.nanquantile(D, 0.9)
onsets = {p: [t for t in range(len(c)) if c[t] and t >= 21 and not c[t - 21:t].any()] for p, c in chg.items()}
res, full, vff_paths = {}, [], {}
for e, d in EST:
    Wl, Ml, Vl = [], [], []; Mall, Vall, Hall, pm = [], [], [], []
    for f in sorted(glob.glob(f"{S}/{d}/out/*.csv")):
        p = os.path.basename(f)[:-4]; x = evaldays(pd.read_csv(f, parse_dates=["date"]))
        assert (x.date.values == evaldays(roll[p]).date.values).all()
        Wt = x[[f"wt_{k}" for k in FIX]].values; mem = Wt @ FA; nv = x.neff_vff.values
        assert np.abs(mem - x.neff_mff_t.values).max() < 0.01, "sum_k w_k N_k must equal the saved neff_mff_t"
        vff_paths.setdefault(p, []).append(nv)
        Mall.append(mem); Vall.append(nv); Hall.append(1 / (Wt @ (1 / FA))); pm.append(mem.mean())
        for t in onsets[p]:
            if t + LAGS[0] < 0 or t + LAGS[-1] >= len(x): continue
            sl = t + LAGS; Wl.append(Wt[sl]); Ml.append(mem[sl]); Vl.append(nv[sl])
    Wl, Ml, Vl = np.array(Wl), np.array(Ml), np.array(Vl)
    res[e] = dict(W=Wl.mean(0), M=np.median(Ml, 0), Mq=np.percentile(Ml, [25, 75], 0), V=np.median(Vl, 0), Vq=np.percentile(Vl, [25, 75], 0), n=len(Wl))
    M, V, H = map(np.concatenate, (Mall, Vall, Hall))
    full.append(dict(estimator=e, mff_wavg_median=np.median(M), mff_wavg_q25=np.percentile(M, 25), mff_wavg_q75=np.percentile(M, 75),
                     mff_wavg_mean=M.mean(), pair_mean_min=min(pm), pair_mean_max=max(pm), mff_harmonic_median=np.median(H),
                     vff_median=np.median(V), vff_mean=V.mean()))
vff_same = max(np.abs(a - b).max() for v in vff_paths.values() for a in v for b in v)
# ------------------------------------------------------------------ figure: Chapter 6 weights_fig() / nbar_axis() style
# (experiments/plot_mff_weights.py on bank-vff-study): plasma memory colours, short memory at the bottom of the stack,
# weight-averaged candidate N_eff lines without a band (MFF and VFF in the Chapter 6 strategy colours), dashed #111111 event marker, 6.1 in width, same rc.
H = 3.1
with plt.rc_context(RC):
    fig, axs = plt.subplots(2, 3, figsize=(W, H), sharex=True, sharey="row", gridspec_kw={"height_ratios": [1.35, 1]})
    for ci, (e, _) in enumerate(EST):
        r = res[e]; a1, a2 = axs[0, ci], axs[1, ci]
        a1.stackplot(LAGS, *r["W"].T, colors=MEMC, lw=0, zorder=2)
        a1.set_ylim(0, 1); a1.set_title(e)
        a2.plot(LAGS, r["V"], color=VFFC, lw=1.1, zorder=3); a2.plot(LAGS, r["M"], color=MFFC, lw=1.1, zorder=4)
        a2.set_yscale("log"); a2.set_ylim(30, 2600); a2.set_yticks(FIX); a2.set_yticklabels([str(k) for k in FIX]); a2.minorticks_off()
        a2.grid(True, axis="y", color=GRID, lw=0.5, zorder=0); a2.set_axisbelow(True)
        for a in (a1, a2):
            a.set_xlim(LAGS[0], LAGS[-1]); a.set_xticks([-50, 0, 50, 100])
            a.axvline(0, color="#111111", lw=0.8, ls=(0, (4, 3)), zorder=6)
    axs[0, 0].set_ylabel("Mean MFF weight")
    axs[1, 0].set_ylabel("Weight-averaged\ncandidate " + r"$N_{\mathrm{eff}}$")
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in MEMC] + [Line2D([], [], color=MFFC, lw=1.1), Line2D([], [], color=VFFC, lw=1.1)]
    fig.legend(h, [r"$N_{\mathrm{eff}}=%d$" % m for m in FIX] + ["MFF", "VFF"], loc="upper center", ncol=7, frameon=False, fontsize=7,
               bbox_to_anchor=(0.53, 0.955), handlelength=1.2, columnspacing=1.2)
    fig.text(0.53, 0.01, "Trading days from the start of a large-change episode", ha="center", fontsize=8)
    fig.subplots_adjust(left=0.12, right=0.985, top=1 - 0.665 / H, bottom=0.42 / H, hspace=0.14, wspace=0.08)
    fig.savefig(os.path.join(HERE, "images", "ch6_practical_weights_ess.png"))
    plt.close(fig)
# ------------------------------------------------------------------ values and checks
rows = []
for e, _ in EST:
    r = res[e]
    for j, l in enumerate(LAGS):
        rows.append(dict(estimator=e, lag=l, **{f"w_{k}": r["W"][j, i] for i, k in enumerate(FIX)}, mff_wavg_ess_median=r["M"][j],
                         mff_wavg_ess_q25=r["Mq"][0, j], mff_wavg_ess_q75=r["Mq"][1, j], vff_ess_median=r["V"][j], n_episodes=r["n"]))
pd.DataFrame(rows).to_csv(os.path.join(HERE, "results", "weights_ess_values.csv"), index=False)
pd.DataFrame(full).to_csv(os.path.join(HERE, "results", "weights_ess_full_period.csv"), index=False)
at = lambda l: int(np.where(LAGS == l)[0][0])
print(f"episode starts (all pairs): {sum(len(v) for v in onsets.values())}; with full window: " + ", ".join(f"{e} {res[e]['n']}" for e, _ in EST))
print(f"CHECK VFF controller N_eff identical across the three estimators: max |diff| = {vff_same:.3g}")
for e, _ in EST:
    r = res[e]
    print(f"{e:16s} lag -21/0/+21/+63/+126 | ESS39+99 weight " + "/".join(f"{r['W'][at(l), :2].sum():.2f}" for l in (-21, 0, 21, 63, 126))
          + " | MFF wavg ESS median " + "/".join(f"{r['M'][at(l)]:.0f}" for l in (-21, 0, 21, 63, 126))
          + " | VFF " + "/".join(f"{r['V'][at(l)]:.0f}" for l in (-21, 0, 21, 63, 126)))
print(pd.DataFrame(full).round(0).to_string(index=False))
