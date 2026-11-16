"""Main-text figure: responsiveness to large estimated dependence changes (t4-scored MFF; practical-application style).
Revision of plot_dependence_response_box.py. On each pair's large-change days (top 10% of |rho99[t] - rho99[t-21]|, evaluation
days only; saved rho columns on row t are held before day t; rows in the pair's own trading days):
  R_t^A = |rho^A[t] - rho^A[t-21]| / |rho99[t] - rho99[t-21]|,  A = fixed ESS 39, fixed ESS 399, VFF, MFF (rho_mff_t).
Boxes: each pair's median R (one value per pair). Diamonds: pooled pair-day median (all large-change pair-days together), the
figure quoted in the text. SPY/IVV under the Hermite series has per-pair median exactly 0 for all four strategies (its Hermite
estimates sit at the rho = 1 boundary, so they do not move); it cannot go on a log axis, so it is left out of the Hermite boxes
(31 pairs) and drawn on a separate zero stub. The pooled diamonds include it.
Writes dependence_response_main.csv, dependence_response_by_pair.csv, tables/ch6_practical_direction_agreement.{tex,md} and
images/ch6_practical_response.png. From saved outputs only. Usage: python3 plot_main_response.py <scratch dir>"""
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
HERE = os.path.dirname(os.path.abspath(__file__)); S = sys.argv[1]
src = open(os.path.join(HERE, "plot_ch6style.py")).read()
exec(src[src.index("INK, GRID"):src.index("W = 6.1") + len("W = 6.1")])
EST = [("RMPL", "practical_rmpl"), ("Counting matrix", "practical_xiao"), ("Hermite series", "practical_hermite")]
SER = [("fixed99", r"Fixed, $N_{\mathrm{eff}}=99$", "rho_fixed_99", "#A6A6A6"), ("fixed199", r"Fixed, $N_{\mathrm{eff}}=199$", "rho_fixed_199", "#6E6E6E"), ("fixed399", r"Fixed, $N_{\mathrm{eff}}=399$", "rho_fixed_399", "#333333"),
       ("vff", r"VFF, $\alpha=0.02$", "rho_vff", "#8E6BB8"), ("mff", "MFF", "rho_mff_t", "#C2185B")]
roll = {os.path.basename(f)[:-4]: pd.read_csv(f, parse_dates=["date"]) for f in sorted(glob.glob(f"{S}/practical_roll2/out/*.csv"))}
rows, pooled = [], []
for e, d in EST:
    rat = {k: [] for k, *_ in SER}; same = {k: [] for k, *_ in SER}
    for f in sorted(glob.glob(f"{S}/{d}/out/*.csv")):
        p = os.path.basename(f)[:-4]; x = pd.read_csv(f, parse_dates=["date"]); r = roll[p]
        assert (x.date.values == r.date.values).all()
        dr = (r.rho_roll_99 - r.rho_roll_99.shift(21)).values; ev = (np.arange(len(x)) >= 560) & (x.date >= "2008-01-01").values
        c = ev & (np.abs(dr) >= np.nanquantile(np.abs(dr)[ev], 0.90))
        for k, _, col, _ in SER:
            dm = (x[col] - x[col].shift(21)).values; R = np.abs(dm[c]) / np.abs(dr[c]); sd = np.sign(dm[c]) == np.sign(dr[c])
            rat[k].append(R); same[k].append(sd)
            rows.append(dict(estimator=e, pair=p, series=k, median_ratio=np.median(R), same_direction=sd.mean(), n_days=int(c.sum())))
    for k, *_ in SER:
        R = np.concatenate(rat[k]); sd = np.concatenate(same[k])
        pooled.append(dict(estimator=e, series=k, pooled_median_ratio=np.median(R), pooled_same_direction=sd.mean(), n_pair_days=len(R)))
Bp = pd.DataFrame(rows); Pl = pd.DataFrame(pooled)
Bp.to_csv(os.path.join(HERE, "results", "dependence_response_by_pair.csv"), index=False)
# ------------------------------------------------------------------ checks against the verified values
REF_MED = {"RMPL": (1.14, 0.19, 0.31, 0.60), "Counting matrix": (1.37, 0.25, 0.39, 0.71), "Hermite series": (1.42, 0.25, 0.39, 0.49)}
REF_DIR = {"RMPL": (85, 93, 93, 90), "Counting matrix": (88, 97, 97, 94), "Hermite series": (82, 91, 91, 90)}
wm = wd = 0
for e in REF_MED:
    g = Pl[Pl.estimator == e].set_index("series")
    for j, k in enumerate(("fixed39", "fixed399", "vff", "mff")):
        if k not in g.index: continue
        wm = max(wm, abs(round(g.loc[k, "pooled_median_ratio"], 2) - REF_MED[e][j])); wd = max(wd, abs(round(100 * g.loc[k, "pooled_same_direction"]) - REF_DIR[e][j]))
print(f"CHECK pooled medians vs verified (2 dp): max |diff| = {wm:.2f}; direction agreement vs verified (whole %): max |diff| = {wd:.0f}")
zero = Bp[Bp.median_ratio == 0]
print("per-pair medians equal to 0:", zero[["estimator", "pair", "series"]].values.tolist())
assert set(map(tuple, zero[["estimator", "pair"]].values)) <= {("Hermite series", "SPY_IVV")}
# ------------------------------------------------------------------ figure
G = len(SER) + 1   # rows per estimator group, including the gap
with plt.rc_context(RC):   # linear axis: SPY/IVV's zero is an ordinary point, so every box uses all 32 pairs
    fig, ax = plt.subplots(figsize=(W, 3.7))
    yt = []
    for ei, (e, _) in enumerate(EST):
        for si, (k, lab, _, col) in enumerate(SER):
            y = ei * G + si; sub = Bp[(Bp.estimator == e) & (Bp.series == k)]; v = sub.median_ratio.values
            main = k in ("vff", "mff")
            ax.boxplot(v, positions=[y], orientation="horizontal", widths=0.78 if main else 0.5, patch_artist=True,
                       medianprops=dict(color="#111111", lw=2.0 if main else 1.5),   # box style of the Chapter 6 family-switch figure
                       boxprops=dict(facecolor=col, edgecolor="#555555", lw=0.9, alpha=0.85 if main else 0.45),
                       whiskerprops=dict(color="#555555", lw=0.9), capprops=dict(color="#555555", lw=0.9),
                       flierprops=dict(marker="o", ms=3.0, mfc="none", mec="#555555", mew=0.8))
            pm = Pl[(Pl.estimator == e) & (Pl.series == k)].pooled_median_ratio.item()
            ax.plot(pm, y, marker="D", ms=3.4, mfc="white", mec="#111111", mew=0.7, ls="none", zorder=6)
        yt.append(ei * G + (len(SER) - 1) / 2)
    for si, (k, *_, col) in enumerate(SER):   # SPY/IVV under the Hermite series: per-pair median exactly 0 (estimate pinned at rho = 1)
        ax.plot(0, 2 * G + si, marker="o", ms=3.6, mfc="white", mec=col, mew=1.0, ls="none", zorder=7)
    ax.text(0.025, 2 * G - 0.55, "SPY/IVV (no movement)", fontsize=6, color="#52514e", ha="left", va="center")
    ax.axvline(1.0, color="#9a9a9a", lw=0.9, ls=(0, (3, 2)), zorder=0)
    ax.set_xlim(-0.03, 1.2); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0"])
    ax.set_yticks(yt); ax.set_yticklabels([e for e, _ in EST]); ax.tick_params(axis="y", length=0); ax.invert_yaxis()
    ax.grid(True, axis="x", color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
    for ei in range(1, len(EST)): ax.axhline(ei * G - 1, color="#ecebe7", lw=0.6, zorder=0)
    ax.set_xlabel("Relative change in estimated Spearman correlation")
    h = [Patch(facecolor=c, edgecolor="#555555", lw=0.9, alpha=0.45 if k.startswith("fixed") else 0.85) for k, *_, c in SER]
    fig.legend(h, [l for _, l, _, _ in SER], loc="upper center", ncol=len(SER), frameon=False, fontsize=6.8,
               bbox_to_anchor=(0.575, 1.0), columnspacing=1.0, handlelength=1.2)
    h2 = [Line2D([], [], marker="D", ms=3.4, mfc="white", mec="#111111", ls="none"), Line2D([], [], color="#9a9a9a", lw=0.9, ls=(0, (3, 2)))]
    fig.legend(h2, ["Pooled pair-day median", "Equal movement"], loc="upper center", ncol=2, frameon=False, fontsize=6.8,
               bbox_to_anchor=(0.575, 0.955), columnspacing=1.5, handlelength=1.6)
    fig.subplots_adjust(left=0.165, right=0.975, top=0.85, bottom=0.14)
    fig.savefig(os.path.join(HERE, "images", "ch6_practical_response.png"))
    plt.close(fig)
# ------------------------------------------------------------------ values, direction-agreement table, summary
out = []
for e, _ in EST:
    for k, *_ in SER:
        sub = Bp[(Bp.estimator == e) & (Bp.series == k)]; v = sub.median_ratio
        pl = Pl[(Pl.estimator == e) & (Pl.series == k)].iloc[0]
        out.append(dict(estimator=e, series=k, n_pairs_in_box=len(v), box_median=v.median(), box_q25=v.quantile(0.25), box_q75=v.quantile(0.75),
                        box_min=v.min(), box_max=v.max(), pooled_median_ratio=pl.pooled_median_ratio, pooled_same_direction=pl.pooled_same_direction,
                        perpair_same_direction_median=sub.same_direction.median(), n_pair_days=pl.n_pair_days))
O = pd.DataFrame(out); O.to_csv(os.path.join(HERE, "results", "dependence_response_main.csv"), index=False)
os.makedirs(os.path.join(HERE, "results", "tables"), exist_ok=True)
hdr = ["Estimator"] + [l for _, l, _, _ in SER]
tex = [r"\begin{tabular}{l" + "c" * len(SER) + "}", r"\toprule", " & ".join(hdr) + r" \\", r"\midrule"]
md = ["| Estimator | " + " | ".join(l for _, l, _, _ in SER) + " |", "|---|" + "---:|" * len(SER)]
for e, _ in EST:
    g = O[O.estimator == e].set_index("series"); v = [f"{100 * g.loc[k, 'pooled_same_direction']:.0f}" for k, *_ in SER]
    tex.append(f"{e} & " + " & ".join(f"{x}\\%" for x in v) + r" \\"); md.append(f"| {e} | " + " | ".join(f"{x}%" for x in v) + " |")
tex += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(HERE, "results", "tables", "ch6_practical_direction_agreement.tex"), "w").write("\n".join(tex) + "\n")
open(os.path.join(HERE, "results", "tables", "ch6_practical_direction_agreement.md"), "w").write("\n".join(md) + "\n")
print(O.round(3).to_string(index=False))
for e, _ in EST:
    g = Bp[(Bp.estimator == e)].pivot(index="pair", columns="series", values="median_ratio")
    print(f"{e:16s} pairs with per-pair median MFF > VFF: {int((g.mff > g.vff).sum())}/32 (ties {int((g.mff == g.vff).sum())}); "
          f"pairs with MFF < fixed 99: {int((g.mff < g.fixed99).sum())}/32; pairs with MFF > fixed 199: {int((g.mff > g.fixed199).sum())}/32")
assert [int((Bp[Bp.estimator == e].pivot(index="pair", columns="series", values="median_ratio").pipe(lambda g: g.mff > g.vff)).sum()) for e, _ in EST] == [29, 27, 23]
