"""Compare MFF and VFF responses on each pair's largest dependence-change days."""
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from real_common import EST, EVAL_FROM, EVAL_ROW, GRID, RC, W, box_style, image, outputs, plt, read, result, verbose_print

SER = [("fixed99", r"Fixed, $N_{\mathrm{eff}}=99$", "rho_fixed_99", "#A6A6A6"), ("fixed199", r"Fixed, $N_{\mathrm{eff}}=199$", "rho_fixed_199", "#6E6E6E"),
       ("fixed399", r"Fixed, $N_{\mathrm{eff}}=399$", "rho_fixed_399", "#333333"), ("vff", r"VFF, $\alpha=0.02$", "rho_vff", "#8E6BB8"),
       ("mff", "MFF", "rho_mff_t", "#C2185B")]
REF_MED = {"RMPL": (1.14, 0.19, 0.31, 0.60), "Counting matrix": (1.37, 0.25, 0.39, 0.71), "Hermite series": (1.42, 0.25, 0.39, 0.49)}
REF_DIR = {"RMPL": (85, 93, 93, 90), "Counting matrix": (88, 97, 97, 94), "Hermite series": (82, 91, 91, 90)}


def ratios(S):
    roll = {p: read(f) for p, f in outputs(S, "practical_roll2").items()}
    rows, pooled = [], []
    for e, d in EST:
        rat, same = {k: [] for k, *_ in SER}, {k: [] for k, *_ in SER}
        for p, f in outputs(S, d).items():
            x, r = read(f), roll[p]
            assert (x.date.values == r.date.values).all()
            dr = (r.rho_roll_99 - r.rho_roll_99.shift(21)).values
            ev = (np.arange(len(x)) >= EVAL_ROW) & (x.date >= EVAL_FROM).values
            c = ev & (np.abs(dr) >= np.nanquantile(np.abs(dr)[ev], 0.90))
            for k, _, col, _ in SER:
                dm = (x[col] - x[col].shift(21)).values
                R, sd = np.abs(dm[c]) / np.abs(dr[c]), np.sign(dm[c]) == np.sign(dr[c])
                rat[k].append(R)
                same[k].append(sd)
                rows.append(dict(estimator=e, pair=p, series=k, median_ratio=np.median(R), same_direction=sd.mean(), n_days=int(c.sum())))
        for k, *_ in SER:
            R, sd = np.concatenate(rat[k]), np.concatenate(same[k])
            pooled.append(dict(estimator=e, series=k, pooled_median_ratio=np.median(R), pooled_same_direction=sd.mean(), n_pair_days=len(R)))
    return pd.DataFrame(rows), pd.DataFrame(pooled)


def checks(Bp, Pl):
    wm = wd = 0
    for e in REF_MED:
        g = Pl[Pl.estimator == e].set_index("series")
        for j, k in enumerate(("fixed39", "fixed399", "vff", "mff")):
            if k in g.index:
                wm = max(wm, abs(round(g.loc[k, "pooled_median_ratio"], 2) - REF_MED[e][j]))
                wd = max(wd, abs(round(100 * g.loc[k, "pooled_same_direction"]) - REF_DIR[e][j]))
    print(f"CHECK pooled medians vs verified (2 dp): max |diff| = {wm:.2f}; direction agreement vs verified (whole %): max |diff| = {wd:.0f}")
    zero = Bp[Bp.median_ratio == 0]
    verbose_print("per-pair medians equal to 0:", zero[["estimator", "pair", "series"]].values.tolist())
    assert set(map(tuple, zero[["estimator", "pair"]].values)) <= {("Hermite series", "SPY_IVV")}


def figure(Bp, Pl):
    G = len(SER) + 1
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(W, 3.7))
        yt = []
        for ei, (e, _) in enumerate(EST):
            for si, (k, _, _, col) in enumerate(SER):
                y = ei * G + si
                main = k in ("vff", "mff")
                v = Bp[(Bp.estimator == e) & (Bp.series == k)].median_ratio.values
                ax.boxplot(v, positions=[y], orientation="horizontal", widths=0.78 if main else 0.5, patch_artist=True,
                           **box_style(col, 0.85 if main else 0.45, median_lw=2.0 if main else 1.5, flier_ms=3.0))
                pm = Pl[(Pl.estimator == e) & (Pl.series == k)].pooled_median_ratio.item()
                ax.plot(pm, y, marker="D", ms=3.4, mfc="white", mec="#111111", mew=0.7, ls="none", zorder=6)
            yt.append(ei * G + (len(SER) - 1) / 2)
        for si, (k, *_, col) in enumerate(SER):
            ax.plot(0, 2 * G + si, marker="o", ms=3.6, mfc="white", mec=col, mew=1.0, ls="none", zorder=7)
        ax.text(0.025, 2 * G - 0.55, "SPY/IVV (no movement)", fontsize=6, color="#52514e", ha="left", va="center")
        ax.axvline(1.0, color="#9a9a9a", lw=0.9, ls=(0, (3, 2)), zorder=0)
        ax.set_xlim(-0.03, 1.2)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0"])
        ax.set_yticks(yt)
        ax.set_yticklabels([e for e, _ in EST])
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
        ax.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for ei in range(1, len(EST)):
            ax.axhline(ei * G - 1, color="#ecebe7", lw=0.6, zorder=0)
        ax.set_xlabel("Relative change in estimated Spearman correlation")
        h = [Patch(facecolor=c, edgecolor="#555555", lw=0.9, alpha=0.45 if k.startswith("fixed") else 0.85) for k, *_, c in SER]
        fig.legend(h, [l for _, l, _, _ in SER], loc="upper center", ncol=len(SER), frameon=False, fontsize=6.8,
                   bbox_to_anchor=(0.575, 1.0), columnspacing=1.0, handlelength=1.2)
        h2 = [Line2D([], [], marker="D", ms=3.4, mfc="white", mec="#111111", ls="none"), Line2D([], [], color="#9a9a9a", lw=0.9, ls=(0, (3, 2)))]
        fig.legend(h2, ["Pooled pair-day median", "Equal movement"], loc="upper center", ncol=2, frameon=False, fontsize=6.8,
                   bbox_to_anchor=(0.575, 0.955), columnspacing=1.5, handlelength=1.6)
        fig.subplots_adjust(left=0.165, right=0.975, top=0.85, bottom=0.14)
        fig.savefig(image("ch6_practical_response.png"))
        plt.close(fig)


def summary(Bp, Pl):
    out = []
    for e, _ in EST:
        for k, *_ in SER:
            sub = Bp[(Bp.estimator == e) & (Bp.series == k)]
            v = sub.median_ratio
            pl = Pl[(Pl.estimator == e) & (Pl.series == k)].iloc[0]
            out.append(dict(estimator=e, series=k, n_pairs_in_box=len(v), box_median=v.median(), box_q25=v.quantile(0.25), box_q75=v.quantile(0.75),
                            box_min=v.min(), box_max=v.max(), pooled_median_ratio=pl.pooled_median_ratio, pooled_same_direction=pl.pooled_same_direction,
                            perpair_same_direction_median=sub.same_direction.median(), n_pair_days=pl.n_pair_days))
    return pd.DataFrame(out)


def direction_table(O):
    tex = [r"\begin{tabular}{l" + "c" * len(SER) + "}", r"\toprule", " & ".join(["Estimator"] + [l for _, l, _, _ in SER]) + r" \\", r"\midrule"]
    md = ["| Estimator | " + " | ".join(l for _, l, _, _ in SER) + " |", "|---|" + "---:|" * len(SER)]
    for e, _ in EST:
        g = O[O.estimator == e].set_index("series")
        v = [f"{100 * g.loc[k, 'pooled_same_direction']:.0f}" for k, *_ in SER]
        tex.append(f"{e} & " + " & ".join(f"{x}\\%" for x in v) + r" \\")
        md.append(f"| {e} | " + " | ".join(f"{x}%" for x in v) + " |")
    tex += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(result("tables"), exist_ok=True)
    open(result("tables/ch6_practical_direction_agreement.tex"), "w").write("\n".join(tex) + "\n")
    open(result("tables/ch6_practical_direction_agreement.md"), "w").write("\n".join(md) + "\n")


def main():
    Bp, Pl = ratios(sys.argv[1])
    Bp.to_csv(result("dependence_response_by_pair.csv"), index=False)
    checks(Bp, Pl)
    figure(Bp, Pl)
    O = summary(Bp, Pl)
    O.to_csv(result("dependence_response_main.csv"), index=False)
    direction_table(O)
    verbose_print(O.round(3).to_string(index=False))
    counts = []
    for e, _ in EST:
        g = Bp[(Bp.estimator == e)].pivot(index="pair", columns="series", values="median_ratio")
        counts.append(int((g.mff > g.vff).sum()))
        verbose_print(f"{e:16s} pairs with per-pair median MFF > VFF: {counts[-1]}/32 (ties {int((g.mff == g.vff).sum())}); "
              f"pairs with MFF < fixed 99: {int((g.mff < g.fixed99).sum())}/32; pairs with MFF > fixed 199: {int((g.mff > g.fixed199).sum())}/32")
    assert counts == [29, 27, 23]


if __name__ == "__main__":
    main()
