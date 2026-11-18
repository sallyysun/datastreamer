"""Section 6.6 response figures: how far MFF, VFF and fixed memories move on each pair's largest dependence changes, and
MFF weights and effective memory around those episodes. Usage: python3 plots_response.py <workdir>"""
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from real_common import EST, EVAL_FROM, EVAL_ROW, FIX, GRID, RC, W, box_style, change, eval_days, image, outputs, plt, read, result, verbose_print

SER = [("fixed99", r"Fixed, $N_{\mathrm{eff}}=99$", "rho_fixed_99", "#A6A6A6"), ("fixed199", r"Fixed, $N_{\mathrm{eff}}=199$", "rho_fixed_199", "#6E6E6E"),
       ("fixed399", r"Fixed, $N_{\mathrm{eff}}=399$", "rho_fixed_399", "#333333"), ("vff", r"VFF, $\alpha=0.02$", "rho_vff", "#8E6BB8"),
       ("mff", "MFF", "rho_mff_t", "#C2185B")]
REF_MED = {"RMPL": (1.14, 0.19, 0.31, 0.60), "Counting matrix": (1.37, 0.25, 0.39, 0.71), "Hermite series": (1.42, 0.25, 0.39, 0.49)}
REF_DIR = {"RMPL": (85, 93, 93, 90), "Counting matrix": (88, 97, 97, 94), "Hermite series": (82, 91, 91, 90)}
FA = np.array(FIX, float)
MEMC = [plt.get_cmap("plasma")(v) for v in (0.82, 0.64, 0.47, 0.30, 0.10)]
MFFC, VFFC = "#C2185B", "#8E6BB8"
LAGS = np.arange(-63, 127)


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


def response_checks(Bp, Pl):
    wm = wd = 0
    for e in REF_MED:
        g = Pl[Pl.estimator == e].set_index("series")
        for j, k in enumerate(("fixed39", "fixed399", "vff", "mff")):
            if k in g.index:
                wm = max(wm, abs(round(g.loc[k, "pooled_median_ratio"], 2) - REF_MED[e][j]))
                wd = max(wd, abs(round(100 * g.loc[k, "pooled_same_direction"]) - REF_DIR[e][j]))
    verbose_print(f"Max reference differences  median ratio {wm:.2f}, direction agreement {wd:.0f} pp")
    zero = Bp[Bp.median_ratio == 0]
    assert set(map(tuple, zero[["estimator", "pair"]].values)) <= {("Hermite series", "SPY_IVV")}


def response_figure(Bp, Pl):
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


def response_summary(Bp, Pl):
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


def plot_response(S):
    """How far each series moves relative to the rolling Spearman on large-change days."""
    Bp, Pl = ratios(S)
    Bp.to_csv(result("dependence_response_by_pair.csv"), index=False)
    response_checks(Bp, Pl)
    response_figure(Bp, Pl)
    O = response_summary(Bp, Pl)
    O.to_csv(result("dependence_response_main.csv"), index=False)
    direction_table(O)
    counts = []
    for e, _ in EST:
        g = Bp[(Bp.estimator == e)].pivot(index="pair", columns="series", values="median_ratio")
        counts.append(int((g.mff > g.vff).sum()))
        verbose_print(f"{e:16s} MFF > VFF {counts[-1]}/32, ties {int((g.mff == g.vff).sum())}, "
                      f"MFF < fixed 99 {int((g.mff < g.fixed99).sum())}/32, MFF > fixed 199 {int((g.mff > g.fixed199).sum())}/32")
    assert counts == [29, 27, 23]


def eval_rows(x):
    return eval_days(x).reset_index(drop=True)


def episode_starts(roll):
    starts = {}
    for p, x in roll.items():
        D = eval_rows(x.assign(D=change(x))).D.values
        c = D >= np.nanquantile(D, 0.9)
        starts[p] = [t for t in range(len(c)) if c[t] and t >= 21 and not c[t - 21:t].any()]
    return starts


def episodes(S, roll, starts):
    res, full, vff_paths = {}, [], {}
    for e, d in EST:
        Wl, Ml, Vl, Mall, Vall, Hall, pm = [], [], [], [], [], [], []
        for p, f in outputs(S, d).items():
            x = eval_rows(read(f))
            assert (x.date.values == eval_rows(roll[p]).date.values).all()
            Wt = x[[f"wt_{k}" for k in FIX]].values
            mem, nv = Wt @ FA, x.neff_vff.values
            assert np.abs(mem - x.neff_mff_t.values).max() < 0.01, "sum_k w_k N_k must equal the saved neff_mff_t"
            vff_paths.setdefault(p, []).append(nv)
            Mall.append(mem)
            Vall.append(nv)
            Hall.append(1 / (Wt @ (1 / FA)))
            pm.append(mem.mean())
            for t in starts[p]:
                if t + LAGS[0] >= 0 and t + LAGS[-1] < len(x):
                    sl = t + LAGS
                    Wl.append(Wt[sl])
                    Ml.append(mem[sl])
                    Vl.append(nv[sl])
        Wl, Ml, Vl = np.array(Wl), np.array(Ml), np.array(Vl)
        res[e] = dict(W=Wl.mean(0), M=np.median(Ml, 0), Mq=np.percentile(Ml, [25, 75], 0), V=np.median(Vl, 0), Vq=np.percentile(Vl, [25, 75], 0), n=len(Wl))
        M, V, H = map(np.concatenate, (Mall, Vall, Hall))
        full.append(dict(estimator=e, mff_wavg_median=np.median(M), mff_wavg_q25=np.percentile(M, 25), mff_wavg_q75=np.percentile(M, 75),
                         mff_wavg_mean=M.mean(), pair_mean_min=min(pm), pair_mean_max=max(pm), mff_harmonic_median=np.median(H),
                         vff_median=np.median(V), vff_mean=V.mean()))
    vff_same = max(np.abs(a - b).max() for v in vff_paths.values() for a in v for b in v)
    return res, full, vff_same


def weights_figure(res):
    H = 3.1
    with plt.rc_context(RC):
        fig, axs = plt.subplots(2, 3, figsize=(W, H), sharex=True, sharey="row", gridspec_kw={"height_ratios": [1.35, 1]})
        for ci, (e, _) in enumerate(EST):
            r, a1, a2 = res[e], axs[0, ci], axs[1, ci]
            a1.stackplot(LAGS, *r["W"].T, colors=MEMC, lw=0, zorder=2)
            a1.set_ylim(0, 1)
            a1.set_title(e)
            a2.plot(LAGS, r["V"], color=VFFC, lw=1.1, zorder=3)
            a2.plot(LAGS, r["M"], color=MFFC, lw=1.1, zorder=4)
            a2.set_yscale("log")
            a2.set_ylim(30, 2600)
            a2.set_yticks(FIX)
            a2.set_yticklabels([str(k) for k in FIX])
            a2.minorticks_off()
            a2.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
            a2.set_axisbelow(True)
            for a in (a1, a2):
                a.set_xlim(LAGS[0], LAGS[-1])
                a.set_xticks([-50, 0, 50, 100])
                a.axvline(0, color="#111111", lw=0.8, ls=(0, (4, 3)), zorder=6)
        axs[0, 0].set_ylabel("Mean MFF weight")
        axs[1, 0].set_ylabel("Weight-averaged\ncandidate " + r"$N_{\mathrm{eff}}$")
        h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in MEMC] + [Line2D([], [], color=MFFC, lw=1.1), Line2D([], [], color=VFFC, lw=1.1)]
        fig.legend(h, [r"$N_{\mathrm{eff}}=%d$" % m for m in FIX] + ["MFF", "VFF"], loc="upper center", ncol=7, frameon=False, fontsize=7,
                   bbox_to_anchor=(0.53, 0.955), handlelength=1.2, columnspacing=1.2)
        fig.text(0.53, 0.01, "Trading days from the start of a large-change episode", ha="center", fontsize=8)
        fig.subplots_adjust(left=0.12, right=0.985, top=1 - 0.665 / H, bottom=0.42 / H, hspace=0.14, wspace=0.08)
        fig.savefig(image("ch6_practical_weights_ess.png"))
        plt.close(fig)


def weights_values(res):
    rows = []
    for e, _ in EST:
        r = res[e]
        for j, l in enumerate(LAGS):
            rows.append(dict(estimator=e, lag=l, **{f"w_{k}": r["W"][j, i] for i, k in enumerate(FIX)}, mff_wavg_ess_median=r["M"][j],
                             mff_wavg_ess_q25=r["Mq"][0, j], mff_wavg_ess_q75=r["Mq"][1, j], vff_ess_median=r["V"][j], n_episodes=r["n"]))
    return pd.DataFrame(rows)


def plot_weights_ess(S):
    """MFF weights and effective memory around large-change episodes."""
    roll = {p: read(f) for p, f in outputs(S, "practical_roll2").items()}
    starts = episode_starts(roll)
    res, full, vff_same = episodes(S, roll, starts)
    weights_figure(res)
    weights_values(res).to_csv(result("weights_ess_values.csv"), index=False)
    pd.DataFrame(full).to_csv(result("weights_ess_full_period.csv"), index=False)
    verbose_print(f"Episode starts {sum(len(v) for v in starts.values())}, max VFF ESS difference {vff_same:.3g}")


def main():
    plot_weights_ess(sys.argv[1])
    plot_response(sys.argv[1])


if __name__ == "__main__":
    main()
