"""Section 6.6 loss figures: regret against static and one-switch oracles, loss on large-change days and by VIX regime,
Hedge against equal weights, and dependence strength against the MFF advantage. Usage: python3 plots_loss.py <workdir>"""
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

from real_common import (ECOL, EST, FIX, GRID, RC, STRATEGY_COLOUR, W, box_style, change, eval_days, image, outputs, plt, read,
                         result, verbose_print)

DOT_FIXED = (99, 199, 399)
DOT_FIXED_STYLE = {39: ("#A6A6A6", "D", 5.0), 99: ("#8a8a8a", "D", 7.5),
                   199: ("#6E6E6E", "s", 5.5), 399: ("#333333", "p", 6.5)}
C_OTHER, C_BEST, C_VFF, C_MFF = "#cfcfcf", "#333333", "#8E6BB8", "#C2185B"
SWITCH_LABEL = {"RMPL": "RMPL", "Counting matrix": "Counting matrix", "Hermite series": "Hermite"}
REGRET_FIX = [99, 199, 399]
REGRET_BARS = [f"fixed{k}" for k in REGRET_FIX] + ["VFF", "MFF"]
REGRET_XMAX = 7.5
CHANGE_FIXED = tuple(map(int, os.environ.get("FIXED", ",".join(map(str, DOT_FIXED))).split(",")))
CHANGE_TAG = "" if CHANGE_FIXED == DOT_FIXED else "_" + "_".join(map(str, CHANGE_FIXED))
CHANGE_REF = {"RMPL": (-6.93, -1.61), "Counting matrix": (-6.36, -1.13), "Hermite series": (-3.92, -3.68)}
VIX_REGIMES = [("calm", "Calm (VIX < 20)"), ("mid", "Moderate (20 ≤ VIX < 25)"), ("stress", "Stress (VIX ≥ 25)")]
VIX_REF = {"RMPL": (-1.79, -2.65, -8.67), "Counting matrix": (-2.40, -1.56, -5.94), "Hermite series": (-3.64, -2.45, -5.75)}
EMK = {"RMPL": "^", "Counting matrix": "o", "Hermite series": "s"}
PAIR_LABEL = {"XOM_CVX": "XOM/CVX", "SPY_IVV": "SPY/IVV"}
WIN = int(os.environ.get("WIN", "99"))


def x_grid(ax):
    ax.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def estimator_rows(ax):
    ax.set_yticks(range(len(EST)))
    ax.set_yticklabels([e for e, _ in EST])
    ax.tick_params(axis="y", length=0)


def save(fig, name):
    fig.savefig(image(name))
    plt.close(fig)


def dot_strategies(fixed=DOT_FIXED):
    return [(f"fixed{k}", rf"Fixed, $N_{{\mathrm{{eff}}}}={k}$", f"lt_fixed_{k}", *DOT_FIXED_STYLE[k])
            for k in fixed] + [
        ("vff", r"VFF, $\alpha=0.02$", "lt_vff", STRATEGY_COLOUR["vff_slow"], "^", 8.0),
        ("mff", "MFF", "lt_mffmix_t", STRATEGY_COLOUR["mff"], "o", 7.0),
    ]


def dot_strategy(ax, x, y, k, col, mk, ms):
    if k == "fixed99":   # hollow and on top: it nearly coincides with MFF for RMPL and the counting matrix
        ax.plot(x, y, marker=mk, ms=ms, mfc="none", mec=col, mew=1.1, ls="none", zorder=6)
    else:
        ax.plot(x, y, marker=mk, ms=ms, color=col, mec="white", mew=0.6, ls="none", zorder=3 if k.startswith("fixed") else 4)


def dot_panel(ax, val, ST, span, title):
    """One row per estimator, each strategy's level on a common-span axis; val(e, k) gives the level."""
    allv = [val(e, k) for e, _ in EST for k, *_ in ST]
    mid = 0.5 * (max(allv) + min(allv))
    for ei, (e, _) in enumerate(EST):
        xs = [val(e, k) for k, *_ in ST]
        ax.plot([min(xs), max(xs)], [ei, ei], color="#cfcec9", lw=1.6, zorder=2, solid_capstyle="round")
        for k, _, _, col, mk, ms in ST:
            dot_strategy(ax, val(e, k), ei, k, col, mk, ms)
    ax.set_xlim(mid - span / 2, mid + span / 2)
    ax.set_ylim(len(EST) - 0.45, -0.55)
    ax.set_title(title)
    estimator_rows(ax)
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    x_grid(ax)
    for ei in range(1, len(EST)):
        ax.axhline(ei - 0.5, color="#ecebe7", lw=0.6, zorder=1)


def dot_legend(fig, ST):
    h = [Line2D([], [], marker=mk, ls="none", ms=ms, mfc="none", mec=c, mew=1.1) if k == "fixed99" else
         Line2D([], [], marker=mk, ls="none", ms=ms, color=c, mec="white") for k, _, _, c, mk, ms in ST]
    fig.legend(h, [l for _, l, *_ in ST], loc="upper center", ncol=len(ST), frameon=False, fontsize=7, bbox_to_anchor=(0.55, 1.0))


def plot_loss_levels(conditions, strategies, loss, out, *, height, span_padding, label_y, top, bottom, hspace):
    """Loss by estimator and strategy, one panel per condition."""
    span = max(max(v) - min(v) for v in ([loss(e, k, c) for e, _ in EST for k, *_ in strategies] for c, _ in conditions))
    span *= span_padding
    with plt.rc_context(RC):
        fig, axes = plt.subplots(len(conditions), 1, figsize=(W, height), squeeze=False)
        for ax, (condition, title) in zip(axes[:, 0], conditions):
            dot_panel(ax, lambda e, k: loss(e, k, condition), strategies, span, title)
        fig.supxlabel(r"Mean logarithmic prediction loss ($\times10^{-3}$ nats per day per pair)", fontsize=8, y=label_y)
        dot_legend(fig, strategies)
        fig.subplots_adjust(left=0.19, right=0.975, top=top, bottom=bottom, hspace=hspace)
        save(fig, out)


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
    verbose_print(f"Static oracle vs minimum fixed loss, max difference {worst:.2e}")
    return pd.DataFrame(rows).groupby(["estimator", "m", "method"]).regret.mean()


def regret_panel(ax, R, m, title):
    bw = 0.13
    for ei, (e, _) in enumerate(EST):
        best = min((f"fixed{k}" for k in REGRET_FIX), key=lambda k: R[(e, m, k)])
        for i, k in enumerate(REGRET_BARS):
            v = R[(e, m, k)]
            y = ei + (i - (len(REGRET_BARS) - 1) / 2) * bw
            col = C_VFF if k == "VFF" else C_MFF if k == "MFF" else C_BEST if k == best else C_OTHER
            ax.barh(y, min(v, REGRET_XMAX), height=bw * 0.62, color=col, edgecolor="#555555", lw=0.6, zorder=3)
            if v > REGRET_XMAX:
                ax.plot([REGRET_XMAX * 0.955, REGRET_XMAX * 0.975], [y + bw * 0.3, y - bw * 0.3], color="white", lw=1.2, zorder=5)
                ax.text(REGRET_XMAX * 1.01, y, f"{v:.1f}", ha="left", va="center", fontsize=5.4, color="#52514e", clip_on=False)
            if k == best:
                ax.text(max(v, 0) + 0.12, y, k[5:], ha="left", va="center", fontsize=5.4, color=C_BEST)
    ax.axvline(0, color="#52514e", lw=0.8, zorder=4)
    ax.set_xlim(-3.0, REGRET_XMAX)
    ax.set_ylim(len(EST) - 0.5, -0.5)
    ax.set_title(title)
    estimator_rows(ax)
    x_grid(ax)


def plot_final_regret(S):
    """Static and one-switch regret by estimator and strategy."""
    R = regrets(S)
    R.to_csv(result("final_regret_values.csv"))
    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, 2, figsize=(W, 3.1), sharey=True, sharex=True)
        for ax, (m, title) in zip(axs, ((0, "Static regret ($m=0$)"), (1, "One-switch regret ($m=1$)"))):
            regret_panel(ax, R, m, title)
        fig.supxlabel(r"Average regret ($\times10^{-3}$ nats per day per pair)", fontsize=8, y=0.02)
        h = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="#555555", lw=0.6) for c in (C_OTHER, C_BEST, C_VFF, C_MFF)]
        fig.legend(h, [r"Fixed $N_{\mathrm{eff}}$ = 99, 199, 399 (top to bottom)", "Fixed, lowest average regret (labelled)",
                       "VFF", "MFF"], loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.55, 1.0),
                   handlelength=1.2, columnspacing=1.5)
        fig.subplots_adjust(left=0.19, right=0.965, top=0.8, bottom=0.15, wspace=0.12)
        save(fig, "ch6_practical_final_regret.png")


def np_mean_pairs(M, X):
    """Mean over pairs of each pair's mean of X over its days in mask M (pairs x days)."""
    return np.nanmean([np.nanmean(np.where(M[i], X[i], np.nan)) for i in range(len(X))])


def change_day_sets(S):
    roll = {p: eval_days(read(f).pipe(lambda x: x.assign(change=change(x)))).set_index("date") for p, f in outputs(S, "practical_roll2").items()}
    pairs = list(roll)
    idx = pd.DatetimeIndex(sorted(set().union(*[d.index for d in roll.values()])))
    CH = np.stack([roll[p].change.reindex(idx).values for p in pairs])
    present = ~np.isnan(CH)
    chg = (CH >= np.nanquantile(CH, 0.90, axis=1, keepdims=True)) & present
    return pairs, idx, [("other", "Other days", (~chg) & present), ("change", "Large estimated dependence-change days (top 10% per pair)", chg)]


def plot_change_days(S):
    """Loss on each pair's top 10% of 21-day dependence changes, and on other days."""
    ST = dot_strategies(CHANGE_FIXED)
    pairs, idx, sets = change_day_sets(S)
    rows = []
    for e, d in EST:
        P = {p: eval_days(read(f)).set_index("date") for p, f in outputs(S, d).items()}
        for k, _, c, *_ in ST:
            X = np.stack([-P[p][c].reindex(idx).values for p in pairs])
            rows += [dict(estimator=e, strategy=k, days=sk, loss=1e3 * np_mean_pairs(M, X)) for sk, _, M in sets]
    Lv = pd.DataFrame(rows)
    Lv.to_csv(result(f"change_days_levels{CHANGE_TAG}.csv"), index=False)

    def val(e, k, s):
        return Lv[(Lv.estimator == e) & (Lv.strategy == k) & (Lv.days == s)].loss.iloc[0]
    plot_loss_levels([(key, title) for key, title, _ in sets], ST, val, f"ch6_practical_final_change_days{CHANGE_TAG}.png",
                     height=3.4, span_padding=1.15, label_y=0.015, top=0.86, bottom=0.13, hspace=0.6)
    if 199 in CHANGE_FIXED:
        max_diff = max(abs(val(e, "mff", s) - val(e, "fixed199", s) - r)
                       for e, ref in CHANGE_REF.items() for s, r in zip(("change", "other"), ref))
        verbose_print(f"MFF − fixed 199 vs reference, max difference {max_diff:.3f}")


def plot_vix_regimes(S):
    """Loss by VIX regime."""
    ST = dot_strategies()
    vix = pd.read_csv(f"{S}/rerun/vix.csv", index_col=0).iloc[:, 0]
    vix.index = pd.to_datetime(vix.index)
    rows, ndays = [], {}
    for e, d in EST:
        for p, f in outputs(S, d).items():
            x = eval_days(read(f))
            v = vix.reindex(pd.DatetimeIndex(x.date)).ffill().values
            assert not np.isnan(v).any()
            for rk, m in {"calm": v < 20, "mid": (v >= 20) & (v < 25), "stress": v >= 25}.items():
                if e == "RMPL":
                    ndays[rk] = ndays.get(rk, 0) + int(m.sum())
                rows += [dict(estimator=e, pair=p, regime=rk, strategy=k, loss=-1e3 * x[c].values[m].mean()) for k, _, c, *_ in ST]
    Lv = pd.DataFrame(rows).groupby(["estimator", "regime", "strategy"]).loss.mean().reset_index()
    Lv.to_csv(result("vix_levels.csv"), index=False)

    def val(e, k, r):
        return Lv[(Lv.estimator == e) & (Lv.regime == r) & (Lv.strategy == k)].loss.iloc[0]
    max_diff = max(abs(round(val(e, "mff", r) - val(e, "vff", r), 2) - ref[i])
                   for e, ref in VIX_REF.items() for i, (r, _) in enumerate(VIX_REGIMES))
    verbose_print(f"MFF − VFF vs reference, max difference {max_diff:.3f}")
    plot_loss_levels(VIX_REGIMES, ST, val, "ch6_practical_appendix_vix.png",
                     height=4.6, span_padding=1.12, label_y=0.012, top=0.9, bottom=0.1, hspace=0.62)
    verbose_print("VIX pair-days", ndays)


def plot_hedge_equal(S):
    """Hedge MFF minus an equal-weight mixture of the same densities, across pairs."""
    CI = pd.read_csv(result("hedge_change_results.csv"))
    CI = CI[(CI.analysis == "hedge-equal") & (CI.days == "full")].set_index("estimator")
    vals = {}
    for e, d in EST:
        v = []
        for f in outputs(S, d).values():
            x = eval_days(read(f))
            eq = -np.log(np.mean(np.exp(np.stack([x[f"lt_fixed_{k}"].values for k in FIX])), axis=0))
            v.append(1e3 * np.mean(-x.lt_mffmix_t.values - eq))
        vals[e] = np.array(v)
    verbose_print("CHECK mean over pairs vs hedge_change_results.csv: max |diff| = %.2e" % max(abs(vals[e].mean() - CI.loc[e, "est"]) for e, _ in EST))
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(W, 1.9))
        for i, (e, _) in enumerate(EST):
            ax.boxplot(vals[e], positions=[i], orientation="horizontal", widths=0.5, patch_artist=True, **box_style(ECOL[e], 0.85))
            ax.plot([CI.loc[e, "lo"], CI.loc[e, "hi"]], [i - 0.36, i - 0.36], color="#111111", lw=1.6, solid_capstyle="butt", zorder=5)
            ax.plot(CI.loc[e, "est"], i - 0.36, marker="D", ms=4.2, mfc="white", mec="#111111", mew=0.8, ls="none", zorder=6)
        ax.axvline(0, color="#8a8984", lw=0.7, ls=(0, (3, 2)), zorder=1)
        estimator_rows(ax)
        ax.set_ylim(len(EST) - 0.55, -0.65)
        x_grid(ax)
        ax.set_xlabel(r"Mean prediction-loss difference: Hedge $-$ Equal ($\times10^{-3}$ nats/day)", fontsize=8)
        h = [Patch(facecolor="#b3b3b3", edgecolor="#555555", lw=0.9), Line2D([], [], color="#111111", lw=1.6, marker="D", ms=4.2, mfc="white", mec="#111111")]
        fig.legend(h, ["Spread across the 32 pairs", "Mean with 95% interval"], loc="upper center", ncol=2, frameon=False, fontsize=7,
                   bbox_to_anchor=(0.58, 1.0), columnspacing=2.0, handlelength=1.6)
        fig.subplots_adjust(left=0.19, right=0.975, top=0.85, bottom=0.25)
        save(fig, "ch6_practical_hedge_equal_mini.png")
    for e, _ in EST:
        v = vals[e]
        verbose_print(f"{e:16s} mean {v.mean():+.2f} [{CI.loc[e,'lo']:+.2f}, {CI.loc[e,'hi']:+.2f}] median {np.median(v):+.2f} "
                      f"IQR [{np.percentile(v,25):+.2f}, {np.percentile(v,75):+.2f}] range [{v.min():+.2f}, {v.max():+.2f}] pairs Hedge better {int((v < 0).sum())}/32")


def rolling_spearman_before(x, y, w):
    """Exact Spearman of the previous w pairs, held before each day."""
    out = np.full(len(x), np.nan)
    for t in range(w, len(x)):
        out[t] = np.corrcoef(pd.Series(x[t - w:t]).rank().values, pd.Series(y[t - w:t]).rank().values)[0, 1]
    return out


def strength_points(S):
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


def strength_panel(ax, g, e):
    s = spearmanr(g.mean_rho, g.mff_minus_vff)
    b1, b0 = np.polyfit(g.mean_rho, g.mff_minus_vff, 1)
    xs = np.linspace(g.mean_rho.min(), g.mean_rho.max(), 50)
    ax.axhline(0, color="#52514e", lw=0.8, zorder=1)
    ax.plot(xs, b0 + b1 * xs, color=ECOL[e], lw=0.9, ls=(0, (4, 2)), alpha=0.6, zorder=2)
    ax.text(0.04, 0.05, rf"$r_s = {s.correlation:.2f}$", transform=ax.transAxes, fontsize=7, color="#52514e", ha="left", va="bottom")
    gf, sp = g[g.pair != "SPY_IVV"], g[g.pair == "SPY_IVV"]
    ax.scatter(gf.mean_rho, gf.mff_minus_vff, s=16, marker=EMK[e], color=ECOL[e], edgecolor="white", lw=0.4, alpha=0.9, zorder=3)
    ax.scatter(sp.mean_rho, sp.mff_minus_vff, s=16, marker=EMK[e], facecolor="white", edgecolor=ECOL[e], lw=0.9, zorder=3)
    for _, r in g[g.pair.isin(PAIR_LABEL)].iterrows():
        spy = r.pair == "SPY_IVV"
        ax.annotate(PAIR_LABEL[r.pair], (r.mean_rho, r.mff_minus_vff), xytext=(-4, -4) if spy else (5, -3), textcoords="offset points",
                    fontsize=6.5, color="#52514e", ha="right" if spy else "left", va="top" if spy else "baseline")
    ax.set_title(e)
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(-28, 3)
    ax.set_xlim(0.42, 1.02)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
    return s


def plot_strength_vs_advantage(S):
    """Mean rolling dependence against the MFF - VFF loss difference."""
    Df = strength_points(S)
    Df.to_csv(result(f"strength_vs_advantage_w{WIN}.csv"), index=False)
    stats = {}
    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, 3, figsize=(W, 2.35), sharex=True, sharey=True)
        for ax, (e, _) in zip(axs, EST):
            stats[e] = strength_panel(ax, Df[Df.estimator == e], e)
        fig.supylabel("Mean prediction-loss difference:\n" + r"MFF $-$ VFF ($\times10^{-3}$ nats/day)", fontsize=8, x=0.012)
        fig.supxlabel(f"Mean rolling Spearman correlation ({WIN}-day window)", fontsize=8, y=0.015)
        fig.subplots_adjust(left=0.115, right=0.985, top=0.88, bottom=0.2, wspace=0.08)
        save(fig, "ch6_practical_strength_vs_advantage.png")
    for e, s in stats.items():
        verbose_print(f"{e:16s} Spearman rank correlation {s.correlation:+.3f} (p = {s.pvalue:.4f}); pairs below zero: {int((Df[Df.estimator == e].mff_minus_vff < 0).sum())}/32")
    r = Df.drop_duplicates("pair").set_index("pair").mean_rho
    verbose_print("mean rho range:", round(r.min(), 2), "-", round(r.max(), 2), "| SPY/IVV:", round(r["SPY_IVV"], 3))
    ex = Df[Df.pair != "SPY_IVV"]
    for e, _ in EST:
        g = ex[ex.estimator == e]
        s = spearmanr(g.mean_rho, g.mff_minus_vff)
        verbose_print(f"   excluding SPY/IVV: {e:16s} {s.correlation:+.2f} (p = {s.pvalue:.4f})")


def main():
    S = sys.argv[1]
    plot_final_regret(S)
    plot_change_days(S)
    plot_vix_regimes(S)
    plot_hedge_equal(S)
    plot_strength_vs_advantage(S)


if __name__ == "__main__":
    main()
