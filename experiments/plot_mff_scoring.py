"""Chapter 6.3 MFF scoring function (exploratory): squared vs log prediction loss for the weight update. Gaussian copula,
KLL margins, delta = 1e-4, eta = 1 for both (main MFF; squared's dev-selected eta is also 1). results/scores_seeds_eval.csv
(probe_strategies SCORES=1, seeds 42-101; see analyze_scores.py). Whole run, mean over seeds, ± 2 SE.
  ch6_mff_scoring_bars_<est>.png  3 x 3: rows = rho_S RMSE / H^2 / excess log loss, columns = scenarios, two bars
  ch6_mff_scoring_diff.png        paired difference squared - log (mean ± 2 SE), rows = metrics, x = scenarios,
                                  one marker per estimator
Usage (repo root): python3 experiments/plot_mff_scoring.py -> images/ch6/"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "experiments")
import plot_constant_ess as ce  # noqa: E402

plt, core = ce.plt, ce.core
SC = [("constant", "Constant"), ("drift", "Gradual drift"), ("jump", "Abrupt jump")]
MET = [("rmse", r"Spearman RMSE ($\times10^{-2}$)", 100), ("h2_x1000", r"$H^2$ ($\times10^{-3}$)", 1),
       ("excess_x1000", r"Excess log loss ($\times10^{-3}$)", 1)]
EST = [("rmpl", "RMPL", "mle"), ("xiao", "Counting Matrix", "xiao"), ("hermite", "Hermite Series", "hermite")]
RULES = [("squared", "Squared-loss MFF", "#9a9a9a"), ("log", "Log-loss MFF", "#C2185B")]
T = os.environ.get("DENS_TAG", "")   # mix: MFF density = mixture
ev = pd.read_csv(core.R + f"scores_seeds_eval{T}.csv")
ev = ev[(ev.window == "whole") & (ev.method.isin(["squared", "log"])) & np.isclose(ev.eta, 1.0)]


def vals(sc, e, rule, met, s):
    return ev[(ev.scenario == sc) & (ev.estimator == e) & (ev.method == rule)].set_index("seed")[met] * s


def bars(e, el):
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(3, 3, figsize=(core.W, 5.0))
        for ri, (met, ml, s) in enumerate(MET):
            for ci, (sc, scl) in enumerate(SC):
                ax = axs[ri, ci]; top = 0
                for i, (rule, _, col) in enumerate(RULES):
                    v = vals(sc, e, rule, met, s)
                    mu, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
                    ax.bar(i, mu, width=0.62, color=col, zorder=3)
                    ax.errorbar(i, mu, yerr=2 * se, color="#52514e", lw=0.8, capsize=0, zorder=4)
                    top = max(top, mu + 2 * se)
                ax.set_ylim(0, 1.15 * top); ax.set_xlim(-0.7, 1.7); ax.set_xticks([])
                ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
                if ri == 0: ax.set_title(scl)
            axs[ri, 0].set_ylabel(ml)
        h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in RULES]
        fig.legend(h, [l for _, l, _ in RULES], loc="upper center", ncol=2, frameon=False, fontsize=7,
                   bbox_to_anchor=(0.54, 1.0), title=f"{el} estimator", title_fontsize=7, handlelength=1.2)
        fig.subplots_adjust(left=0.12, right=0.985, top=0.87, bottom=0.03, hspace=0.18, wspace=0.32)
        out = f"images/ch6/ch6_mff_scoring_bars_{e}" + (f"_{T}" if T else "") + ".png"
        fig.savefig(out); plt.close(fig)
    print("wrote", out)


def bars_tight(e, el):
    """Compact version of bars(): narrower panels, adjacent bars, one-line legend. -> ..._tight.png"""
    SHORT = {"rmse": r"RMSE ($\times10^{-2}$)", "h2_x1000": r"$H^2$ ($\times10^{-3}$)", "excess_x1000": "Excess log loss\n" + r"($\times10^{-3}$)"}
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(3, 3, figsize=(core.W * 0.78, 3.7))
        for ri, (met, ml, s) in enumerate(MET):
            for ci, (sc, scl) in enumerate(SC):
                ax = axs[ri, ci]; top = 0
                for i, (rule, _, col) in enumerate(RULES):
                    v = vals(sc, e, rule, met, s)
                    mu, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
                    ax.bar(i, mu, width=0.38, color=col, zorder=3)
                    ax.errorbar(i, mu, yerr=2 * se, color="#52514e", lw=0.8, capsize=0, zorder=4)
                    top = max(top, mu + 2 * se)
                ax.set_ylim(0, 1.1 * top); ax.set_xlim(-0.55, 1.55); ax.set_xticks([])
                ax.yaxis.set_major_locator(plt.MaxNLocator(3)); ax.tick_params(axis="y", labelsize=6.5, pad=1)
                ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
                if ri == 0: ax.set_title(scl, fontsize=8, pad=3)
            axs[ri, 0].set_ylabel(SHORT[met], fontsize=7)
        h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in RULES]
        fig.legend(h, [l for _, l, _ in RULES], loc="upper center", ncol=2, frameon=False, fontsize=7,
                   bbox_to_anchor=(0.56, 1.0), handlelength=1.2, columnspacing=1.5)
        fig.subplots_adjust(left=0.16, right=0.99, top=0.88, bottom=0.02, hspace=0.12, wspace=0.42)
        out = f"images/ch6/ch6_mff_scoring_bars_{e}" + (f"_{T}" if T else "") + "_tight.png"
        fig.savefig(out); plt.close(fig)
    print("wrote", out)


def diffs():
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(1, 3, figsize=(core.W, 2.4))
        for ax, (met, ml, s) in zip(axs, MET):
            for j, (e, el, ck) in enumerate(EST):
                for ci, (sc, _) in enumerate(SC):
                    d = vals(sc, e, "squared", met, s) - vals(sc, e, "log", met, s)
                    mu, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
                    x = ci + (j - 1) * 0.22
                    ax.errorbar(x, mu, yerr=2 * se, color=ce.COL[ck], marker="o", ms=3.5, lw=1.0, capsize=0, zorder=3)
            ax.axhline(0, color="#333333", lw=0.8, zorder=2)
            ax.set_xticks(range(3)); ax.set_xticklabels(["Constant", "Gradual\ndrift", "Abrupt\njump"], fontsize=7)
            ax.tick_params(axis="x", length=0); ax.set_xlim(-0.5, 2.5)
            ax.set_title(ml, fontsize=8)
            ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        axs[0].set_ylabel("Squared-loss MFF $-$ log-loss MFF")
        h = [plt.Line2D([], [], color=ce.COL[ck], marker="o", ms=3.5, lw=1.0) for _, _, ck in EST]
        fig.legend(h, [el for _, el, _ in EST], loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.54, 1.0))
        fig.subplots_adjust(left=0.1, right=0.985, top=0.76, bottom=0.17, wspace=0.32)
        out = "images/ch6/ch6_mff_scoring_diff" + (f"_{T}" if T else "") + ".png"
        fig.savefig(out); plt.close(fig)
    print("wrote", out)


def bars_norm(e, el):
    """One row: per scenario, the three metrics divided by squared-loss MFF's mean (squared = 1). Both bars and their
    ± 2 SE are divided by the same squared-loss mean (the SEs are of the separate means, not of the paired difference;
    paired differences: ch6_mff_scoring_diff). -> ch6_mff_scoring_norm_<est>[_tag].png"""
    ML = ["Spearman\nRMSE", "$H^2$", "Excess\nlog loss"]
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(1, 3, figsize=(core.W * 0.85, 2.3), sharey=True)
        for ax, (sc, scl) in zip(axs, SC):
            for mi, (met, _, s) in enumerate(MET):
                vs = vals(sc, e, "squared", met, s); vl = vals(sc, e, "log", met, s)
                ms = vs.mean()
                for i, (v, col) in enumerate([(vs, RULES[0][2]), (vl, RULES[1][2])]):
                    x = mi + (i - 0.5) * 0.34
                    ax.bar(x, v.mean() / ms, width=0.3, color=col, zorder=3)
                    ax.errorbar(x, v.mean() / ms, yerr=2 * v.std(ddof=1) / np.sqrt(len(v)) / ms, color="#52514e", lw=0.8, capsize=0, zorder=4)
            ax.axhline(1, color="#333333", lw=0.6, ls=(0, (3, 2)), zorder=2)
            ax.set_xticks(range(3)); ax.set_xticklabels(ML, fontsize=6.8); ax.tick_params(axis="x", length=0)
            ax.set_xlim(-0.55, 2.55); ax.set_ylim(0, 1.25)
            ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
            ax.set_title(scl, fontsize=8, pad=3)
        axs[0].set_ylabel("Relative to squared-loss MFF", fontsize=7)
        h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in RULES]
        fig.legend(h, [l for _, l, _ in RULES], loc="upper center", ncol=2, frameon=False, fontsize=7,
                   bbox_to_anchor=(0.55, 1.0), handlelength=1.2, columnspacing=1.5)
        fig.subplots_adjust(left=0.1, right=0.99, top=0.8, bottom=0.17, wspace=0.08)
        out = f"images/ch6/ch6_mff_scoring_norm_{e}" + (f"_{T}" if T else "") + ".png"
        fig.savefig(out); plt.close(fig)
    print("wrote", out)


def ratio_fig(e, el, B=10000):
    """One row: log-loss / squared-loss MFF ratio of mean error per metric (squared = dashed line at 1), with a 95%
    paired percentile bootstrap interval (seeds resampled jointly for both scoring rules, ratio of resampled means;
    B = 10000, rng seed 20261115). Labels: percentage reduction. -> ch6_mff_scoring_ratio_<est>[_tag].png"""
    rng = np.random.default_rng(20261115)
    ML = ["Spearman\nRMSE", "$H^2$", "Excess\nlog loss"]
    rows = []
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(1, 3, figsize=(core.W * 0.8, 2.25), sharey=True)
        for ax, (sc, scl) in zip(axs, SC):
            for mi, (met, _, s) in enumerate(MET):
                vs = vals(sc, e, "squared", met, s); vl = vals(sc, e, "log", met, s).reindex(vs.index)
                S, Lg = vs.values, vl.values
                r = Lg.mean() / S.mean()
                idx = rng.integers(0, len(S), size=(B, len(S)))
                rb = Lg[idx].mean(axis=1) / S[idx].mean(axis=1)
                lo, hi = np.percentile(rb, [2.5, 97.5])
                rows.append((sc, met, r, lo, hi))
                ax.bar(mi, r, width=0.62, color=RULES[1][2], zorder=3)
                ax.errorbar(mi, r, yerr=[[r - lo], [hi - r]], color="#52514e", lw=0.9, capsize=0, zorder=4)
                ax.text(mi, lo - 0.03, f"\u2212{100 * (1 - r):.0f}%", ha="center", va="top", fontsize=6.3, color="white", zorder=5)
            ax.axhline(1, color="#555555", lw=0.8, ls=(0, (3, 2)), zorder=2)
            ax.set_xticks(range(3)); ax.set_xticklabels(ML, fontsize=6.8); ax.tick_params(axis="x", length=0)
            ax.set_xlim(-0.55, 2.55); ax.set_ylim(0, 1.12)
            ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
            ax.set_title(scl, fontsize=8, pad=3)
        axs[0].set_ylabel("Log-loss / squared-loss MFF", fontsize=7)
        fig.subplots_adjust(left=0.1, right=0.99, top=0.88, bottom=0.18, wspace=0.08)
        out = f"images/ch6/ch6_mff_scoring_ratio_{e}" + (f"_{T}" if T else "") + ".png"
        fig.savefig(out); plt.close(fig)
    print("wrote", out)
    for sc, met, r, lo, hi in rows:
        print(f"{sc:9s} {met:13s} ratio {r:.3f}  95% paired bootstrap [{lo:.3f}, {hi:.3f}]  reduction {100*(1-r):4.1f}% [{100*(1-hi):.1f}, {100*(1-lo):.1f}]")


if os.environ.get("RATIO"):
    ratio_fig("rmpl", "RMPL")
elif os.environ.get("NORM"):
    bars_norm("rmpl", "RMPL")
elif os.environ.get("TIGHT"):
    bars_tight("rmpl", "RMPL")
else:
    for e, el, _ in EST: bars(e, el)
    diffs()
