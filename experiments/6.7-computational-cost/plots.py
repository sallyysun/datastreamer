"""Section 6.7 figures: retained state of the selected configurations, RMSE against retained state, processing time,
the two KLL implementations, and the RMSE gain from a larger budget."""
import itertools
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402
import plot_style as ce  # noqa: E402

plt = ce.plt
SC = [(s, C.SCENARIO_LABEL[s]) for s in C.SCENARIOS]
SEED_KEY = ["family", "scenario", "estimator", "approx", "gain", "seed"]
APPROX_LABELS = {"rmpl": "KLL $k$", "xiao": "bins", "hermite": "order"}
IMPL = {"mpl_kll": (ce.KLL_OWN, "v", "Hand-written KLL"), "mpl_dskll": (ce.ESTIMATOR["rmpl"], "^", "Apache DataSketches KLL")}


def budget_estimators(labels=C.EST_LABEL):
    return [("mpl_dskll" if e == "rmpl" else e, labels[e], ce.ESTIMATOR[e], C.EST_MARKER[e]) for e in C.ESTIMATORS]


def log2_axis(set_scale, set_ticks, set_labels, ticks):
    set_scale("log", base=2)
    set_ticks(ticks)
    set_labels([f"{v:g}" for v in ticks])


def budget_guides(ax):
    for budget in C.BUDGETS:
        ax.axvline(budget, color=ce.REF, lw=0.5, ls=":", zorder=0)


def seed_rmse(path):
    return pd.read_csv(path).set_index(SEED_KEY).rmse.sort_index()


def plot_storage():
    """Largest retained state of the validation-selected configurations at each budget."""
    sel = pd.read_csv(ce.R + "budget_selected.csv")
    sel = sel.assign(kib=sel.peak_val / 1024)
    EST = budget_estimators()
    bw = 0.26
    with plt.rc_context(ce.RC):
        fig, ax = plt.subplots(figsize=(ce.W * 0.72, 2.5))
        for i, b in enumerate(C.BUDGETS):
            ax.hlines(b, i - 0.45, i + 0.45, color=ce.EDGE, lw=0.8, ls=(0, (3, 2)), zorder=2)
            ax.text(i - 0.45, b * 1.07, "budget", va="bottom", ha="left", fontsize=6, color=ce.EDGE)
            for j, (e, lab, col, _) in enumerate(EST):
                top = sel[(sel.estimator == e) & (sel.budget_kib == b)].kib.max()
                xp = i + (j - 1) * bw
                ax.bar(xp, top, width=bw * 0.9, color=col, alpha=0.85, zorder=3, label=lab if i == 0 else None)
                ax.text(xp, top * 0.9, f"{top:.1f}", ha="center", va="top", fontsize=5.6, color="white")
        log2_axis(ax.set_yscale, ax.set_yticks, ax.set_yticklabels, [2, 4, 8, 16, 32, 64, 128, 256])
        ax.set_ylim(2, 400)
        ax.set_xlim(-0.55, 2.75)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{b} KiB" for b in C.BUDGETS])
        ax.tick_params(axis="x", length=0)
        ax.set_xlabel("Storage budget", fontsize=7.5)
        ax.set_ylabel("Maximum selected\nretained state (KiB)", fontsize=7.5)
        ce.grid(ax, "y")
        h, l = ax.get_legend_handles_labels()
        ce.top_legend(fig, h, l, ncol=3, fontsize=6.5, handlelength=1.2, columnspacing=1.0)
        fig.subplots_adjust(left=0.15, right=0.97, top=0.86, bottom=0.17)
        ce.save(fig, "ch6_budget_storage_n15k.png")


def frontier(val, SD, f, s, e):
    """(retained KiB, mean test RMSE x 10^-2) per approximation size, at its validation-best gain."""
    v = val[(val.family == f) & (val.scenario == s) & (val.estimator == e)]
    return [(r.peak_bytes / 1024, 100 * SD.loc[(f, s, e, r.approx, r.gain)].mean()) for _, r in v.loc[v.groupby("approx").rmse.idxmin()].iterrows()]


def memory_panel(ax, EST, fr):
    for e, lab, col, mk in EST:
        ax.plot([p[0] for p in fr[e]], [p[1] for p in fr[e]], color=col, marker=mk, ms=3.0, lw=1.1, label=lab)
    best = [min(p[1] for p in fr[e]) for e, *_ in EST]
    top = max(p[1] for p in fr["xiao"])
    lo = min(best)
    top = max(top, max(best) + 0.25 * (top - lo))
    ax.set_ylim(lo - 0.08 * (top - lo), top + 0.3 * (top - lo))
    budget_guides(ax)
    log2_axis(ax.set_xscale, ax.set_xticks, ax.set_xticklabels, [2, 8, 32, 128])
    ce.grid(ax, "y")
    ax.tick_params(labelsize=6.5)


def plot_memory_curves():
    """Test RMSE against retained state at each size's validation-best gain."""
    sw = pd.read_csv(ce.R + "budget_sweep_n15k.csv")
    val, SD = sw[sw.split == "val"], seed_rmse(ce.R + "budget_seeds_n15k.csv")
    EST = budget_estimators({e: f"{C.EST_LABEL[e]} ({APPROX_LABELS[e]})" for e in C.ESTIMATORS})
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(4, 3, figsize=(ce.W, 6.4), sharex=True)
        for (i, f), (j, (s, sl)) in itertools.product(enumerate(C.FAMILIES_T31), enumerate(SC)):
            memory_panel(axs[i, j], EST, {e: frontier(val, SD, f, s, e) for e, *_ in EST})
            if i == 0:
                axs[i, j].set_title(sl, fontsize=8, pad=3, loc="left")
            if j == 0:
                axs[i, j].set_ylabel(C.FAMILY_LONG[f], fontsize=7.5)
        h, l = axs[0, 0].get_legend_handles_labels()
        ce.top_legend(fig, h, l, ncol=3, handlelength=1.6, columnspacing=1.2)
        ce.bottom_label(fig, "Retained state (KiB, log scale); dotted lines mark the 16, 64 and 256 KiB budgets", fontsize=7.5)
        fig.text(0.005, 0.5, "Spearman RMSE ($\\times10^{-2}$)", rotation=90, va="center", fontsize=7.5)
        fig.subplots_adjust(left=0.1, right=0.99, top=0.93, bottom=0.07, hspace=0.18, wspace=0.22)
        ce.save(fig, "ch6_budget_memory_curves_n15k.png")


def plot_processing_time():
    """Time per observation against retained state."""
    T = pd.read_csv(ce.R + "budget_time_table_n15k.csv")
    with plt.rc_context(ce.RC):
        fig, ax = plt.subplots(figsize=(ce.W * 0.62, 2.5))
        for e, lab, c, m in budget_estimators():
            x = T[T.estimator == e]
            ax.plot(x.kib, x.us, color=c, marker=m, ms=3, lw=1.1, label=lab)
        log2_axis(ax.set_xscale, ax.set_xticks, ax.set_xticklabels, [1, 4, 16, 64, 256])
        log2_axis(ax.set_yscale, ax.set_yticks, ax.set_yticklabels, [0.25, 1, 4, 16, 64])
        budget_guides(ax)
        ce.grid(ax)
        ax.set_xlabel("Retained state (KiB)", fontsize=7.5)
        ax.set_ylabel("Time per observation (µs)", fontsize=7.5)
        ax.legend(frameon=False, fontsize=6.8, loc="upper left")
        fig.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.17)
        ce.save(fig, "ch6_budget_processing_time.png")


def kll_inputs():
    F, S = C.FAMILIES_T31, C.SCENARIOS
    own_s = pd.concat([pd.read_csv(ce.R + f"budget_sweep_n15k_ownkll_{s}_{f}.csv") for s in S for f in F])
    own_d = pd.concat([pd.read_csv(ce.R + f"budget_seeds_n15k_ownkll_{s}_{f}.csv") for s in S for f in F])
    ds_s, ds_d = pd.read_csv(ce.R + "budget_sweep_n15k.csv"), pd.read_csv(ce.R + "budget_seeds_n15k.csv")
    sw = pd.concat([own_s, ds_s[ds_s.estimator == "mpl_dskll"]])
    sd = pd.concat([own_d, ds_d[ds_d.estimator == "mpl_dskll"]])
    SD = sd.set_index(SEED_KEY).rmse.sort_index()
    T = pd.concat([pd.read_csv(ce.R + f"budget_timing_n15k_kll_rep{r}.csv") for r in range(1, 6)])
    TT = T.groupby(["estimator", "approx"]).agg(us=("mean_ns", lambda x: x.mean() / 1e3), us_sd=("mean_ns", lambda x: x.std() / 1e3)).reset_index()
    return sw[sw.split == "val"], SD, TT.set_index(["estimator", "approx"]).us, sorted(own_s.approx.unique())


def kll_table(val, SD, t, K):
    rows = []
    for k in K:
        diffs, kib, rm = [], {}, {"mpl_kll": [], "mpl_dskll": []}
        for f, s in itertools.product(C.FAMILIES_T31, C.SCENARIOS):
            x = {}
            for e in IMPL:
                v = val[(val.family == f) & (val.scenario == s) & (val.estimator == e) & (val.approx == k)]
                r = v.loc[v.rmse.idxmin()]
                x[e] = 100 * SD.loc[(f, s, e, k, r.gain)].values
                kib[e] = max(kib.get(e, 0), r.peak_bytes / 1024)
                rm[e].append(x[e].mean())
            diffs.append((x["mpl_kll"] - x["mpl_dskll"]).mean())
        rows.append(dict(k=k, kib_own=kib["mpl_kll"], kib_ds=kib["mpl_dskll"], ratio_kib=kib["mpl_kll"] / kib["mpl_dskll"],
                         us_own=t[("mpl_kll", k)], us_ds=t[("mpl_dskll", k)], ratio_us=t[("mpl_dskll", k)] / t[("mpl_kll", k)],
                         d_mean=np.mean(diffs), d_min=np.min(diffs), d_max=np.max(diffs), d_absmax=np.max(np.abs(diffs)),
                         rmse_own=np.mean(rm["mpl_kll"]), rmse_ds=np.mean(rm["mpl_dskll"])))
    return pd.DataFrame(rows)


def kll_storage(ax, R):
    for e, col in (("mpl_kll", "kib_own"), ("mpl_dskll", "kib_ds")):
        c, m, lab = IMPL[e]
        over = R[col] > 256
        ax.plot(R.k, R[col], color=c, lw=1.1, zorder=2)
        ax.scatter(R.k[~over], R[col][~over], color=c, marker=m, s=12, zorder=3, label=lab)
        ax.scatter(R.k[over], R[col][over], facecolor="white", edgecolor=c, marker=m, s=12, lw=0.8, zorder=3)
    ax.axhline(256, color=ce.EDGE, lw=0.8, ls=(0, (3, 2)))
    ax.text(9, 256 * 1.12, "256 KiB budget", fontsize=5.8, color=ce.EDGE)
    log2_axis(ax.set_yscale, ax.set_yticks, ax.set_yticklabels, [2, 8, 32, 128, 512])
    ax.set_ylabel("Retained state (KiB)", fontsize=7)
    ax.set_title("(a) Storage", fontsize=8, pad=3, loc="left")


def kll_accuracy(ax, R):
    ax.fill_between(R.k, R.d_min, R.d_max, color=ce.KLL_OWN, alpha=0.18, lw=0)
    ax.plot(R.k, R.d_mean, color=ce.KLL_OWN, marker="o", ms=2.6, lw=1.1)
    ax.axhline(0, color=ce.DARK, lw=0.6)
    ax.set_ylim(-0.3, 0.3)
    ax.set_ylabel("RMSE difference ($\\times10^{-2}$)\nhand-written − DataSketches", fontsize=6.5)
    ax.set_title("(b) Accuracy", fontsize=8, pad=3, loc="left")


def kll_timing(ax, R):
    for e, col in (("mpl_kll", "us_own"), ("mpl_dskll", "us_ds")):
        c, m, _ = IMPL[e]
        ax.plot(R.k, R[col], color=c, marker=m, ms=3, lw=1.1)
    log2_axis(ax.set_yscale, ax.set_yticks, ax.set_yticklabels, [0.5, 2, 8, 32])
    ax.set_ylabel("Time per observation (µs)", fontsize=7)
    ax.set_title("(c) Processing time", fontsize=8, pad=3, loc="left")


def plot_kll_comparison():
    """Hand-written vs DataSketches KLL: storage, accuracy and time at matched capacity."""
    R = kll_table(*kll_inputs())
    R.to_csv(ce.R + "budget_kll_n15k_table.csv", index=False)
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(1, 3, figsize=(ce.W, 2.4))
        kll_storage(axs[0], R)
        kll_accuracy(axs[1], R)
        kll_timing(axs[2], R)
        for ax in axs:
            log2_axis(ax.set_xscale, ax.set_xticks, ax.set_xticklabels, [8, 64, 512, 4096])
            ce.grid(ax)
        h = [plt.Line2D([], [], color=IMPL[e][0], marker=IMPL[e][1], lw=1.1, ms=3) for e in IMPL]
        h.append(plt.Line2D([], [], color=ce.REF, marker="v", ls="", mfc="white", ms=3.5))
        ce.top_legend(fig, h, [IMPL[e][2] for e in IMPL] + ["exceeds 256 KiB"], ncol=3, fontsize=6.5, handlelength=1.5, columnspacing=1.2)
        ce.bottom_label(fig, "KLL capacity $k$ (log scale)", fontsize=6.3)
        fig.subplots_adjust(left=0.08, right=0.99, top=0.8, bottom=0.17, wspace=0.42)
        ce.save(fig, "ch6_budget_kll_comparison.png")


def gain_panel(ax, EST, P, s, lo, hi):
    pos, data, cols, alphas = [], [], [], []
    for (i, (e, _, c, _)), (j, b) in itertools.product(enumerate(EST), enumerate((64, 256))):
        pos.append(i + (j - 0.5) * 0.36)
        cols.append(c)
        alphas.append(0.35 if b == 64 else 0.85)
        data.append(P[(P.scenario == s) & (P.estimator == e) & (P.budget == b)].pct.values)
    bp = ax.boxplot(data, positions=pos, widths=0.3, whis=1.5, patch_artist=True, showfliers=True,
                    medianprops=dict(color=ce.DARK, lw=1.2), whiskerprops=dict(color=ce.EDGE, lw=0.8),
                    capprops=dict(color=ce.EDGE, lw=0.8), flierprops=dict(marker="o", ms=2.4, mfc="none", mec=ce.EDGE, mew=0.6))
    for p, c, a in zip(bp["boxes"], cols, alphas):
        p.set_facecolor(c)
        p.set_alpha(a)
        p.set_edgecolor(ce.EDGE)
        p.set_linewidth(0.7)
    ax.axhline(0, color=ce.DARK, lw=0.6, zorder=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels([label for _, label, *_ in EST], fontsize=6.6)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(np.floor(lo) - 0.3, np.ceil(hi) + 0.3)
    ce.grid(ax, "y")


def plot_gain_scenarios():
    """RMSE reduction at 64 and 256 KiB relative to 16 KiB."""
    P = pd.read_csv(ce.R + "budget_gain.csv")
    EST = budget_estimators(C.EST_LABEL2)
    titles = [("constant", "(a) Constant dependence"), ("drift", "(b) Gradual drift"), ("jump", "(c) Abrupt jump")]
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(1, 3, figsize=(ce.W, 2.4), sharey=True)
        for ax, (s, t) in zip(axs, titles):
            gain_panel(ax, EST, P, s, P.pct.min(), P.pct.max())
            ax.set_title(t, fontsize=7.8, pad=3, loc="left")
        axs[0].set_ylabel("RMSE reduction vs 16 KiB (%)", fontsize=7.5)
        h = [ce.swatch(color=ce.REF, alpha=0.35), ce.swatch(color=ce.REF, alpha=0.85)]
        ce.top_legend(fig, h, ["64 KiB", "256 KiB"], ncol=2, fontsize=6.6, handlelength=1.2, x=0.54)
        fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.13, wspace=0.06)
        ce.save(fig, "ch6_budget_accuracy_gain_scenarios.png")
    rows = [dict(scenario=s, estimator=e, budget=b, median=x.pct.median(), min=x.pct.min(), max=x.pct.max(),
                 mc_se_median=x.pct_mc_se.median(), n_sig=int((x.pct > 2 * x.pct_mc_se).sum()),
                 n_neg_sig=int((x.pct < -2 * x.pct_mc_se).sum()))
            for (s, e, b), x in P.groupby(["scenario", "estimator", "budget"])]
    pd.DataFrame(rows).to_csv(ce.R + "budget_gain_scenarios.csv", index=False)


def main():
    plot_storage()
    plot_memory_curves()
    plot_processing_time()
    plot_kll_comparison()
    plot_gain_scenarios()


if __name__ == "__main__":
    main()
