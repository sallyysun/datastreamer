"""Section 6.2 appendix figures from the final E1 candidate set (results/budget_{sweep,seeds}_final.csv: matched paths, gains
cum / 0.0005-0.10 on constant and drift, 0.005-0.10 on jump; validation seeds 1000-1039, test 42-141).
(1) ch6_budget_memory_curves.png: test RMSE against retained state for every approximation size (CM bins, HS order, RMPL KLL k
    on DataSketches), each at its validation-best gain; families x scenarios; mean over the 100 test seeds (no error bars).
(2) ch6_budget_kll_comparison.png: RMPL on the hand-written vs DataSketches KLL at matched k: family-averaged test RMSE (per
    seed, mean over the four families, each at its validation-best gain for that k; ±2 SE over seeds) against retained state.
Usage (from experiments/): python3 plot_budget_appendix.py"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import plot_constant_ess as ce  # noqa: E402

plt, core = ce.plt, ce.core
sw = pd.read_csv("results/budget_sweep_ext.csv")
SD = pd.read_csv("results/budget_seeds_ext.csv").set_index(["family", "scenario", "estimator", "approx", "gain", "seed"]).rmse.sort_index()
val = sw[sw.split == "val"]
FAMS = [("gaussian", "Gaussian"), ("t", "Student's $t_4$"), ("clayton", "Clayton"), ("gumbel", "Gumbel")]
SC = [("constant", "Constant"), ("drift", "Gradual drift"), ("jump", "Abrupt jump")]
E = [("mpl_dskll", "RMPL (KLL $k$)", ce.COL["mle"], "^"), ("xiao", "Counting Matrix (bins)", ce.COL["xiao"], "o"),
     ("hermite", "Hermite Series (order)", ce.COL["hermite"], "s")]


def frontier(f, s, e):
    v = val[(val.family == f) & (val.scenario == s) & (val.estimator == e)]
    b = v.loc[v.groupby("approx").rmse.idxmin()]
    out = []
    for _, r in b.iterrows():
        x = 100 * SD.loc[(f, s, e, r.approx, r.gain)]
        out.append((r.approx, r.peak_bytes / 1024, x.mean(), x.std(ddof=1) / 10, x.values))
    return out


with plt.rc_context(ce.RC):
    fig, axs = plt.subplots(4, 3, figsize=(core.W, 6.4), sharex=True)
    for i, (f, fl) in enumerate(FAMS):
        for j, (s, sl) in enumerate(SC):
            ax = axs[i, j]
            top = 0
            for e, lab, col, mk in E:
                fr = frontier(f, s, e)
                ax.plot([p[1] for p in fr], [p[2] for p in fr], color=col, marker=mk, ms=3.0, lw=1.1, label=lab)
                if e == "xiao": top = max(p[2] for p in fr)
            lo = min(min(p[2] for p in frontier(f, s, e)) for e, *_ in E)
            top = max(top, max(min(p[2] for p in frontier(f, s, e)) for e, *_ in E) + 0.25 * (top - lo))
            ax.set_ylim(lo - 0.08 * (top - lo), top + 0.3 * (top - lo))
            for b in (16, 64, 256): ax.axvline(b, color="#8a8984", lw=0.5, ls=":", zorder=0)
            ax.set_xscale("log", base=2); ax.set_xticks([2, 8, 32, 128]); ax.set_xticklabels(["2", "8", "32", "128"])
            ax.grid(True, axis="y", color=ce.GRID, lw=0.5); ax.set_axisbelow(True)
            ax.tick_params(labelsize=6.5)
            if i == 0: ax.set_title(sl, fontsize=8, pad=3, loc="left")
            if j == 0: ax.set_ylabel(fl, fontsize=7.5)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.53, 1.0), handlelength=1.6, columnspacing=1.2)
    fig.text(0.53, 0.01, "Retained state (KiB, log scale); dotted lines mark the 16, 64 and 256 KiB budgets", ha="center", fontsize=7.5)
    fig.text(0.005, 0.5, "Spearman RMSE ($\\times10^{-2}$)", rotation=90, va="center", fontsize=7.5)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.93, bottom=0.07, hspace=0.18, wspace=0.22)
    fig.savefig("../images/ch6/6.2/ch6_budget_memory_curves.png"); plt.close(fig)

KE = [("mpl_kll", "Hand-written KLL", "#6250d6", "v"), ("mpl_dskll", "Apache DataSketches KLL", ce.COL["mle"], "^")]
rows = []
with plt.rc_context(ce.RC):
    fig, axs = plt.subplots(1, 3, figsize=(core.W, 2.35))
    for ax, (s, sl) in zip(axs, SC):
        for e, lab, col, mk in KE:
            pts = []
            for k in [k for k in sorted(sw[sw.estimator == e].approx.unique()) if k <= 1024]:   # matched capacities only
                per, kib = [], []
                for f, _ in FAMS:
                    v = val[(val.family == f) & (val.scenario == s) & (val.estimator == e) & (val.approx == k)]
                    r = v.loc[v.rmse.idxmin()]
                    per.append(100 * SD.loc[(f, s, e, k, r.gain)].values); kib.append(r.peak_bytes / 1024)
                a = np.mean(per, axis=0)
                pts.append((k, max(kib), a.mean(), a.std(ddof=1) / 10))
                rows.append(dict(scenario=s, impl=e, k=k, kib=max(kib), rmse=a.mean(), se=a.std(ddof=1) / 10))
            pts = [p for p in pts if p[0] >= 32]
            ax.errorbar([p[1] for p in pts], [p[2] for p in pts], yerr=[2 * p[3] for p in pts], color=col, marker=mk, ms=3.2,
                        lw=1.1, elinewidth=0.7, capsize=0, label=lab)
            for p in pts:
                if p[0] == 32: ax.annotate(f"$k$={p[0]}", (p[1], p[2]), xytext=(3, 4), textcoords="offset points", fontsize=5.8, color=col)
        ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16, 32, 64]); ax.set_xticklabels(["2", "4", "8", "16", "32", "64"])
        ax.set_xlim(1.6, 110)
        ax.grid(True, axis="y", color=ce.GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_title(f"({'abc'[SC.index((s, sl))]}) {sl}", fontsize=8, pad=3, loc="left")
    axs[0].set_ylabel("Spearman RMSE ($\\times10^{-2}$)", fontsize=7.5)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.54, 1.0), handlelength=1.6, columnspacing=1.4)
    fig.text(0.54, 0.01, "Retained state of RMPL (KiB, log scale)", ha="center", fontsize=7.5)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.8, bottom=0.19, wspace=0.28)
    fig.savefig("../images/ch6/6.2/ch6_budget_kll_comparison.png"); plt.close(fig)
pd.DataFrame(rows).to_csv("results/budget_kll_familyavg.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
