"""Section 6.2/6.7 trade-off figure. (a, b) Percentage reduction in test Spearman RMSE relative to each estimator's own 16 KiB
configuration, at 64 and 256 KiB, one point per generating family x dependence scenario (12 points per estimator and budget;
the spread is variation across experimental conditions, not Monte Carlo uncertainty); black bar = median across the 12.
Per cell: 100 * (mean RMSE_16 - mean RMSE_b) / mean RMSE_16 over the 100 paired test seeds, configurations chosen on validation
seeds. (a) n = 5000, (b) n = 15,000. (c) Processing time per observation against retained state for every tested approximation
size (5-run mean, gain 0.02, n = 5000 streams; results/budget_time_table.csv).
Usage (from experiments/): python3 plot_budget_tradeoff.py -> ../images/ch6/6.2/ch6_budget_tradeoff.png,
results/budget_tradeoff_summary.csv"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import plot_constant_ess as ce  # noqa: E402

plt, core = ce.plt, ce.core
E = [("mpl_dskll", "RMPL", ce.COL["mle"], "^"), ("xiao", "Counting\nMatrix", ce.COL["xiao"], "o"),
     ("hermite", "Hermite\nSeries", ce.COL["hermite"], "s")]
SETS = [("", "_ext", "n = 5000"), ("_n15k", "_n15k", "n = 15,000")]
rows = []
for suf, src, nl in SETS:
    C = pd.read_csv(f"results/budget_audit_cells{suf}.csv")
    SD = pd.read_csv(f"results/budget_seeds{src}.csv").set_index(["family", "scenario", "estimator", "approx", "gain", "seed"]).rmse.sort_index()
    for (f, s, e), x in C.groupby(["family", "scenario", "estimator"]):
        x = x.set_index("budget_kib")
        base = SD.loc[(f, s, e, int(x.loc[16].approx), float(x.loc[16].gain))].values
        for b in (64, 256):
            y = SD.loc[(f, s, e, int(x.loc[b].approx), float(x.loc[b].gain))].values
            d = base - y
            rows.append(dict(n=nl, family=f, scenario=s, estimator=e, budget=b, pct=100 * d.mean() / base.mean(),
                             pct_mc_se=100 * d.std(ddof=1) / np.sqrt(len(d)) / base.mean(),
                             kib_16=x.loc[16].peak_val / 1024, kib_b=x.loc[b].peak_val / 1024))
P = pd.DataFrame(rows)
P.to_csv("results/budget_tradeoff_summary.csv", index=False)
T = pd.read_csv("results/budget_time_table.csv")

rng = np.random.default_rng(1)
with plt.rc_context(ce.RC):
    fig = plt.figure(figsize=(core.W, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.45, left=0.08, right=0.99, top=0.84, bottom=0.2)
    axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1], sharey=axa); axc = fig.add_subplot(gs[0, 2])
    for ax, (_, _, nl), tag in zip([axa, axb], SETS, ["(a)", "(b)"]):
        for i, (e, lab, col, mk) in enumerate(E):
            for j, b in enumerate((64, 256)):
                y = P[(P.n == nl) & (P.estimator == e) & (P.budget == b)].pct.values
                x = i + (j - 0.5) * 0.36
                ax.scatter(x + rng.uniform(-0.06, 0.06, len(y)), y, s=9, marker=mk, facecolor=col if b == 256 else "white",
                           edgecolor=col, lw=0.7, alpha=0.9, zorder=3)
                ax.hlines(np.median(y), x - 0.13, x + 0.13, color="#111111", lw=1.4, zorder=4)
        ax.axhline(0, color="#222222", lw=0.6, zorder=2)
        ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l, _, _ in E], fontsize=6.5); ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, 2.6); ax.set_ylim(-1, 10)
        ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        ax.set_title(f"{tag} Accuracy gain, {nl}", fontsize=7.5, pad=3, loc="left")
    plt.setp(axb.get_yticklabels(), visible=False)
    axa.set_ylabel("RMSE reduction vs 16 KiB (%)", fontsize=7)
    for e, lab, col, mk in E:
        x = T[T.estimator == e]
        axc.plot(x.kib, x.us, color=col, marker=mk, ms=2.8, lw=1.0, label=lab.replace("\n", " "))
    axc.set_xscale("log", base=2); axc.set_yscale("log", base=2)
    axc.set_xticks([1, 4, 16, 64, 256]); axc.set_xticklabels(["1", "4", "16", "64", "256"])
    axc.set_yticks([0.25, 1, 4, 16, 64]); axc.set_yticklabels(["0.25", "1", "4", "16", "64"])
    for b in (16, 64, 256): axc.axvline(b, color="#8a8984", lw=0.5, ls=":", zorder=0)
    axc.grid(True, color=ce.GRID, lw=0.5); axc.set_axisbelow(True)
    axc.set_xlabel("Retained state (KiB)", fontsize=7); axc.set_ylabel("µs per observation", fontsize=7)
    axc.set_title("(c) Processing time", fontsize=7.5, pad=3, loc="left")
    h = [plt.Line2D([], [], color=c, marker=m, ls="-", ms=3, lw=1) for _, _, c, m in E]
    h += [plt.Line2D([], [], color="#555555", marker="o", ls="", mfc="white", ms=3.2), plt.Line2D([], [], color="#555555", marker="o", ls="", ms=3.2),
          plt.Line2D([], [], color="#111111", lw=1.4)]
    fig.legend(h, ["RMPL", "Counting Matrix", "Hermite Series", "64 KiB", "256 KiB", "median of 12 conditions"], loc="upper center", ncol=6,
               frameon=False, fontsize=6.2, bbox_to_anchor=(0.53, 1.0), handlelength=1.4, columnspacing=1.0)
    fig.text(0.29, 0.02, "Points: generating family × dependence scenario (variation across conditions)", ha="center", fontsize=6, color="#444444")
    fig.savefig("../images/ch6/6.2/ch6_budget_tradeoff.png"); plt.close(fig)

g = P.groupby(["n", "estimator", "budget"]).agg(median=("pct", "median"), min=("pct", "min"), max=("pct", "max"),
                                               over1=("pct", lambda v: int((v > 1).sum())), mc_se_med=("pct_mc_se", "median"),
                                               mc_se_max=("pct_mc_se", "max"))
print(g.round(2).to_string())
print("all cells:", P.groupby(["n", "budget"]).pct.agg(["median", "max"]).round(2).to_string())
