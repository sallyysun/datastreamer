# Figures for the matched-budget study. See notes/studies/budget_mpl.md.
# Usage: python3 experiments/plot_budget.py  (from repo root; reads experiments/results/budget_*.csv)
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")
import os
# BUDGET_TAG=matched reads results/budget_*_matched.csv and writes to figures/matched/ (the primary protocol).
TAG = os.environ.get("BUDGET_TAG", "")
R = "experiments/results/"
OUT = "notes/thesis/figures/" + (TAG + "/" if TAG else "")
SUF = ("_" + TAG) if TAG else ""
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
import sys
MODE = sys.argv[1] if len(sys.argv) > 1 else "ds"   # ds (thesis) | own (hand-written KLL) | both
OWN = MODE == "own"
MPL = "mpl_kll" if OWN else "mpl_dskll"
COL = {"xiao": "#2a78d6", "hermite": "#eb6834", MPL: "#1baf7a"}
if MODE == "both":   # same MPL, two sketch implementations
    COL = {"xiao": "#9ec5f4", "hermite": "#f4b39a", "mpl_kll": "#6250d6", "mpl_dskll": "#1baf7a"}
LAB = {"xiao": "Xiao (bins B)", "hermite": "Hermite (order N)", "mpl_kll": "MPL + hand-written KLL (k)", "mpl_dskll": "MPL + KLL sketch (k; Apache DataSketches)"}
MK = {"xiao": "o", "hermite": "s", "mpl_kll": "v", "mpl_dskll": "^"}
BUDGETS = [16, 64, 256]

df = pd.read_csv(R + "budget_sweep" + SUF + ".csv")
val, test = df[df.split == "val"], df[df.split == "test"]


def frontier(fam, sc, est, metric="rmse"):
    """Test error per approximation setting, at the gain that was best on validation."""
    v = val[(val.family == fam) & (val.scenario == sc) & (val.estimator == est)]
    best = v.loc[v.groupby("approx")[metric].idxmin(), ["approx", "gain"]]
    t = test[(test.family == fam) & (test.scenario == sc) & (test.estimator == est)]
    return t.merge(best, on=["approx", "gain"]).sort_values("approx")


def style(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(labelsize=8, colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)


# What each family is, and what it means for the Gaussian-copula MPL.
FAMLAB = {"gaussian": "Gaussian copula\n(MPL correctly specified)",
          "clayton": "Clayton copula\n(lower-tail dependence;\nMPL misspecified)",
          "gumbel": "Gumbel copula\n(upper-tail dependence;\nMPL misspecified)",
          "t": "Student-$t_4$ copula\n(symmetric tail dependence;\nMPL misspecified)"}
SCLAB = {"constant": r"constant: $\rho_S = 0.58$",
         "drift": (r"drift: $\rho_S$ from $0.19$ to $0.79$ (Gaussian)" if TAG == "matched"
                   else r"drift: $\rho_S$ from $-0.48$ to $0.79$ (Gaussian)"),
         "jump": r"jump: $\rho_S$ 0.19 $\to$ 0.79 at $t$ = 2500"}


def fig_memory(fams, out):
    scs = ["constant", "drift", "jump"]
    fig, axes = plt.subplots(len(fams), len(scs), figsize=(11, 2.75 * len(fams) + 1.0), sharex=True,
                             squeeze=False)
    fig.patch.set_facecolor(SURF)
    for i, fam in enumerate(fams):
        for j, sc in enumerate(scs):
            ax = axes[i, j]
            style(ax, SCLAB[sc] if i == 0 else "")
            for b in BUDGETS:
                ax.axvline(b, color=MUTED, lw=0.8, ls=":", alpha=0.7)
            for est in COL:
                f = frontier(fam, sc, est)
                ax.errorbar(f.peak_bytes / 1024, 100 * f.rmse, yerr=200 * f.se_rmse, color=COL[est],
                            marker=MK[est], ms=4.5, lw=2, capsize=0, label=LAB[est])
            ex = frontier(fam, sc, "mpl_exact")
            ax.axhline(100 * ex.rmse.iloc[0], color=COL[MPL], lw=1.2, ls="--",
                       label="MPL, exact PIT (reference)")
            ax.set_xscale("log")
            ref = frontier(fam, sc, "xiao").rmse.max()
            ax.set_ylim(top=100 * ref * 1.3)   # k = 8, 16 run off the top; the crossing is what matters
            if j == 0:
                ax.set_ylabel(FAMLAB[fam] + "\n\n" + r"test RMSE ($\times10^{-2}$)", fontsize=9, color=INK)
            if i == len(fams) - 1:
                ax.set_xlabel("measured peak state (KiB, log)", fontsize=9, color=MUTED)
    if MODE == "both":
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=8.5,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
    else:
        axes[0, 0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("RMSE against memory: each point one approximation setting at its validation-best gain\n"
                 "(100 test seeds, ±2 SE; dotted lines are the 16 / 64 / 256 KiB budgets)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.045 if MODE == "both" else 0, 1, 1 - 0.55 / fig.get_figheight()))
    fig.savefig(OUT + out, dpi=150)


def fig_runtime():
    tm = pd.read_csv(R + "budget_timing" + SUF + ".csv")
    scs = ["constant", "drift", "jump"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=False)
    fig.patch.set_facecolor(SURF)
    for ax, sc in zip(axes, scs):
        style(ax, f"gaussian · {sc}")
        for est in COL:
            f = frontier("gaussian", sc, est).merge(tm[tm.estimator == est][["approx", "mean_ns"]], on="approx")
            ax.plot(f.mean_ns / 1000, 100 * f.rmse, color=COL[est], marker=MK[est], ms=4.5, lw=2, label=LAB[est])
            for _, r in f.iloc[[0, -1]].iterrows():
                ax.annotate(f"{int(r.approx)}", (r.mean_ns / 1000, 100 * r.rmse), fontsize=7, color=MUTED,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_ylim(top=100 * frontier("gaussian", sc, "xiao").rmse.max() * 1.3)
        ax.set_xlabel(r"time per tick, update + estimate ($\mu$s, log)", fontsize=9, color=MUTED)
    axes[0].set_ylabel(r"test RMSE ($\times10^{-2}$)", fontsize=9, color=MUTED)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("RMSE against processing time (one core; end labels are B, N, k)", fontsize=11, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT + "ch6_budget_runtime.png", dpi=150)


def fig_recovery():
    p = pd.read_csv(R + "budget_paths" + SUF + ".csv")
    fams = ["gaussian", "clayton"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    fig.patch.set_facecolor(SURF)
    for ax, fam in zip(axes, fams):
        style(ax, FAMLAB[fam].replace("\n", " ") + " · jump")
        tr = None
        for est in COL:
            v = val[(val.family == fam) & (val.scenario == "jump") & (val.estimator == est)
                    & (val.peak_bytes <= 64 * 1024)]
            b = v.loc[v.post_rmse.idxmin()]
            s = p[(p.family == fam) & (p.estimator == est) & (p.approx == b.approx) & (p.gain == b.gain)]
            s = s[(s.t >= 2200) & (s.t <= 3200)]
            tr = s
            ax.plot(s.t, s.est, color=COL[est], lw=2, label=f"{LAB[est].split(' (')[0]} [{int(b.approx)}, {b.gain:g}]")
        ax.plot(tr.t, tr.truth, color=INK, lw=1.2, ls="--", label=r"true $\rho_S$")
        ax.set_xlabel("tick", fontsize=9, color=MUTED)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    axes[0].set_ylabel(r"mean estimate of $\rho_S$ (100 seeds)", fontsize=9, color=MUTED)
    fig.suptitle("Recovery after the jump at t = 2500: configurations within 64 KiB, tuned on post-change RMSE "
                 "[approx, gain]", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT + "ch6_budget_recovery.png", dpi=150)


def fig_k():
    """Same MPL, two KLL implementations: test RMSE against the sketch parameter k (top rows) and the
    memory each holds at that k (bottom). At equal k the accuracy is the same; only the storage differs."""
    import matplotlib.gridspec as gridspec
    scs, fams = ["constant", "drift", "jump"], ["gaussian", "clayton"]
    fig = plt.figure(figsize=(11, 9.2))
    fig.patch.set_facecolor(SURF)
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 0.9], hspace=0.45, wspace=0.25)
    for i, fam in enumerate(fams):
        for j, sc in enumerate(scs):
            ax = fig.add_subplot(gs[i, j])
            style(ax, SCLAB[sc] if i == 0 else "")
            for est in ["mpl_kll", "mpl_dskll"]:
                f = frontier(fam, sc, est)
                ax.errorbar(f.approx, 100 * f.rmse, yerr=200 * f.se_rmse, color=COL[est], marker=MK[est], ms=4.5,
                            lw=2, capsize=0, label=LAB[est])
            ex = frontier(fam, sc, "mpl_exact")
            ax.axhline(100 * ex.rmse.iloc[0], color=MUTED, lw=1.1, ls="--", label="MPL, exact PIT (reference)")
            ax.set_xscale("log", base=2)
            ax.set_xticks([8, 32, 128, 512]); ax.set_xticklabels(["8", "32", "128", "512"])
            ax.set_ylim(top=100 * frontier(fam, sc, "mpl_dskll").rmse.iloc[2] * 1.15)   # k = 8, 16 run off the top
            if j == 0:
                ax.set_ylabel(FAMLAB[fam] + "\n\n" + r"test RMSE ($\times10^{-2}$)", fontsize=9, color=INK)
            if i == 1:
                ax.set_xlabel("sketch parameter k (log scale)", fontsize=9, color=MUTED)
    ax = fig.add_subplot(gs[2, :])
    style(ax, "Retained state at each k (both margins; measured peak)")
    for est in ["mpl_kll", "mpl_dskll"]:
        f = frontier("gaussian", "constant", est)
        ax.plot(f.approx, f.peak_bytes / 1024, color=COL[est], marker=MK[est], ms=5, lw=2, label=LAB[est])
        for a, b in zip(f.approx, f.peak_bytes / 1024):
            if a in (64, 1024): ax.annotate(f"{b:.1f} KiB", (a, b), textcoords="offset points", xytext=(6, -10 if est == "mpl_dskll" else 6), fontsize=8, color=COL[est])
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks([8, 16, 32, 64, 128, 256, 512, 1024]); ax.set_xticklabels(["8", "16", "32", "64", "128", "256", "512", "1024"])
    from matplotlib.ticker import FixedLocator, NullLocator
    ax.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50, 100])); ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels(["2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("sketch parameter k (log scale)", fontsize=9, color=MUTED)
    ax.set_ylabel("KiB (log)", fontsize=9, color=MUTED)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("The same MPL on two KLL implementations: the same accuracy at the same k from k = 128, with 4-6x more memory for the hand-written sketch\n"
                 "(100 test seeds, ±2 SE; each point the validation-best gain at that k)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(OUT + "ch6_budget_rmse_vs_k.png", dpi=150, bbox_inches="tight")


if MODE == "both":
    fig_k()
    fig_memory(["gaussian", "clayton"], "ch6_budget_memory_ds_vs_own.png")
    fig_memory(["gaussian", "clayton", "gumbel", "t"], "ch6_budget_memory_all_ds_vs_own.png")
    print("wrote", OUT + "ch6_budget_memory{,_all}_ds_vs_own.png")
elif OWN:   # comparison figures only; the thesis figures use DataSketches
    fig_memory(["gaussian", "clayton"], "ch6_budget_memory_ownkll.png")
    fig_memory(["gaussian", "clayton", "gumbel", "t"], "ch6_budget_memory_all_ownkll.png")
    print("wrote", OUT + "ch6_budget_memory{,_all}_ownkll.png")
else:
    fig_memory(["gaussian", "clayton"], "ch6_budget_memory.png")                       # main text
    fig_memory(["gaussian", "clayton", "gumbel", "t"], "ch6_budget_memory_all.png")    # appendix
    fig_runtime()
    fig_recovery()
    print("wrote", OUT + "ch6_budget_{memory,memory_all,runtime,recovery}.png")
