# Figures for the matched-budget study. See notes/studies/budget_mpl.md.
# Usage: python3 experiments/plot_budget.py  (from repo root; reads experiments/results/budget_*.csv)
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")
R = "experiments/results/"
OUT = "notes/thesis/figures/"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"xiao": "#2a78d6", "hermite": "#eb6834", "mpl_kll": "#1baf7a"}
LAB = {"xiao": "Xiao (bins B)", "hermite": "Hermite (order N)", "mpl_kll": "MPL + KLL (capacity k)"}
MK = {"xiao": "o", "hermite": "s", "mpl_kll": "^"}
BUDGETS = [16, 64, 256]

df = pd.read_csv(R + "budget_sweep.csv")
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


def fig_memory(fams, out):
    scs = ["constant", "drift", "jump"]
    fig, axes = plt.subplots(len(fams), len(scs), figsize=(11, 2.75 * len(fams) + 1.0), sharex=True,
                             squeeze=False)
    fig.patch.set_facecolor(SURF)
    for i, fam in enumerate(fams):
        for j, sc in enumerate(scs):
            ax = axes[i, j]
            style(ax, f"{fam} · {sc}")
            for b in BUDGETS:
                ax.axvline(b, color=MUTED, lw=0.8, ls=":", alpha=0.7)
            for est in COL:
                f = frontier(fam, sc, est)
                ax.errorbar(f.peak_bytes / 1024, 100 * f.rmse, yerr=200 * f.se_rmse, color=COL[est],
                            marker=MK[est], ms=4.5, lw=2, capsize=0, label=LAB[est])
            ex = frontier(fam, sc, "mpl_exact")
            ax.axhline(100 * ex.rmse.iloc[0], color=COL["mpl_kll"], lw=1.2, ls="--",
                       label="MPL, exact PIT (oracle)")
            ax.set_xscale("log")
            ref = frontier(fam, sc, "xiao").rmse.max()
            ax.set_ylim(top=100 * ref * 1.3)   # k = 8, 16 run off the top; the crossing is what matters
            if j == 0:
                ax.set_ylabel(r"test RMSE ($\times10^{-2}$)", fontsize=9, color=MUTED)
            if i == len(fams) - 1:
                ax.set_xlabel("measured peak state (KiB, log)", fontsize=9, color=MUTED)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("RMSE against memory: each point one approximation setting at its validation-best gain\n"
                 "(100 test seeds, ±2 SE; dotted lines are the 16 / 64 / 256 KiB budgets)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / fig.get_figheight()))
    fig.savefig(OUT + out, dpi=150)


def fig_runtime():
    tm = pd.read_csv(R + "budget_timing.csv")
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
    p = pd.read_csv(R + "budget_paths.csv")
    fams = ["gaussian", "clayton"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    fig.patch.set_facecolor(SURF)
    for ax, fam in zip(axes, fams):
        style(ax, f"{fam} · jump")
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


fig_memory(["gaussian", "clayton"], "ch6_budget_memory.png")                       # main text
fig_memory(["gaussian", "clayton", "gumbel", "t"], "ch6_budget_memory_all.png")    # appendix
fig_runtime()
fig_recovery()
print("wrote", OUT + "ch6_budget_{memory,memory_all,runtime,recovery}.png")
