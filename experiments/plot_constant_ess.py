"""Chapter 6, "Effective sample size under constant dependence": RMSE of rho_S against N_eff for the five fixed arms and
the no-forgetting arm, Xiao / Hermite / RMPL, Gaussian copula, constant path (rho_S = 0.582), 60 seeds (42-101).
Data: results/nocum_arms[_fam].csv (probe_nocum_arms.cpp). Its fixed arms reproduce thesis_bank_matched500 exactly, so
the no-forgetting arm runs on the same streams, inputs and margins as the E3 fixed arms.
Usage (repo root): python3 experiments/plot_constant_ess.py
  -> notes/thesis/figures/matched/ch6_fig_constant_ess.png, experiments/results/constant_ess_2026-11-12.txt"""
import math
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "experiments")
import plot_ch6_core as core  # noqa: E402  (style, colours, paths)

plt = core.plt
ARMS = ["fixed_39", "fixed_99", "fixed_199", "fixed_399", "fixed_1999", "nocum"]
LAB = ["39", "99", "199", "399", "1999", "No\nforgetting"]
EST = [("xiao", "Counting Matrix"), ("hermite", "Hermite Series"), ("mle", "RMPL")]
SC = 100   # RMSE x 10^-2


def load(f, src="nocum"):
    """src = "nocum": results/nocum_arms (RMPL on the banks' EW-500 margins; the first approved figures).
    src = "kll": results/fixed_ess_seeds (probe_fixed_ess; RMPL on cumulative DataSketches KLL k = 256 margins)."""
    if src == "kll":
        d = pd.read_csv(core.R + f"fixed_ess_seeds{core.sfx(f)}.csv", dtype={"neff": str})
        d = d[(d.grid == "main") & (d.window == "whole")].copy()
        d["component"] = d.estimator.replace({"rmpl": "mle"})
        d["method"] = d.neff.map(lambda v: "nocum" if v == "nocum" else f"fixed_{round(float(v))}")
        return d
    d = pd.read_csv(core.R + f"nocum_arms{core.sfx(f)}.csv")
    return d[d.window == "whole"]


def table():
    lines = ["Constant dependence (rho_S = 0.582), whole run t = 500-4999, 60 seeds (42-101), Linux/libstdc++.",
             "Source: results/nocum_arms[_fam].csv (probe_nocum_arms.cpp); fixed arms identical to thesis_bank_matched500.",
             "RMSE of rho_S x 10^-2, mean (SE over seeds). Last column: paired no forgetting - N_eff 1999 (x 10^-3, SE); "
             "W = no forgetting lower by > 2 SE.", ""]
    hdr = f"{'family':9s} {'estimator':9s} " + " ".join(f"{a:>12s}" for a in ARMS) + "   nocum-1999 (x1e-3)"
    for sc in ["constant", "drift", "jump"]:
        lines += [f"[{sc}]", hdr]
        for f in core.FAMS:
            d = load(f)
            for e, _ in EST:
                p = d[(d.scenario == sc) & (d.component == e)].pivot(index="seed", columns="method", values="rmse")
                cells = " ".join(f"{SC*p[a].mean():6.2f}({SC*p[a].std(ddof=1)/math.sqrt(len(p)):4.2f})" for a in ARMS)
                m, se = core.paired(1000 * p.nocum, 1000 * p.fixed_1999)
                v = "W" if m < -2 * se else "L" if m > 2 * se else "T"
                lines.append(f"{f:9s} {e:9s} {cells}   {m:+7.2f} ({se:4.2f}) {v}")
        lines.append("")
    open(core.R + "constant_ess_2026-11-12.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


# The first accepted look (2026-11-12): original Chapter 6 palette and type, 6.1 x 2.35 in at the printed width.
COL = {"xiao": "#2a78d6", "hermite": "#eb6834", "mle": "#1baf7a"}
INK, GRID = "#0b0b0b", "#e4e3df"
RC = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
      "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "normal", "axes.labelsize": 8,
      "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
      "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e",
      "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.titlelocation": "left",
      "axes.titlepad": 6, "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white",
      "figure.dpi": 300, "savefig.dpi": 300}


def figure(scenario="constant", out=core.OUT + "ch6_fig_constant_ess.png", ylim=13.5, src="nocum", label_nocum=False, est=None):
    """constant: value labels at N_eff 1999 and no forgetting. drift: the minimum RMSE labelled (and, with
    label_nocum, the no-forgetting RMSE)."""
    d = load("gaussian", src); d = d[d.scenario == scenario]
    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, 3, figsize=(core.W, 2.35), sharey=True)
        x = np.array([0, 1, 2, 3, 4, 5.35])   # a gap before no forgetting: it is not a point on the N_eff scale
        for ax, (e, title) in zip(axs, est or EST):
            p = d[d.component == e].pivot(index="seed", columns="method", values="rmse") * SC
            mu = p[ARMS].mean(); se = p[ARMS].std(ddof=1) / math.sqrt(len(p))
            col = COL[e]
            ax.plot(x[:5], mu[ARMS[:5]], color=col, lw=1.4, zorder=3)
            ax.errorbar(x[:5], mu[ARMS[:5]], yerr=2 * se[ARMS[:5]], fmt="o", ms=3.8, color=col, lw=1, capsize=0, zorder=4)
            ax.errorbar(x[5], mu["nocum"], yerr=2 * se["nocum"], fmt="D", ms=4.2, color=col, mfc="white", mew=1.3, lw=1,
                        capsize=0, zorder=4)
            if scenario == "constant":
                for a, xi in [("fixed_1999", x[4]), ("nocum", x[5])]:
                    ax.text(xi, mu[a] + 0.75, f"{mu[a]:.2f}", ha="center", va="bottom", fontsize=6.8, color=INK)
            else:
                k = int(np.argmin(mu[ARMS[:5]].values))
                print(f"  {title} {scenario}: minimum {mu[ARMS[k]]:.2f} at N_eff {ARMS[k][6:]}")
                ax.text(x[k], mu[ARMS[k]] - 0.75 * ylim / 13.5, f"{mu[ARMS[k]]:.2f}", ha="center", va="top",
                        fontsize=6.8, color=INK)
                if label_nocum:
                    ax.text(x[5], mu["nocum"] + 0.75 * ylim / 13.5, f"{mu['nocum']:.2f}", ha="center", va="bottom",
                            fontsize=6.8, color=INK)
            m, s = core.paired(p.nocum, p.fixed_1999)
            if scenario == "constant": print(f"  {title}: no forgetting - 1999 = {m:+.2f} ± {2*s:.2f} (x1e-2, paired 2 SE)")
            ax.set_title(title)
            ax.set_xticks(x); ax.set_xticklabels(LAB, fontsize=7)
            ax.set_xlim(-0.45, 6.0); ax.set_ylim(0, ylim)
            ax.grid(True, axis="y", color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
            ax.tick_params(axis="x", length=0)
        axs[0].set_ylabel(r"Spearman RMSE ($\times10^{-2}$)")
        fig.text(0.53, 0.015, r"Effective Sample Size, $N_{\mathrm{eff}}$", ha="center", fontsize=8)
        fig.subplots_adjust(left=0.08, right=0.975, top=0.9, bottom=0.24, wspace=0.12)
        fig.savefig(out)
        plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    table()
    figure()
    figure("drift", core.OUT + "ch6_fig_drift_ess.png", ylim=21)
