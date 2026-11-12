"""Two candidate recovery figures for Section 6.2 (Gaussian abrupt jump, 60 seeds), from results/jump_recovery.csv and
results/fixed_ess_jumpbias.csv (analyze_jump_recovery.py, probe_fixed_ess JUMPBIAS=1). Style of the approved figures.
  1. images/ch6/ch6_jump_recovery_time.png: 90% recovery time against N_eff (log-log), one marker per estimator, with the
     exponential-forgetting prediction ln(0.1) / ln(1 - 2/(N + 1)).
  2. images/ch6/ch6_jump_bias_time.png: normalised absolute bias B(k) of the mean prequential estimate around the jump,
     one panel per estimator, the N_eff colours of ch6_jump_ess_rmse, threshold 0.1 dashed.
Usage (repo root): python3 experiments/plot_jump_recovery.py"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "experiments")
import plot_constant_ess as ce  # noqa: E402

plt, core = ce.plt, ce.core
EST = [("rmpl", "RMPL", "mle", "^"), ("xiao", "Counting Matrix", "xiao", "o"), ("hermite", "Hermite Series", "hermite", "s")]
CMAP = plt.get_cmap("plasma")   # the N_eff colours of plot_jump_time.py
MEM = [("39.000000", "39", CMAP(0.82), "-"), ("99.000000", "99", CMAP(0.64), "-"), ("199.000000", "199", CMAP(0.47), "-"),
       ("399.000000", "399", CMAP(0.30), "-"), ("1999.000000", "1999", CMAP(0.10), "-"),
       ("nocum", "No forgetting", "#0b0b0b", (0, (3, 1.6)))]
JUMP = 2500
PRE, POST = 6 / np.pi * np.arcsin(0.1), 6 / np.pi * np.arcsin(0.4)
GAP = abs(PRE - POST)
os.makedirs("images/ch6", exist_ok=True)


def recovery_time():
    r = pd.read_csv(core.R + "jump_recovery.csv")
    r = r[(r.copula == "gaussian") & (r.neff != "No forgetting")].copy()
    r["N"] = r.neff.astype(float)
    with plt.rc_context(ce.RC):
        fig, ax = plt.subplots(figsize=(core.W * 0.62, 2.6))
        nn = np.geomspace(30, 2600, 200)
        ax.plot(nn, np.log(0.1) / np.log(1 - 2 / (nn + 1)), color="#9a9a9a", lw=0.9, ls=(0, (4, 2)), zorder=2,
                label="Theoretical")   # ln 0.1 / ln((N - 1)/(N + 1)), Chapter 5
        for (e, lab, ckey, mk), off in zip(EST, [0.94, 1.0, 1.06]):   # small horizontal offsets so markers don't hide
            g = r[r.estimator == e].sort_values("N")
            ax.plot(g.N * off, g.recovery, ls="none", marker=mk, ms=4.5, color=ce.COL[ckey], label=lab, zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([39, 99, 199, 399, 1999]); ax.set_xticklabels(["39", "99", "199", "399", "1999"])
        ax.set_yticks([50, 100, 200, 500, 1000, 2000]); ax.set_yticklabels(["50", "100", "200", "500", "1000", "2000"])
        ax.minorticks_off(); ax.set_xlim(30, 2700); ax.set_ylim(35, 3000)
        ax.set_xlabel(r"Effective sample size, $N_{\mathrm{eff}}$")
        ax.set_ylabel("90% recovery time (observations)")
        ax.grid(True, color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False, fontsize=7, handlelength=1.6, borderaxespad=0.3, labelspacing=0.35)
        fig.subplots_adjust(left=0.16, right=0.97, top=0.96, bottom=0.17)
        out = "images/ch6/ch6_jump_recovery_time.png"
        fig.savefig(out); plt.close(fig)
    print("wrote", out)


def bias_time(before=300, after=500):
    d = pd.read_csv(core.R + "fixed_ess_jumpbias.csv", dtype={"neff": str})
    d = d[(d.t >= JUMP - before) & (d.t <= JUMP + after)]
    with plt.rc_context(ce.RC):
        fig, axs = plt.subplots(1, 3, figsize=(core.W, 2.6), sharey=True)
        for ax, (e, title, _, _) in zip(axs, EST):
            for k, lab, col, ls in MEM:
                g = d[(d.estimator == e) & (d.neff == k)].sort_values("t")
                rho = np.where(g.t < JUMP, PRE, POST)
                est = rho + g.mean_err.values                       # mean prequential estimate
                ax.plot(g.t - JUMP, np.abs(est - POST) / GAP, color=col, ls=ls, lw=0.9, label=lab, zorder=3)
            ax.axhline(0.1, color="#52514e", lw=0.7, ls=(0, (1, 1.5)), zorder=2)
            ax.axvline(0, color="#52514e", lw=0.8, ls=(0, (4, 3)), zorder=2)
            ax.set_title(title)
            ax.set_xlim(-before, after); ax.set_ylim(0, 1.08); ax.set_xticks([-250, 0, 250, 500])
            ax.grid(True, axis="y", color=ce.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        axs[0].set_ylabel(r"Normalised bias $B(k)$")
        axs[0].text(-before + 10, 0.13, "0.1", fontsize=6.8, color="#52514e")
        fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=6, frameon=False, fontsize=7,
                   bbox_to_anchor=(0.53, 1.0), handlelength=2.0, columnspacing=1.1, title=r"$N_{\mathrm{eff}}$",
                   title_fontsize=7)
        fig.text(0.53, 0.015, "Observations relative to the jump", ha="center", fontsize=8)
        fig.subplots_adjust(left=0.08, right=0.975, top=0.78, bottom=0.18, wspace=0.12)
        out = "images/ch6/ch6_jump_bias_time.png"
        fig.savefig(out); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    recovery_time()
    bias_time()
