"""Section 6.7: hand-written vs Apache DataSketches KLL inside RMPL at matched capacity k, n = 15,000 streams.
Same streams, gains, seeds and selection: at each k, each implementation uses its validation-best gain per family x scenario.
(a) Retained state (peak, both margins + RMPL state) against k; the dashed line is the 256 KiB budget and hollow markers exceed it.
(b) Paired test-RMSE difference, hand-written minus DataSketches (×10⁻²): the line is the mean over the 12 family x scenario
    conditions, the band their range (variation across conditions).
(c) Time per observation (update + estimate, gain 0.02, one thread, mean of 5 runs, both implementations in the same runs).
k = 16384 retains every observation at n = 15,000 (full-retention reference).
Inputs: results/budget_{sweep,seeds}_n15k_ownkll_<scenario>_<family>.csv, results/budget_{sweep,seeds}_n15k.csv (DataSketches),
results/budget_timing_n15k_kll_rep{1..5}.csv. Usage (from experiments/): python3 plot_budget_kll_n15k.py
-> ../images/ch6/6.7/ch6_budget_kll_comparison.png, results/budget_kll_n15k_table.csv"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import plot_constant_ess as ce  # noqa: E402

plt, core = ce.plt, ce.core
F, S = ["gaussian", "t", "clayton", "gumbel"], ["constant", "drift", "jump"]
own_s = pd.concat([pd.read_csv(f"results/budget_sweep_n15k_ownkll_{s}_{f}.csv") for s in S for f in F])
own_d = pd.concat([pd.read_csv(f"results/budget_seeds_n15k_ownkll_{s}_{f}.csv") for s in S for f in F])
ds_s, ds_d = pd.read_csv("results/budget_sweep_n15k.csv"), pd.read_csv("results/budget_seeds_n15k.csv")
sw = pd.concat([own_s, ds_s[ds_s.estimator == "mpl_dskll"]]); sd = pd.concat([own_d, ds_d[ds_d.estimator == "mpl_dskll"]])
val = sw[sw.split == "val"]
SD = sd.set_index(["family", "scenario", "estimator", "approx", "gain", "seed"]).rmse.sort_index()
T = pd.concat([pd.read_csv(f"results/budget_timing_n15k_kll_rep{r}.csv") for r in range(1, 6)])
TT = T.groupby(["estimator", "approx"]).agg(us=("mean_ns", lambda x: x.mean() / 1e3), us_sd=("mean_ns", lambda x: x.std() / 1e3)).reset_index()
K = sorted(own_s.approx.unique())
rows = []
for k in K:
    diffs, kib, rm = [], {}, {"mpl_kll": [], "mpl_dskll": []}
    for f in F:
        for s in S:
            x = {}
            for e in ("mpl_kll", "mpl_dskll"):
                v = val[(val.family == f) & (val.scenario == s) & (val.estimator == e) & (val.approx == k)]
                r = v.loc[v.rmse.idxmin()]
                x[e] = 100 * SD.loc[(f, s, e, k, r.gain)].values
                kib[e] = max(kib.get(e, 0), r.peak_bytes / 1024)
                rm[e].append(x[e].mean())
            diffs.append((x["mpl_kll"] - x["mpl_dskll"]).mean())
    t = TT.set_index(["estimator", "approx"]).us
    rows.append(dict(k=k, kib_own=kib["mpl_kll"], kib_ds=kib["mpl_dskll"], ratio_kib=kib["mpl_kll"] / kib["mpl_dskll"],
                     us_own=t[("mpl_kll", k)], us_ds=t[("mpl_dskll", k)], ratio_us=t[("mpl_dskll", k)] / t[("mpl_kll", k)],
                     d_mean=np.mean(diffs), d_min=np.min(diffs), d_max=np.max(diffs), d_absmax=np.max(np.abs(diffs)),
                     rmse_own=np.mean(rm["mpl_kll"]), rmse_ds=np.mean(rm["mpl_dskll"])))
R = pd.DataFrame(rows)
R.to_csv("results/budget_kll_n15k_table.csv", index=False)
print(R.round(3).to_string(index=False))

C = {"mpl_kll": ("#6250d6", "v", "Hand-written KLL"), "mpl_dskll": (ce.COL["mle"], "^", "Apache DataSketches KLL")}
with plt.rc_context(ce.RC):
    fig, axs = plt.subplots(1, 3, figsize=(core.W, 2.4))
    for e, col in (("mpl_kll", "kib_own"), ("mpl_dskll", "kib_ds")):
        c, m, lab = C[e]
        over = R[col] > 256
        axs[0].plot(R.k, R[col], color=c, lw=1.1, zorder=2)
        axs[0].scatter(R.k[~over], R[col][~over], color=c, marker=m, s=12, zorder=3, label=lab)
        axs[0].scatter(R.k[over], R[col][over], facecolor="white", edgecolor=c, marker=m, s=12, lw=0.8, zorder=3)
    axs[0].axhline(256, color="#444444", lw=0.8, ls=(0, (3, 2)))
    axs[0].text(9, 256 * 1.12, "256 KiB budget", fontsize=5.8, color="#444444")
    axs[0].set_yscale("log", base=2); axs[0].set_yticks([2, 8, 32, 128, 512]); axs[0].set_yticklabels(["2", "8", "32", "128", "512"])
    axs[0].set_ylabel("Retained state (KiB)", fontsize=7)
    axs[0].set_title("(a) Storage", fontsize=8, pad=3, loc="left")
    axs[1].fill_between(R.k, R.d_min, R.d_max, color="#6250d6", alpha=0.18, lw=0)
    axs[1].plot(R.k, R.d_mean, color="#6250d6", marker="o", ms=2.6, lw=1.1)
    axs[1].axhline(0, color="#222222", lw=0.6)
    axs[1].set_ylim(-0.3, 0.3)
    axs[1].set_ylabel("RMSE difference ($\\times10^{-2}$)\nhand-written − DataSketches", fontsize=6.5)
    axs[1].set_title("(b) Accuracy", fontsize=8, pad=3, loc="left")
    for e, col in (("mpl_kll", "us_own"), ("mpl_dskll", "us_ds")):
        c, m, lab = C[e]
        axs[2].plot(R.k, R[col], color=c, marker=m, ms=3, lw=1.1)
    axs[2].set_yscale("log", base=2); axs[2].set_yticks([0.5, 2, 8, 32]); axs[2].set_yticklabels(["0.5", "2", "8", "32"])
    axs[2].set_ylabel("Time per observation (µs)", fontsize=7)
    axs[2].set_title("(c) Processing time", fontsize=8, pad=3, loc="left")
    for ax in axs:
        ax.set_xscale("log", base=2); ax.set_xticks([8, 64, 512, 4096]); ax.set_xticklabels(["8", "64", "512", "4096"])
        ax.grid(True, color=ce.GRID, lw=0.5); ax.set_axisbelow(True)
    h = [plt.Line2D([], [], color=C[e][0], marker=C[e][1], lw=1.1, ms=3) for e in C]
    h.append(plt.Line2D([], [], color="#777777", marker="v", ls="", mfc="white", ms=3.5))
    fig.legend(h, [C[e][2] for e in C] + ["exceeds 256 KiB"], loc="upper center", ncol=3, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.53, 1.0), handlelength=1.5, columnspacing=1.2)
    fig.text(0.53, 0.01, "KLL capacity $k$ (log scale)",
             ha="center", fontsize=6.3)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.8, bottom=0.17, wspace=0.42)
    fig.savefig("../images/ch6/6.7/ch6_budget_kll_comparison.png"); plt.close(fig)
print("wrote ../images/ch6/6.7/ch6_budget_kll_comparison.png")
