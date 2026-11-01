"""Figure for thesis section 6.3: mean bank weights and weight-averaged arm memory around the jump
(Gaussian DGP, Xiao components, squared and log banks). Reads results/thesis_bank_weights.csv.
Usage (from experiments/): python3 plot_thesis_bank.py -> ../notes/thesis/figures/ch6_bank_weights.png"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

d = pd.read_csv("results/thesis_bank_weights.csv")
d = d[(d.copula == "gaussian") & (d.component == "xiao")]
fig, ax = plt.subplots(2, 2, figsize=(10, 5.5), sharex=True)
for j, (bank, title) in enumerate([("bank", "squared-loss bank"), ("bank_log", "log-loss bank")]):
    g = d[d.bank == bank]
    w = g.pivot(index="t", columns="mem", values="weight")
    for m in w.columns:
        ax[0, j].plot(w.index, w[m], lw=1.2, label=f"M = {m}")
    ax[0, j].axvline(2500, color="k", ls=":", lw=0.8)
    ax[0, j].set_title(title)
    ax[0, j].set_ylabel("mean weight")
    mem = (w * w.columns.values).sum(axis=1)
    ax[1, j].plot(mem.index, mem, color="k", lw=1.2)
    ax[1, j].axvline(2500, color="k", ls=":", lw=0.8)
    ax[1, j].set_yscale("log")
    ax[1, j].set_ylabel(r"$\sum_k w_k M_k$ (ticks)")
    ax[1, j].set_xlabel("tick")
ax[0, 0].legend(fontsize=8, ncol=2)
fig.tight_layout()
os.makedirs("../notes/thesis/figures", exist_ok=True)
fig.savefig("../notes/thesis/figures/ch6_bank_weights.png", dpi=150)
