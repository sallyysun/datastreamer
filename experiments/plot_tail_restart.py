"""Soft vs hard oracle restart against best fixed memory, per pi switch (notes/studies/tail_dependence.md, step 3b).
Needs results/tail_soft_gamma.csv (./tail_soft 40 8080)."""
import csv, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

SOFT, HARD, INK, INK2, GRID, BG = "#1baf7a", "#4a3aa7", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
rows = list(csv.DictReader(open("results/tail_soft_gamma.csv")))
by = {}
for r in rows:
    by.setdefault(r["pi0"], []).append(r)
sw = sorted(by, key=float)

plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=BG)
ax.set_facecolor(BG); ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.axhspan(1, 4, color="#f3f2ee", lw=0)
ax.axhline(1, color=INK2, lw=1.2, ls="--")
ax.text(len(sw) - 0.55, 1.04, "never restart (best fixed memory)", color=INK2, fontsize=10, ha="right", va="bottom")
ax.text(len(sw) - 0.55, 3.7, "worse than never restarting", color=INK2, fontsize=10, ha="right", va="top")

gh, gs = [], []
for i, p in enumerate(sw):
    g = by[p]
    h = [float(r["hard"]) / float(r["best_fixed"]) for r in g]
    s = [float(r["soft"]) / float(r["best_fixed"]) for r in g]
    gh.append(statistics.geometric_mean(h)); gs.append(statistics.geometric_mean(s))
    for k, (hv, sv) in enumerate(zip(h, s)):
        j = (k - 3.5) * 0.025
        ax.plot(i - 0.13 + j, hv, "o", color=HARD, ms=5, alpha=0.45, mew=0)
        ax.plot(i + 0.13 + j, sv, "o", color=SOFT, ms=5, alpha=0.45, mew=0)
x = range(len(sw))
ax.plot([i - 0.13 for i in x], gh, color=HARD, lw=2, marker="o", ms=9, mec=BG, mew=1.5)
ax.plot([i + 0.13 for i in x], gs, color=SOFT, lw=2, marker="o", ms=9, mec=BG, mew=1.5)
ax.text(0.05, gh[0], "  hard restart\n  (throw away all old data)", color=INK, fontsize=10, va="center")
ax.text(-0.4, 0.6, "soft restart\n(keep some old data)", color=INK, fontsize=10, va="top")

ax.set_yscale("log"); ax.set_ylim(0.17, 4)
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_yticks([0.2, 0.3, 0.5, 1, 2, 3]); ax.set_yticklabels(["0.2", "0.3", "0.5", "1", "2", "3"])
dchi = [statistics.mean(abs(float(r["dchi"])) for r in by[p] if float(r["q"]) == 0.05) for p in sw]
ax.set_xticks(list(x))
ax.set_xticklabels([f"{float(p):.2g}→{1 - float(p):.2g}\n{d:.2f}" for p, d in zip(sw, dchi)])
ax.set_xlim(-0.5, len(sw) - 0.5)
ax.set_xlabel("Switch in π, smallest to largest\n(second row: size of the tail change, q = 0.05)")
ax.set_ylabel("Error over the run, relative to never restarting (log)")
ax.set_title("Soft restart beats both alternatives in all 48 settings", loc="left", fontweight="bold", color=INK)
fig.text(0.01, 0.01, "Small dots: the 8 settings per switch (4 thresholds × 2 tails), 40 seeds each. Large dots: geometric mean. "
         "Error = MSE vs χ(q), ticks 10k–40k (switch at 20k). Oracle: true change time, best γ.", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("../notes/studies/figures/tail_restart.png", dpi=160, facecolor=BG)
