"""Clayton-Gumbel mixture summary figure (notes/studies/tail_dependence.md, steps 3a-3b).
Needs results/tail_mixture.txt (./tail_mixture) and results/tail_soft_gamma.csv (./tail_soft 40 8080)."""
import csv, math, re, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOW, UP, OVR, SOFT, HARD = "#2a78d6", "#eb6834", "#52514e", "#1baf7a", "#4a3aa7"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TH, ZE, RHO, Q = 1.4186, 1.7118, 0.58192, 0.05

def c_clay(a): return (2 * a ** -TH - 1) ** (-1 / TH)
def c_gumb(a): return math.exp(-(2 * (-math.log(a)) ** ZE) ** (1 / ZE))
def chi_l(pi): return (pi * c_clay(Q) + (1 - pi) * c_gumb(Q)) / Q
def chi_u(pi):
    b = 1 - Q
    return (1 - 2 * b + pi * c_clay(b) + (1 - pi) * c_gumb(b)) / Q

mc = []
txt = open("results/tail_mixture.txt").read()
for m in re.finditer(r"^path pi=(\S+).*?mc (\S+) \(z.*?\n.*?q=0\.05 \S+/(\S+).*?\n.*?q=0\.05 \S+/(\S+)", txt, re.M):
    mc.append(tuple(map(float, m.groups())))

rows = list(csv.DictReader(open("results/tail_soft_gamma.csv")))
by = {}
for r in rows:
    by.setdefault(r["pi0"], []).append(r)
sw = sorted(by, key=float)
dpi = list(range(len(sw)))
hard = [statistics.geometric_mean(float(r["hard"]) / float(r["best_fixed"]) for r in by[p]) for p in sw]
soft = [statistics.geometric_mean(float(r["soft"]) / float(r["best_fixed"]) for r in by[p]) for p in sw]
dchi05 = [statistics.mean(abs(float(r["dchi"])) for r in by[p] if float(r["q"]) == Q) for p in sw]

plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=BG)
for ax in (a, b):
    ax.set_facecolor(BG); ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)

xs = [i / 100 for i in range(101)]
a.plot(xs, [RHO] * len(xs), color=OVR, lw=2, ls="--")
a.plot(xs, [chi_l(x) for x in xs], color=LOW, lw=2)
a.plot(xs, [chi_u(x) for x in xs], color=UP, lw=2)
for pi, rho, cl, cu in mc:
    a.plot(pi, rho, "o", color=OVR, ms=7, mec=BG, mew=1.5)
    a.plot(pi, cl, "o", color=LOW, ms=7, mec=BG, mew=1.5)
    a.plot(pi, cu, "o", color=UP, ms=7, mec=BG, mew=1.5)
a.text(0.02, RHO + 0.015, "overall dependence (Spearman ρ) — unchanged", color=INK, fontsize=10, va="bottom")
a.text(0.12, chi_l(0.12) - 0.03, "lower-tail dependence (Clayton-like)", color=INK, fontsize=10, va="top")
a.text(0.12, chi_u(0.12) + 0.03, "upper-tail dependence (Gumbel-like)", color=INK, fontsize=10, va="bottom")
a.set_xlim(0, 1); a.set_ylim(0, 0.7)
a.set_xticks([0, 0.25, 0.5, 0.75, 1]); a.set_xticklabels(["0\nall Gumbel", "0.25", "0.5", "0.75", "1\nall Clayton"])
a.set_xlabel("Clayton weight π")
a.set_ylabel("Dependence")
a.set_title("Changing π moves dependence between the tails", loc="left", fontweight="bold", color=INK)

b.axhline(1, color=INK2, lw=1.2, ls="--")
b.text(dpi[-1], 1.05, "never restart", color=INK2, fontsize=10, ha="right", va="bottom")
b.plot(dpi, hard, color=HARD, lw=2, marker="o", ms=8, mec=BG, mew=1.5)
b.plot(dpi, soft, color=SOFT, lw=2, marker="o", ms=8, mec=BG, mew=1.5)
b.text(dpi[0] + 0.15, hard[0], "restart from scratch", color=INK, fontsize=10, va="center")
b.text(dpi[-1], soft[-1] - 0.15, "restart, keeping some old data", color=INK, fontsize=10, ha="right", va="top")
b.set_xticks(dpi)
b.set_xticklabels([f"{float(p):.2g}→{1 - float(p):.2g}\n{d:.2f}" for p, d in zip(sw, dchi05)], fontsize=10)
b.set_xlim(-0.4, len(sw) - 0.6); b.set_ylim(0, 3)
b.set_xlabel("Switch in π, smallest to largest\n(second row: size of the tail change, q = 0.05)")
b.set_ylabel("Error after the switch\n(relative to never restarting)")
b.set_title("Bigger switches make restarting worthwhile", loc="left", fontweight="bold", color=INK)

fig.tight_layout()
fig.savefig("../notes/studies/figures/tail_mixture.png", dpi=160, facecolor=BG)
print("mc points", len(mc), "dchi05", [round(x, 3) for x in dchi05])
