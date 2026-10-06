"""Figure for step 3b of notes/studies/tail_dependence.md (run ./tail_soft 40 8080 first)."""
import csv, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

TAU = 20000
SOFT, HARD, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"

rows = list(csv.DictReader(open("results/tail_soft_gamma.csv")))
chi_pre = {}
for r in csv.DictReader(open("results/tail_soft.csv")):
    chi_pre[(r["pi0"], r["tail"], r["q"])] = float(r["chi_pre"])

def var_old(q, m, chi0):
    # Var(J/K) at TAU: v0 / (q * ESS) with ESS = (sum w)^2 / sum w^2 over the TAU pre-change ticks.
    if math.isinf(m):
        ess = TAU
    else:
        l = 1 - 2 / (m + 1)
        ess = ((1 - l**TAU) / (1 - l)) ** 2 / ((1 - l ** (2 * TAU)) / (1 - l * l))
    return chi0 * (1 - chi0) / (q * ess)

d, ratio_s, ratio_h, emp, lo, hi, p_old, p_new = [], [], [], [], [], [], [], []
for r in rows:
    q, m, dchi, v = float(r["q"]), float(r["m_star"]), abs(float(r["dchi"])), float(r["v_post"])
    fixed = float(r["best_fixed"])
    d.append(dchi); ratio_s.append(float(r["soft"]) / fixed); ratio_h.append(float(r["hard"]) / fixed)
    emp.append(float(r["n0_star"])); lo.append(float(r["boot_n0_lo"])); hi.append(float(r["boot_n0_hi"]))
    p_old.append(v / dchi**2)
    p_new.append(v / (dchi**2 + var_old(q, m, chi_pre[(r["pi0"], r["tail"], r["q"])])))

def fit(pred):
    x = [math.log(p) for p in pred]; y = [math.log(e) for e in emp]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y)); sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    rat = sorted(e / p for e, p in zip(emp, pred))
    within2 = sum(0.5 <= e / p <= 2 for e, p in zip(emp, pred))
    return sxy / sxx, sxy**2 / (sxx * syy), rat[len(rat) // 2], within2
for name, p in (("v/dchi^2", p_old), ("v/(dchi^2+Var)", p_new)):
    s, r2, med, w2 = fit(p)
    print(f"{name:16s} slope {s:.3f}  R2 {r2:.3f}  median emp/pred {med:.2f}  within x2 {w2}/48")

import statistics
groups = {}
for r, dd, rs, rh in zip(rows, d, ratio_s, ratio_h):
    groups.setdefault(r["pi0"], []).append((dd, rs, rh))
gx, gs, gh = [], [], []
for k in sorted(groups, key=float):
    g = groups[k]
    gx.append(statistics.mean(x[0] for x in g))
    gs.append(statistics.geometric_mean(x[1] for x in g))
    gh.append(statistics.geometric_mean(x[2] for x in g))

plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5), facecolor="#fcfcfb")
for ax in (a, b):
    ax.set_facecolor("#fcfcfb"); ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)

a.axhline(1, color=INK2, lw=1.2, ls="--")
a.text(0.02, 1.05, "never restart", color=INK2, fontsize=10, va="bottom")
a.plot(gx, gh, color=HARD, lw=2, marker="o", ms=8, mec="#fcfcfb", mew=1.5)
a.plot(gx, gs, color=SOFT, lw=2, marker="o", ms=8, mec="#fcfcfb", mew=1.5)
a.text(gx[0] + 0.01, gh[0], "throw away all old data", color=INK, fontsize=10, va="center")
a.text(gx[-1], gs[-1] - 0.09, "keep some old data", color=INK, fontsize=10, ha="right", va="top")
a.set_ylim(0, 3); a.set_xlim(0, 0.5)
a.set_xlabel("Size of the change in tail dependence")
a.set_ylabel("Error after the change\n(relative to never restarting)")
a.set_title("Keeping some old data always wins", loc="left", fontweight="bold", color=INK)

lim = (0.1, 400)
b.plot(lim, lim, color=INK2, lw=1.2, ls="--")
b.text(150, 80, "perfect\nprediction", color=INK2, fontsize=10, ha="left", va="top")
b.scatter(p_new, emp, s=50, color=SOFT, edgecolor="#fcfcfb", lw=1.5, zorder=3)
b.set_xscale("log"); b.set_yscale("log"); b.set_xlim(lim); b.set_ylim(lim)
b.set_xticks([1, 10, 100]); b.set_xticklabels(["1", "10", "100"])
b.set_yticks([1, 10, 100]); b.set_yticklabels(["1", "10", "100"])
b.grid(True, axis="x", color=GRID, lw=0.8)
b.set_xlabel("Old tail events to keep: formula")
b.set_ylabel("Old tail events to keep: best in simulation")
b.set_title("The formula predicts how much to keep", loc="left", fontweight="bold", color=INK)

fig.tight_layout()
fig.savefig("../notes/studies/figures/tail_soft.png", dpi=160, facecolor=fig.get_facecolor())
