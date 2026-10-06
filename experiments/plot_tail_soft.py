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

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.2), facecolor="#fcfcfb")
for ax in (a, b):
    ax.set_facecolor("#fcfcfb"); ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)

a.axhline(1, color=INK2, lw=1.2, ls="--")
a.text(0.535, 1.04, "best fixed memory (no restart)", color=INK2, ha="right", va="bottom", fontsize=9)
a.scatter(d, ratio_h, s=42, color=HARD, edgecolor="#fcfcfb", lw=1.5, label="Hard restart (γ = 0)", zorder=3)
a.scatter(d, ratio_s, s=42, color=SOFT, edgecolor="#fcfcfb", lw=1.5, label="Soft restart (best γ)", zorder=3)
a.set_yscale("log"); a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); a.set_yticks([0.2, 0.5, 1, 2, 3]); a.set_yticklabels(["0.2", "0.5", "1", "2", "3"])
a.set_xlabel("Size of tail change |Δχ(q)|"); a.set_ylabel("Post-change MSE ÷ best fixed memory  (log)")
a.set_title("A. Partial restart beats both extremes", loc="left", fontweight="bold", color=INK)
a.text(0.06, 2.6, "hard restart worse\nthan never restarting", color=HARD, fontsize=9)
a.text(0.12, 0.25, "soft restart\nwins all 48 cells", color=SOFT, fontsize=9)
a.legend(frameon=False, loc="upper right")

lim = (0.1, 400)
b.plot(lim, lim, color=INK2, lw=1.2, ls="--")
b.fill_between(lim, [x / 2 for x in lim], [x * 2 for x in lim], color=GRID, alpha=0.6, lw=0)
b.text(300, 220, "y = x", color=INK2, fontsize=9, ha="right", va="top")
b.text(0.13, 0.33, "×2 band", color=INK2, fontsize=9)
b.scatter(p_old, emp, s=42, facecolor="none", edgecolor=INK2, lw=1.2, label="v₁ / Δχ²  (known χ₀)", zorder=3)
b.errorbar(p_new, emp, yerr=[[e - l for e, l in zip(emp, lo)], [h - e for e, h in zip(emp, hi)]], fmt="none",
           ecolor=SOFT, alpha=0.5, lw=1.2, zorder=3)
b.scatter(p_new, emp, s=42, color=SOFT, edgecolor="#fcfcfb", lw=1.5, label="v₁ / (Δχ² + Var χ̂₀)  (corrected)", zorder=4)
b.set_xscale("log"); b.set_yscale("log"); b.set_xlim(lim); b.set_ylim(lim)
b.set_xlabel("Predicted optimal retained tail events n₀*"); b.set_ylabel("Empirical optimal n₀*  (oracle γ × E[K_τ])")
b.set_title("B. Retained old evidence follows the shrinkage law", loc="left", fontweight="bold", color=INK)
b.legend(frameon=False, loc="upper left")

fig.text(0.01, 0.01, "Clayton–Gumbel mixture at ρ_S = 0.58, π₀ → 1−π₀, q ∈ {.01,.02,.05,.10}, lower & upper tail; "
         "40 seeds, known changepoint. Whiskers: bootstrap range of empirical n₀* (γ grid is coarse, ~×2 steps).",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("../notes/studies/figures/tail_soft.png", dpi=160, facecolor=fig.get_facecolor())
