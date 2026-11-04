"""Path figure for the family bank through a shape switch at constant rho_S (one run, seed 42), in the style of
ch6_bank_path.png. Row 1: chi_L(0.05) of the truth, the family bank, the Hermite own-density bank and the corner
counter (M = 1999). Row 2: the family bank's weights summed by copula shape (stacked). Row 3: its mean forgetting factor
lambda_bar = sum_k w_k lambda_k (retention; the arms have lambda = 0.98, 0.995, 0.999, i.e. M = 99, 399, 1999).
Reads results/family_path_<regime>_42.csv (probe_family_path).
Usage (from experiments/): python3 plot_family_path.py -> ../notes/thesis/figures/ch6_family_path.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BANK, HERM, CNT = "#2a78d6", "#1baf7a", "#eb6834"
SHAPES = [("w_gaussian", "Gaussian", "#6250d6"), ("w_t", "$t_4$", "#eda100"), ("w_gumbel", "Gumbel", "#008300"),
          ("w_clayton", "Clayton", "#e87ba4"), ("rot", "rotated (neg. dep.)", "#c3c2b7")]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})
regs = [("gauss2clayton", "Gaussian → Clayton"), ("clayton2gumbel", "Clayton → Gumbel"), ("gauss2t", "Gaussian → $t_4$")]

fig, ax = plt.subplots(3, 3, figsize=(13, 8.2), sharex=True, gridspec_kw={"height_ratios": [1.6, 1, 0.8]})
for j, (r, title) in enumerate(regs):
    p = pd.read_csv(f"results/family_path_{r}_42.csv")
    p = p[p.t >= 1000]
    a = ax[0, j]
    a.plot(p.t, p.counter_L, color=CNT, lw=1.0, alpha=0.9, label="corner counter, M = 1999")
    a.plot(p.t, p.hermite_L, color=HERM, lw=1.2, label="Hermite own-density bank")
    a.plot(p.t, p.family_L, color=BANK, lw=1.8, label="family bank")
    a.plot(p.t, p.true_L, color=INK, lw=1.2, ls="--", label=r"true $\chi_L(0.05)$")
    a.set_title(title, color=INK)
    if j == 0: a.set_ylabel(r"$\chi_L(0.05)$")
    a.grid(color=GRID, lw=0.6)

    a = ax[1, j]
    rot = p.w_clayton90 + p.w_clayton270 + p.w_gumbel90 + p.w_gumbel270
    ys = [p[k] if k != "rot" else rot for k, _, _ in SHAPES]
    a.stackplot(p.t, *ys, colors=[c for _, _, c in SHAPES], labels=[l for _, l, _ in SHAPES], edgecolor="white", lw=0.3)
    a.set_ylim(0, 1)
    if j == 0: a.set_ylabel("family weight\n(stacked)")

    a = ax[2, j]
    # Mean forgetting factor lambda_bar = sum_k w_k lambda_k, drawn as 1 - lambda_bar on an inverted log axis so that
    # up = longer memory and the arms' lambda = 0.98, 0.995, 0.999 are evenly spaced.
    a.plot(p.t, 1 - p["lambda"], color=BANK, lw=1.4)
    a.set_yscale("log")
    a.set_ylim(0.025, 0.0008)
    a.set_yticks([0.02, 0.005, 0.001], ["0.980", "0.995", "0.999"])
    a.minorticks_off()
    a.grid(color=GRID, lw=0.6)
    if j == 0: a.set_ylabel("mean forgetting\nfactor $\\bar\\lambda$")
    a.set_xlabel("tick (switch at 5000)")
    for row in range(3): ax[row, j].axvline(5000, color=INK2, lw=0.7, ls=":")
ax[0, 0].legend(fontsize=7.5, frameon=False, loc="upper left")
ax[1, 2].legend(fontsize=7.5, frameon=True, framealpha=0.9, loc="lower right")
fig.suptitle(r"Family bank through a shape switch at constant $\rho_S$ = 0.58 (one run, seed 42): tail dependence, "
             "which family it believes, and its forgetting factor", fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("../notes/thesis/figures/ch6_family_path.png", dpi=160)
print("wrote ch6_family_path.png")
