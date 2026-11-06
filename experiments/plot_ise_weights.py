"""How the scoring loss changes the bank's weights and its copula accuracy, one seed (42).
Columns: Xiao, Hermite, MPL arms. Rows 1-3: stacked weights (N_eff) under
  (1) MoM log loss  -log c_Gauss(rho_k)        eta 1,    delta 3e-4  (the thesis log bank)
  (2) own-density loss  -log c_hat_k           eta 1,    delta 3e-4  (same tuning: isolates the loss)
  (3) own-density loss  -log c_hat_k           eta 0.25, delta 3e-3  (*_bank_own as tuned)
Row 4: density ISE (normal-score scale) of each bank's predicted copula (log bank: Gaussian at its rho;
own banks: mixture of the arms' own densities). Reads results/ise_path_<copula>_<scenario>.csv
(probe_ise_path). See budget_mpl.md §19. Usage (from experiments/): python3 plot_ise_weights.py <copula> <scenario>"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cop, sc = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("gaussian", "jump")
d = pd.read_csv(f"results/ise_path_{cop}_{sc}.csv")
NEFF = [39, 99, 199, 399, 1999]
blues = plt.cm.Blues(np.linspace(0.35, 0.95, len(NEFF)))
COMP = [("xiao", "Xiao arms"), ("hermite", "Hermite arms"), ("mle", "MPL arms")]
BANKS = [("log", r"log loss $-\log c_{\mathrm{Gauss}}(\hat\rho_k)$, η=1, δ=3·10⁻⁴", "#d95f02"),
         ("own_eta1", r"own density $-\log \hat c_k$, η=1, δ=3·10⁻⁴", "#7570b3"),
         ("own", r"own density $-\log \hat c_k$, η=0.25, δ=3·10⁻³ (as tuned)", "#1b9e77")]
TITLE = {("gaussian", "jump"): "Gaussian jump (ρ_S 0.19 → 0.79 at t = 2500)",
         ("clayton2gumbel", "shape"): "Clayton → Gumbel at t = 2500, ρ_S fixed at 0.58",
         ("clayton", "constant"): "Clayton, constant ρ_S = 0.58"}
change = sc in ("jump", "shape")
fig, ax = plt.subplots(4, 3, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1.1]})
for j, (c, ctitle) in enumerate(COMP):
    for r, (bt, blabel, col) in enumerate(BANKS):
        g = d[(d.comp == c) & (d.bank_type == bt)]
        w = np.vstack([g[f"w{k}"].values for k in range(5)])
        ax[r, j].stackplot(g.t.values, w, colors=blues, edgecolor="white", linewidth=0.2,
                           labels=[f"N_eff = {n}" for n in NEFF])
        ax[r, j].set_ylim(0, 1)
        if j == 0:
            ax[r, j].set_ylabel(["log loss\nη=1, δ=3·10⁻⁴", "own density\nη=1, δ=3·10⁻⁴", "own density, tuned\nη=0.25, δ=3·10⁻³"][r], fontsize=8.5)
        e = g[g.ise >= 0]
        m = 1e3 * e.ise[e.t >= 500].mean()
        ax[3, j].plot(e.t, 1e3 * e.ise.rolling(5, center=True, min_periods=1).mean(), color=col, lw=1.4,
                      label=f"{['log loss', 'own, η=1', 'own, tuned'][r]}  (mean {m:.2f})")
    ax[0, j].set_title(ctitle, fontsize=11)
    ax[3, j].set_yscale("log")
    ax[3, j].legend(fontsize=7.5, loc="upper left", framealpha=0.85)
    ax[3, j].set_xlabel("tick")
    for r in range(4):
        ax[r, j].grid(alpha=0.25, lw=0.5)
        if change:
            ax[r, j].axvline(2500, color="k", ls=":", lw=0.8)
lo = min(a.get_ylim()[0] for a in ax[3]); hi = max(a.get_ylim()[1] for a in ax[3])
for a in ax[3]:
    a.set_ylim(lo, hi)
ax[3, 0].set_ylabel("density ISE\n" + r"($\times10^{-3}$, log)", fontsize=8.5)
h, l = ax[0, 2].get_legend_handles_labels()
ax[0, 2].legend(h[::-1], l[::-1], fontsize=7, loc="center right", framealpha=0.9)
fig.suptitle(f"Which loss scores the arms: weights and copula accuracy, one seed (42) — {TITLE.get((cop, sc), cop + ' ' + sc)}", fontsize=11)
fig.tight_layout()
out = f"../notes/thesis/figures/ch6_ise_weights_{cop}_{sc}.png"
fig.savefig(out, dpi=150)
print(out)
