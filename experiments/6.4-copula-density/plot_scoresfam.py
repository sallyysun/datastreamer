"""Heatmap: log-loss advantage of MFF weighting across generating (rows) x working (columns) copula families.
results/scoresfam_cells.csv (analyze_scoresfam.py). Cell value = (squared - log) / squared x 100, whole run: positive
(blue) = log loss better; hatched-free pale text = within 2 SE. Rows of panels = metrics; columns = scenarios.
Usage: python3 plot_scoresfam.py [estimator ...] -> images/ch6_scoresfam_heatmap_<est>.png (default: all three)"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import config as cfg  # noqa: E402
import plot_style as ce  # noqa: E402

plt, core = ce.plt, ce.core
C = pd.read_csv(core.R + "scoresfam_cells.csv")
C = C[C.window == "whole"]
FAM = cfg.FAMILIES_T31
FL = [cfg.FAMILY_MATH[f] for f in FAM]
SC = [(s, cfg.SCENARIO_LABEL[s]) for s in cfg.SCENARIOS]
MET = [("rmse", "Spearman RMSE"), ("h2_x1000", "$H^2$"), ("excess_x1000", "Excess log loss")]


def fig_thesis(e):
    """Thesis version: same cells, no internal title (caption carries the method), directional colourbar, larger labels.
    -> images/ch6_scoresfam_heatmap_<est>.png"""
    lim = 60
    ML = {"rmse": "Spearman RMSE", "h2_x1000": "Squared Hellinger", "excess_x1000": "Excess log loss"}
    FS = FL
    with plt.rc_context(ce.RC):
        f, axs = plt.subplots(3, 3, figsize=(core.W, 6.0))
        for ri, (m, _) in enumerate(MET):
            for ci, (sc, scl) in enumerate(SC):
                ax = axs[ri, ci]
                A = np.full((4, 4), np.nan); S = np.zeros((4, 4), bool)
                for i, g in enumerate(FAM):
                    for j, w in enumerate(FAM):
                        r = C[(C.gen == g) & (C.working == w) & (C.scenario == sc) & (C.estimator == e) & (C.metric == m)].iloc[0]
                        A[i, j] = 100 * r["diff"] / r["sq"]; S[i, j] = r["verdict"] != "tie"
                im = ax.imshow(np.clip(A, -lim, lim), cmap="RdBu", vmin=-lim, vmax=lim)
                for i in range(4):
                    for j in range(4):
                        strong = abs(np.clip(A[i, j], -lim, lim)) > 38
                        col = ("white" if strong else "#111111") if S[i, j] else "#8a8a8a"
                        ax.text(j, i, f"{A[i, j]:+.0f}".replace("-", "\u2212"), ha="center", va="center", fontsize=7.2,
                                color=col, fontweight=("bold" if i == j else "normal"))
                ax.set_xticks(range(4)); ax.set_yticks(range(4))
                ax.set_xticklabels(FS if ri == 2 else [], fontsize=7.2, rotation=35, ha="right", rotation_mode="anchor")
                ax.set_yticklabels(FS if ci == 0 else [], fontsize=7.2)
                for sp in ax.spines.values(): sp.set_visible(False)
                ax.tick_params(length=0, pad=2)
                if ri == 0: ax.set_title(scl, fontsize=8.5, pad=4)
            axs[ri, 0].annotate(ML[m], xy=(0, 0.5), xycoords="axes fraction", xytext=(-44, 0), textcoords="offset points",
                                rotation=90, ha="center", va="center", fontsize=8)
        f.text(0.03, 0.6, "Generating family", rotation=90, ha="center", va="center", fontsize=8)
        f.text(0.555, 0.142, "Estimation family", ha="center", fontsize=8)
        cax = f.add_axes([0.3, 0.078, 0.5, 0.016])
        cb = f.colorbar(im, cax=cax, orientation="horizontal", ticks=[-60, -30, 0, 30, 60])
        cb.ax.set_xticklabels(["\u221260", "\u221230", "0\n(no difference)", "30", "60"], fontsize=6.8)
        cb.ax.tick_params(length=2, pad=1.5)
        cb.set_label("Relative error reduction with logarithmic loss (%)", fontsize=7.5, labelpad=2)
        cax.text(-0.02, 0.5, "Squared loss\nbetter", transform=cax.transAxes, ha="right", va="center", fontsize=7, color="#a51c30")
        cax.text(1.02, 0.5, "Log loss\nbetter", transform=cax.transAxes, ha="left", va="center", fontsize=7, color="#1d4f91")
        f.subplots_adjust(left=0.2, right=0.99, top=0.96, bottom=0.225, hspace=0.07, wspace=0.05)
        out = f"images/ch6_scoresfam_heatmap_{e}.png"
        f.savefig(out); plt.close(f)
    print("wrote", out)


for e in (sys.argv[1:] or ["rmpl", "xiao", "hermite"]):
    fig_thesis(e)
