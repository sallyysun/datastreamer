"""Heatmap: log-loss advantage of MFF weighting across generating (rows) x working (columns) copula families.
results/scoresfam_cells.csv (analyze_scoresfam.py). Cell value = (squared - log) / squared x 100, whole run: positive
(blue) = log loss better; hatched-free pale text = within 2 SE. Rows of panels = metrics; columns = scenarios.
Usage (repo root): python3 experiments/plot_scoresfam.py [estimator] -> images/ch6/explore_scoresfam_heatmap_<est>.png"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "experiments")
import plot_constant_ess as ce  # noqa: E402

plt, core = ce.plt, ce.core
C = pd.read_csv(core.R + "scoresfam_cells.csv")
C = C[C.window == "whole"]
FAM = ["gaussian", "t", "clayton", "gumbel"]
FL = ["Gaussian", "$t_4$", "Clayton", "Gumbel"]
SC = [("constant", "Constant"), ("drift", "Gradual drift"), ("jump", "Abrupt jump")]
MET = [("rmse", "Spearman RMSE"), ("h2_x1000", "$H^2$"), ("excess_x1000", "Excess log loss")]
EL = {"rmpl": "RMPL", "xiao": "Counting Matrix", "hermite": "Hermite Series", "cf": "CopulaFit"}


def fig(e):
    with plt.rc_context(ce.RC):
        f, axs = plt.subplots(3, 3, figsize=(core.W, 6.0))
        lim = 60
        for ri, (m, ml) in enumerate(MET):
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
                        ax.text(j, i, f"{A[i, j]:+.0f}", ha="center", va="center", fontsize=6.3,
                                color=("#111111" if S[i, j] else "#9a9a9a"), fontweight=("bold" if i == j else "normal"))
                ax.set_xticks(range(4)); ax.set_yticks(range(4))
                ax.set_xticklabels(FL if ri == 2 else [], fontsize=6.5, rotation=30); ax.set_yticklabels(FL if ci == 0 else [], fontsize=6.5)
                for s in ax.spines.values(): s.set_visible(False)
                ax.tick_params(length=0)
                if ri == 0: ax.set_title(scl, fontsize=8)
                if ci == 0: ax.set_ylabel(f"{ml}\ngenerating family", fontsize=7)
        for ax in axs[2]: ax.set_xlabel("working family", fontsize=7)
        cb = f.colorbar(im, cax=f.add_axes([0.9, 0.2, 0.015, 0.6]))
        cb.set_label("(squared − log) / squared, %  (positive: log loss better)", fontsize=6.5)
        f.suptitle(f"{EL[e]}: log-loss vs squared-loss MFF weighting (grey = within 2 SE; bold = matched family)", fontsize=7.5, y=0.985)
        f.subplots_adjust(left=0.13, right=0.87, top=0.93, bottom=0.1, hspace=0.12, wspace=0.08)
        out = f"images/ch6/explore_scoresfam_heatmap_{e}.png"
        f.savefig(out); plt.close(f)
    print("wrote", out)


def fig_thesis(e):
    """Thesis version: same cells, no internal title (caption carries the method), directional colourbar, larger labels.
    -> images/ch6/ch6_scoresfam_heatmap_<est>.png"""
    lim = 60
    ML = {"rmse": "Spearman RMSE", "h2_x1000": "Squared Hellinger", "excess_x1000": "Excess log loss"}
    FS = ["Gaussian", "$t_4$", "Clayton", "Gumbel"]
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
        out = f"images/ch6/ch6_scoresfam_heatmap_{e}.png"
        f.savefig(out); plt.close(f)
    print("wrote", out)


if os.environ.get("THESIS"):
    for e in (sys.argv[1:] or ["rmpl"]): fig_thesis(e)
else:
    for e in (sys.argv[1:] or ["rmpl", "cf"]): fig(e)
