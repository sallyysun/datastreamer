"""Chapter 6 core figures (docs/chapter6_visual_plan.md), sized for the printed page (6.1 in wide, 8-9 pt text).
Reads only final, validated results:
  E1  results/budget_sweep_final.csv, budget_seeds_final.csv (make_budget_final.py)
  E2  results/memory_sweep_matched.csv
  E3  results/thesis_bank_matched500[_fam].csv
  E4  results/thesis_bank_weights_{hedge0,density4}_s{42,57,72,87}.csv
  E5  results/family_bank_<s>_density3[_fam].csv (constant, drift), *_density3b* (jump, switches, mixture),
      results/family_bank_path_<s>_density2_claygum_mix.csv
Usage (repo root): python3 experiments/plot_ch6_core.py [e1 e2 e3 e4 e5 e6]  -> notes/thesis/figures/matched/ch6_fig_<e>.png"""
import glob
import math
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "experiments")
import ch6_style as st  # noqa: E402  (palette, font, sizes; sets rcParams)
plt = st.plt

R, OUT = "experiments/results/", "notes/thesis/figures/matched/"
W = st.W
INK, MUTED, GRID, SHADE, SPINE_ = st.INK, st.MUTED, st.GRID, st.SHADE, st.SPINE
EST, METHOD, BASE, THIRD = st.EST, st.METHOD, st.BASE, st.THIRD
FAMS = ["gaussian", "clayton", "gumbel", "t"]
FL = {"gaussian": "Gaussian", "clayton": "Clayton", "gumbel": "Gumbel", "t": "$t_4$"}
REG = {"constant": "Constant", "drift": "Gradual drift", "jump": "Abrupt jump"}
sfx = lambda f: "" if f == "gaussian" else "_" + f


def zero(ax, horizontal=False):
    (ax.axhline if horizontal else ax.axvline)(0, color=INK, lw=1.3, zorder=1)


def tidy(ax, grid_axis="x"):
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def paired(a, b):
    d = (a - b).dropna()
    return d.mean(), d.std(ddof=1) / math.sqrt(len(d))


# ---------------------------------------------------------------- E1
def e1():
    df = pd.read_csv(R + "budget_sweep_final.csv"); sd = pd.read_csv(R + "budget_seeds_final.csv")
    val = df[df.split == "val"]; cap = 65536
    rows = []
    for s in ["constant", "drift", "jump"]:
        for f in FAMS:
            r = {}
            for e in ["xiao", "hermite", "mpl_dskll"]:
                v = val[(val.family == f) & (val.scenario == s) & (val.estimator == e) & (val.peak_bytes <= cap)]
                k = v.loc[v.rmse.idxmin()]
                x = sd[(sd.family == f) & (sd.scenario == s) & (sd.estimator == e) & (sd.approx == k.approx) & (np.isclose(sd.gain, k.gain))]
                r[e] = x.set_index("seed").rmse
            oth = min(["xiao", "hermite"], key=lambda e: r[e].mean())
            m, se = paired(100 * r["mpl_dskll"], 100 * r[oth])
            rows.append((s, f, m, se, oth))
    fig, ax = plt.subplots(figsize=(W, 3.5))
    y, ticks, labels = 0, [], []
    for gi, s in enumerate(["constant", "drift", "jump"]):
        if gi: y += 0.8
        ax.text(-0.27, y - 0.62, {"constant": "Constant dependence", "drift": "Gradual drift", "jump": "Abrupt jump"}[s],
                fontsize=8, fontweight="bold", color=INK, transform=ax.get_yaxis_transform(), ha="left")
        for (s2, f, m, se, oth) in rows:
            if s2 != s: continue
            sig = abs(m) > 2 * se
            ax.errorbar(m, y, xerr=2 * se, fmt="o", ms=4.5, color=INK, mfc=INK if sig else "white", lw=1.2, capsize=0, zorder=3)
            ax.text(1.01, y, f"vs {'Xiao' if oth == 'xiao' else 'Hermite'}", fontsize=7, color=MUTED, va="center",
                    transform=ax.get_yaxis_transform())
            ticks.append(y); labels.append(FL[f]); y += 1
    zero(ax)
    ax.axvspan(-0.8, 0, color=SHADE, zorder=0)
    ax.set_xlim(-0.8, 0.5); ax.set_ylim(y - 0.3, -1.2)
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    ax.set_xlabel(r"MPL minus the better of Xiao and Hermite: test RMSE of $\hat\rho_S$ ($\times10^{-2}$), paired mean $\pm$ 2 SE")
    ax.text(0.02, 1.0, "← MPL more accurate", transform=ax.transAxes, fontsize=7.5, color=MUTED, va="bottom")
    ax.text(1.0, 1.0, "Xiao/Hermite more accurate →", transform=ax.transAxes, fontsize=7.5, color=MUTED, va="bottom", ha="right")
    ax.plot([], [], "o", color=INK, ms=4.5, label="beyond 2 SE"); ax.plot([], [], "o", color=INK, mfc="white", ms=4.5, label="within 2 SE")
    ax.tick_params(axis="y", length=0)
    tidy(ax)
    ax.legend(frameon=False, loc="lower left", fontsize=7, handletextpad=0.2)
    fig.subplots_adjust(left=0.27, right=0.88, top=0.93, bottom=0.14)
    fig.savefig(OUT + "ch6_fig_e1.png")


# ---------------------------------------------------------------- E2
def e2():
    ms = pd.read_csv(R + "memory_sweep_matched.csv")
    T = 5000; t = np.arange(500, T); r = 0.2 + 0.6 * t / (T - 1)
    d_rms = np.sqrt(np.mean(((3 / np.pi) / np.sqrt(1 - r ** 2 / 4) * 0.6 / (T - 1)) ** 2))
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.9), sharey=True)
    for ax, (est, key) in zip(axs, [("xiao_ew", "xiao"), ("hermite_ew", "hermite")]):
        g = ms[(ms.estimator == est) & (ms.scenario == "drift") & (ms.window == "all")].sort_values("mem_ticks")
        n = g.mem_ticks.values
        i = int(np.argmin(g.mse.values)); c2, c1, _ = np.polyfit(np.log(n[i - 1:i + 2]), g.mse.values[i - 1:i + 2], 2)
        n_meas = np.exp(-c1 / (2 * c2))
        w = (n >= 50) & (n <= 2000); s2 = np.median((g["var"].values * n)[w])
        n_pred = (2 * s2 / d_rms ** 2) ** (1 / 3)
        keep = n <= 12000
        ax.plot(n[keep], 1e3 * g["var"].values[keep], **st.VAR)
        ax.plot(n[keep], 1e3 * g.bias2.values[keep], **st.BIAS2)
        ax.plot(n[keep], 1e3 * g.mse.values[keep], color=EST[key], lw=1.8, marker=st.EST_MARKER[key], ms=3.2)
        ax.axvline(n_meas, color=INK, lw=0.9)
        ax.axvline(n_pred, color=INK, lw=0.9, ls=(0, (3, 2)))
        ax.set_xscale("log"); ax.set_ylim(0, 26)
        ax.set_title(st.EST_LABEL[key])
        tidy(ax, "y")
        print(f"  E2 {key}: measured {n_meas:.0f}, predicted {n_pred:.0f}")
        if key == "xiao":
            ax.set_ylabel(r"Error ($\times10^{-3}$)")
            ax.text(58, 16.5, "MSE", color=EST[key])
            ax.text(1500, 2.3, "variance", color=st.VAR["color"])
            ax.text(2900, 19, "squared bias", color=st.BIAS2["color"], ha="right")
    axs[1].plot([], [], color=INK, lw=0.9, label="measured optimum")
    axs[1].plot([], [], color=INK, lw=0.9, ls=(0, (3, 2)), label="predicted optimum")
    fig.legend(*axs[1].get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(0.99, 1.0), ncol=2, handlelength=1.8)
    fig.supxlabel(r"Effective sample size, $N_{\mathrm{eff}}$ (log scale)", fontsize=st.LABEL, y=0.02)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.82, bottom=0.2, wspace=0.08)
    fig.savefig(OUT + "ch6_fig_e2.png")


# ---------------------------------------------------------------- E3
def e3():
    fixed = ["fixed_39", "fixed_99", "fixed_199", "fixed_399", "fixed_1999"]
    res = {}
    for f in FAMS:
        d = pd.read_csv(R + f"thesis_bank_matched500{sfx(f)}.csv"); d = d[d.window == "whole"]
        for sc in ["constant", "drift", "jump"]:
            for c in ["xiao", "hermite", "mle"]:
                p = d[(d.scenario == sc) & (d.component == c)].pivot(index="seed", columns="method", values="rmse")
                best = min(fixed, key=lambda m: p[m].mean())
                res[(sc, f, c)] = paired(1000 * p["bank_log"], 1000 * p[best]) + (best,)
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.6), sharey=True, sharex=True)
    off = {"xiao": -0.22, "hermite": 0.0, "mle": 0.22}
    for ax, sc in zip(axs, ["constant", "drift", "jump"]):
        ax.axvspan(-22, 0, color=SHADE, zorder=0)
        wins = losses = 0
        for yi, f in enumerate(FAMS):
            for c in ["xiao", "hermite", "mle"]:
                m, se, _ = res[(sc, f, c)]
                wins += m < -2 * se; losses += m > 2 * se
                ax.errorbar(m, yi + off[c], xerr=2 * se, fmt="o", ms=3.5, color=EST[c], lw=1.1, capsize=0, zorder=3)
        zero(ax)
        ax.set_title({"constant": "Constant", "drift": "Gradual drift", "jump": "Abrupt jump"}[sc]
                     + f"\nbank better {wins}/12, worse {losses}/12", fontsize=8)
        ax.set_xlim(-22, 13)
        tidy(ax)
    axs[0].set_yticks(range(4)); axs[0].set_yticklabels([FL[f] for f in FAMS]); axs[0].set_ylim(3.5, -0.6)
    axs[0].tick_params(axis="y", length=0)
    fig.supxlabel(r"log bank minus hindsight-best fixed memory: RMSE of $\hat\rho_S$ ($\times10^{-3}$), paired mean $\pm$ 2 SE (60 seeds); shaded = bank better",
                  fontsize=8, y=0.1)
    for c, lab in [("xiao", "Xiao"), ("hermite", "Hermite"), ("mle", "recursive MPL")]:
        axs[2].plot([], [], "o", color=EST[c], ms=3.5, label=lab)
    fig.legend(*axs[2].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.01),
               handletextpad=0.2, fontsize=7.5)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.83, bottom=0.27, wspace=0.08)
    fig.savefig(OUT + "ch6_fig_e3.png")


# ---------------------------------------------------------------- E4
def e4():
    def w(tag):
        d = pd.concat([pd.read_csv(R + f"thesis_bank_weights_{tag}_s{b}.csv") for b in (42, 57, 72, 87)])
        return d[d.bank == "bank_log"].groupby(["t", "mem"]).weight.mean().unstack()
    H, F = w("hedge0"), w("density4")
    t = H.index.values
    fig, axs = plt.subplots(3, 1, figsize=(W, 5.0), sharex=True, gridspec_kw={"height_ratios": [0.55, 1.5, 1.2]})
    rho = np.where(t < 2500, 6 / np.pi * np.arcsin(0.1), 6 / np.pi * np.arcsin(0.4))
    axs[0].plot(t, rho, color=INK, lw=1.3)
    axs[0].set_ylabel(r"$\rho_S$"); axs[0].set_ylim(0, 1); axs[0].set_yticks([0.19, 0.79])
    axs[0].set_title("(a) True dependence")
    for lab, Wt, col in [("fixed share", F, METHOD), ("plain Hedge", H, BASE)]:
        mem = (Wt * np.array(Wt.columns, float)).sum(axis=1)
        axs[1].plot(t, mem, color=col, lw=1.8)
        axs[1].text(4030, mem.iloc[-1], lab, color=col if col != BASE else MUTED, va="center")
    axs[1].set_yscale("log"); axs[1].set_ylim(30, 2600)
    axs[1].set_yticks([39, 99, 199, 399, 1999]); axs[1].set_yticklabels(["39", "99", "199", "399", "1999"]); axs[1].minorticks_off()
    axs[1].set_ylabel(r"$\Sigma_k\, w_k N_{\mathrm{eff},k}$")
    axs[1].set_title("(b) Weighted memory of the bank")
    for Wt, col in [(F, METHOD), (H, BASE)]:
        axs[2].plot(t, Wt[39], color=col, lw=1.6, ls="-")
        axs[2].plot(t, Wt[[399, 1999]].sum(axis=1), color=col, lw=1.2, ls=(0, (3, 2)))
    axs[2].plot([], [], color=MUTED, lw=1.6, label=r"$N_{\mathrm{eff}} = 39$")
    axs[2].plot([], [], color=MUTED, lw=1.2, ls=(0, (3, 2)), label=r"$N_{\mathrm{eff}} \geq 399$")
    axs[2].legend(loc="upper right", handlelength=2.2)
    axs[2].set_ylim(-0.03, 1.05); axs[2].set_ylabel("Arm weight")
    axs[2].set_title("(c) Weight on the shortest and longest arms")
    axs[2].set_xlabel("Observation, $t$")
    for ax in axs:
        ax.axvline(2500, color=SPINE_, lw=0.8, ls=":")
        tidy(ax, "y")
    axs[2].set_xlim(2000, 4000)
    fig.subplots_adjust(left=0.13, right=0.82, top=0.95, bottom=0.09, hspace=0.5)
    fig.savefig(OUT + "ch6_fig_e4.png")


# ---------------------------------------------------------------- E5
def load_density(c, tag):
    fs = [f for f in sorted(glob.glob(R + f"family_bank_[0-9]*_{tag}{sfx(c)}.csv")) if f.endswith(f"_{tag}{sfx(c)}.csv")]
    return pd.concat([pd.read_csv(f) for f in fs])


def fbdiff(d, sc, w, comp="mle_bank_log_point", met="excess_x1000"):
    p = d[(d.scenario == sc) & (d.window == w) & (d.metric == met)].pivot_table(index="seed", columns="method", values="value")
    return paired(p["family_bank"], p[comp])


def e5():
    base = {}
    for c in FAMS:
        a = load_density(c, "density3"); b = load_density(c, "density3b")
        base[c] = pd.concat([a[a.scenario.isin(["constant", "drift"])], b])
    sw = {c: load_density(c, "density3b") for c in ["gauss2t", "clayton2gumbel", "gauss2clayton", "claygum_mix"]}
    fig = plt.figure(figsize=(W, 5.1))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.8, 1.6], height_ratios=[1, 1.1], hspace=0.55, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    for yi, sc in enumerate(["constant", "drift", "jump"]):
        m, se = fbdiff(base["gaussian"], sc, "whole")
        ax.errorbar(m, yi, xerr=2 * se, fmt="o", ms=4.5, color=METHOD, lw=1.2, capsize=0, zorder=3)
    zero(ax); ax.set_xlim(-0.5, 2.6); ax.set_ylim(2.6, -0.6)
    ax.set_yticks(range(3)); ax.set_yticklabels(["Constant", "Drift", "Jump"]); ax.tick_params(axis="y", length=0)
    ax.set_title("(a) Gaussian truth"); ax.set_xlabel(r"Difference ($\times10^{-3}$)"); tidy(ax)
    ax = fig.add_subplot(gs[0, 1])
    labels, yi = [], 0
    for c in ["clayton", "gumbel", "t"]:
        for sc in ["constant", "drift", "jump"]:
            m, se = fbdiff(base[c], sc, "whole")
            ax.errorbar(m, yi, xerr=2 * se, fmt="o", ms=4.5, color=METHOD, lw=1.2, capsize=0, zorder=3)
            labels.append(f"{FL[c]}, {sc}"); yi += 1
        yi += 0.4
    ys = [i + 0.4 * (i // 3) for i in range(9)]
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=st.SMALL + 0.5); ax.set_ylim(max(ys) + 0.6, -0.6)
    ax.tick_params(axis="y", length=0)
    zero(ax); ax.set_xlim(-115, 8)
    ax.set_title("(b) Non-Gaussian truth"); ax.set_xlabel(r"Difference ($\times10^{-3}$)"); tidy(ax)
    wins = ["pre", "rec_0_100", "rec_100_500", "post"]
    wlab = ["before", "0–100", "100–500", "after"]
    regs = [("gauss2t", r"Gauss. $\to t_4$"), ("clayton2gumbel", r"Clay. $\to$ Gumb."),
            ("gauss2clayton", r"Gauss. $\to$ Clay."), ("claygum_mix", "Mixture")]
    sub = gs[1, :].subgridspec(1, 4, wspace=0.12)
    for ri, (key, lab) in enumerate(regs):
        ax = fig.add_subplot(sub[0, ri])
        ms_, ses = zip(*[fbdiff(sw[key], "shape", w) for w in wins])
        ax.errorbar(range(4), ms_, yerr=2 * np.array(ses), fmt="o", ms=4, color=METHOD, lw=1.2, capsize=0, zorder=3)
        zero(ax, horizontal=True)
        ax.set_ylim(-90, 60); ax.set_xticks(range(4)); ax.set_xticklabels(wlab, rotation=40, fontsize=st.SMALL, ha="right")
        ax.set_title(("(c) " if ri == 0 else "") + lab, fontsize=st.LABEL)
        if ri == 0: ax.set_ylabel(r"Difference ($\times10^{-3}$)")
        else: ax.set_yticklabels([])
        tidy(ax, "y")
    fig.subplots_adjust(left=0.14, right=0.99, top=0.94, bottom=0.12)
    fig.savefig(OUT + "ch6_fig_e5.png")


# ---------------------------------------------------------------- E6 (mixture)
def e6():
    d = load_density("claygum_mix", "density3b")
    wins = ["pre", "rec_0_100", "rec_100_500", "rec_500_1500", "post"]
    wlab = ["before", "0–100", "100–500", "500–1500", "after"]
    p = d[d.metric == "excess_x1000"].pivot_table(index=["window", "seed"], columns="method", values="value")
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    for m, lab, col, mk in [("family_bank", "family bank", METHOD, "o"), ("mle_bank_log_point", "single Gaussian", BASE, "s"),
                            ("arm_t_1999", "best fixed arm", st.FOURTH, "^")]:
        mu = [p.loc[w][m].mean() for w in wins]; se = [p.loc[w][m].std(ddof=1) / math.sqrt(60) for w in wins]
        ax.errorbar(range(5), mu, yerr=2 * np.array(se), fmt="-" + mk, ms=4, color=col, lw=1.2, capsize=0)
    fl = [p.loc[w]["gauss_floor_expected"].mean() for w in wins]
    ax.plot(range(5), fl, color=INK, lw=0.9, ls=(0, (3, 2)))
    ax.set_xticks(range(5)); ax.set_xticklabels(wlab, rotation=35, ha="right")
    ax.set_ylabel(r"Excess log loss ($\times10^{-3}$)"); ax.set_ylim(0, 42)
    ax.set_title("(a) Excess log loss")
    for yy, lab, col in [(18.2, "single Gaussian", MUTED), (13.8, "Gaussian floor", INK),
                         (10.6, "family bank", METHOD), (6.8, "best fixed arm", st.FOURTH)]:
        ax.text(4.25, yy, lab, color=col, va="center")
    ax.set_xlim(-0.3, 7.0); tidy(ax, "y")
    ax = axs[1]
    pth = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(R + "family_bank_path_[0-9]*_density2_claygum_mix.csv"))]) \
        .groupby(["t", "series"]).value.mean().unstack()
    t = pth.index.values
    ax.plot(t, pth["chiU_true"], color=INK, lw=1.4)
    ax.plot(t, pth["chiU_family"], color=METHOD, lw=1.4)
    ax.plot(t, pth["chiU_ratio399"], color=THIRD, lw=1.2)
    ax.text(700, 0.235, "truth", color=INK); ax.text(700, 0.115, "family bank", color=METHOD)
    ax.text(3150, 0.335, "corner counter", color=THIRD)
    ax.axvline(2500, color=SPINE_, lw=0.8, ls=":")
    ax.set_xlim(500, 5000); ax.set_ylim(0, 0.5); ax.set_xticks([1000, 2000, 3000, 4000, 5000])
    ax.set_ylabel(r"$\chi_U(0.05)$"); ax.set_xlabel("Observation, $t$")
    ax.set_title("(b) Upper-tail dependence"); tidy(ax, "y")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.25, wspace=0.3)
    fig.savefig(OUT + "ch6_fig_e6.png")


# ---------------------------------------------------------------- level ("which is lower") versions
def _e1_cells(cap=65536):
    df = pd.read_csv(R + "budget_sweep_final.csv"); sd = pd.read_csv(R + "budget_seeds_final.csv")
    val = df[df.split == "val"]; out = {}
    for s in ["constant", "drift", "jump"]:
        for f in FAMS:
            r = {}
            for e in ["xiao", "hermite", "mpl_dskll"]:
                v = val[(val.family == f) & (val.scenario == s) & (val.estimator == e) & (val.peak_bytes <= cap)]
                k = v.loc[v.rmse.idxmin()]
                x = sd[(sd.family == f) & (sd.scenario == s) & (sd.estimator == e) & (sd.approx == k.approx) & (np.isclose(sd.gain, k.gain))]
                r[e] = 100 * x.set_index("seed").rmse
            oth = min(["xiao", "hermite"], key=lambda e: r[e].mean())
            out[(s, f)] = ({e: r[e].mean() for e in r}, paired(r["mpl_dskll"], r[oth]), oth)
    return out


def _levels(fig, axs, cells, regimes, keys, style_, span, unit, diff_lab):
    """One panel per regime, same horizontal span (only the offset differs); paired difference ± 2 SE at each row end."""
    for ax, s in zip(axs, regimes):
        vals_all = [v for f in FAMS for v in cells[(s, f)][0].values()]
        lo = min(vals_all) - 0.05 * span
        ax.set_xlim(lo, lo + 1.5 * span)            # same units per cm in every panel; the right third holds the numbers
        ax.axvspan(lo + 1.08 * span, lo + 1.5 * span, color="white", zorder=2)
        ax.set_xticks([t for t in ax.get_xticks() if lo <= t <= lo + 1.05 * span])
        for yi, f in enumerate(FAMS):
            vals, (m, se), _ = cells[(s, f)]
            ax.plot([min(vals.values()), max(vals.values())], [yi, yi], color=GRID, lw=3.5, zorder=1, solid_capstyle="round")
            for k in keys:
                mk, col, ms = style_[k]
                ax.plot(vals[k], yi, mk, color=col, ms=ms, zorder=3, mec="white", mew=0.6)
            ax.text(0.995, yi, f"{m:+.2f} ± {2 * se:.2f}" if abs(m) < 10 else f"{m:+.1f} ± {2 * se:.1f}",
                    transform=ax.get_yaxis_transform(), fontsize=6.5, va="center", ha="right",
                    color=INK if abs(m) > 2 * se else MUTED)
        ax.set_title({"constant": "Constant dependence", "drift": "Gradual drift", "jump": "Abrupt jump"}[s])
        ax.set_xlabel(unit); tidy(ax)
    axs[0].set_yticks(range(4)); axs[0].set_yticklabels([FL[f] for f in FAMS]); axs[0].set_ylim(3.6, -0.75)
    axs[0].tick_params(axis="y", length=0)
    for ax in axs: ax.text(0.995, -0.62, diff_lab, transform=ax.get_yaxis_transform(), fontsize=6.3, ha="right", color=MUTED)


def e1_levels():
    cells = _e1_cells()
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.5), sharey=True)
    st = {"xiao": ("o", EST["xiao"], 6), "hermite": ("s", EST["hermite"], 5.5), "mpl_dskll": ("^", EST["mpl_dskll"], 6.5)}
    _levels(fig, axs, cells, ["constant", "drift", "jump"], ["xiao", "hermite", "mpl_dskll"], st, 1.2,
            r"test RMSE of $\hat\rho_S$ ($\times10^{-2}$)", "MPL − best other")
    for e, lab in [("xiao", "Xiao"), ("hermite", "Hermite"), ("mpl_dskll", "recursive MPL")]:
        axs[1].plot([], [], st[e][0], color=st[e][1], ms=6, label=lab)
    fig.legend(*axs[1].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.01))
    fig.text(0.01, 0.985, "Lower is better. 64 KiB budget, validation-tuned, 100 test seeds. All panels share one scale (only the offsets differ);\n"
             "row-end numbers: MPL minus the better of Xiao/Hermite, paired mean ± 2 SE (grey = within 2 SE).",
             fontsize=6.8, color=MUTED, va="top")
    fig.subplots_adjust(left=0.1, right=0.99, top=0.78, bottom=0.29, wspace=0.12)
    fig.savefig(OUT + "ch6_fig_e1_levels.png")


def e3_levels():
    fixed = ["fixed_39", "fixed_99", "fixed_199", "fixed_399", "fixed_1999"]
    cells = {}
    for f in FAMS:
        d = pd.read_csv(R + f"thesis_bank_matched500{sfx(f)}.csv")
        for sc in ["constant", "drift", "jump"]:
            p = d[(d.scenario == sc) & (d.window == "whole") & (d.component == "mle")].pivot(index="seed", columns="method", values="rmse") * 1000
            best = min(fixed, key=lambda m: p[m].mean())
            cells[(sc, f)] = ({"bank": p["bank_log"].mean(), "best": p[best].mean(), "vff": p["vff_slow"].mean()},
                              paired(p["bank_log"], p[best]), best)
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.5), sharey=True)
    st = {"bank": ("o", METHOD, 6.5), "best": ("D", BASE, 5.5), "vff": ("s", THIRD, 5.5)}
    _levels(fig, axs, cells, ["constant", "drift", "jump"], ["vff", "best", "bank"], st, 42,
            r"RMSE of $\hat\rho_S$ ($\times10^{-3}$)", "bank − best fixed")
    for k, lab in [("bank", "log bank (adaptive)"), ("best", "best fixed memory (hindsight)"), ("vff", "slow VFF")]:
        axs[1].plot([], [], st[k][0], color=st[k][1], ms=6, label=lab)
    fig.legend(*axs[1].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.01))
    fig.text(0.01, 0.985, "Lower is better. Recursive MPL, whole run, 60 seeds. All panels share one scale (only the offsets differ);\n"
             "row-end numbers: log bank minus best fixed memory, paired mean ± 2 SE.", fontsize=6.8, color=MUTED, va="top")
    fig.subplots_adjust(left=0.1, right=0.99, top=0.78, bottom=0.29, wspace=0.12)
    fig.savefig(OUT + "ch6_fig_e3_levels.png")


# ---------------------------------------------------------------- grouped-column alternatives
def e1_columns():
    cells = _e1_cells()
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.9))
    est = ["xiao", "hermite", "mpl_dskll"]
    bw = 0.27
    for ax, s in zip(axs, ["constant", "drift", "jump"]):
        for fi, f in enumerate(FAMS):
            vals = cells[(s, f)][0]
            for ei, e in enumerate(est):
                ax.bar(fi + (ei - 1) * bw, vals[e], width=bw * 0.92, color=EST[e], label=st.EST_LABEL[e] if fi == 0 else None, zorder=3)
        ax.set_xticks(range(4)); ax.set_xticklabels(["Gauss.", "Clay.", "Gumb.", "$t_4$"])
        ax.set_title(REG[s])
        ax.set_ylim(0, {"constant": 2.0, "drift": 5.0, "jump": 9.0}[s]); tidy(ax, "y"); ax.tick_params(axis="x", length=0)
    axs[0].set_ylabel(r"Spearman RMSE ($\times10^{-2}$)")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.53, -0.01))
    fig.subplots_adjust(left=0.1, right=0.99, top=0.9, bottom=0.24, wspace=0.32)
    fig.savefig(OUT + "ch6_fig_e1_columns.png")


def e3_columns():
    """Recursive MPL: log bank, hindsight-best fixed memory (per cell) and slow VFF, averaged over the four families."""
    fixed = ["fixed_39", "fixed_99", "fixed_199", "fixed_399", "fixed_1999"]
    agg = {}
    for sc in ["constant", "drift", "jump"]:
        rows = []
        for f in FAMS:
            d = pd.read_csv(R + f"thesis_bank_matched500{sfx(f)}.csv")
            p = d[(d.scenario == sc) & (d.window == "whole") & (d.component == "mle")].pivot(index="seed", columns="method", values="rmse") * 1000
            best = min(fixed, key=lambda m: p[m].mean())
            rows.append(pd.DataFrame({"bank": p["bank_log"], "best": p[best], "vff": p["vff_slow"]}))
        a = sum(rows) / len(rows)
        agg[sc] = ({k: a[k].mean() for k in a}, paired(a["bank"], a["best"]), paired(a["bank"], a["vff"]))
    fig, ax = plt.subplots(figsize=(W * 0.7, 3.0))
    items = [("bank", "log bank", METHOD), ("best", "best fixed memory", BASE), ("vff", "slow VFF", THIRD)]
    bw = 0.27
    for si, sc in enumerate(["constant", "drift", "jump"]):
        for ki, (k, lab, col) in enumerate(items):
            ax.bar(si + (ki - 1) * bw, agg[sc][0][k], width=bw * 0.92, color=col, label=lab if si == 0 else None, zorder=3)
    ax.set_xticks(range(3)); ax.set_xticklabels([REG[s] for s in ["constant", "drift", "jump"]]); ax.tick_params(axis="x", length=0)
    ax.set_ylabel(r"Spearman RMSE ($\times10^{-3}$)"); ax.set_ylim(0, 95); tidy(ax, "y")
    ax.legend(loc="upper left")
    fig.subplots_adjust(left=0.15, right=0.98, top=0.96, bottom=0.12)
    fig.savefig(OUT + "ch6_fig_e3_columns.png")
    for sc in agg:
        (m1, s1), (m2, s2) = agg[sc][1], agg[sc][2]
        print(f"  E3 {sc}: bank {agg[sc][0]['bank']:.1f}, best fixed {agg[sc][0]['best']:.1f}, slow VFF {agg[sc][0]['vff']:.1f};"
              f" bank-best {m1:+.1f}±{2*s1:.1f}, bank-VFF {m2:+.1f}±{2*s2:.1f} (paired, 2 SE)")


if __name__ == "__main__":
    which = sys.argv[1:] or ["e1_columns", "e2", "e3_columns", "e4", "e5", "e6"]
    for w in which:
        globals()[w]()
        print("wrote", OUT + f"ch6_fig_{w}.png")
