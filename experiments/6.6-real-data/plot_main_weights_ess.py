"""Plot MFF weights and effective memory around large dependence changes."""
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from real_common import EST, FIX, GRID, RC, W, change, image, outputs, plt, read, result, verbose_print

FA = np.array(FIX, float)
MEMC = [plt.get_cmap("plasma")(v) for v in (0.82, 0.64, 0.47, 0.30, 0.10)]
MFFC, VFFC = "#C2185B", "#8E6BB8"
LAGS = np.arange(-63, 127)


def evaldays(x):
    x = x.iloc[560:]
    return x[x.date >= "2008-01-01"].reset_index(drop=True)


def episode_starts(roll):
    starts = {}
    for p, x in roll.items():
        D = evaldays(x.assign(D=change(x))).D.values
        c = D >= np.nanquantile(D, 0.9)
        starts[p] = [t for t in range(len(c)) if c[t] and t >= 21 and not c[t - 21:t].any()]
    return starts


def episodes(S, roll, starts):
    res, full, vff_paths = {}, [], {}
    for e, d in EST:
        Wl, Ml, Vl, Mall, Vall, Hall, pm = [], [], [], [], [], [], []
        for p, f in outputs(S, d).items():
            x = evaldays(read(f))
            assert (x.date.values == evaldays(roll[p]).date.values).all()
            Wt = x[[f"wt_{k}" for k in FIX]].values
            mem, nv = Wt @ FA, x.neff_vff.values
            assert np.abs(mem - x.neff_mff_t.values).max() < 0.01, "sum_k w_k N_k must equal the saved neff_mff_t"
            vff_paths.setdefault(p, []).append(nv)
            Mall.append(mem)
            Vall.append(nv)
            Hall.append(1 / (Wt @ (1 / FA)))
            pm.append(mem.mean())
            for t in starts[p]:
                if t + LAGS[0] >= 0 and t + LAGS[-1] < len(x):
                    sl = t + LAGS
                    Wl.append(Wt[sl])
                    Ml.append(mem[sl])
                    Vl.append(nv[sl])
        Wl, Ml, Vl = np.array(Wl), np.array(Ml), np.array(Vl)
        res[e] = dict(W=Wl.mean(0), M=np.median(Ml, 0), Mq=np.percentile(Ml, [25, 75], 0), V=np.median(Vl, 0), Vq=np.percentile(Vl, [25, 75], 0), n=len(Wl))
        M, V, H = map(np.concatenate, (Mall, Vall, Hall))
        full.append(dict(estimator=e, mff_wavg_median=np.median(M), mff_wavg_q25=np.percentile(M, 25), mff_wavg_q75=np.percentile(M, 75),
                         mff_wavg_mean=M.mean(), pair_mean_min=min(pm), pair_mean_max=max(pm), mff_harmonic_median=np.median(H),
                         vff_median=np.median(V), vff_mean=V.mean()))
    vff_same = max(np.abs(a - b).max() for v in vff_paths.values() for a in v for b in v)
    return res, full, vff_same


def figure(res):
    H = 3.1
    with plt.rc_context(RC):
        fig, axs = plt.subplots(2, 3, figsize=(W, H), sharex=True, sharey="row", gridspec_kw={"height_ratios": [1.35, 1]})
        for ci, (e, _) in enumerate(EST):
            r, a1, a2 = res[e], axs[0, ci], axs[1, ci]
            a1.stackplot(LAGS, *r["W"].T, colors=MEMC, lw=0, zorder=2)
            a1.set_ylim(0, 1)
            a1.set_title(e)
            a2.plot(LAGS, r["V"], color=VFFC, lw=1.1, zorder=3)
            a2.plot(LAGS, r["M"], color=MFFC, lw=1.1, zorder=4)
            a2.set_yscale("log")
            a2.set_ylim(30, 2600)
            a2.set_yticks(FIX)
            a2.set_yticklabels([str(k) for k in FIX])
            a2.minorticks_off()
            a2.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
            a2.set_axisbelow(True)
            for a in (a1, a2):
                a.set_xlim(LAGS[0], LAGS[-1])
                a.set_xticks([-50, 0, 50, 100])
                a.axvline(0, color="#111111", lw=0.8, ls=(0, (4, 3)), zorder=6)
        axs[0, 0].set_ylabel("Mean MFF weight")
        axs[1, 0].set_ylabel("Weight-averaged\ncandidate " + r"$N_{\mathrm{eff}}$")
        h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in MEMC] + [Line2D([], [], color=MFFC, lw=1.1), Line2D([], [], color=VFFC, lw=1.1)]
        fig.legend(h, [r"$N_{\mathrm{eff}}=%d$" % m for m in FIX] + ["MFF", "VFF"], loc="upper center", ncol=7, frameon=False, fontsize=7,
                   bbox_to_anchor=(0.53, 0.955), handlelength=1.2, columnspacing=1.2)
        fig.text(0.53, 0.01, "Trading days from the start of a large-change episode", ha="center", fontsize=8)
        fig.subplots_adjust(left=0.12, right=0.985, top=1 - 0.665 / H, bottom=0.42 / H, hspace=0.14, wspace=0.08)
        fig.savefig(image("ch6_practical_weights_ess.png"))
        plt.close(fig)


def values(res):
    rows = []
    for e, _ in EST:
        r = res[e]
        for j, l in enumerate(LAGS):
            rows.append(dict(estimator=e, lag=l, **{f"w_{k}": r["W"][j, i] for i, k in enumerate(FIX)}, mff_wavg_ess_median=r["M"][j],
                             mff_wavg_ess_q25=r["Mq"][0, j], mff_wavg_ess_q75=r["Mq"][1, j], vff_ess_median=r["V"][j], n_episodes=r["n"]))
    return pd.DataFrame(rows)


def main():
    S = sys.argv[1]
    roll = {p: read(f) for p, f in outputs(S, "practical_roll2").items()}
    starts = episode_starts(roll)
    res, full, vff_same = episodes(S, roll, starts)
    figure(res)
    values(res).to_csv(result("weights_ess_values.csv"), index=False)
    pd.DataFrame(full).to_csv(result("weights_ess_full_period.csv"), index=False)
    verbose_print(f"Episode starts {sum(len(v) for v in starts.values())}, max VFF ESS difference {vff_same:.3g}")


if __name__ == "__main__":
    main()
