"""Section 6.2 / 6.7 storage - time - accuracy analysis with the extended RMPL KLL grid (k = 2048, 4096, 8192; TAG kllext).
Accuracy: results/budget_{sweep,seeds}_final.csv plus results/budget_{sweep,seeds}_kllext_{cd,j}_<family>.csv (RMPL only; the
k = 1024 rows must reproduce the final sweep exactly). Timing: results/budget_timing_rep{1..5}.csv (probe_budget time, gain 0.02,
Gaussian drift, one thread, 5 seeds x 4500 ticks, update + estimate) and budget_timing_g{0,0.005,0.01}.csv (selected gains).
RMSE per approximation size: each family at its validation-best gain for that size; per test seed averaged over the four
families; mean ± SE over 100 seeds. Usage (from experiments/): python3 analyze_budget_time.py ->
results/budget_time_2026-11-15.md, results/budget_time_table.csv, ../images/ch6/6.2/ch6_budget_storage_time.png"""
import math
import sys
import numpy as np
import pandas as pd

R = "results/"
FAMS = ["gaussian", "t", "clayton", "gumbel"]
SC = ["constant", "drift", "jump"]
E = ["mpl_dskll", "xiao", "hermite"]
NAME = {"mpl_dskll": "RMPL", "xiao": "Counting Matrix", "hermite": "Hermite Series"}
APX = {"mpl_dskll": "k", "xiao": "bins", "hermite": "order"}
BUD = {16: 16384, 64: 65536, 256: 262144}
M = []
P = M.append

fs, fd = pd.read_csv(R + "budget_sweep_final.csv"), pd.read_csv(R + "budget_seeds_final.csv")
xs = pd.concat([pd.read_csv(R + f"budget_sweep_kllext_{t}_{f}.csv") for t in ("cd", "j") for f in FAMS])
xd = pd.concat([pd.read_csv(R + f"budget_seeds_kllext_{t}_{f}.csv") for t in ("cd", "j") for f in FAMS])
k = ["family", "scenario", "estimator", "approx", "gain", "seed"]
chk = xd[xd.approx == 1024].merge(fd, on=k, suffixes=("_x", "_f"))
chs = xs[xs.approx == 1024].merge(fs, on=["family", "scenario", "split", "estimator", "approx", "gain"], suffixes=("_x", "_f"))
P("# Storage, processing time and accuracy, extended RMPL grid (2026-11-15)\n")
P(f"Reproduction of RMPL k = 1024 by the extension run: {len(chk)} per-seed test rows, max |ΔRMSE| {np.abs(chk.rmse_x - chk.rmse_f).max():.1e}; "
  f"{len(chs)} summary rows, max |ΔRMSE| {np.abs(chs.rmse_x - chs.rmse_f).max():.1e}, peak bytes equal {bool((chs.peak_bytes_x == chs.peak_bytes_f).all())}.\n")
sw = pd.concat([fs, xs[xs.approx > 1024]], ignore_index=True)
sd = pd.concat([fd, xd[xd.approx > 1024]], ignore_index=True)
val = sw[sw.split == "val"]
SD = sd.set_index(k).rmse.sort_index()

# ---------- timing
reps = [pd.read_csv(R + f"budget_timing_rep{r}.csv") for r in range(1, 6)]
T = pd.concat([d.assign(rep=i) for i, d in enumerate(reps)])
TS = T.groupby(["estimator", "approx"]).agg(peak=("peak_bytes", "max"), t=("mean_ns", "mean"), t_sd=("mean_ns", "std"),
                                            t_min=("mean_ns", "min"), t_max=("mean_ns", "max"), p95=("p95_ns", "mean"),
                                            upd=("update_mean_ns", "mean")).reset_index()
G = {g: pd.read_csv(R + f"budget_timing_g{g}.csv").set_index(["estimator", "approx"]).mean_ns for g in ("0", "0.005", "0.01")}
old = pd.read_csv(R + "budget_timing_final.csv").set_index(["estimator", "approx"]).mean_ns


def acc(s, e, a):
    per = []
    for f in FAMS:
        v = val[(val.family == f) & (val.scenario == s) & (val.estimator == e) & (val.approx == a)]
        g = float(v.loc[v.rmse.idxmin()].gain)
        per.append(100 * SD.loc[(f, s, e, a, g)].values)
    x = np.mean(per, axis=0)
    return x.mean(), x.std(ddof=1) / math.sqrt(len(x))


rows = []
for _, r in TS[TS.estimator.isin(E)].iterrows():
    d = dict(estimator=r.estimator, approx=int(r.approx), kib=r.peak / 1024, us=r.t / 1e3, us_sd=r.t_sd / 1e3,
             us_cv=r.t_sd / r.t, p95_us=r.p95 / 1e3, upd_share=r.upd / r.t,
             us_g0=G["0"].get((r.estimator, r.approx), np.nan) / 1e3, us_g005=G["0.005"].get((r.estimator, r.approx), np.nan) / 1e3,
             us_g01=G["0.01"].get((r.estimator, r.approx), np.nan) / 1e3, us_old=old.get((r.estimator, r.approx), np.nan) / 1e3)
    for s in SC:
        d[f"rmse_{s}"], d[f"se_{s}"] = acc(s, r.estimator, int(r.approx))
    rows.append(d)
TT = pd.DataFrame(rows).sort_values(["estimator", "approx"])
TT.to_csv(R + "budget_time_table.csv", index=False)

P("## 1. Timing table (gain 0.02; mean of 5 repeated runs ± SD across runs; RMSE family-averaged at the validation-best gain per size, ×10⁻²)\n")
P("| Estimator | size | retained KiB | µs / obs (± SD, 5 runs) | p95 µs | update share | RMSE constant | RMSE drift | RMSE jump |")
P("|---|---|---|---|---|---|---|---|---|")
for e in E:
    for _, r in TT[TT.estimator == e].iterrows():
        P(f"| {NAME[e]} | {APX[e]} = {r.approx} | {r.kib:.2f} | {r.us:.2f} ± {r.us_sd:.2f} | {r.p95_us:.2f} | {r.upd_share:.0%} | "
          f"{r.rmse_constant:.2f} | {r.rmse_drift:.2f} | {r.rmse_jump:.2f} |")
P(f"\nTiming stability: coefficient of variation across the 5 runs median {TT.us_cv.median():.1%}, max {TT.us_cv.max():.1%} "
  f"({NAME[TT.loc[TT.us_cv.idxmax()].estimator]} {APX[TT.loc[TT.us_cv.idxmax()].estimator]} = {TT.loc[TT.us_cv.idxmax()].approx}). "
  f"Ratio to the earlier single run (budget_timing_final.csv, k ≤ 1024): median {np.nanmedian(TT.us / TT.us_old):.2f}, "
  f"range {np.nanmin(TT.us / TT.us_old):.2f}–{np.nanmax(TT.us / TT.us_old):.2f}.")

P("\n## 2. Effect of the gain on time (ratio to gain 0.02, same run conditions, one run each)\n")
P("| Estimator | cumulative (g = 0) | g = 0.005 | g = 0.01 |")
P("|---|---|---|---|")
for e in E:
    x = TT[TT.estimator == e]
    P(f"| {NAME[e]} | {np.median(x.us_g0 / x.us):.2f} ({np.min(x.us_g0 / x.us):.2f}–{np.max(x.us_g0 / x.us):.2f}) | "
      f"{np.median(x.us_g005 / x.us):.2f} ({np.min(x.us_g005 / x.us):.2f}–{np.max(x.us_g005 / x.us):.2f}) | "
      f"{np.median(x.us_g01 / x.us):.2f} ({np.min(x.us_g01 / x.us):.2f}–{np.max(x.us_g01 / x.us):.2f}) |")

# ---------- budget re-selection with the extended grid (RMPL)
P("\n## 3. Budget selections with the extended RMPL grid (validation only, as before)\n")
sel = []
for f in FAMS:
    for s in SC:
        for b, cap in BUD.items():
            for e in E:
                v = val[(val.family == f) & (val.scenario == s) & (val.estimator == e) & (val.peak_bytes <= cap)]
                r = v.loc[v.rmse.idxmin()]
                x = 100 * SD.loc[(f, s, e, int(r.approx), float(r.gain))]
                tt = TT[(TT.estimator == e) & (TT.approx == int(r.approx))].iloc[0]
                sel.append(dict(family=f, scenario=s, budget=b, estimator=e, approx=int(r.approx), gain=float(r.gain), kib=r.peak_bytes / 1024,
                                rmse=x.mean(), x=x.values, us=tt.us, us_sel=tt[{0.0: "us_g0", 0.005: "us_g005", 0.01: "us_g01"}.get(float(r.gain), "us")]))
S = pd.DataFrame(sel)
P("| Estimator | budget | selected sizes (cells) | retained KiB max | µs at selected configs (g = 0.02) min–max | µs at selected gains min–max |")
P("|---|---|---|---|---|---|")
for e in E:
    for b in BUD:
        x = S[(S.estimator == e) & (S.budget == b)]
        P(f"| {NAME[e]} | {b} KiB | " + ", ".join(f"{a} ({n})" for a, n in x.approx.value_counts().sort_index().items())
          + f" | {x.kib.max():.2f} | {x.us.min():.2f}–{x.us.max():.2f} | {x.us_sel.min():.2f}–{x.us_sel.max():.2f} |")
red = []
for (f, s, e), x in S.groupby(["family", "scenario", "estimator"]):
    x = x.set_index("budget")
    for lo, hi in [(16, 64), (64, 256), (16, 256)]:
        d = x.loc[lo].x - x.loc[hi].x
        red.append(dict(family=f, scenario=s, estimator=e, step=f"{lo}->{hi}", red=d.mean(), se=d.std(ddof=1) / 10))
Rd = pd.DataFrame(red)
P("\nRMSE reduction 16 → 256 KiB with the extended RMPL grid (×10⁻², paired):\n")
for e in E:
    x = Rd[(Rd.estimator == e) & (Rd.step == "16->256")]
    P(f"- {NAME[e]}: median {x.red.median():+.3f}, max {x.red.max():+.3f}; 64 → 256 KiB max {Rd[(Rd.estimator == e) & (Rd.step == '64->256')].red.max():+.3f}")
x = Rd[Rd.step == "16->256"]
P(f"- All 36: median {x.red.median():+.3f}, max {x.red.max():+.3f}")

# ---------- dominance
P("\n## 4. Dominated configurations (family-averaged RMSE per scenario; a configuration is dominated if another, of any estimator, "
  "is no larger in retained KiB and µs and has RMSE lower by more than 2 SE of the difference treated as unpaired)\n")
for s in SC:
    X = TT[["estimator", "approx", "kib", "us", f"rmse_{s}", f"se_{s}"]].rename(columns={f"rmse_{s}": "r", f"se_{s}": "se"})
    dom = []
    for i, a in X.iterrows():
        for j, b in X.iterrows():
            if i != j and b.kib <= a.kib and b.us <= a.us and (a.r - b.r) > 2 * math.hypot(a.se, b.se):
                dom.append(f"{NAME[a.estimator]} {a.approx} (by {NAME[b.estimator]} {b.approx})"); break
    best = X.loc[X.r.idxmin()]
    P(f"- **{s}**: lowest RMSE {best.r:.2f} ({NAME[best.estimator]} {APX[best.estimator]} = {best.approx}, {best.kib:.1f} KiB, {best.us:.2f} µs). "
      f"Dominated: {len(dom)}/{len(X)} — " + "; ".join(dom))
    for cap_us in (1.0, 2.0, 5.0):
        y = X[X.us <= cap_us]
        if len(y):
            b2 = y.loc[y.r.idxmin()]
            P(f"  - within {cap_us:g} µs: best {NAME[b2.estimator]} {APX[b2.estimator]} = {b2.approx} (RMSE {b2.r:.2f}, {b2.kib:.1f} KiB, {b2.us:.2f} µs)")

open(R + "budget_time_2026-11-15.md", "w").write("\n".join(M) + "\n")
print("\n".join(M))

# ---------- figure: storage vs time
sys.path.insert(0, ".")
import plot_constant_ess as ce  # noqa: E402
plt, core = ce.plt, ce.core
ST = {"mpl_dskll": (ce.COL["mle"], "^"), "xiao": (ce.COL["xiao"], "o"), "hermite": (ce.COL["hermite"], "s")}
with plt.rc_context(ce.RC):
    fig, ax = plt.subplots(figsize=(core.W * 0.72, 2.6))
    for e in E:
        x = TT[TT.estimator == e]
        c, m = ST[e]
        ax.plot(x.kib, x.us, color=c, marker=m, ms=3.4, lw=1.1, label=NAME[e])
        lab = {"mpl_dskll": "$k$ = {}", "xiao": "{} bins", "hermite": "order {}"}[e]
        for n, (_, r) in enumerate(x.iloc[[0, -1]].iterrows()):
            off = {("mpl_dskll", 0): (-4, 6), ("xiao", 0): (5, -3), ("hermite", 0): (2, -9),
                   ("mpl_dskll", 1): (-14, 7), ("xiao", 1): (5, -2), ("hermite", 1): (5, -2)}[(e, n)]
            ax.annotate(lab.format(r.approx), (r.kib, r.us), xytext=off, textcoords="offset points", fontsize=5.8, color=c,
                        ha="right" if (e, n) == ("mpl_dskll", 0) else "left")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128", "256"]); ax.set_xlim(0.9, 700)
    yt = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
    ax.set_yticks(yt); ax.set_yticklabels([f"{v:g}" for v in yt])
    ax.grid(True, color=ce.GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_xlabel("Retained state (KiB, log scale)", fontsize=7.5); ax.set_ylabel("Time per observation (µs, log scale)", fontsize=7.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.17)
    fig.savefig("../images/ch6/6.2/ch6_budget_storage_time.png"); plt.close(fig)
print("wrote ../images/ch6/6.2/ch6_budget_storage_time.png")
