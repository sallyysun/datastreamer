"""Section 6.2 (E1) audit and consolidation, no simulation. Candidates = results/budget_{sweep,seeds}_final.csv (matched sweep,
g in {0.005, 0.01, 0.02, 0.05, 0.10}, plus the extension g in {cum, 0.0005, 0.001, 0.002} for constant and drift; jump cells
only have the original grid). Per family x scenario x budget each estimator's (approx, gain) is chosen on validation seeds
1000-1039 among candidates whose validation peak retained state fits the cap (the analyze_budget_gainext.py rule), then scored
on test seeds 42-141. Paired SEs use the per-seed test RMSEs (same seeds for every configuration and family). Timing:
results/budget_timing_final.csv (probe_budget time: Gaussian drift, g = 0.02, single thread, 5 seeds x 4500 ticks,
update + estimate). Retained-state breakdown: scratch diagnostic mem_breakdown.cpp (see the summary).
Usage (from experiments/): python3 audit_budget.py -> results/budget_audit_2026-11-15.md, results/budget_audit_cells.csv,
results/budget_audit_familyavg.csv, ../images/ch6/6.2/ch6_budget_accuracy_storage.png"""
import math
import os
import sys
import numpy as np
import pandas as pd

R = "results/"
BUD = {16: 16384, 64: 65536, 256: 262144}
E = ["mpl_dskll", "xiao", "hermite"]
NAME = {"mpl_dskll": "RMPL", "xiao": "Counting Matrix", "hermite": "Hermite Series"}
SHORT = {"mpl_dskll": "RMPL", "xiao": "CM", "hermite": "HS"}
FAMS = ["gaussian", "t", "clayton", "gumbel"]
FL = {"gaussian": "Gaussian", "t": "t4", "clayton": "Clayton", "gumbel": "Gumbel"}
SC = ["constant", "drift", "jump"]

sw = pd.read_csv(R + "budget_sweep_final.csv")
sd = pd.read_csv(R + "budget_seeds_final.csv")
tm = pd.read_csv(R + "budget_timing_final.csv")
val, tst = sw[sw.split == "val"], sw[sw.split == "test"]
SD = sd.set_index(["family", "scenario", "estimator", "approx", "gain", "seed"]).rmse.sort_index()
M = []
P = M.append

# ---------- 1. design checks
P("# Section 6.2 (E1) memory-constraint audit, 2026-11-15\n")
P("Platform of the runs: Linux x86_64, libstdc++, g++ -O2 (budget_gainext_2026-11-11.txt header). No new simulation was run for this audit.\n")
P("## 1. Design verification\n")
nseed = sd.groupby(["family", "scenario", "estimator", "approx", "gain"]).seed.agg(["count", "min", "max"])
P(f"- Test seeds per configuration: {nseed['count'].min()}–{nseed['count'].max()}, range {nseed['min'].min()}–{nseed['max'].max()} "
  f"(per-seed file holds test seeds only; validation enters only through the summary rows, split = val).")
P(f"- Families: {sorted(sw.family.unique())}; scenarios: {sorted(sw.scenario.unique())}; estimators in the candidate set: {sorted(sw.estimator.unique())}.")
for s in SC:
    g = sorted(sw[(sw.scenario == s) & (sw.estimator == 'xiao')].gain.unique())
    P(f"- Gain grid, {s}: {', '.join('cum' if x == 0 else f'{x:g}' for x in g)}")
for e, lab in [("xiao", "bins B"), ("hermite", "order N"), ("mpl_dskll", "KLL k")]:
    P(f"- Approximation grid, {NAME[e]} ({lab}): {[int(a) for a in sorted(sw[sw.estimator == e].approx.unique())]}")
P("- Validation seeds 1000–1039 (n_val = 40) and test seeds 42–141 (n_test = 100): from the probe_budget sweep invocation "
  "(`sweep 100 40`, kValSeed = 1000, kTestSeed = 42) recorded in the run header.\n")


def pick(f, s, e, cap):
    v = val[(val.family == f) & (val.scenario == s) & (val.estimator == e) & (val.peak_bytes <= cap)]
    r = v.loc[v.rmse.idxmin()]
    return int(r.approx), float(r.gain), int(r.peak_bytes), float(r.rmse)


def seeds(f, s, e, a, g):
    return SD.loc[(f, s, e, a, g)]


def glab(g): return "cum" if g == 0 else f"{g:g}"


rows = []
for f in FAMS:
    for s in SC:
        for b, cap in BUD.items():
            for e in E:
                a, g, pk, vr = pick(f, s, e, cap)
                t = tst[(tst.family == f) & (tst.scenario == s) & (tst.estimator == e) & (tst.approx == a) & np.isclose(tst.gain, g)].iloc[0]
                x = seeds(f, s, e, a, g)
                tt = tm[(tm.estimator == e) & (tm.approx == a)].iloc[0]
                rows.append(dict(family=f, scenario=s, budget_kib=b, estimator=e, approx=a, gain=g, val_rmse=100 * vr,
                                 rmse=100 * x.mean(), se=100 * x.std(ddof=1) / math.sqrt(len(x)), n=len(x),
                                 peak_val=pk, peak_test=int(t.peak_bytes), cap=cap, util=100 * max(pk, t.peak_bytes) / cap,
                                 tick_ns=tt.mean_ns, update_ns=tt.update_mean_ns, p95_ns=tt.p95_ns))
C = pd.DataFrame(rows)
C.to_csv(R + "budget_audit_cells.csv", index=False)
assert len(C) == 108

P("## 2. Selection uses validation only\n")
P("- `pick()` (identical to analyze_budget_gainext.py and plot_budget.py) filters split = val rows and takes the minimum validation RMSE "
  "among candidates whose validation peak bytes ≤ cap. Test rows enter only after selection. Confirmed by reading the code; no test quantity "
  "feeds a configuration choice.")
P(f"- All 108 selected configurations also fit their cap on the test seeds: max test peak / cap = {(C.peak_test / C.cap).max():.3f}; "
  f"test peak = validation peak in {(C.peak_test == C.peak_val).sum()}/108 (RMPL's DataSketches compaction is randomised but its peak is deterministic here).")
P("- Exception to 'validation only': the *comparator* in 'RMPL − better of CM/HS' is chosen on test means (draft §6.2 says so). Section 5 reports the validation-chosen comparator as well.\n")

# ---------- 3. accuracy table
P("## 3. Test Spearman RMSE (×10⁻², mean ± SE over 100 test seeds) at the validated configuration\n")
P("Cell entries: RMSE ± SE [approx, gain]. Approx = KLL k (RMPL), bins B (CM), order N (HS).\n")
P("| Scenario | Family | Budget | RMPL | Counting Matrix | Hermite Series |")
P("|---|---|---|---|---|---|")
for s in SC:
    for f in FAMS:
        for b in BUD:
            c = C[(C.family == f) & (C.scenario == s) & (C.budget_kib == b)].set_index("estimator")
            P(f"| {s} | {FL[f]} | {b} | " + " | ".join(f"{c.loc[e].rmse:.2f} ± {c.loc[e].se:.2f} [{c.loc[e].approx}, {glab(c.loc[e].gain)}]" for e in E) + " |")

# ---------- 4. storage effect
P("\n## 4. Effect of storage (RMSE reduction ×10⁻², positive = more storage helps; paired over the 100 test seeds)\n")
imp = []
for f in FAMS:
    for s in SC:
        for e in E:
            cfg = {b: C[(C.family == f) & (C.scenario == s) & (C.budget_kib == b) & (C.estimator == e)].iloc[0] for b in BUD}
            x = {b: seeds(f, s, e, int(cfg[b].approx), float(cfg[b].gain)) for b in BUD}
            for lo, hi in [(16, 64), (64, 256), (16, 256)]:
                d = 100 * (x[lo] - x[hi])
                imp.append(dict(family=f, scenario=s, estimator=e, step=f"{lo}->{hi}", red=d.mean(), se=d.std(ddof=1) / math.sqrt(len(d)),
                                same_cfg=(cfg[lo].approx == cfg[hi].approx) and (cfg[lo].gain == cfg[hi].gain),
                                cfg_lo=f"{cfg[lo].approx},{glab(cfg[lo].gain)}", cfg_hi=f"{cfg[hi].approx},{glab(cfg[hi].gain)}"))
I = pd.DataFrame(imp)
I.to_csv(R + "budget_audit_storage.csv", index=False)
P("| Step | Estimator | cells with a different configuration | max reduction (cell) | median | mean | min (cell) | cells reduction > 2 SE | cells worse by > 2 SE |")
P("|---|---|---|---|---|---|---|---|---|")
for step in ["16->64", "64->256", "16->256"]:
    for e in E:
        x = I[(I.step == step) & (I.estimator == e)]
        mx, mn = x.loc[x.red.idxmax()], x.loc[x.red.idxmin()]
        P(f"| {step} | {NAME[e]} | {(~x.same_cfg).sum()}/12 | {mx.red:+.3f} ± {mx.se:.3f} ({FL[mx.family]} {mx.scenario}) | {x.red.median():+.3f} | "
          f"{x.red.mean():+.3f} | {mn.red:+.3f} ± {mn.se:.3f} ({FL[mn.family]} {mn.scenario}) | {(x.red > 2 * x.se).sum()}/12 | {(x.red < -2 * x.se).sum()}/12 |")
for step in ["16->64", "64->256", "16->256"]:
    x = I[I.step == step]
    P(f"\nAll estimators, {step}: max {x.red.max():+.3f}, median {x.red.median():+.3f}, mean {x.red.mean():+.3f}, "
      f"median over cells where the configuration changed {x[~x.same_cfg].red.median():+.3f} ({(~x.same_cfg).sum()}/36 changed).")
big = I[(I.step == "16->256") & (I.red > 0.12 + 1e-9)]
P(f"\n**Check of 'at most 0.12 ×10⁻²' (16 → 256 KiB):** {len(big)} of 36 estimator-cells exceed 0.12 (unrounded):")
for _, r in big.iterrows():
    P(f"- {NAME[r.estimator]}, {FL[r.family]} {r.scenario}: {r.red:+.3f} ± {r.se:.3f} [{r.cfg_lo} → {r.cfg_hi}]")
cj = I[(I.step == "16->256") & (I.scenario != "jump")]
P(f"- Restricted to constant and drift (the draft's scope): max {cj.red.max():+.3f} ({NAME[cj.loc[cj.red.idxmax()].estimator]}, "
  f"{FL[cj.loc[cj.red.idxmax()].family]} {cj.loc[cj.red.idxmax()].scenario}).")
pairmax = I[I.step != "16->256"]
P(f"- Largest single-step (16→64 or 64→256) reduction: {pairmax.red.max():+.3f} ({NAME[pairmax.loc[pairmax.red.idxmax()].estimator]}, "
  f"{FL[pairmax.loc[pairmax.red.idxmax()].family]} {pairmax.loc[pairmax.red.idxmax()].scenario}, {pairmax.loc[pairmax.red.idxmax()].step}).")
neg = I[I.red < 0]
P(f"- More storage gave a *higher* test RMSE in {len(neg)} estimator-cell-steps (validation selection noise): "
  + "; ".join(f"{NAME[r.estimator]} {FL[r.family]} {r.scenario} {r.step} {r.red:+.3f} ± {r.se:.3f}" for _, r in neg.iterrows()) + ".")

# ---------- 5. RMPL vs best nonparametric
P("\n## 5. RMPL minus the better nonparametric estimator (×10⁻², paired mean ± SE; W/L at 2 SE)\n")
cmp = []
for f in FAMS:
    for s in SC:
        for b in BUD:
            c = C[(C.family == f) & (C.scenario == s) & (C.budget_kib == b)].set_index("estimator")
            xm = seeds(f, s, "mpl_dskll", int(c.loc["mpl_dskll"].approx), float(c.loc["mpl_dskll"].gain))
            out = dict(family=f, scenario=s, budget_kib=b)
            for rule, comp in [("test", min(["xiao", "hermite"], key=lambda e: c.loc[e].rmse)),
                               ("val", min(["xiao", "hermite"], key=lambda e: c.loc[e].val_rmse))]:
                xo = seeds(f, s, comp, int(c.loc[comp].approx), float(c.loc[comp].gain))
                d = 100 * (xm - xo)
                m, se = d.mean(), d.std(ddof=1) / math.sqrt(len(d))
                out.update({f"{rule}_comp": comp, f"{rule}_d": m, f"{rule}_se": se, f"{rule}_v": "W" if m < -2 * se else "L" if m > 2 * se else "T"})
            cmp.append(out)
K = pd.DataFrame(cmp)
K.to_csv(R + "budget_audit_rmpl_vs_best.csv", index=False)
for b in BUD:
    k = K[K.budget_kib == b]
    P(f"- {b} KiB: test-chosen comparator W/L/T = {(k.test_v == 'W').sum()}/{(k.test_v == 'L').sum()}/{(k.test_v == 'T').sum()}; "
      f"validation-chosen = {(k.val_v == 'W').sum()}/{(k.val_v == 'L').sum()}/{(k.val_v == 'T').sum()}; comparator differs in "
      f"{(k.test_comp != k.val_comp).sum()} cells.")
pat = {b: tuple(K[K.budget_kib == b].sort_values(["family", "scenario"]).test_v) for b in BUD}
P(f"- Cell-by-cell verdicts identical at 16, 64 and 256 KiB: {pat[16] == pat[64] == pat[256]}.")
P("\n64 KiB table (draft Table tab:ch6-e1) recomputed:\n")
P("| Scenario | Family | RMPL g | CM | HS | RMPL | RMPL − better (test-chosen) | RMPL − better (validation-chosen) |")
P("|---|---|---|---|---|---|---|---|")
for s in SC:
    for f in ["gaussian", "clayton", "gumbel", "t"]:
        c = C[(C.family == f) & (C.scenario == s) & (C.budget_kib == 64)].set_index("estimator")
        k = K[(K.family == f) & (K.scenario == s) & (K.budget_kib == 64)].iloc[0]
        P(f"| {s} | {FL[f]} | {glab(c.loc['mpl_dskll'].gain)} | {c.loc['xiao'].rmse:.2f} | {c.loc['hermite'].rmse:.2f} | {c.loc['mpl_dskll'].rmse:.2f} | "
          f"{k.test_d:+.2f} ± {k.test_se:.2f} {k.test_v} ({SHORT[k.test_comp]}) | {k.val_d:+.2f} ± {k.val_se:.2f} {k.val_v} ({SHORT[k.val_comp]}) |")

# ---------- 6. memory
P("\n## 6. Retained state of the selected configurations (bytes; peak over every tick, max over seeds)\n")
P("| Estimator | Budget | selected approx (cells) | peak bytes min–max | utilisation of cap min–median–max |")
P("|---|---|---|---|---|")
for e in E:
    for b in BUD:
        x = C[(C.estimator == e) & (C.budget_kib == b)]
        ap = x.approx.value_counts().sort_index()
        P(f"| {NAME[e]} | {b} KiB | " + ", ".join(f"{a} ({n})" for a, n in ap.items()) + f" | {x.peak_val.min():,}–{x.peak_val.max():,} | "
          f"{x.util.min():.1f}% – {x.util.median():.1f}% – {x.util.max():.1f}% |")

bd = open(R + "budget_mem_breakdown.csv").read().splitlines()
P("\nComponent breakdown (diag_mem_breakdown.cpp, 5000 Gaussian draws, cumulative mode; results/budget_mem_breakdown.csv):\n")
P("- **RMPL** (`RecursiveMPL::store_bytes`): object itself 712 B (copula parameter, Fisher-information / gain state, counters, config, "
  "and inline shells of the unused margin types) + both DataSketches KLL margins (counting allocator, actual heap) + 112 B of heap held "
  "by unused margin objects. k = 256: 712 + 2 × 2,848 + 112 = 6,520 B; k = 512: 712 + 2 × 5,108 + 112 = 11,040 B; k = 1024: 712 + 2 × 8,808 + 112 = 18,440 B. "
  "No shared or external state exists in the standalone runs. Identical for g = cum and g = 0.005.")
P("- **Counting Matrix**: object 680 B + B² + 2B count doubles + 2(B − 1) cut-points + 32 B of rank-window shells; its peak also includes the "
  "start-up calibration buffer (4,336 B at B = 30), which is released after the cut-points freeze, so the steady state is smaller than the reported peak (B = 30: 8,856 vs 13,192 B).")
P("- **Hermite Series**: object 520 B + A and W ((N + 1)² doubles each) + a_x, a_y, z + 64 B. W is a constant of the order and could be shared; counting it makes Hermite's figure conservative (×≈2).")
P("- Excluded everywhere: transient working memory inside update()/estimate() and allocator headers (malloc check within 0–7%, budget_mpl.md §24).")

# ---------- 7. timing
P("\n## 7. Processing time per observation at the selected configurations (µs; update + estimate, Gaussian drift, g = 0.02, one thread)\n")
P("| Estimator | Budget | mean tick time min–median–max | update share of tick (median) |")
P("|---|---|---|---|")
for e in E:
    for b in BUD:
        x = C[(C.estimator == e) & (C.budget_kib == b)]
        P(f"| {NAME[e]} | {b} KiB | {x.tick_ns.min() / 1e3:.2f} – {x.tick_ns.median() / 1e3:.2f} – {x.tick_ns.max() / 1e3:.2f} | "
          f"{(x.update_ns / x.tick_ns).median():.0%} |")
P("\nFull timing grid (µs, mean / p95; update only in brackets):\n")
P("| Estimator | approx | peak bytes | mean | p95 | update |")
P("|---|---|---|---|---|---|")
for _, r in tm[tm.estimator.isin(E + ["mpl_kll"])].iterrows():
    P(f"| {NAME.get(r.estimator, 'RMPL, own KLL')} | {r.approx} | {r.peak_bytes:,} | {r.mean_ns / 1e3:.2f} | {r.p95_ns / 1e3:.2f} | {r.update_mean_ns / 1e3:.2f} |")
old = pd.read_csv(R + "budget_timing.csv").set_index(["estimator", "approx"]).mean_ns
new = tm.set_index(["estimator", "approx"]).mean_ns
rr = (new / old).dropna()
P(f"\nRun-to-run reproducibility: budget_timing_final (2026-11-10) / budget_timing (2026-11-09, same code paths for these keys): ratio median "
  f"{rr.median():.2f}, range {rr.min():.2f}–{rr.max():.2f}.")

# ---------- 8. KLL implementations
P("\n## 8. Hand-written KLL vs Apache DataSketches KLL inside RMPL, matched k\n")
P("Bytes and times: timing grid above. RMSE: test RMSE at the gain validation picks for each implementation at that k "
  "(mean over the 12 family × scenario cells, and the largest absolute cell difference).\n")
P("| k | own bytes | DS bytes | own / DS | own µs | DS µs | DS / own time | mean RMSE own | mean RMSE DS | max abs cell diff |")
P("|---|---|---|---|---|---|---|---|---|---|")
kl = []
for k in sorted(tm[tm.estimator == "mpl_kll"].approx):
    ro, rd = tm[(tm.estimator == "mpl_kll") & (tm.approx == k)].iloc[0], tm[(tm.estimator == "mpl_dskll") & (tm.approx == k)].iloc[0]
    ro_, rd_ = [], []
    for f in FAMS:
        for s in SC:
            for e, lst in [("mpl_kll", ro_), ("mpl_dskll", rd_)]:
                v = val[(val.family == f) & (val.scenario == s) & (val.estimator == e) & (val.approx == k)]
                g = float(v.loc[v.rmse.idxmin()].gain)
                lst.append(100 * seeds(f, s, e, k, g).mean())
    ro_, rd_ = np.array(ro_), np.array(rd_)
    kl.append(dict(k=k, own_b=ro.peak_bytes, ds_b=rd.peak_bytes, ratio=ro.peak_bytes / rd.peak_bytes))
    P(f"| {k} | {ro.peak_bytes:,} | {rd.peak_bytes:,} | {ro.peak_bytes / rd.peak_bytes:.2f} | {ro.mean_ns / 1e3:.2f} | {rd.mean_ns / 1e3:.2f} | "
      f"{rd.mean_ns / ro.mean_ns:.2f} | {ro_.mean():.2f} | {rd_.mean():.2f} | {np.abs(ro_ - rd_).max():.2f} |")

# ---------- 9. family average for the figure
fa = []
for s in SC:
    for b in BUD:
        for e in E:
            per = []
            for f in FAMS:
                c = C[(C.family == f) & (C.scenario == s) & (C.budget_kib == b) & (C.estimator == e)].iloc[0]
                per.append(seeds(f, s, e, int(c.approx), float(c.gain)).values)
            a = 100 * np.mean(per, axis=0)      # per test seed: average over the four families
            fa.append(dict(scenario=s, budget_kib=b, estimator=e, mean=a.mean(), se=a.std(ddof=1) / math.sqrt(len(a)),
                           peak_mean=C[(C.scenario == s) & (C.budget_kib == b) & (C.estimator == e)].peak_val.mean()))
FA = pd.DataFrame(fa)
FA.to_csv(R + "budget_audit_familyavg.csv", index=False)
P("\n## 9. Family-averaged test RMSE for the figure (×10⁻²)\n")
P("Per test seed, average the four families' RMSEs at their own validated configurations; mean ± SE over the 100 seeds "
  "(seed s of every family is one replication, so the SE does not treat 400 family–seed combinations as independent).\n")
P("| Scenario | Budget | RMPL | Counting Matrix | Hermite Series |")
P("|---|---|---|---|---|")
for s in SC:
    for b in BUD:
        x = FA[(FA.scenario == s) & (FA.budget_kib == b)].set_index("estimator")
        P(f"| {s} | {b} KiB | " + " | ".join(f"{x.loc[e]['mean']:.3f} ± {x.loc[e].se:.3f}" for e in E) + " |")

open(R + "budget_audit_2026-11-15.md", "w").write("\n".join(M) + "\n")
print("\n".join(M))

# ---------- figure
sys.path.insert(0, ".")
import plot_constant_ess as ce  # noqa: E402
plt, core = ce.plt, ce.core
COLS = {"mpl_dskll": ce.COL["mle"], "xiao": ce.COL["xiao"], "hermite": ce.COL["hermite"]}
MK = {"mpl_dskll": "^", "xiao": "o", "hermite": "s"}
OFF = {"mpl_dskll": 0.0, "xiao": -0.06, "hermite": 0.06}
os.makedirs("../images/ch6/6.2", exist_ok=True)
with plt.rc_context(ce.RC):
    fig, axs = plt.subplots(1, 3, figsize=(core.W, 2.35))
    for ax, s, t in zip(axs, SC, ["(a) Constant", "(b) Gradual drift", "(c) Abrupt jump"]):
        for e in E:
            x = FA[(FA.scenario == s) & (FA.estimator == e)].sort_values("budget_kib")
            xp = np.log2(x.budget_kib.values) + OFF[e]
            ax.errorbar(xp, x["mean"], yerr=2 * x.se, color=COLS[e], marker=MK[e], ms=3.6, lw=1.1, capsize=0, elinewidth=0.8, label=NAME[e])
        ax.set_xticks(np.log2(list(BUD))); ax.set_xticklabels([f"{b}" for b in BUD]); ax.set_xlim(3.6, 8.4)
        ax.grid(True, axis="y", color=ce.GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_title(t, fontsize=8, pad=3, loc="left")
    axs[0].set_ylabel("Spearman RMSE ($\\times10^{-2}$)", fontsize=7.5)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.54, 1.0), handlelength=1.6, columnspacing=1.4)
    fig.text(0.54, 0.01, "Storage budget (KiB)", ha="center", fontsize=7.5)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.8, bottom=0.19, wspace=0.3)
    fig.savefig("../images/ch6/6.2/ch6_budget_accuracy_storage.png"); plt.close(fig)
print("wrote ../images/ch6/6.2/ch6_budget_accuracy_storage.png")
