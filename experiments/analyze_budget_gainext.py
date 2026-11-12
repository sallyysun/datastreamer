"""E1 with the extended gain grid (budget_mpl.md §25). Candidates = the final matched sweep (g in {0.005, ..., 0.10}) plus the
extension run (TAG gainext: g in {0 = cumulative, 0.0005, 0.001, 0.002}, and 0.005 again as a reproduction check), constant and
drift cells; jump cells keep the matched sweep (their validation picks were interior, g = 0.01). Per family x scenario x budget,
each estimator's (approximation, gain) is chosen on validation seeds 1000-1039 within the memory cap; DataSketches-MPL is compared
on test seeds 42-141 with Xiao, Hermite and the better of the two (paired, W/L/T at 2 SE).
Usage (from experiments/): python3 analyze_budget_gainext.py"""
import math
import numpy as np
import pandas as pd

BUD = {"16KiB": 16384, "64KiB": 65536, "256KiB": 262144}
E = ["xiao", "hermite", "mpl_dskll"]
NAME = {"xiao": "Xiao", "hermite": "Hermite", "mpl_dskll": "DS-MPL"}
FAMS = ["gaussian", "clayton", "gumbel", "t"]

old = pd.read_csv("results/budget_sweep_matched.csv")
olds = pd.read_csv("results/budget_seeds_matched.csv")
new = pd.concat([pd.read_csv(f"results/budget_sweep_gainext_{f}.csv") for f in FAMS])
news = pd.concat([pd.read_csv(f"results/budget_seeds_gainext_{f}.csv") for f in FAMS])

# Reproduction: the g = 0.005 candidates of the extension must equal the matched sweep exactly.
k = ["family", "scenario", "estimator", "approx", "gain", "seed"]
m = news[np.isclose(news.gain, 0.005)].merge(olds, on=k, suffixes=("_n", "_o"))
mv = new[np.isclose(new.gain, 0.005)].merge(old, on=["family", "scenario", "split", "estimator", "approx", "gain"], suffixes=("_n", "_o"))
print(f"Reproduction of g = 0.005: per-seed test rows {len(m)}, max |diff| {np.abs(m.rmse_n - m.rmse_o).max():.1e}; "
      f"summary rows {len(mv)}, max |diff| rmse {np.abs(mv.rmse_n - mv.rmse_o).max():.1e}, peak_bytes equal {bool((mv.peak_bytes_n == mv.peak_bytes_o).all())}")

ext = new[~np.isclose(new.gain, 0.005)]
exts = news[~np.isclose(news.gain, 0.005)]
sweep = pd.concat([old, ext], ignore_index=True); seeds = pd.concat([olds, exts], ignore_index=True)
val = sweep[sweep.split == "val"]


def pick(f, s, e, cap, grid):
    v = val[(val.family == f) & (val.scenario == s) & (val.estimator == e) & (val.peak_bytes <= cap)]
    if grid == "original": v = v[v.gain >= 0.005 - 1e-12]
    r = v.loc[v.rmse.idxmin()]
    return int(r.approx), float(r.gain), int(r.peak_bytes)


def test_rmse(f, s, e, a, g):
    x = seeds[(seeds.family == f) & (seeds.scenario == s) & (seeds.estimator == e) & (seeds.approx == a) & (np.isclose(seeds.gain, g))]
    return x.set_index("seed").rmse


def glab(g): return "cum" if g == 0 else f"{g:g}"


summary = {}
for grid in ["original", "extended"]:
    print(f"\n######## {grid} gain grid" + (" (g >= 0.005)" if grid == "original" else " (adds cumulative, 0.0005, 0.001, 0.002 for constant and drift)"))
    for b, cap in BUD.items():
        rows = []; wlt = {"xiao": [0, 0, 0], "hermite": [0, 0, 0], "best": [0, 0, 0]}; best_cnt = {e: 0 for e in E}
        for f in FAMS:
            for s in ["constant", "drift", "jump"]:
                g = grid if s != "jump" else "original"
                pk = {e: pick(f, s, e, cap, g) for e in E}
                r = {e: test_rmse(f, s, e, pk[e][0], pk[e][1]) for e in E}
                mu = {e: 100 * r[e].mean() for e in E}
                best_cnt[min(E, key=lambda e: mu[e])] += 1
                oth = min(["xiao", "hermite"], key=lambda e: mu[e])
                row = {"family": f, "scenario": s}
                for e in E: row[NAME[e]] = f"{mu[e]:.2f} [{pk[e][0]}, g={glab(pk[e][1])}]"
                for key, comp in [("xiao", "xiao"), ("hermite", "hermite"), ("best", oth)]:
                    d = 100 * (r["mpl_dskll"] - r[comp]).dropna(); mm, se = d.mean(), d.std(ddof=1) / math.sqrt(len(d))
                    v = 0 if mm < -2 * se else 1 if mm > 2 * se else 2
                    wlt[key][v] += 1
                    if key == "best":
                        row["DS-MPL − better other"] = f"{mm:+.2f} ± {se:.2f} ({NAME[comp]}) {'WLT'[v]}"
                        summary[(grid, b, f, s)] = (mm, se, 'WLT'[v], comp)
                rows.append(row)
        print(f"\n== {b}: test RMSE x1e-2 (100 seeds) [approx, gain chosen on validation]; DS-MPL minus the better of Xiao/Hermite (mean ± SE) ==")
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"lowest test RMSE: " + ", ".join(f"{NAME[e]} {best_cnt[e]}/12" for e in E)
              + f" | DS-MPL W/L/T vs Xiao {wlt['xiao']}, vs Hermite {wlt['hermite']}, vs the better of the two {wlt['best']}")

print("\n######## Changes (extended − original grid), DS-MPL minus the better of Xiao/Hermite, by cell")
for b in BUD:
    ch = []
    for f in FAMS:
        for s in ["constant", "drift"]:
            o, n = summary[("original", b, f, s)], summary[("extended", b, f, s)]
            if o[2] != n[2]: ch.append(f"{f}/{s}: {o[2]} ({o[0]:+.2f}) -> {n[2]} ({n[0]:+.2f} vs {NAME[n[3]]})")
    print(f"  {b}: " + ("; ".join(ch) if ch else "no verdict changes"))

print("\n######## Accuracy against memory (extended grid; validation-tuned test RMSE x1e-2 at each budget)")
for f in FAMS:
    for s in ["constant", "drift"]:
        line = f"  {f:8s} {s:8s}"
        for e in E:
            vals = []
            for b, cap in BUD.items():
                a, g, _ = pick(f, s, e, cap, "extended")
                vals.append(100 * test_rmse(f, s, e, a, g).mean())
            line += f" | {NAME[e]:7s} " + " / ".join(f"{v:.2f}" for v in vals)
        print(line + "   (16 / 64 / 256 KiB)")

print("\n######## Diagnostic: MPL margins at the extended grid (validation-tuned within 256 KiB; exact PIT is not budget-feasible)")
for f in FAMS:
    for s in ["constant", "drift"]:
        out = []
        for e in ["mpl_dskll", "mpl_kll", "mpl_exact", "mpl_ew"]:
            v = val[(val.family == f) & (val.scenario == s) & (val.estimator == e)]
            r = v.loc[v.rmse.idxmin()]
            out.append(f"{e} {100 * test_rmse(f, s, e, int(r.approx), float(r.gain)).mean():.2f} (g={glab(r.gain)})")
        print(f"  {f:8s} {s:8s} | " + " | ".join(out))
