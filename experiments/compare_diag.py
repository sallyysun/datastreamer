"""Paired comparison of staged diagnostic runs (results/thesis_bank_diag_<tag>[_fam].csv) against a reference tag.
Usage: python3 compare_diag.py REF TAG [TAG ...] [--comp xiao,hermite,mle]   notes/studies/ew_marginals.md."""
import sys, math
import pandas as pd

args = [a for a in sys.argv[1:] if not a.startswith("--")]
comps = None
for a in sys.argv[1:]:
    if a.startswith("--comp="): comps = a.split("=", 1)[1].split(",")
ref, tags = args[0], args[1:]
FAM = ["gaussian", "clayton", "gumbel", "t"]

def load(tag):
    fr = []
    for f in FAM:
        suf = "" if f == "gaussian" else "_" + f
        # TAG may be a diagnostic tag (diag_<tag>), a final tag (final500), or a path prefix (contains '/')
        p = (f"{tag}{suf}.csv" if "/" in tag else
             f"results/thesis_bank_{tag}{suf}.csv" if tag.startswith(("final", "matched")) else f"results/thesis_bank_diag_{tag}{suf}.csv")
        try: fr.append(pd.read_csv(p))
        except FileNotFoundError: pass
    return pd.concat(fr)

R = load(ref)
METHODS = ["fixed_39", "fixed_199", "fixed_1999", "bank", "bank_log", "uniform", "vff_slow", "vff_fast"]
for tag in tags:
    T = load(tag)
    m = R.merge(T, on=["copula", "scenario", "seed", "component", "window", "method"], suffixes=("_r", "_t"))
    if comps: m = m[m.component.isin(comps)]
    print(f"\n=== {tag} - {ref}: paired mean difference (RMSE x1000; excess log loss x1000), seeds 1-30 x 4 families ===")
    for metric, sc in (("rmse", 1000), ("excess_x1000", 1)):
        w = m[m.window == "whole"]
        rows = []
        for (c, s), g in w.groupby(["component", "scenario"]):
            row = {"comp": c, "scen": s}
            for meth in METHODS:
                h = g[g.method == meth]
                d = sc * (h[f"{metric}_t"] - h[f"{metric}_r"])
                d = d.dropna()
                row[meth] = f"{d.mean():+6.1f}±{d.std(ddof=1)/math.sqrt(len(d)):.1f}" if len(d) > 1 else ""
            rows.append(row)
        print(f"-- {metric}")
        print(pd.DataFrame(rows).to_string(index=False))
    j = m[(m.scenario == "jump") & m.method.isin(["bank", "bank_log", "fixed_199"])]
    rows = []
    for (c, meth), g in j.groupby(["component", "method"]):
        row = {"comp": c, "method": meth}
        for win in ["pre", "rec_0_100", "rec_100_500", "rec_500_1500"]:
            h = g[g.window == win]; d = 1000 * (h.rmse_t - h.rmse_r)
            row[win] = f"{d.mean():+6.1f}±{d.std(ddof=1)/math.sqrt(len(d)):.1f}"
        rows.append(row)
    print("-- jump recovery windows, RMSE x1000")
    print(pd.DataFrame(rows).to_string(index=False))
