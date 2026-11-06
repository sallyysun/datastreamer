# Tables for the matched-budget study from results/budget_sweep.csv. See notes/studies/budget_mpl.md.
import warnings; warnings.filterwarnings("ignore")
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/budget_sweep.csv")
BUDGETS = {"16KiB": 16 * 1024, "64KiB": 64 * 1024, "256KiB": 256 * 1024}
KEY = ["estimator", "approx", "gain"]
val = df[df.split == "val"].set_index(["family", "scenario"] + KEY)
test = df[df.split == "test"].set_index(["family", "scenario"] + KEY)

def pick(fam, sc, est, cap, metric="rmse", gain=None):
    v = val.loc[(fam, sc, est)].reset_index()
    v = v[v.peak_bytes <= cap]
    if gain is not None:
        v = v[v.gain == gain]
    if v.empty:
        return None
    b = v.loc[v[metric].idxmin()]
    return test.loc[(fam, sc, est, b.approx, b.gain)], b.approx, b.gain

rows = []
for (fam, sc), _ in df.groupby(["family", "scenario"]):
    for bname, cap in BUDGETS.items():
        for est in ["xiao", "hermite", "mpl_kll", "mpl_dskll"]:
            r = pick(fam, sc, est, cap)
            if r is None:
                continue
            t, a, g = r
            rows.append(dict(family=fam, scenario=sc, budget=bname, estimator=est, approx=int(a), gain=g,
                             peak_kib=t.peak_bytes / 1024, rmse=t.rmse, se=t.se_rmse, post=t.post_rmse,
                             delay=t.recovery_delay))
tuned = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 40); pd.set_option("display.max_rows", 500)
print("== Validation-tuned, test RMSE (x1e-2) ==")
w = tuned.assign(cell=lambda d: d.apply(lambda r: f"{100*r.rmse:.2f}±{100*r.se:.2f} [{r.approx},{r.gain:g}]", axis=1))
print(w.pivot_table(index=["family", "scenario", "budget"], columns="estimator", values="cell", aggfunc="first"))
print("\n== jump: post-change RMSE (x1e-2) and recovery delay (ticks), tuned on post-change RMSE ==")
rows = []
for fam in df.family.unique():
    for bname, cap in BUDGETS.items():
        for est in ["xiao", "hermite", "mpl_kll", "mpl_dskll"]:
            r = pick(fam, "jump", est, cap, metric="post_rmse")
            if r is None:
                continue
            t, a, g = r
            rows.append(dict(family=fam, budget=bname, estimator=est,
                             cell=f"{100*t.post_rmse:.2f}±{100*t.se_post:.2f} d={t.recovery_delay:.0f} [{int(a)},{g:g}]"))
print(pd.DataFrame(rows).pivot_table(index=["family", "budget"], columns="estimator", values="cell", aggfunc="first"))
print("\n== Common gain, Gaussian: best approx within 64KiB at each gain, test RMSE (x1e-2) ==")
rows = []
for sc in ["constant", "drift", "jump"]:
    for g in sorted(df.gain.unique()):
        for est in ["xiao", "hermite", "mpl_kll", "mpl_dskll"]:
            t, a, _ = pick("gaussian", sc, est, BUDGETS["64KiB"], gain=g)
            rows.append(dict(scenario=sc, gain=g, estimator=est, cell=f"{100*t.rmse:.2f} [{int(a)}]"))
print(pd.DataFrame(rows).pivot_table(index=["scenario", "gain"], columns="estimator", values="cell", aggfunc="first"))
print("\n== Marginal ablation (MPL, test RMSE x1e-2 at the val-best gain for each variant) ==")
rows = []
for (fam, sc), _ in df.groupby(["family", "scenario"]):
    r = {"family": fam, "scenario": sc}
    for est, a in [("mpl_exact", 0), ("mpl_window", 500), ("mpl_kll", 64), ("mpl_kll", 256),
                   ("mpl_dskll", 64), ("mpl_dskll", 256), ("mpl_fit_kll", 256)]:
        v = val.loc[(fam, sc, est, a)].reset_index()
        g = v.loc[v.rmse.idxmin()].gain
        t = test.loc[(fam, sc, est, a, g)]
        r[f"{est.replace('mpl_', '')}{'' if est=='mpl_exact' else a}"] = f"{100*t.rmse:.2f}±{100*t.se_rmse:.2f}"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
