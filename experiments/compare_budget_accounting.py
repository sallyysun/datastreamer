"""Matched-budget winners before and after the 2026-11-09 memory-accounting fix (budget_mpl.md §24).
Usage: python3 compare_budget_accounting.py OLD.csv [OLD_t.csv] NEW.csv  -> stdout."""
import sys
import pandas as pd

*old_paths, new_path = sys.argv[1:]
old = pd.concat([pd.read_csv(p) for p in old_paths])
old = old.drop_duplicates(["family", "scenario", "split", "estimator", "approx", "gain"], keep="last")
new = pd.read_csv(new_path)
BUD = {"16KiB": 16 * 1024, "64KiB": 64 * 1024, "256KiB": 256 * 1024}
ESTS = ["xiao", "hermite", "mpl_kll", "mpl_dskll"]

def tuned(df):
    val = df[df.split == "val"].set_index(["family", "scenario", "estimator", "approx", "gain"])
    test = df[df.split == "test"].set_index(["family", "scenario", "estimator", "approx", "gain"])
    rows = []
    for (f, s), _ in df.groupby(["family", "scenario"]):
        for b, cap in BUD.items():
            for e in ESTS:
                v = val.loc[(f, s, e)].reset_index()
                v = v[v.peak_bytes <= cap]
                if v.empty:
                    rows.append(dict(family=f, scenario=s, budget=b, est=e, rmse=float("nan"), cfg="none"))
                    continue
                k = v.loc[v.rmse.idxmin()]
                t = test.loc[(f, s, e, k.approx, k.gain)]
                rows.append(dict(family=f, scenario=s, budget=b, est=e, rmse=100 * t.rmse, se=100 * t.se_rmse,
                                 kib=t.peak_bytes / 1024, cfg=f"{int(k.approx)},{k.gain:g}"))
    return pd.DataFrame(rows)

o, n = tuned(old), tuned(new)
m = o.merge(n, on=["family", "scenario", "budget", "est"], suffixes=("_old", "_new"))
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
for b in BUD:
    print(f"\n== {b}: test RMSE x1e-2 [knob,gain] old -> new ==")
    t = m[m.budget == b]
    t = t.assign(cell=t.apply(lambda r: f"{r.rmse_old:.2f}[{r.cfg_old}] -> {r.rmse_new:.2f}[{r.cfg_new}]", axis=1))
    print(t.pivot_table(index=["family", "scenario"], columns="est", values="cell", aggfunc="first"))
    for tag, lab in (("old", "before"), ("new", "after")):
        for mpl in ("mpl_dskll", "mpl_kll"):
            wins = 0
            for (f, s), g in t.groupby(["family", "scenario"]):
                r = g.set_index("est")[f"rmse_{tag}"]
                cand = r[["xiao", "hermite", mpl]].dropna()
                wins += cand.idxmin() == mpl
            print(f"  {lab}: {mpl} lowest of (xiao, hermite, {mpl}) in {wins}/12 cells")
