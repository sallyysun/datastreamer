"""Held-out check of the tuned ForgettingBank (probe_bank_tune on seeds 42-101) against every single
fixed memory and both VFF rates (from probe_thesis_bank, same seeds and data), drift and jump only.
Paired differences x1e-3, mean +- 2 SE over seeds. Negative = bank better.
Usage (from experiments/): python3 tune_bank_eval.py [rmse|excess_x1000] [window]"""
import sys
import pandas as pd

metric = sys.argv[1] if len(sys.argv) > 1 else "rmse"
window = sys.argv[2] if len(sys.argv) > 2 else "whole"
sc = 1000 if metric == "rmse" else 1
TUNED = {"sq": (1.0, 1e-4), "log": (1.0, 3e-4)}
OLD = {"sq": (0.5, 1e-4), "log": (1.0, 4e-4)}
sfx = ["", "_clayton", "_gumbel", "_t"]
t = pd.concat(pd.read_csv(f"results/bank_tune_42{s}.csv") for s in sfx)
t = t[t.window == window]
parts = [t[t.method.str.startswith("fixed")]]
for lab, grid in [("tuned", TUNED), ("old", OLD)]:
    for m, (e, dl) in grid.items():
        parts.append(t[(t.method == m) & (t.eta == e) & ((t.delta - dl).abs() < 1e-12)].assign(method=f"{m}_{lab}"))
v = pd.concat(pd.read_csv(f"results/thesis_bank{s}.csv") for s in sfx)
v = v[(v.window == window) & v.scenario.isin(["drift", "jump"]) & v.method.isin(["vff_slow", "vff_fast"])]
d = pd.concat(parts + [v])
w = d.pivot_table(index=["copula", "scenario", "component", "seed"], columns="method", values=metric)
fixed = [c for c in w.columns if c.startswith("fixed_")]

def ci(x):
    return x.mean() * sc, 2 * x.std(ddof=1) / len(x) ** .5 * sc

pd.set_option("display.width", 250, "display.max_rows", 200)
rows, tally = [], {}
for key, g in w.groupby(level=[0, 1, 2]):
    r = dict(zip(["copula", "scenario", "component"], key))
    best = g[fixed].mean().idxmin()
    bv = min(["vff_slow", "vff_fast"], key=lambda c: g[c].mean())
    r["best_fixed"] = best
    for b in ["sq_tuned", "log_tuned"]:
        for ref, lab in [(best, "hind"), (bv, "vff")] + [(f, f) for f in fixed]:
            mu, h = ci(g[b] - g[ref])
            outcome = "win" if mu + h < 0 else "loss" if mu - h > 0 else "tie"
            tally.setdefault((b, lab), {"win": 0, "loss": 0, "tie": 0})[outcome] += 1
            if lab in ("hind", "vff"):
                r[f"{b}-{lab}"] = f"{mu:+.1f}±{h:.1f}"
        mu, h = ci(g[b] - g[b.replace("tuned", "old")])
        r[f"{b}-old"] = f"{mu:+.1f}±{h:.1f}"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
print("\nwin/loss/tie over the 24 drift+jump cells (+-2SE):")
tl = pd.DataFrame(tally).T
print(tl.to_string())

# One memory committed to in advance across both scenarios: pick the fixed M with the lowest mean
# over drift and jump for each (family, component); compare the banks with it, pooled.
print("\nbank minus the one fixed M best on drift+jump pooled, per family x component (cells and pooled):")
out = []
for (cop, comp), g in w.groupby(level=[0, 2]):
    m = g[fixed].mean().idxmin()
    for b in ["sq_tuned", "log_tuned"]:
        res = {"copula": cop, "component": comp, "M": m, "bank": b}
        for scn, gg in g.groupby(level=1):
            mu, h = ci(gg[b] - gg[m]); res[scn] = f"{mu:+.1f}±{h:.1f}"
        out.append(res)
print(pd.DataFrame(out).to_string(index=False))
