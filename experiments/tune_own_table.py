"""Own-density bank (ForgettingBank Loss::Own) from probe_bank_tune output.
1. Tuning (seeds 1-30, drift + jump): regret against the cell's best fixed memory in excess log
   loss of the bank's mixture of own densities (excess_own_x1000), grid over eta, delta.
2. Held-out (seeds 42-101): own bank at the chosen (eta, delta) against every fixed memory and the
   tuned MoM log bank, on RMSE and on excess_own. Paired, +-2 SE; W/L/T counts over cells.
3. Shape switches (gauss2t, clayton2gumbel; seeds 42-101): excess_own and weight-averaged memory per
   window, fixed arms vs the two banks.
Usage (from experiments/): python3 tune_own_table.py [eta delta]"""
import sys
import pandas as pd

pd.set_option("display.width", 250, "display.max_rows", 300)
FAMS = ["", "_clayton", "_gumbel", "_t"]
cell = ["copula", "scenario", "component"]

def load(first, sfx):
    return pd.concat(pd.read_csv(f"results/bank_tune_{first}{s}.csv") for s in sfx)

# 1. tuning
d = load(1, FAMS)
d = d[d.window == "whole"]
fx = d[d.method.str.startswith("fixed")]
for metric in ["excess_own_x1000", "rmse"]:
    sc = 1000 if metric == "rmse" else 1
    best = fx.groupby(cell + ["method"])[metric].mean().groupby(level=[0, 1, 2]).min().rename("best")
    b = d[d.method == "own"].groupby(cell + ["eta", "delta"])[metric].mean().reset_index().merge(best.reset_index(), on=cell)
    b["reg"] = (b[metric] - b.best) * sc
    print(f"\n== own bank, tuning seeds 1-30: mean regret vs best fixed memory in {metric} (rows eta, cols delta)")
    print(b.pivot_table(index="eta", columns="delta", values="reg", aggfunc="mean").round(2).to_string())
    for by in ["component", "scenario"]:
        p = b.groupby([by, "eta", "delta"]).reg.mean().reset_index()
        print(f"best per {by}:\n" + p.loc[p.groupby(by).reg.idxmin()].round(3).to_string(index=False))

eta = float(sys.argv[1]) if len(sys.argv) > 2 else 1.0
delta = float(sys.argv[2]) if len(sys.argv) > 2 else 3e-4
LOG = (1.0, 3e-4)

def pick(t):
    own = t[(t.method == "own") & (t.eta == eta) & ((t.delta - delta).abs() < 1e-12)].assign(method="own_bank")
    log = t[(t.method == "log") & (t.eta == LOG[0]) & ((t.delta - LOG[1]).abs() < 1e-12)].assign(method="log_bank")
    return pd.concat([t[t.method.str.startswith("fixed")], own, log])

# 2. held out
t = load(42, FAMS)
t = pick(t[t.window == "whole"])
for metric in ["rmse", "excess_own_x1000"]:
    sc = 1000 if metric == "rmse" else 1
    w = t.pivot_table(index=cell + ["seed"], columns="method", values=metric)
    rows = []
    for c, g in w.groupby(level=[0, 1, 2]):
        best = g[[x for x in g.columns if x.startswith("fixed")]].mean().idxmin()
        r = {"copula": c[0], "scenario": c[1], "component": c[2], "own": g.own_bank.mean() * sc,
             "log": g.log_bank.mean() * sc, "best_fixed": f"{g[best].mean() * sc:.1f} ({best[6:]})"}
        for o in [x for x in g.columns if x != "own_bank"]:
            diff = (g.own_bank - g[o]) * sc
            m, se = diff.mean(), 2 * diff.std(ddof=1) / len(diff) ** .5
            r[o] = "W" if m + se < 0 else "L" if m - se > 0 else "T"
        r["own-log"] = f"{((g.own_bank - g.log_bank) * sc).mean():+.1f}"
        rows.append(r)
    R = pd.DataFrame(rows)
    print(f"\n== held-out 42-101, own bank (eta={eta}, delta={delta}) on {metric}; W = own bank better")
    print(R.to_string(index=False))
    cols = [c for c in R.columns if c.startswith("fixed") or c == "log_bank"]
    print("W/L/T over cells:", {c: "/".join(str((R[c] == k).sum()) for k in "WLT") for c in cols})

# 3. shape
try:
    s = load(42, ["_gauss2t", "_clayton2gumbel"])
except FileNotFoundError:
    sys.exit()
s = pick(s)
print("\n== shape switches (rho_S fixed, family changes at 2500), seeds 42-101")
for metric in ["excess_own_x1000", "mem", "rmse"]:
    p = s.groupby(["copula", "component", "window", "method"])[metric].mean().unstack("method")
    print(f"\n-- {metric}")
    print(p.round(1 if metric != "rmse" else 4).to_string())
