"""Tables for the family bank (probe_family_bank, probe_bank_regret ... family). Paired mean +- 2 SE over
seeds; W/L/T = family bank lower / higher / within 2 SE. Note: notes/studies/thesis_bank_family.md.
Usage (from experiments/): python3 family_bank_table.py [first_seed=42]"""
import sys
import pandas as pd

first = sys.argv[1] if len(sys.argv) > 1 else "42"
fams = ["gaussian", "clayton", "gumbel", "t"]
shapes = ["gauss2t", "clayton2gumbel", "gauss2clayton"]
sfx = lambda c: "" if c == "gaussian" else "_" + c
d = pd.concat(pd.read_csv(f"results/family_bank_{first}{sfx(c)}.csv") for c in fams + shapes)
banks = ["xiao_bank_log", "hermite_bank_log", "mle_bank_log", "xiao_bank_own", "hermite_bank_own", "mle_bank_own"]
short = {b: b.replace("_bank_", "_") for b in banks}


def ci(x):
    x = x.dropna()
    return f"{x.mean():+.1f} ± {2 * x.std(ddof=1) / len(x) ** .5:.1f}"


def wlt(x):
    h = 2 * x.std(ddof=1) / len(x) ** .5
    return "W" if x.mean() < -h else "L" if x.mean() > h else "T"


def table(metric, win, scale, title, refs):
    w = d[(d.window == win) & (d.metric == metric)].pivot_table(index=["copula", "scenario", "seed"], columns="method",
                                                               values="value") * scale
    print(f"\n## {title}, window {win}: family_bank, and family_bank minus ...\n")
    print("| copula | scenario | family | " + " | ".join("− " + short.get(r, r) for r in refs) + " |")
    print("| --- | --- | --- | " + " | ".join("---" for _ in refs) + " |")
    count = {r: {"W": 0, "L": 0, "T": 0} for r in refs}
    for (c, s), g in w.groupby(level=[0, 1]):
        cells = []
        for r in refs:
            if r == "best_arm":
                arms = [x for x in g.columns if x.startswith("arm_")]
                best = g[arms].mean().idxmin()
                x = g.family_bank - g[best]
                cells.append(f"{ci(x)} ({best[4:]})")
            else:
                x = g.family_bank - g[r]
                cells.append(ci(x))
            count[r][wlt(x)] += 1
        print(f"| {c} | {s} | {g.family_bank.mean():.1f} | " + " | ".join(cells) + " |")
    print("\nW/L/T over cells: " + "; ".join(f"{short.get(r, r)} {v['W']}/{v['L']}/{v['T']}" for r, v in count.items()))


table("rmse", "whole", 1000, "rho_S RMSE x1e-3", banks + ["best_arm"])
for win in ["rec_0_100", "rec_100_500"]:
    table("rmse", win, 1000, "rho_S RMSE x1e-3", ["xiao_bank_log", "mle_bank_log", "hermite_bank_own"])
table("excess_x1000", "whole", 1, "Excess log loss x1e-3 nats/tick at the true PIT", banks)
for win in ["rec_0_100", "rec_100_500"]:
    table("excess_x1000", win, 1, "Excess log loss x1e-3", ["mle_bank_log", "hermite_bank_own", "mle_bank_own"])
for tail in ["chiL", "chiU"]:
    for win in ["whole", "rec_0_100", "rec_100_500", "rec_500_1500"]:
        table(f"{tail}_rmse", win, 1000, f"{tail}(0.05) RMSE x1e-3 vs ratio estimator", ["ratio_99", "ratio_399", "ratio_1999"])

print("\n## Shape switches: ticks after the switch for the new shape's summed weight to reach 50% / 90%\n")
print("| switch | new shape | weight at t=2499 | t50 median [IQR] | t90 median [IQR] | never 50% | never 90% |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for c in shapes:
    s = pd.read_csv(f"results/family_bank_switch_{first}_{c}.csv")
    q = lambda x: f"{x[x >= 0].median():.0f} [{x[x >= 0].quantile(.25):.0f}, {x[x >= 0].quantile(.75):.0f}]"
    print(f"| {c} | {s.new_shape[0]} | {s.pre_weight_new.mean():.3f} | {q(s.t50)} | {q(s.t90)} | "
          f"{(s.t50 < 0).sum()}/{len(s)} | {(s.t90 < 0).sum()}/{len(s)} |")

print("\n## Seed-mean shape weights and memory, averaged over ticks (pre = [500, 2500), post = [2500, 5000))\n")
print("| copula | scenario | span | " + " | ".join(["gaussian", "t", "clayton", "clayton90", "gumbel", "gumbel90"]) + " | mem |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for c in fams + shapes:
    p = pd.read_csv(f"results/family_bank_path_{first}{sfx(c)}.csv")
    for sc, g in p.groupby("scenario"):
        for span, (a, b) in [("pre", (500, 2500)), ("post", (2500, 5000))]:
            h = g[(g.t >= a) & (g.t < b)].groupby("series").value.mean()
            print(f"| {c} | {sc} | {span} | " + " | ".join(f"{h['w_' + k]:.2f}" for k in
                  ["gaussian", "t", "clayton", "clayton90", "gumbel", "gumbel90"]) + f" | {h['mem']:.0f} |")

print("\n## Regret of the family bank's mixture log loss against B_{T,m}(delta) (bound at eta = 1)\n")
r = pd.concat(pd.read_csv(f"results/bank_regret_family_{first}{sfx(c)}.csv") for c in fams)
print("| bank | scenario | m | mean | 95% | max | bound | max/bound | viol |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for (b, sc, m), g in r.groupby(["bank", "scenario", "m"]):
    if m not in (0, 1, 2, 5, 20, 50): continue
    print(f"| {b} | {sc} | {m} | {g.regret.mean():.2f} | {g.regret.quantile(.95):.2f} | {g.regret.max():.2f} | "
          f"{g.bound.mean():.1f} | {(g.regret / g.bound).max():.2f} | {(g.regret > g.bound).sum()}/{len(g)} |")
