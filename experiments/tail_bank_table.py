"""Tables for probe_tail_bank: chi(q) RMSE and bias (x1e-3, mean +- 2 SE over seeds) of the family bank, the
EW corner counter tail_chi and the bank's fixed single-shape arms, against the exact finite-q chi of the true
regime. Note: notes/studies/thesis_bank_tails.md.
Usage (from experiments/): python3 tail_bank_table.py [first_seed=42]"""
import sys
import pandas as pd

first = sys.argv[1] if len(sys.argv) > 1 else "42"
regimes = ["gauss2clayton", "clayton2gumbel", "gauss2t", "mix", "gaussian"]
d = pd.concat(pd.read_csv(f"results/tail_bank_{r}_{first}.csv") for r in regimes)
main = ["family_bank", "tail_chi_399", "tail_chi_1999", "tail_chi_4999", "tail_chi_1999_known"]
wins = ["whole", "pre", "post_0_500", "post_500_2000", "post_2000_5000"]


def ms(x, col):
    x = x[col] * 1000
    return f"{x.mean():.0f} ± {2 * x.std(ddof=1) / len(x) ** .5:.0f}"


for col in ["rmse", "bias"]:
    print(f"\n## chi(q) {col} x1e-3 (mean ± 2 SE over seeds), by window\n")
    print("| regime | q | tail | method | " + " | ".join(wins) + " |")
    print("| --- | --- | --- | --- | " + " | ".join("---" for _ in wins) + " |")
    for (r, q, tl), g in d.groupby(["regime", "q", "tail"], sort=False):
        arms = g[(g.window == "whole") & g.method.str.startswith("arm_")].groupby("method").rmse.mean()
        best = arms.idxmin()
        for m in main + [best]:
            h = g[g.method == m]
            label = m if m != best else f"best arm ({m[4:]})"
            print(f"| {r} | {q:.2f} | {tl} | {label} | " + " | ".join(ms(h[h.window == w], col) for w in wins) + " |")

print("\n## Paired whole-run RMSE difference, family_bank minus each counter (x1e-3; W = bank lower by > 2 SE)\n")
print("| regime | q | tail | − tail_chi_399 | − tail_chi_1999 | − tail_chi_4999 | − best arm |")
print("| --- | --- | --- | --- | --- | --- | --- |")
cnt = {k: [0, 0, 0] for k in ["tail_chi_399", "tail_chi_1999", "tail_chi_4999", "best arm"]}
for (r, q, tl), g in d[d.window == "whole"].groupby(["regime", "q", "tail"], sort=False):
    w = g.pivot_table(index="seed", columns="method", values="rmse") * 1000
    best = w[[c for c in w.columns if c.startswith("arm_")]].mean().idxmin()
    cells = []
    for k, c in [("tail_chi_399",) * 2, ("tail_chi_1999",) * 2, ("tail_chi_4999",) * 2, ("best arm", best)]:
        x = w.family_bank - w[c]
        h = 2 * x.std(ddof=1) / len(x) ** .5
        cnt[k][0 if x.mean() < -h else 1 if x.mean() > h else 2] += 1
        cells.append(f"{x.mean():+.0f} ± {h:.0f}" + (f" ({c[4:]})" if k == "best arm" else ""))
    print(f"| {r} | {q:.2f} | {tl} | " + " | ".join(cells) + " |")
print("\nW/L/T: " + "; ".join(f"{k} {v[0]}/{v[1]}/{v[2]}" for k, v in cnt.items()))

print("\n## Ticks after the switch for the seed-mean chi to cross halfway from the old target to the new\n")
c = pd.concat(pd.read_csv(f"results/tail_bank_cross_{r}_{first}.csv") for r in regimes if r != "gaussian")
c = c[(c.new - c.old).abs() > 1e-3]
print(c.pivot_table(index=["regime", "q", "tail", "old", "new"], columns="method", values="ticks_to_half", sort=False)
      [main].to_string())

print("\n## Seed-mean shape weights of the family bank (pre = [1000, 5000), post_late = [7000, 10000))\n")
for r in regimes:
    p = pd.read_csv(f"results/tail_bank_path_{r}_{first}.csv")
    p = p[p.series.str.startswith("w_")]
    pre = p[(p.t >= 1000) & (p.t < 5000)].groupby("series").value.mean()
    post = p[p.t >= 7000].groupby("series").value.mean()
    print(f"- {r}: pre " + ", ".join(f"{k[2:]} {v:.2f}" for k, v in pre.items() if v >= 0.02)
          + "; post " + ", ".join(f"{k[2:]} {v:.2f}" for k, v in post.items() if v >= 0.02))

g = d[(d.regime == "gaussian") & (d.window == "whole") & (d.method.str.startswith("arm_gaussian"))]
print("\n## Sanity: Gaussian arms' mean chi on the pure Gaussian path (lambda = 0)\n")
print(g.groupby(["q", "tail", "method"])["mean"].mean().round(4).to_string())
