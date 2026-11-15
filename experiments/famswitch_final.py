"""Section 6.5 final numbers from saved probe_family_bank outputs (density3b switches, density3 constant paths; EW margins
N_eff,m = 500; seeds 42-101 in four 15-seed chunks). Delta L = L(Gaussian-only MFF, gauss_bank) - L(family-augmented MFF,
family_bank), excess log loss x1e-3 nats/obs at the true PIT; positive favours family augmentation. Windows are per-tick
means, so ticks 2600-4999 = (2500 post - 100 rec_0_100) / 2400 per seed (exact).
Usage (from experiments/): python3 famswitch_final.py -> results/famswitch_final.csv, results/famswitch_summary_2026-11-15.md"""
import math
import numpy as np
import pandas as pd

R = "results/"
CH = (42, 57, 72, 87)
REG = [("gauss2t", "Gaussian → t4", "gaussian", "t"), ("gauss2clayton", "Gaussian → Clayton", "gaussian", "clayton"),
       ("clayton2gumbel", "Clayton → Gumbel", "clayton", "gumbel")]
ms = lambda x: (x.mean(), 2 * x.std(ddof=1) / math.sqrt(len(x)), len(x))
rows = []; M = []
M.append("# Section 6.5 — family-switching results (verified from saved outputs)\n")
M.append("## Family identification (density3b; 60 replications)\n")
M.append("| Switch | t50 median [IQR] | reached 50% | t90 median [IQR] | reached 90% | original-family weight, pre (500–2499) | new-family weight, post (2500–4999) | new-family weight, late (3000–4999) |")
M.append("|---|---|---|---|---|---|---|---|")
for reg, rl, old, new in REG:
    sw = pd.concat([pd.read_csv(R + f"family_bank_switch_{s}_density3b_{reg}.csv") for s in CH]); sw = sw[sw.scenario == "shape"]
    q = lambda x: f"{x.median():g} [{x.quantile(.25):g}, {x.quantile(.75):g}]"
    p = pd.concat([pd.read_csv(R + f"family_bank_path_{s}_density3b_{reg}.csv") for s in CH]); p = p[p.scenario == "shape"]
    w = lambda fam, a, b: p[(p.series == f"w_{fam}") & (p.t >= a) & (p.t < b)].groupby("t").value.mean().mean()
    M.append(f"| {rl} | {q(sw.t50[sw.t50 >= 0])} | {(sw.t50 >= 0).sum()}/{len(sw)} | {q(sw.t90[sw.t90 >= 0])} | {(sw.t90 >= 0).sum()}/{len(sw)} | "
             f"{w(old, 500, 2500):.3f} | {w(new, 2500, 5000):.3f} | {w(new, 3000, 5000):.3f} |")

M.append("\n## Stability under constant families (density3; no change; 60 replications)\n")
M.append("False switch = any other shape's summed weight reaches 50% (90%) at some tick in 2500–4999; correct-family weight = mean over ticks 500–4999.\n")
M.append("| Truth | correct-family weight | another shape reaches 50% | another shape reaches 90% | shapes involved (50%) |")
M.append("|---|---|---|---|---|")
for f, sfx in [("gaussian", ""), ("t", "_t"), ("clayton", "_clayton"), ("gumbel", "_gumbel")]:
    sw = pd.concat([pd.read_csv(R + f"family_bank_switch_{s}_density3{sfx}.csv") for s in CH]); sw = sw[sw.scenario == "constant"]
    pt = pd.concat([pd.read_csv(R + f"family_bank_path_{s}_density3{sfx}.csv") for s in CH]); pt = pt[(pt.scenario == "constant") & (pt.t >= 500)]
    wc = pt[pt.series == f"w_{f}"].groupby("t").value.mean().mean()
    by = sw[sw.t50 >= 0].groupby("new_shape").seed.nunique().to_dict()
    M.append(f"| {f} | {wc:.3f} | {sw[sw.t50 >= 0].seed.nunique()}/60 | {sw[sw.t90 >= 0].seed.nunique()}/60 | {by} |")

M.append("\n## Predictive accuracy: ΔL = L(Gaussian-only MFF) − L(family-augmented MFF), ×10⁻³ nats/obs, mean ± 2 SE (60 paired replications; positive favours family augmentation)\n")
M.append("| Switch | first 100 after (2500–2599) | remaining 2400 (2600–4999) | full post-switch (2500–4999) | L Gaussian-only (post) | L family (post) |")
M.append("|---|---|---|---|---|---|")
for reg, rl, _, _ in REG:
    d = pd.concat([pd.read_csv(R + f"family_bank_{s}_density3b_{reg}.csv") for s in CH])
    d = d[(d.metric == "excess_x1000") & d.method.isin(["family_bank", "gauss_bank"])]
    x = d.pivot_table(index=["seed", "method"], columns="window", values="value").reset_index()
    x["late"] = (2500 * x["post"] - 100 * x["rec_0_100"]) / 2400
    g = x[x.method == "gauss_bank"].set_index("seed"); f = x[x.method == "family_bank"].set_index("seed")
    cells = []
    for w in ["rec_0_100", "late", "post"]:
        m, s2, n = ms(g[w] - f[w]); cells.append(f"{m:+.1f} ± {s2:.1f}")
        rows.append(dict(regime=reg, window=w, dL=m, dL_2se=s2, n=n, L_gauss=g[w].mean(), L_family=f[w].mean()))
    M.append(f"| {rl} | " + " | ".join(cells) + f" | {g['post'].mean():.2f} | {f['post'].mean():.2f} |")
d = pd.concat([pd.read_csv(R + f"family_bank_{s}_density3.csv") for s in CH])
d = d[(d.scenario == "constant") & (d.window == "whole") & (d.metric == "excess_x1000")]
x = d.pivot_table(index="seed", columns="method", values="value")
m, s2, n = ms(x.family_bank - x.gauss_bank)
rows.append(dict(regime="gaussian_constant", window="whole", dL=-m, dL_2se=s2, n=n, L_gauss=x.gauss_bank.mean(), L_family=x.family_bank.mean()))
M.append(f"\nCost of family augmentation under constant Gaussian dependence (ticks 500–4999): L(family) − L(Gaussian-only) = **{m:+.2f} ± {s2:.2f}** ×10⁻³ nats/obs "
         f"(L family {x.family_bank.mean():.2f}, Gaussian-only {x.gauss_bank.mean():.2f}).")
pd.DataFrame(rows).to_csv(R + "famswitch_final.csv", index=False)
open(R + "famswitch_summary_2026-11-15.md", "w").write("\n".join(M) + "\n")
print("\n".join(M))
