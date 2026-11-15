"""Section 6.5: initial family identification on the constant paths (no change), family bank from a uniform start.
results/family_bank_initid_42_initid[_fam].csv (probe_family_bank INITID=1 TAG=initid FB_SCENARIOS=constant; seeds 42-101;
config unchanged; the run reproduces density3's constant-path false-switch rows and family-bank metrics exactly).
t = number of observations processed (weights held before observation t). t50/t90: first t at which the true family's
summed weight (over memories 99/399/1999) reaches 0.5/0.9; failures = never within 5000 (counted as +inf in medians/IQRs).
Sustained: t_sus = last t with weight < 0.5, plus 1 (from then on the true family keeps >= 50%).
Usage (from experiments/): python3 famswitch_initid.py -> results/famswitch_initid_2026-11-15.md"""
import numpy as np
import pandas as pd

R = "results/"
FAM = [("gaussian", "", "Gaussian"), ("t", "_t", "Student's t4"), ("clayton", "_clayton", "Clayton"), ("gumbel", "_gumbel", "Gumbel")]


def q(x):
    x = np.sort(np.asarray(x, float))
    f = lambda p: np.quantile(x, p, method="linear") if np.isfinite(np.quantile(x, p)) else np.inf
    s = lambda v: "never" if not np.isfinite(v) else f"{v:g}"
    return f"{s(f(.5))} [{s(f(.25))}, {s(f(.75))}]"


L = ["# Initial copula-family identification (constant families, 60 replications, uniform start)\n",
     "| Truth | t50 median [IQR] | t90 median [IQR] | t50 ≤ 100 / 250 / 500 / 1000 | t90 ≤ 100 / 250 / 500 / 1000 | sustained ≥50% from, median [IQR] | never below 50% after first crossing | share of later obs. ≥50% (median) | mean weight at n = 100 / 250 / 500 / 1000 |",
     "|---|---|---|---|---|---|---|---|---|"]
for f, s, fl in FAM:
    d = pd.read_csv(R + f"family_bank_initid_42_initid{s}.csv")
    t50 = np.where(d.t50 >= 0, d.t50, np.inf); t90 = np.where(d.t90 >= 0, d.t90, np.inf)
    sus = np.where(d.t50 >= 0, d.last_below50 + 1, np.inf)
    n = len(d)
    within = lambda t: " / ".join(f"{int((t <= h).sum())}" for h in (100, 250, 500, 1000))
    never_below = int(((d.t50 >= 0) & (d.last_below50 < d.t50)).sum())
    share = np.where(d.t50 >= 0, d.ticks_ge50_after_t50 / (5000 - d.t50), np.nan)
    L.append(f"| {fl} | {q(t50)} | {q(t90)} | {within(t50)} (of {n}) | {within(t90)} | {q(sus)} | {never_below}/{n} | {np.nanmedian(share):.3f} | "
             f"{d.w100.mean():.2f} / {d.w250.mean():.2f} / {d.w500.mean():.2f} / {d.w1000.mean():.2f} |")
    fail50, fail90 = int((d.t50 < 0).sum()), int((d.t90 < 0).sum())
    L.append(f"|  | never reached: {fail50} | never reached: {fail90} | | | | | min share {np.nanmin(share):.3f} | |")
open(R + "famswitch_initid_2026-11-15.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
for f, s, fl in FAM:
    d = pd.read_csv(R + f"family_bank_initid_42_initid{s}.csv")
    late = d[(d.t50 >= 0) & (d.last_below50 + 1 > 1000)]
    print(fl, "replications whose sustained identification starts after n = 1000:", len(late), "| last_below50 values:", sorted(late.last_below50.tolist())[:12])
