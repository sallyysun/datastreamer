"""Section 6.5 (adaptation to copula-family changes): collate the existing family-bank shape-switch results. Reads only
saved outputs of probe_family_bank (no new runs):
  results/family_bank_switch_<s>[_TAG]_<regime>.csv  per seed: ticks after the switch for the new shape's summed weight
                                                     to reach 50% (t50) / 90% (t90); -1 = never. Constant paths: the
                                                     same for every other shape from tick 2500 (false switches).
  results/family_bank_<s>[_TAG]_<regime>.csv          per seed x window x method x metric (excess log loss etc.)
Runs compared: original (rank-window PIT, untagged, seeds 42-101 in one file), 'density' (EW margins, §32), and
density2 / density3 / density3b (EW margins, matched paths, seeds 42-101 in four chunks 42/57/72/87; density3b is the
latest for the switches, density3 for constant paths).
Usage (from experiments/): python3 analyze_famswitch.py -> results/famswitch_2026-11-15.txt"""
import math
import os
import numpy as np
import pandas as pd

R = "results/"
REG = [("gauss2t", "Gaussian -> t4"), ("gauss2clayton", "Gaussian -> Clayton"), ("clayton2gumbel", "Clayton -> Gumbel")]
L = []; P = lambda s="": L.append(s)


def load(kind, tag, reg):
    if tag in ("density2", "density3", "density3b"):
        fs = [R + f"family_bank_{kind}{s}_{tag}_{reg}.csv" if reg else R + f"family_bank_{kind}{s}_{tag}.csv" for s in ("_42", "_57", "_72", "_87")]
    elif tag == "density":
        fs = [R + f"family_bank_{kind}_42_density_{reg}.csv"]
    else:
        fs = [R + f"family_bank_{kind}_42_{reg}.csv" if reg else R + f"family_bank_{kind}_42.csv"]
    fs = [f.replace("family_bank_switch", "family_bank_switch").replace("family_bank__", "family_bank_") for f in fs]
    return pd.concat([pd.read_csv(f) for f in fs if os.path.exists(f)])


P("Family-switch adaptation (family bank: 8 shapes x memories 99/399/1999, eta = 1, delta = 2.1e-4, own copula log loss).")
P("Switch at tick 2500 of 5000 at constant rho_S (d = 0.6, rho_S ~ 0.58). t50 / t90: ticks after the switch until the")
P("new family's summed weight first reaches 50% / 90%; success = reached within the remaining 2500 ticks.")
P()
P("=== Switch times by run version (median [IQR], success out of n) ===")
for tag, lab in [("orig", "original (rank-window PIT, 2026-11-01)"), ("density", "EW margins, density run (§32)"),
                 ("density2", "EW margins, matched, density2"), ("density3b", "EW margins, matched, density3b (latest)")]:
    P(f"-- {lab}")
    for reg, rl in REG:
        d = load("switch_", tag, reg) if False else None
        fs = []
        try:
            if tag == "orig": d = pd.read_csv(R + f"family_bank_switch_42_{reg}.csv")
            elif tag == "density": d = pd.read_csv(R + f"family_bank_switch_42_density_{reg}.csv")
            else: d = pd.concat([pd.read_csv(R + f"family_bank_switch_{s}_{tag}_{reg}.csv") for s in (42, 57, 72, 87)])
        except FileNotFoundError as ex:
            P(f"   {rl}: missing ({ex.filename})"); continue
        d = d[d.scenario == "shape"]
        ok50, ok90 = d.t50 >= 0, d.t90 >= 0
        q = lambda x: f"{x.median():.0f} [{x.quantile(.25):.0f}, {x.quantile(.75):.0f}]"
        P(f"   {rl:22s} n={len(d):3d}  t50 {q(d.t50[ok50]):18s} success {ok50.sum()}/{len(d)}   t90 {q(d.t90[ok90]):18s} success {ok90.sum()}/{len(d)}"
          f"   pre-switch weight on new family {d.pre_weight_new.mean():.4f}")
latest_sw = {reg: pd.concat([pd.read_csv(R + f"family_bank_switch_{s}_density3b_{reg}.csv") for s in (42, 57, 72, 87)]) for reg, _ in REG}

P(); P("=== False switches on constant paths (density3; no change): seeds (of 60) in which another shape reaches 50% / 90% of the")
P("weight within the 2500 ticks after tick 2500; correct-family weight = path mean over ticks 500-4990 (seed mean) ===")
FAM = [("gaussian", "", "gaussian"), ("t", "_t", "t"), ("clayton", "_clayton", "clayton"), ("gumbel", "_gumbel", "gumbel")]
for f, sfx, shape in FAM:
    sw = pd.concat([pd.read_csv(R + f"family_bank_switch_{s}_density3{sfx}.csv") for s in (42, 57, 72, 87)])
    sw = sw[sw.scenario == "constant"]
    by50 = sw[sw.t50 >= 0].groupby("new_shape").seed.nunique().to_dict()
    by90 = sw[sw.t90 >= 0].groupby("new_shape").seed.nunique().to_dict()
    any50 = sw[sw.t50 >= 0].seed.nunique(); any90 = sw[sw.t90 >= 0].seed.nunique()
    pt = pd.concat([pd.read_csv(R + f"family_bank_path_{s}_density3{sfx}.csv") for s in (42, 57, 72, 87)])
    pt = pt[(pt.scenario == "constant") & (pt.t >= 500)]
    w = pt[pt.series == f"w_{shape}"].groupby("t").value.mean().mean()
    P(f"{f:9s} correct-family weight {w:.3f}; any shape >=50%: {any50}/60 seeds {by50}; >=90%: {any90}/60 {by90}")

P(); P("=== Excess log loss (x1e-3 nats/obs), density3b switches, paired over 60 seeds: mean (2 SE) ===")
P("family = family bank mixture; gaussMFF = gauss_bank (same Hedge/fixed-share machinery, Gaussian arms 99/399/1999,")
P("delta = 3/(2*4969)); single = MPL log bank point density (Gaussian at its combined rho, Chapter 6 log bank, delta 1e-4);")
P("Gmix = MPL log bank's Gaussian mixture (log-bank weights).")
rows = []
for reg, rl in REG + [("gaussian_constant", "Gaussian, constant (baseline)")]:
    if reg == "gaussian_constant":
        d = pd.concat([pd.read_csv(R + f"family_bank_{s}_density3.csv") for s in (42, 57, 72, 87)]); d = d[d.scenario == "constant"]
        wins = ["whole"]
    else:
        d = pd.concat([pd.read_csv(R + f"family_bank_{s}_density3b_{reg}.csv") for s in (42, 57, 72, 87)])
        wins = ["pre", "rec_0_100", "rec_100_500", "post", "whole"]
    d = d[d.metric == "excess_x1000"]
    P(f"-- {rl}")
    for w in wins:
        x = d[d.window == w].pivot_table(index="seed", columns="method", values="value")
        cells = []
        for m, ml in [("family_bank", "family"), ("gauss_bank", "gaussMFF"), ("mle_bank_log_point", "single"), ("mle_bank_log_mix", "Gmix")]:
            cells.append(f"{ml} {x[m].mean():7.2f}")
        def pd_(a, b):
            dd = x[a] - x[b]; return dd.mean(), 2 * dd.std(ddof=1) / math.sqrt(len(dd))
        a1, s1 = pd_("family_bank", "gauss_bank"); a2, s2 = pd_("family_bank", "mle_bank_log_point"); a3, s3 = pd_("family_bank", "mle_bank_log_mix")
        rows.append(dict(regime=reg, window=w, family=x.family_bank.mean(), family_se=x.family_bank.std(ddof=1) / math.sqrt(len(x)),
                         gaussMFF=x.gauss_bank.mean(), gaussMFF_se=x.gauss_bank.std(ddof=1) / math.sqrt(len(x)),
                         single=x.mle_bank_log_point.mean(), d_gaussMFF=a1, d_gaussMFF_2se=s1, d_single=a2, d_single_2se=s2, d_gmix=a3, d_gmix_2se=s3, n=len(x)))
        P(f"   {w:12s} " + "  ".join(cells) + f" | family-gaussMFF {a1:+7.2f} ({s1:.2f})  family-single {a2:+7.2f} ({s2:.2f})  family-Gmix {a3:+7.2f} ({s3:.2f})")
pd.DataFrame(rows).to_csv(R + "famswitch_density.csv", index=False)
open(R + "famswitch_2026-11-15.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
