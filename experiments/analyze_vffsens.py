"""VFF learning-rate sensitivity (appendix to Section 6.3.2), from results/strategies_seeds_vffsens[_fam].csv and
results/strategies_extremes_vffsens[_fam].csv (probe_strategies.cpp, TAG=vffsens VFF_RATES=0.005,...,0.2; same streams,
KLL margins, MFF and fixed arms as strategies_2026-11-12, which it reproduces exactly at alpha 0.02 / 0.1).
alpha = 0.02 is the development choice; every "best alpha" here is chosen on the evaluation seeds (HINDSIGHT).
Family averages: per seed, the mean over the four families; paired SEs from those per-seed means.
Usage (from experiments/): python3 analyze_vffsens.py -> results/vffsens_2026-11-12.txt, results/vffsens_summary.csv,
results/vffsens_extremes.csv"""
import math
import os
import numpy as np
import pandas as pd

R = "results/"
T = ("_" + os.environ["DENS_TAG"]) if os.environ.get("DENS_TAG") else ""   # mix: MFF density = mixture
FAMS = [("gaussian", ""), ("clayton", "_clayton"), ("gumbel", "_gumbel"), ("t", "_t")]
FL = {"gaussian": "Gaussian", "clayton": "Clayton", "gumbel": "Gumbel", "t": "t4", "mean4": "mean of 4"}
EST = [("rmpl", "RMPL"), ("xiao", "Counting Matrix"), ("hermite", "Hermite Series")]
RATES = ["0.005", "0.01", "0.02", "0.05", "0.075", "0.1", "0.2"]
FIXED = ["fixed_39", "fixed_99", "fixed_199", "fixed_399", "fixed_1999"]
MET = [("rmse", "Spearman RMSE (x 10^-2)", 100), ("excess_x1000", "Excess log loss (x 10^-3 nats/obs)", 1),
       ("h2_x1000", "Squared Hellinger (x 10^-3)", 1)]
SCEN = ["constant", "drift", "jump"]
WINS = {"constant": ["whole"], "drift": ["whole"], "jump": ["whole", "pre", "rec_0_100", "rec_100_500", "post"]}

d = pd.concat([pd.read_csv(R + f"strategies_seeds_vffsens{T[1:]}{s}.csv") for _, s in FAMS])
L = []; P = lambda s="": L.append(s)
se = lambda x: x.std(ddof=1) / math.sqrt(len(x))
wtl = lambda x: "W" if x.mean() < -2 * se(x) else "L" if x.mean() > 2 * se(x) else "T"   # W = MFF lower


def table(m, scale, sc, w, e, fam):
    """per-seed table: VFF at each alpha, MFF, hindsight-best fixed; fam = family or 'mean4'."""
    def one(f):
        p = d[(d.copula == f) & (d.scenario == sc) & (d.window == w) & (d.estimator == e)].pivot(index="seed", columns="strategy", values=m) * scale
        best = min(FIXED, key=lambda k: p[k].mean())
        out = pd.DataFrame({a: p[f"vff_{a}"] for a in RATES}); out["mff"] = p["mff"]; out["fixed"] = p[best]
        return out
    return sum(one(f) for f, _ in FAMS) / 4 if fam == "mean4" else one(fam)


rows = []
for m, _, scale in MET:
    for sc in SCEN:
        for w in WINS[sc]:
            for e, _ in EST:
                for fam in [f for f, _ in FAMS] + ["mean4"]:
                    t = table(m, scale, sc, w, e, fam)
                    for k in RATES + ["mff", "fixed"]:
                        r = dict(metric=m, scenario=sc, window=w, estimator=e, copula=fam, strategy=k if k in ("mff", "fixed") else f"vff_{k}",
                                 mean=t[k].mean(), se=se(t[k]))
                        if k in RATES:
                            dd = t["mff"] - t[k]
                            r.update(mff_minus=dd.mean(), mff_minus_se=se(dd), verdict=wtl(dd))
                        rows.append(r)
S = pd.DataFrame(rows)
S.to_csv(R + f"vffsens_summary{T}.csv", index=False)

P("VFF learning-rate sensitivity. 60 seeds (42-101), matched paths, prequential forecasts, KLL k = 256 margins (shared by")
P("RMPL arms, VFF controller and MFF scoring), MFF eta = 1, delta = 1e-4 (log loss) fixed throughout. Linux/libstdc++.")
P("alpha = 0.02 = development choice. 'best alpha' = HINDSIGHT on the evaluation seeds. MFF - VFF: paired, mean ± 2 SE;")
P("W/T/L = MFF lower / within 2 SE / higher. Reproduction: alpha 0.02, 0.1, MFF and fixed rows equal strategies_seeds exactly.")
for m, mlab, scale in MET:
    P(); P("#" * 110); P(f"# {mlab}"); P("#" * 110)
    for sc in SCEN:
        for w in WINS[sc]:
            P(); P(f"=== {sc}, window {w}: family mean (VFF by alpha | MFF | Fixed hindsight), then MFF - VFF(alpha) ===")
            for e, el in EST:
                g = S[(S.metric == m) & (S.scenario == sc) & (S.window == w) & (S.estimator == e) & (S.copula == "mean4")].set_index("strategy")
                vals = " ".join(f"{g.loc['vff_' + a, 'mean']:7.2f}" for a in RATES)
                P(f"{el:16s} a=" + " ".join(f"{a:>7s}" for a in RATES) + f" | {'MFF':>7s} {'Fixed':>7s}")
                P(f"{'':16s}   {vals} | {g.loc['mff', 'mean']:7.2f} {g.loc['fixed', 'mean']:7.2f}")
                P(f"{'':16s}   " + " ".join(f"{g.loc['vff_' + a, 'mff_minus']:+6.2f}{g.loc['vff_' + a, 'verdict']}" for a in RATES)
                  + "   (MFF - VFF)")
                P(f"{'':16s}   " + " ".join(f"±{2 * g.loc['vff_' + a, 'mff_minus_se']:5.2f} " for a in RATES))

# best alpha (hindsight) per scenario and estimator; compromise; MFF comparisons (whole run, family mean)
P(); P("=" * 110); P("BEST TESTED ALPHA (HINDSIGHT, evaluation seeds), whole run, family mean"); P("=" * 110)
best_rows = []
for m, mlab, scale in MET:
    P(f"-- {mlab}")
    for e, el in EST:
        cells = []
        for sc in SCEN:
            g = S[(S.metric == m) & (S.scenario == sc) & (S.window == "whole") & (S.estimator == e) & (S.copula == "mean4")
                  & S.strategy.str.startswith("vff_")].set_index("strategy")["mean"]
            a = g.idxmin()[4:]
            cells.append(f"{sc}: {a:>5s} ({g.min():.2f})")
            best_rows.append(dict(metric=m, estimator=e, scenario=sc, best_alpha=a, value=g.min()))
        # best single alpha across scenarios: minimise the mean over scenarios of VFF(alpha) / VFF(best alpha in scenario)
        rel = {}
        for a in RATES:
            rr = []
            for sc in SCEN:
                g = S[(S.metric == m) & (S.scenario == sc) & (S.window == "whole") & (S.estimator == e) & (S.copula == "mean4")
                      & S.strategy.str.startswith("vff_")].set_index("strategy")["mean"]
                rr.append(g["vff_" + a] / g.min())
            rel[a] = (np.mean(rr), np.max(rr))
        a1 = min(rel, key=lambda a: rel[a][0])
        best_rows.append(dict(metric=m, estimator=e, scenario="all (single alpha)", best_alpha=a1, value=rel[a1][0]))
        P(f"   {el:16s} " + "   ".join(cells) + f"   | best single alpha: {a1} (mean ratio to per-scenario best {rel[a1][0]:.3f}, "
          f"worst {rel[a1][1]:.3f}); 0.05: {rel['0.05'][0]:.3f}/{rel['0.05'][1]:.3f}, 0.075: {rel['0.075'][0]:.3f}/{rel['0.075'][1]:.3f}, "
          f"0.02: {rel['0.02'][0]:.3f}/{rel['0.02'][1]:.3f}")
B = pd.DataFrame(best_rows)
B.to_csv(R + f"vffsens_best_alpha{T}.csv", index=False)

P(); P("=" * 110); P("MFF COMPARISONS, whole run, family mean: MFF - VFF, mean ± 2 SE (W/T/L)"); P("=" * 110)
for m, mlab, scale in MET:
    P(f"-- {mlab}")
    for e, el in EST:
        a1 = B[(B.metric == m) & (B.estimator == e) & (B.scenario == "all (single alpha)")].best_alpha.iloc[0]
        cells = []
        for sc in SCEN:
            ab = B[(B.metric == m) & (B.estimator == e) & (B.scenario == sc)].best_alpha.iloc[0]
            t = table(m, scale, sc, "whole", e, "mean4")
            parts = []
            for lab, a in [("0.02(dev)", "0.02"), (f"best {ab}", ab), (f"single {a1}", a1)]:
                dd = t["mff"] - t[a]
                parts.append(f"{lab} {dd.mean():+.2f}±{2*se(dd):.2f}{wtl(dd)}")
            allw = "".join(wtl(t["mff"] - t[a]) for a in RATES)
            cells.append(f"{sc}: " + ", ".join(parts) + f", every alpha {allw}")
        P(f"   {el}:"); [P("      " + c) for c in cells]

# robustness: MFF vs hindsight-best alpha per scenario, per family x estimator x metric
P(); P("=" * 110); P("ROBUSTNESS: MFF vs the hindsight-best alpha for that family, estimator, scenario and metric (whole run)"); P("=" * 110)
for m, mlab, scale in MET:
    out = {sc: [] for sc in SCEN}
    for sc in SCEN:
        for e, _ in EST:
            for f, _ in FAMS:
                t = table(m, scale, sc, "whole", e, f)
                ab = min(RATES, key=lambda a: t[a].mean())
                out[sc].append(wtl(t["mff"] - t[ab]))
    P(f"-- {mlab}: " + "  ".join(f"{sc} W/T/L {v.count('W')}/{v.count('T')}/{v.count('L')}" for sc, v in out.items())
      + "   (12 cells each: 3 estimators x 4 families)")

# extremes (excess log loss), whole window
X = pd.concat([pd.read_csv(R + f"strategies_extremes_vffsens{T[1:]}{s}.csv") for _, s in FAMS])
X.to_csv(R + f"vffsens_extremes{T}.csv", index=False)
P(); P("=" * 110); P("EXCESS LOG LOSS EXTREMES (whole window, all 60 seeds x 4500 ticks per family; nothing clipped or dropped)"); P("=" * 110)
P("seeds / seed-ticks with |rho_hat| > 0.98 summed over families; max per-tick loss; mean and median of per-tick excess")
P("(averaged over families); share of the summed excess from the worst 1% of seed-ticks (averaged over families).")
for e, el in EST:
    for sc in SCEN:
        g = X[(X.scenario == sc) & (X.estimator == e)]
        P(f"-- {el}, {sc}")
        for k in [f"vff_{a}" for a in RATES] + ["mff"]:
            h = g[g.strategy == k]
            P(f"   {k:10s} seeds {int(h.seeds_extreme.sum()):3d}  seed-ticks {int(h.seedticks_extreme.sum()):5d}  max {h.ex_max.max():9.0f}"
              f"  mean {h.ex_mean.mean():8.2f}  median {h.ex_median.mean():6.2f}  top-1% share {h.top1pct_share.mean():5.2f}")
open(R + ("vffsens_2026-11-12.txt" if not T else f"vffsens{T}_2026-11-14.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
