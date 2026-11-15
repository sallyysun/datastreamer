"""Table A (Section 6.3.2): whole-run Spearman RMSE of Fixed (hindsight-best by RMSE), VFF (alpha = 0.02) and MFF,
averaged over the four copula families, from results/strategies_seeds[_fam].csv (KLL construction, 60 seeds).
Fixed: per family x estimator x scenario, the fixed ESS minimising mean whole-run RMSE; then per seed the mean over the
four families of each strategy; SEs and paired differences from those per-seed means.
Usage (from experiments/): python3 make_table_a.py -> results/ch6_table_a_rmse.tex, results/ch6_table_a_rmse.txt"""
import math
import pandas as pd

FAMS = ["", "_clayton", "_gumbel", "_t"]
FIXED = ["fixed_39", "fixed_99", "fixed_199", "fixed_399", "fixed_1999"]
EST = [("rmpl", "RMPL"), ("xiao", "Counting Matrix"), ("hermite", "Hermite Series")]
SC = [("constant", "Constant"), ("drift", "Gradual drift"), ("jump", "Abrupt jump")]
d = pd.concat([pd.read_csv(f"results/strategies_seeds{s}.csv") for s in FAMS])
d = d[d.window == "whole"]
se = lambda x: x.std(ddof=1) / math.sqrt(len(x))
tex, txt = [], ["Table A: whole-run Spearman RMSE (x 10^-2), mean over 4 families, 60 seeds; SE in brackets.",
                "Fixed = hindsight-best fixed ESS by RMSE per family/estimator/scenario (N in brackets). Paired: mean ± 2 SE.", ""]
for sc, scl in SC:
    for i, (e, el) in enumerate(EST):
        per, bests = [], []
        for f, g in d[(d.scenario == sc) & (d.estimator == e)].groupby("copula"):
            p = g.pivot(index="seed", columns="strategy", values="rmse") * 100
            b = min(FIXED, key=lambda k: p[k].mean()); bests.append(b[6:])
            per.append(pd.DataFrame({"fixed": p[b], "vff": p["vff_slow"], "mff": p["mff"]}))
        a = sum(per) / len(per)
        mu = a.mean(); s = a.apply(se)
        j = mu.idxmin()
        cells = [f"{mu[k]:.2f} ({s[k]:.2f})" for k in ["fixed", "vff", "mff"]]
        cells = [f"\\textbf{{{c}}}" if k == j else c for c, k in zip(cells, ["fixed", "vff", "mff"])]
        tex.append(f"{scl if i == 0 else ''} & {el} & " + " & ".join(cells) + " \\\\")
        d1, d2 = a.mff - a.fixed, a.mff - a.vff
        txt.append(f"{scl:14s} {el:16s} Fixed {mu.fixed:5.2f} ({s.fixed:.2f}) [N {'/'.join(sorted(set(bests)))}]  VFF {mu.vff:5.2f} ({s.vff:.2f})  "
                   f"MFF {mu.mff:5.2f} ({s.mff:.2f})   MFF-Fixed {d1.mean():+.2f}±{2*se(d1):.2f}  MFF-VFF {d2.mean():+.2f}±{2*se(d2):.2f}")
    tex.append("\\addlinespace" if sc != "jump" else "\\bottomrule")
out = ["\\begin{table}[htbp]", "\\centering",
       "\\caption{Whole-run Spearman's $\\rho_S$ RMSE ($\\times10^{-2}$) of fixed forgetting, VFF ($\\alpha=0.02$) and MFF, averaged over the four generating copula families (60 paired replications; standard errors in brackets). Fixed uses the effective sample size with the lowest whole-run RMSE, selected in hindsight for each family, estimator and scenario ($N_{\\mathrm{eff}}=1999$ under constant dependence, 399 under drift and 199 under the jump, in every family). Bold: lowest in the row.}",
       "\\label{tab:strategy-rmse}", "\\begin{tabular}{llrrr}", "\\toprule",
       "Scenario & Estimator & Fixed & VFF & MFF \\\\", "\\midrule"] + tex + ["\\end{tabular}", "\\end{table}"]
open("results/ch6_table_a_rmse.tex", "w").write("\n".join(out) + "\n")
open("results/ch6_table_a_rmse.txt", "w").write("\n".join(txt) + "\n")
print("\n".join(txt)); print(); print("\n".join(out))
