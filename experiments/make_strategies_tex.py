"""LaTeX tables for Section 6.3.2 from results/strategies_summary.csv (analyze_strategies.py).
Usage (from experiments/): python3 make_strategies_tex.py -> results/strategies_tables.tex"""
import os
import pandas as pd
T = ("_" + os.environ["DENS_TAG"]) if os.environ.get("DENS_TAG") else ""

s = pd.read_csv(f"results/strategies_summary{T}.csv").fillna("")
EST = [("rmpl", "RMPL"), ("xiao", "Counting Matrix"), ("hermite", "Hermite Series")]
FAM = [("gaussian", "Gaussian"), ("clayton", "Clayton"), ("gumbel", "Gumbel"), ("t", "Student $t_4$")]
SC = [("constant", "Constant"), ("drift", "Gradual drift"), ("jump", "Abrupt jump")]
MET = {"rmse": ("Spearman's $\\rho_S$ RMSE ($\\times10^{-2}$)", "rmse"), "excess_x1000": ("excess logarithmic loss ($\\times10^{-3}$ nats per observation)", "excess"),
       "h2_x1000": ("squared Hellinger distance ($\\times10^{-3}$)", "h2")}
K = ["fixed", "vff_slow", "vff_fast", "mff"]
out = []


def val(m, sc, e, f, k, w="whole"):
    r = s[(s.metric == m) & (s.scenario == sc) & (s.window == w) & (s.estimator == e) & (s.copula == f) & (s.strategy == k)].iloc[0]
    return r


def fmt(r, bold, se=True):
    x = f"{r['mean']:.2f}" + (f" ({r['se']:.2f})" if se else "")
    return f"\\textbf{{{x}}}" if bold else x


# main: RMSE, family mean
out += ["% Main chapter: RMSE averaged over the four copula families (whole run).",
        "\\begin{table}[htbp]", "\\centering",
        "\\caption{Spearman's $\\rho_S$ RMSE ($\\times10^{-2}$) of fixed forgetting, VFF and MFF, averaged over the four generating copula families (60 Monte Carlo replications; standard errors in brackets). Fixed is the lowest-RMSE fixed effective sample size chosen in hindsight for each family, estimator and scenario: an accuracy reference, not an online procedure. Bold: lowest in the row.}",
        "\\label{tab:strategy-rmse}", "\\begin{tabular}{llrrrr}", "\\toprule",
        "Scenario & Estimator & Fixed & VFF slow & VFF fast & MFF \\\\", "\\midrule"]
for sc, scl in SC:
    for i, (e, el) in enumerate(EST):
        rs = [val("rmse", sc, e, "mean4", k) for k in K]
        best = min(range(4), key=lambda j: rs[j]["mean"])
        out.append(f"{scl if i == 0 else ''} & {el} & " + " & ".join(fmt(r, j == best) for j, r in enumerate(rs)) + " \\\\")
    out.append("\\addlinespace" if sc != "jump" else "\\bottomrule")
out += ["\\end{tabular}", "\\end{table}", ""]

# appendix: per metric, per scenario, family-level
for m, (mlab, tag) in MET.items():
    for sc, scl in SC:
        out += [f"% Appendix: {tag}, {sc}.", "\\begin{table}[htbp]", "\\centering", "\\small",
                f"\\caption{{{scl}: {mlab} by generating family and estimator (whole run; 60 replications; standard errors in brackets). Fixed: hindsight-best fixed effective sample size (in brackets after the value).}}",
                f"\\label{{tab:app-strategy-{tag}-{sc}}}", "\\begin{tabular}{llrrrr}", "\\toprule",
                "Family & Estimator & Fixed ($N_{\\mathrm{eff}}$) & VFF slow & VFF fast & MFF \\\\", "\\midrule"]
        for fi, (f, fl) in enumerate(FAM):
            for i, (e, el) in enumerate(EST):
                rs = [val(m, sc, e, f, k) for k in K]
                fx = f"{rs[0]['mean']:.2f} ({str(rs[0]['best_fixed'])[6:]})"
                out.append(f"{fl if i == 0 else ''} & {el} & {fx} & " + " & ".join(fmt(r, False) for r in rs[1:]) + " \\\\")
            if fi < 3: out.append("\\addlinespace")
        out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
open(f"results/strategies_tables{T}.tex", "w").write("\n".join(out) + "\n")
print(f"wrote results/strategies_tables{T}.tex")
