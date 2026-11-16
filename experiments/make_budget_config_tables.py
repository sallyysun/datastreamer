"""Section 6.7: validation-selected configurations of the final n = 15,000 budget experiment (RMPL on Apache DataSketches KLL).
Source: results/budget_audit_cells_n15k.csv (audit_budget.py with BUDGET_SET=_n15k; selection on validation seeds 1000-1039 within
each cap) and results/budget_sweep_n15k.csv (validation SE, for the 'smallest configuration within 1 SE' check).
Retention factor lambda = 1 - g, g = per-observation gain (Hermite's own parameter is the update weight g; same memory).
Usage (from experiments/): python3 make_budget_config_tables.py -> results/budget_configs_n15k.csv,
results/budget_configs_n15k_main.tex, results/budget_configs_n15k_appendix.tex"""
import numpy as np
import pandas as pd

C = pd.read_csv("results/budget_audit_cells_n15k.csv")
S = pd.read_csv("results/budget_sweep_n15k.csv")
V = S[S.split == "val"]
E = [("mpl_dskll", "RMPL", "k"), ("xiao", "Counting Matrix", "B"), ("hermite", "Hermite Series", "N")]
FAM = [("gaussian", "Gaussian"), ("t", "Student $t_4$"), ("clayton", "Clayton"), ("gumbel", "Gumbel")]
SC = [("constant", "Constant"), ("drift", "Drift"), ("jump", "Jump")]
C["kib"] = C.peak_val / 1024
C["lam"] = 1 - C.gain
rows = []
for _, r in C.iterrows():
    v = V[(V.family == r.family) & (V.scenario == r.scenario) & (V.estimator == r.estimator) & (V.peak_bytes <= r.cap)]
    sel = v[(v.approx == r.approx) & np.isclose(v.gain, r.gain)].iloc[0]
    same = v[np.isclose(v.gain, r.gain)]                           # same gain, smaller approximation sizes
    ok = same[same.rmse <= sel.rmse + sel.se_rmse]                 # within 1 (unpaired) validation SE of the selection
    sm = ok.loc[ok.peak_bytes.idxmin()]
    ok1 = same[same.rmse <= sel.rmse * 1.01]                       # within 1% of the selected validation RMSE
    sm1 = ok1.loc[ok1.peak_bytes.idxmin()]
    rows.append(dict(estimator=r.estimator, family=r.family, scenario=r.scenario, budget_kib=r.budget_kib, approx=int(r.approx),
                     gain=r.gain, retention=r.lam, kib=r.kib, util_pct=100 * r.peak_val / r.cap, val_rmse=r.val_rmse,   # already ×1e-2 in budget_audit_cells
                     test_rmse=r.rmse, smallest_within_1se_approx=int(sm.approx), smallest_within_1se_kib=sm.peak_bytes / 1024,
                     larger_than_needed=bool(sm.peak_bytes < r.peak_val), smallest_within_1pct_approx=int(sm1.approx),
                     smallest_within_1pct_kib=sm1.peak_bytes / 1024, larger_than_1pct=bool(sm1.peak_bytes < r.peak_val)))
D = pd.DataFrame(rows)
D.to_csv("results/budget_configs_n15k.csv", index=False)


def lam(g):
    return "1 (none)" if g == 0 else f"{1 - g:g}"


def rng(x, fmt="{:g}"):
    a, b = x.min(), x.max()
    return fmt.format(a) if a == b else f"{fmt.format(a)}--{fmt.format(b)}"


# ---- main-text table: estimator x budget, approximation range and max KiB; retention by scenario in a footer row
L = [r"\begin{table}[ht]", r"\centering\small",
     r"\caption{Validation-selected configurations at $n = 15{,}000$: range of the approximation size across the four copula families and "
     r"three dependence scenarios, and the largest retained state (KiB). RMPL uses Apache DataSketches KLL margins.}",
     r"\label{tab:ch6-budget-configs}", r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
     r"Estimator & Size & 16\,KiB & 64\,KiB & 256\,KiB \\", r"\midrule"]
for e, el, sym in E:
    x = D[D.estimator == e]
    cells = [f"{rng(x[x.budget_kib == b].approx)} ({x[x.budget_kib == b].kib.max():.1f})" for b in (16, 64, 256)]
    L.append(f"{el} & ${sym}$ & " + " & ".join(cells) + r" \\")
L += [r"\midrule", r"\multicolumn{5}{@{}l}{Selected retention factor $\lambda$ (all estimators and budgets): "
      + "constant 1 (no forgetting); drift " + rng(D[D.scenario == "drift"].retention, "{:g}") + "; jump " + rng(D[D.scenario == "jump"].retention, "{:g}")
      + r"}\\", r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open("results/budget_configs_n15k_main.tex", "w").write("\n".join(L) + "\n")

# ---- appendix table: every combination
A = [r"\begin{longtable}{@{}lllcccccc@{}}",
     r"\caption{Validation-selected configurations for every estimator, copula family, scenario and budget ($n = 15{,}000$; RMPL with "
     r"Apache DataSketches KLL margins). Size: KLL capacity $k$ (RMPL), bins $B$ (Counting Matrix), order $N$ (Hermite Series); "
     r"$\lambda = 1 - g$ is the retention factor (1 = no forgetting); KiB is the measured peak retained state; RMSE is the validation "
     r"RMSE of $\rho_S$ ($\times10^{-2}$, seeds 1000--1039).}\label{tab:ch6-budget-configs-full}\\",
     r"\toprule", r"Estimator & Scenario & Family & Budget & Size & $\lambda$ & KiB & Use (\%) & Val.\ RMSE \\", r"\midrule", r"\endfirsthead",
     r"\toprule", r"Estimator & Scenario & Family & Budget & Size & $\lambda$ & KiB & Use (\%) & Val.\ RMSE \\", r"\midrule", r"\endhead",
     r"\bottomrule", r"\endfoot"]
for e, el, sym in E:
    for s, sl in SC:
        for f, fl in FAM:
            for b in (16, 64, 256):
                r = D[(D.estimator == e) & (D.scenario == s) & (D.family == f) & (D.budget_kib == b)].iloc[0]
                A.append(f"{el} & {sl} & {fl} & {b} & ${sym}={r.approx}$ & {r.retention:g} & {r.kib:.1f} & {r.util_pct:.0f} & {r.val_rmse:.3f} \\\\")
        A.append(r"\addlinespace")
A.append(r"\end{longtable}")
open("results/budget_configs_n15k_appendix.tex", "w").write("\n".join(A) + "\n")

print(open("results/budget_configs_n15k_main.tex").read())
print(D.groupby(["estimator", "budget_kib"]).agg(approx_min=("approx", "min"), approx_max=("approx", "max"), kib_max=("kib", "max"),
                                                 util_max=("util_pct", "max"), util_med=("util_pct", "median"),
                                                 larger=("larger_than_needed", "sum")).round(1).to_string())
print(D.groupby(["scenario"]).retention.agg(lambda s: sorted(set(s))).to_string())
for col, kcol, lab in [("larger_than_needed", "smallest_within_1se_kib", "within 1 validation SE"), ("larger_than_1pct", "smallest_within_1pct_kib", "within 1% RMSE")]:
    print(f"selections larger than the smallest same-gain size {lab}: {int(D[col].sum())}/108")
    for (e, b), g in D.groupby(["estimator", "budget_kib"]):
        x = g[g[col]]
        print(f"  {e:10s} {b:4d} KiB: {len(x)}/12, median excess {np.median(x.kib - x[kcol]) if len(x) else 0:.1f} KiB")
