"""Write the validation-selected budget configuration tables."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402

EST = [("mpl_dskll", "RMPL", "k"), ("xiao", "Counting Matrix", "B"), ("hermite", "Hermite Series", "N")]
FAM = [(f, "Student $t_4$" if f == "t" else C.FAMILY_TEXT[f]) for f in C.FAMILIES_T31]
SC = [(s, s.capitalize()) for s in C.SCENARIOS]
HEAD = r"Estimator & Scenario & Family & Budget & Size & $\lambda$ & KiB & Use (\%) & Val.\ RMSE \\"


def smallest(same, sel_rmse, tol):
    ok = same[same.rmse <= sel_rmse + tol]
    return ok.loc[ok.peak_bytes.idxmin()]


def configs(sel, V):
    sel = sel.assign(kib=sel.peak_val / 1024, lam=1 - sel.gain)
    rows = []
    for _, r in sel.iterrows():
        v = V[(V.family == r.family) & (V.scenario == r.scenario) & (V.estimator == r.estimator) & (V.peak_bytes <= r.cap)]
        s = v[(v.approx == r.approx) & np.isclose(v.gain, r.gain)].iloc[0]
        same = v[np.isclose(v.gain, r.gain)]
        sm, sm1 = smallest(same, s.rmse, s.se_rmse), smallest(same, s.rmse, s.rmse * 0.01)
        rows.append(dict(estimator=r.estimator, family=r.family, scenario=r.scenario, budget_kib=r.budget_kib, approx=int(r.approx),
                         gain=r.gain, retention=r.lam, kib=r.kib, util_pct=100 * r.peak_val / r.cap, val_rmse=r.val_rmse,
                         test_rmse=r.rmse, smallest_within_1se_approx=int(sm.approx), smallest_within_1se_kib=sm.peak_bytes / 1024,
                         larger_than_needed=bool(sm.peak_bytes < r.peak_val), smallest_within_1pct_approx=int(sm1.approx),
                         smallest_within_1pct_kib=sm1.peak_bytes / 1024, larger_than_1pct=bool(sm1.peak_bytes < r.peak_val)))
    return pd.DataFrame(rows)


def span(x, fmt="{:g}"):
    a, b = x.min(), x.max()
    return fmt.format(a) if a == b else f"{fmt.format(a)}--{fmt.format(b)}"


def main_table(D):
    L = [r"\begin{table}[ht]", r"\centering\small",
         r"\caption{Validation-selected configurations at $n = 15{,}000$: range of the approximation size across the four copula families and "
         r"three dependence scenarios, and the largest retained state (KiB). RMPL uses Apache DataSketches KLL margins.}",
         r"\label{tab:ch6-budget-configs}", r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
         r"Estimator & Size & 16\,KiB & 64\,KiB & 256\,KiB \\", r"\midrule"]
    for e, el, sym in EST:
        x = D[D.estimator == e]
        cells = [f"{span(x[x.budget_kib == b].approx)} ({x[x.budget_kib == b].kib.max():.1f})" for b in (16, 64, 256)]
        L.append(f"{el} & ${sym}$ & " + " & ".join(cells) + r" \\")
    L += [r"\midrule", r"\multicolumn{5}{@{}l}{Selected retention factor $\lambda$ (all estimators and budgets): "
          + "constant 1 (no forgetting); drift " + span(D[D.scenario == "drift"].retention) + "; jump " + span(D[D.scenario == "jump"].retention)
          + r"}\\", r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return L


def appendix_table(D):
    A = [r"\begin{longtable}{@{}lllcccccc@{}}",
         r"\caption{Validation-selected configurations for every estimator, copula family, scenario and budget ($n = 15{,}000$; RMPL with "
         r"Apache DataSketches KLL margins). Size: KLL capacity $k$ (RMPL), bins $B$ (Counting Matrix), order $N$ (Hermite Series); "
         r"$\lambda = 1 - g$ is the retention factor (1 = no forgetting); KiB is the measured peak retained state; RMSE is the validation "
         r"RMSE of $\rho_S$ ($\times10^{-2}$, seeds 1000--1039).}\label{tab:ch6-budget-configs-full}\\",
         r"\toprule", HEAD, r"\midrule", r"\endfirsthead", r"\toprule", HEAD, r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for e, el, sym in EST:
        for s, sl in SC:
            for f, fl in FAM:
                for b in (16, 64, 256):
                    r = D[(D.estimator == e) & (D.scenario == s) & (D.family == f) & (D.budget_kib == b)].iloc[0]
                    A.append(f"{el} & {sl} & {fl} & {b} & ${sym}={r.approx}$ & {r.retention:g} & {r.kib:.1f} & {r.util_pct:.0f} & {r.val_rmse:.3f} \\\\")
            A.append(r"\addlinespace")
    return A + [r"\end{longtable}"]


def report(D):
    print(D.groupby(["estimator", "budget_kib"]).agg(approx_min=("approx", "min"), approx_max=("approx", "max"), kib_max=("kib", "max"),
                                                     util_max=("util_pct", "max"), util_med=("util_pct", "median"),
                                                     larger=("larger_than_needed", "sum")).round(1).to_string())
    print(D.groupby(["scenario"]).retention.agg(lambda s: sorted(set(s))).to_string())
    for col, kcol, lab in [("larger_than_needed", "smallest_within_1se_kib", "within 1 validation SE"),
                           ("larger_than_1pct", "smallest_within_1pct_kib", "within 1% RMSE")]:
        print(f"selections larger than the smallest same-gain size {lab}: {int(D[col].sum())}/108")
        for (e, b), g in D.groupby(["estimator", "budget_kib"]):
            x = g[g[col]]
            print(f"  {e:10s} {b:4d} KiB: {len(x)}/12, median excess {np.median(x.kib - x[kcol]) if len(x) else 0:.1f} KiB")


def main():
    sw = pd.read_csv("results/budget_sweep_n15k.csv")
    D = configs(pd.read_csv("results/budget_selected.csv"), sw[sw.split == "val"])
    D.to_csv("results/budget_configs_n15k.csv", index=False)
    open("results/budget_configs_n15k_main.tex", "w").write("\n".join(main_table(D)) + "\n")
    open("results/budget_configs_n15k_appendix.tex", "w").write("\n".join(appendix_table(D)) + "\n")
    if os.environ.get("VERBOSE") == "1":
        report(D)


if __name__ == "__main__":
    main()
