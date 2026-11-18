"""Section 6.7 tables at n = 15,000: assemble the budget sweep and timing runs, select within each storage budget on the
validation seeds, write the configuration tables, and (REPORTS=1) the marginal-estimation ablation."""
import itertools
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
from report_output import format_estimate, reports_enabled, verbose_print, write_report  # noqa: E402
from summary_stats import standard_error  # noqa: E402
import config as C  # noqa: E402

R = "results/"
EST = [("mpl_dskll", "RMPL", "k"), ("xiao", "Counting Matrix", "B"), ("hermite", "Hermite Series", "N")]
FAM = [(f, "Student $t_4$" if f == "t" else C.FAMILY_TEXT[f]) for f in C.FAMILIES_T31]
SC = [(s, s.capitalize()) for s in C.SCENARIOS]
HEAD = r"Estimator & Scenario & Family & Budget & Size & $\lambda$ & KiB & Use (\%) & Val.\ RMSE \\"
MARGIN_ORDER = ["rmpl_oracle", "rmpl_full_emp", "rmpl_seq_emp", "rmpl_seq_mid", "rmpl_seq_emp_t0_500", "rmpl_seq_emp_t0_2000", "rmpl_kll_64",
                "rmpl_kll_256", "rmpl_kll_512", "rmpl_kll_1024", "rmpl_kll_4096", "rmpl_kll_16384", "rmpl_kll_512_t0_2000",
                "rankmom_oracle", "rankmom_full_emp", "rankmom_seq_emp", "cm_oracle", "cm_default", "cm_full_emp", "cm_seq_emp",
                "hs20_oracle", "hs20_default", "hs20_full_emp", "hs20_seq_emp", "hs60_oracle", "hs60_default", "hs60_full_emp", "hs60_seq_emp"]
MARGIN_PAIRS = [("rmpl_seq_emp", "rmpl_full_emp"), ("rmpl_kll_512", "rmpl_seq_emp"), ("rmpl_seq_emp_t0_2000", "rmpl_seq_emp"),
                ("rmpl_seq_mid", "rmpl_seq_emp"), ("rmpl_seq_mid", "cm_default"), ("rmpl_kll_512_t0_2000", "rmpl_kll_512"),
                ("rmpl_seq_emp", "cm_default"), ("rmpl_full_emp", "cm_default"), ("rmpl_oracle", "cm_default")]
MARGIN_KINDS = ("rmpl", "rankmom", "cm", "hs20", "hs60")


def prepare():
    """One sweep per scenario x family into budget_{sweep,seeds}_n15k; the five timing runs averaged."""
    for kind in ("sweep", "seeds"):
        d = pd.concat([pd.read_csv(R + f"budget_{kind}_n15k_{s}_{f}.csv") for s in C.SCENARIOS for f in C.FAMILIES_T31],
                      ignore_index=True)
        d.to_csv(R + f"budget_{kind}_n15k.csv", index=False)
    T = pd.concat([pd.read_csv(R + f"budget_timing_n15k_rep{r}.csv").assign(rep=r) for r in range(1, 6)])
    g = T.groupby(["estimator", "approx"]).agg(kib=("peak_bytes", lambda x: x.max() / 1024), us=("mean_ns", lambda x: x.mean() / 1e3),
                                               us_sd=("mean_ns", lambda x: x.std() / 1e3)).reset_index()
    g["cv"] = g.us_sd / g.us
    g.to_csv(R + "budget_time_table_n15k.csv", index=False)


def select(sw, SD):
    """Per cell and budget, the size and gain with the lowest validation RMSE that fit, and its test RMSE."""
    val, tst = sw[sw.split == "val"], sw[sw.split == "test"]
    rows = []
    for f, s, b, (e, _, _) in itertools.product(C.FAMILIES_T31, C.SCENARIOS, C.BUDGETS, EST):
        cap = b * 1024
        v = val[(val.family == f) & (val.scenario == s) & (val.estimator == e) & (val.peak_bytes <= cap)]
        r = v.loc[v.rmse.idxmin()]
        a, g = int(r.approx), float(r.gain)
        t = tst[(tst.family == f) & (tst.scenario == s) & (tst.estimator == e) & (tst.approx == a) & np.isclose(tst.gain, g)].iloc[0]
        x = SD.loc[(f, s, e, a, g)]
        rows.append(dict(family=f, scenario=s, budget_kib=b, estimator=e, approx=a, gain=g, val_rmse=100 * r.rmse,
                         rmse=100 * x.mean(), se=standard_error(x, scale=100), peak_val=int(r.peak_bytes),
                         peak_test=int(t.peak_bytes), cap=cap, util=100 * max(r.peak_bytes, t.peak_bytes) / cap))
    return pd.DataFrame(rows)


def gains(sel, SD):
    """Percentage RMSE reduction at 64 and 256 KiB relative to the 16 KiB selection, over the paired test seeds."""
    rows = []
    for (f, s, e), x in sel.groupby(["family", "scenario", "estimator"]):
        x = x.set_index("budget_kib")
        base = SD.loc[(f, s, e, int(x.loc[16].approx), float(x.loc[16].gain))].values
        for b in (64, 256):
            d = base - SD.loc[(f, s, e, int(x.loc[b].approx), float(x.loc[b].gain))].values
            rows.append(dict(family=f, scenario=s, estimator=e, budget=b, pct=100 * d.mean() / base.mean(),
                             pct_mc_se=standard_error(d, scale=100) / base.mean()))
    return pd.DataFrame(rows)


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
    for (e, el, sym), (s, sl) in itertools.product(EST, SC):
        for (f, fl), b in itertools.product(FAM, (16, 64, 256)):
            r = D[(D.estimator == e) & (D.scenario == s) & (D.family == f) & (D.budget_kib == b)].iloc[0]
            A.append(f"{el} & {sl} & {fl} & {b} & ${sym}={r.approx}$ & {r.retention:g} & {r.kib:.1f} & {r.util_pct:.0f} & {r.val_rmse:.3f} \\\\")
        A.append(r"\addlinespace")
    return A + [r"\end{longtable}"]


def budgets():
    sw = pd.read_csv(R + "budget_sweep_n15k.csv")
    SD = pd.read_csv(R + "budget_seeds_n15k.csv").set_index(["family", "scenario", "estimator", "approx", "gain", "seed"]).rmse.sort_index()
    sel = select(sw, SD)
    sel.to_csv(R + "budget_selected.csv", index=False)
    gains(sel, SD).to_csv(R + "budget_gain.csv", index=False)
    D = configs(sel, sw[sw.split == "val"])
    D.to_csv(R + "budget_configs_n15k.csv", index=False)
    open(R + "budget_configs_n15k_main.tex", "w").write("\n".join(main_table(D)) + "\n")
    open(R + "budget_configs_n15k_appendix.tex", "w").write("\n".join(appendix_table(D)) + "\n")
    verbose_print(f"Larger than same-gain alternatives (1 SE / 1% RMSE)  "
                  f"{D.larger_than_needed.sum()}/{len(D)} / {D.larger_than_1pct.sum()}/{len(D)}")


def paired(dd):
    return format_estimate(dd.mean(), dd.std(ddof=1) / 10, "+.3f", ".3f")


def margins_report(W):
    A = W["rmse_all"]
    M = ["# Marginal-estimation ablation, Gaussian constant dependence, n = 15,000 (2026-11-16)\n",
         "RMSE of rho_S ×10⁻², 100 paired test seeds. all = obs. 500–14,999, late = 10,000–14,999; Δ = variant − ref, mean ± SE. "
         "full_emp sees the whole sample (diagnostic, not streaming).\n",
         "| Variant | RMSE all | RMSE late | Δ all vs ref ± SE | ref |", "|---|---|---|---|---|"]
    for v in [v for v in MARGIN_ORDER if v in A]:
        r = v.split("_")[0] + "_oracle"
        M.append(f"| {v} | {A[v].mean():.3f} | {W['rmse_late'][v].mean():.3f} | " + ("—" if v == r else paired(A[v] - A[r])) + f" | {r} |")
    chk = (A["rmpl_kll_16384"] - A["rmpl_seq_emp"]).abs().max()
    M.append(f"\nCheck: KLL k = 16384 (never compresses at n = 15,000) vs exact sequential empirical, max |Δ| over seeds = {chk:.2e}.")
    for (a, b), w in itertools.product(MARGIN_PAIRS, ("rmse_all", "rmse_late")):
        M.append(f"- {a} − {b} ({w}): " + paired(W[w][a] - W[w][b]))
    for lab, kind in [("\nInflation of RMSE, sequential empirical margins / true margins: ", "seq_emp"),
                      ("Inflation, full-sample empirical / true margins: ", "full_emp")]:
        M.append(lab + ", ".join(f"{k} {A[f'{k}_{kind}'].mean() / A[f'{k}_oracle'].mean():.3f}" for k in MARGIN_KINDS))
    return M


def margins():
    d = pd.read_csv(R + "margins_seeds.csv")
    W = {w: d.pivot(index="seed", columns="variant", values=w) * 100 for w in ("rmse_all", "rmse_late")}
    write_report(R + "margins_2026-11-16.md", margins_report(W))


def main():
    prepare()
    budgets()
    if reports_enabled():
        margins()


if __name__ == "__main__":
    main()
