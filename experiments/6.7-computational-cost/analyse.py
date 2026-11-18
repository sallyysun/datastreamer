"""Section 6.7 tables at n = 15,000: assemble the budget sweep and timing runs, select within each storage budget on the
validation seeds, write the configuration tables, and (REPORTS=1) the marginal-estimation ablation."""
import itertools
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
from report_output import format_estimate, md_table, reports_enabled, verbose_print, write_report  # noqa: E402
from summary_stats import standard_error  # noqa: E402
import config as C  # noqa: E402

R = "results/"
SEED_KEY = ["family", "scenario", "estimator", "approx", "gain", "seed"]
TIMING_REPEATS = range(1, 6)
LARGER_BUDGETS, BASE_BUDGET = (64, 256), 16
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
    """One sweep per scenario x family into budget_{sweep,seeds}_n15k; the repeated timing runs averaged."""
    for kind in ("sweep", "seeds"):
        runs = [pd.read_csv(R + f"budget_{kind}_n15k_{s}_{f}.csv") for s in C.SCENARIOS for f in C.FAMILIES_T31]
        pd.concat(runs, ignore_index=True).to_csv(R + f"budget_{kind}_n15k.csv", index=False)
    timing = pd.concat([pd.read_csv(R + f"budget_timing_n15k_rep{r}.csv").assign(rep=r) for r in TIMING_REPEATS])
    table = timing.groupby(["estimator", "approx"]).agg(kib=("peak_bytes", lambda x: x.max() / 1024),
                                                        us=("mean_ns", lambda x: x.mean() / 1e3),
                                                        us_sd=("mean_ns", lambda x: x.std() / 1e3)).reset_index()
    table["cv"] = table.us_sd / table.us
    table.to_csv(R + "budget_time_table_n15k.csv", index=False)


def select(sweep, seed_rmse):
    """Per cell and budget, the size and gain with the lowest validation RMSE that fit, and its test RMSE."""
    val, test = sweep[sweep.split == "val"], sweep[sweep.split == "test"]
    rows = []
    for family, scenario, budget, (estimator, _, _) in itertools.product(C.FAMILIES_T31, C.SCENARIOS, C.BUDGETS, EST):
        cap = budget * 1024
        fits = val[(val.family == family) & (val.scenario == scenario) & (val.estimator == estimator) & (val.peak_bytes <= cap)]
        chosen = fits.loc[fits.rmse.idxmin()]
        approx, gain = int(chosen.approx), float(chosen.gain)
        tested = test[(test.family == family) & (test.scenario == scenario) & (test.estimator == estimator)
                      & (test.approx == approx) & np.isclose(test.gain, gain)].iloc[0]
        rmse = seed_rmse.loc[(family, scenario, estimator, approx, gain)]
        peak = max(chosen.peak_bytes, tested.peak_bytes)
        rows.append(dict(family=family, scenario=scenario, budget_kib=budget, estimator=estimator, approx=approx, gain=gain,
                         val_rmse=100 * chosen.rmse, rmse=100 * rmse.mean(), se=standard_error(rmse, scale=100),
                         peak_val=int(chosen.peak_bytes), peak_test=int(tested.peak_bytes), cap=cap, util=100 * peak / cap))
    return pd.DataFrame(rows)


def gains(selected, seed_rmse):
    """Percentage RMSE reduction at the larger budgets relative to the 16 KiB selection, over the paired test seeds."""
    def rmse_at(cell, family, scenario, estimator, budget):
        r = cell.loc[budget]
        return seed_rmse.loc[(family, scenario, estimator, int(r.approx), float(r.gain))].values
    rows = []
    for (family, scenario, estimator), cell in selected.groupby(["family", "scenario", "estimator"]):
        cell = cell.set_index("budget_kib")
        base = rmse_at(cell, family, scenario, estimator, BASE_BUDGET)
        for budget in LARGER_BUDGETS:
            diff = base - rmse_at(cell, family, scenario, estimator, budget)
            rows.append(dict(family=family, scenario=scenario, estimator=estimator, budget=budget,
                             pct=100 * diff.mean() / base.mean(), pct_mc_se=standard_error(diff, scale=100) / base.mean()))
    return pd.DataFrame(rows)


def smallest_within(same_gain, rmse, tolerance):
    ok = same_gain[same_gain.rmse <= rmse + tolerance]
    return ok.loc[ok.peak_bytes.idxmin()]


def configs(selected, val):
    """Each selection with its retention factor and use of the budget, and the smallest same-gain size within 1 SE / 1%."""
    selected = selected.assign(kib=selected.peak_val / 1024, lam=1 - selected.gain)
    rows = []
    for _, r in selected.iterrows():
        fits = val[(val.family == r.family) & (val.scenario == r.scenario) & (val.estimator == r.estimator) & (val.peak_bytes <= r.cap)]
        same_gain = fits[np.isclose(fits.gain, r.gain)]
        chosen = same_gain[same_gain.approx == r.approx].iloc[0]
        within_se = smallest_within(same_gain, chosen.rmse, chosen.se_rmse)
        within_pct = smallest_within(same_gain, chosen.rmse, chosen.rmse * 0.01)
        rows.append(dict(estimator=r.estimator, family=r.family, scenario=r.scenario, budget_kib=r.budget_kib, approx=int(r.approx),
                         gain=r.gain, retention=r.lam, kib=r.kib, util_pct=100 * r.peak_val / r.cap, val_rmse=r.val_rmse,
                         test_rmse=r.rmse, smallest_within_1se_approx=int(within_se.approx),
                         smallest_within_1se_kib=within_se.peak_bytes / 1024, larger_than_needed=bool(within_se.peak_bytes < r.peak_val),
                         smallest_within_1pct_approx=int(within_pct.approx), smallest_within_1pct_kib=within_pct.peak_bytes / 1024,
                         larger_than_1pct=bool(within_pct.peak_bytes < r.peak_val)))
    return pd.DataFrame(rows)


def span(x, fmt="{:g}"):
    lo, hi = x.min(), x.max()
    return fmt.format(lo) if lo == hi else f"{fmt.format(lo)}--{fmt.format(hi)}"


def main_table(table):
    lines = [r"\begin{table}[ht]", r"\centering\small",
             r"\caption{Validation-selected configurations at $n = 15{,}000$: range of the approximation size across the four copula families and "
             r"three dependence scenarios, and the largest retained state (KiB). RMPL uses Apache DataSketches KLL margins.}",
             r"\label{tab:ch6-budget-configs}", r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
             r"Estimator & Size & 16\,KiB & 64\,KiB & 256\,KiB \\", r"\midrule"]
    for estimator, label, symbol in EST:
        rows = table[table.estimator == estimator]
        cells = [f"{span(rows[rows.budget_kib == b].approx)} ({rows[rows.budget_kib == b].kib.max():.1f})" for b in C.BUDGETS]
        lines.append(f"{label} & ${symbol}$ & " + " & ".join(cells) + r" \\")
    retention = ("constant 1 (no forgetting); drift " + span(table[table.scenario == "drift"].retention)
                 + "; jump " + span(table[table.scenario == "jump"].retention))
    lines += [r"\midrule", r"\multicolumn{5}{@{}l}{Selected retention factor $\lambda$ (all estimators and budgets): " + retention + r"}\\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return lines


def appendix_table(table):
    lines = [r"\begin{longtable}{@{}lllcccccc@{}}",
             r"\caption{Validation-selected configurations for every estimator, copula family, scenario and budget ($n = 15{,}000$; RMPL with "
             r"Apache DataSketches KLL margins). Size: KLL capacity $k$ (RMPL), bins $B$ (Counting Matrix), order $N$ (Hermite Series); "
             r"$\lambda = 1 - g$ is the retention factor (1 = no forgetting); KiB is the measured peak retained state; RMSE is the validation "
             r"RMSE of $\rho_S$ ($\times10^{-2}$, seeds 1000--1039).}\label{tab:ch6-budget-configs-full}\\",
             r"\toprule", HEAD, r"\midrule", r"\endfirsthead", r"\toprule", HEAD, r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for (estimator, elabel, symbol), (scenario, slabel) in itertools.product(EST, SC):
        for (family, flabel), budget in itertools.product(FAM, C.BUDGETS):
            r = table[(table.estimator == estimator) & (table.scenario == scenario) & (table.family == family)
                      & (table.budget_kib == budget)].iloc[0]
            lines.append(f"{elabel} & {slabel} & {flabel} & {budget} & ${symbol}={r.approx}$ & {r.retention:g} & {r.kib:.1f} & "
                         f"{r.util_pct:.0f} & {r.val_rmse:.3f} \\\\")
        lines.append(r"\addlinespace")
    return lines + [r"\end{longtable}"]


def budgets():
    sweep = pd.read_csv(R + "budget_sweep_n15k.csv")
    seed_rmse = pd.read_csv(R + "budget_seeds_n15k.csv").set_index(SEED_KEY).rmse.sort_index()
    selected = select(sweep, seed_rmse)
    selected.to_csv(R + "budget_selected.csv", index=False)
    gains(selected, seed_rmse).to_csv(R + "budget_gain.csv", index=False)
    table = configs(selected, sweep[sweep.split == "val"])
    table.to_csv(R + "budget_configs_n15k.csv", index=False)
    open(R + "budget_configs_n15k_main.tex", "w").write("\n".join(main_table(table)) + "\n")
    open(R + "budget_configs_n15k_appendix.tex", "w").write("\n".join(appendix_table(table)) + "\n")
    verbose_print(f"Larger than same-gain alternatives (1 SE / 1% RMSE)  "
                  f"{table.larger_than_needed.sum()}/{len(table)} / {table.larger_than_1pct.sum()}/{len(table)}")


def paired(diff):
    return format_estimate(diff.mean(), standard_error(diff), "+.3f", ".3f")


def margins_report(rmse):
    """RMSE of each marginal-estimation variant, paired differences against its oracle and between chosen pairs."""
    whole, late = rmse["rmse_all"], rmse["rmse_late"]
    rows = []
    for variant in [v for v in MARGIN_ORDER if v in whole]:
        ref = variant.split("_")[0] + "_oracle"
        diff = "—" if variant == ref else paired(whole[variant] - whole[ref])
        rows.append([variant, f"{whole[variant].mean():.3f}", f"{late[variant].mean():.3f}", diff, ref])
    check = (whole["rmpl_kll_16384"] - whole["rmpl_seq_emp"]).abs().max()
    lines = (["# Marginal-estimation ablation, Gaussian constant dependence, n = 15,000 (2026-11-16)\n",
              "RMSE of rho_S ×10⁻², 100 paired test seeds. all = obs. 500–14,999, late = 10,000–14,999; Δ = variant − ref, mean ± SE. "
              "full_emp sees the whole sample (diagnostic, not streaming).\n"]
             + md_table(["Variant", "RMSE all", "RMSE late", "Δ all vs ref ± SE", "ref"], rows)
             + [f"\nCheck: KLL k = 16384 (never compresses at n = 15,000) vs exact sequential empirical, max |Δ| over seeds = {check:.2e}."])
    lines += [f"- {a} − {b} ({w}): " + paired(rmse[w][a] - rmse[w][b]) for (a, b), w in itertools.product(MARGIN_PAIRS, ("rmse_all", "rmse_late"))]
    for label, kind in [("\nInflation of RMSE, sequential empirical margins / true margins: ", "seq_emp"),
                        ("Inflation, full-sample empirical / true margins: ", "full_emp")]:
        lines.append(label + ", ".join(f"{k} {whole[f'{k}_{kind}'].mean() / whole[f'{k}_oracle'].mean():.3f}" for k in MARGIN_KINDS))
    return lines


def margins():
    seeds = pd.read_csv(R + "margins_seeds.csv")
    rmse = {w: seeds.pivot(index="seed", columns="variant", values=w) * 100 for w in ("rmse_all", "rmse_late")}
    write_report(R + "margins_2026-11-16.md", margins_report(rmse))


def main():
    prepare()
    budgets()
    if reports_enabled():
        margins()


if __name__ == "__main__":
    main()
