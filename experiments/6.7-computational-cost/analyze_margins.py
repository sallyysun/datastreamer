"""Summarize paired-seed RMSE across marginal estimation methods."""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from report_output import write_report  # noqa: E402

ORDER = ["rmpl_oracle", "rmpl_full_emp", "rmpl_seq_emp", "rmpl_seq_mid", "rmpl_seq_emp_t0_500", "rmpl_seq_emp_t0_2000", "rmpl_kll_64",
         "rmpl_kll_256", "rmpl_kll_512", "rmpl_kll_1024", "rmpl_kll_4096", "rmpl_kll_16384", "rmpl_kll_512_t0_2000",
         "rankmom_oracle", "rankmom_full_emp", "rankmom_seq_emp", "cm_oracle", "cm_default", "cm_full_emp", "cm_seq_emp",
         "hs20_oracle", "hs20_default", "hs20_full_emp", "hs20_seq_emp", "hs60_oracle", "hs60_default", "hs60_full_emp", "hs60_seq_emp"]
PAIRS = [("rmpl_seq_emp", "rmpl_full_emp"), ("rmpl_kll_512", "rmpl_seq_emp"), ("rmpl_seq_emp_t0_2000", "rmpl_seq_emp"),
         ("rmpl_seq_mid", "rmpl_seq_emp"), ("rmpl_seq_mid", "cm_default"), ("rmpl_kll_512_t0_2000", "rmpl_kll_512"),
         ("rmpl_seq_emp", "cm_default"), ("rmpl_full_emp", "cm_default"), ("rmpl_oracle", "cm_default")]
KINDS = ("rmpl", "rankmom", "cm", "hs20", "hs60")


def paired(dd):
    return f"{dd.mean():+.3f} ± {dd.std(ddof=1) / 10:.3f}"


def report(W):
    A = W["rmse_all"]
    M = ["# Marginal-estimation ablation, Gaussian constant dependence, n = 15,000 (2026-11-16)\n",
         "RMSE of rho_S ×10⁻² over 100 paired test seeds (42–141), cumulative estimation. 'all' = observations 500–14,999; "
         "'late' = 10,000–14,999. Δ = variant minus the stated reference, paired mean ± SE. full_emp uses the whole future sample "
         "(diagnostic only, not streaming).\n",
         "| Variant | RMSE all | RMSE late | Δ all vs ref ± SE | ref |", "|---|---|---|---|---|"]
    for v in [v for v in ORDER if v in A]:
        r = v.split("_")[0] + "_oracle"
        M.append(f"| {v} | {A[v].mean():.3f} | {W['rmse_late'][v].mean():.3f} | " + ("—" if v == r else paired(A[v] - A[r])) + f" | {r} |")
    chk = (A["rmpl_kll_16384"] - A["rmpl_seq_emp"]).abs().max()
    M.append(f"\nCheck: KLL k = 16384 (never compresses at n = 15,000) vs exact sequential empirical, max |Δ| over seeds = {chk:.2e}.")
    for a, b in PAIRS:
        for w in ("rmse_all", "rmse_late"):
            M.append(f"- {a} − {b} ({w}): " + paired(W[w][a] - W[w][b]))
    for lab, kind in [("\nInflation of RMSE, sequential empirical margins / true margins: ", "seq_emp"),
                      ("Inflation, full-sample empirical / true margins: ", "full_emp")]:
        M.append(lab + ", ".join(f"{k} {A[f'{k}_{kind}'].mean() / A[f'{k}_oracle'].mean():.3f}" for k in KINDS))
    return M


def main():
    d = pd.read_csv("results/margins_seeds.csv")
    W = {w: d.pivot(index="seed", columns="variant", values=w) * 100 for w in ("rmse_all", "rmse_late")}
    M = report(W)
    write_report("results/margins_2026-11-16.md", M)


if __name__ == "__main__":
    main()
