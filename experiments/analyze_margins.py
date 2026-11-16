"""Summarise results/margins_seeds.csv (probe_margins.cpp): mean RMSE x1e-2 over 100 paired seeds, and paired differences.
Usage (from experiments/): python3 analyze_margins.py -> results/margins_2026-11-16.md"""
import numpy as np
import pandas as pd

d = pd.read_csv("results/margins_seeds.csv")
W = {w: d.pivot(index="seed", columns="variant", values=w) * 100 for w in ("rmse_all", "rmse_late")}
M = ["# Marginal-estimation ablation, Gaussian constant dependence, n = 15,000 (2026-11-16)\n",
     "RMSE of rho_S ×10⁻² over 100 paired test seeds (42–141), cumulative estimation. 'all' = observations 500–14,999; "
     "'late' = 10,000–14,999. Δ = variant minus the stated reference, paired mean ± SE. full_emp uses the whole future sample "
     "(diagnostic only, not streaming).\n",
     "| Variant | RMSE all | RMSE late | Δ all vs ref ± SE | ref |", "|---|---|---|---|---|"]
REF = lambda v: (v.split("_")[0] + "_oracle") if not v.startswith("hs") else v.split("_")[0] + "_oracle"
order = [v for v in ["rmpl_oracle", "rmpl_full_emp", "rmpl_seq_emp", "rmpl_seq_mid", "rmpl_seq_emp_t0_500", "rmpl_seq_emp_t0_2000", "rmpl_kll_64",
                     "rmpl_kll_256", "rmpl_kll_512", "rmpl_kll_1024", "rmpl_kll_4096", "rmpl_kll_16384", "rmpl_kll_512_t0_2000",
                     "rankmom_oracle", "rankmom_full_emp", "rankmom_seq_emp", "cm_oracle", "cm_default", "cm_full_emp", "cm_seq_emp",
                     "hs20_oracle", "hs20_default", "hs20_full_emp", "hs20_seq_emp", "hs60_oracle", "hs60_default", "hs60_full_emp",
                     "hs60_seq_emp"] if v in W["rmse_all"]]
for v in order:
    r = REF(v)
    dd = W["rmse_all"][v] - W["rmse_all"][r]
    M.append(f"| {v} | {W['rmse_all'][v].mean():.3f} | {W['rmse_late'][v].mean():.3f} | "
             + ("—" if v == r else f"{dd.mean():+.3f} ± {dd.std(ddof=1) / 10:.3f}") + f" | {r} |")
chk = (W["rmse_all"]["rmpl_kll_16384"] - W["rmse_all"]["rmpl_seq_emp"]).abs().max()
M.append(f"\nCheck: KLL k = 16384 (never compresses at n = 15,000) vs exact sequential empirical, max |Δ| over seeds = {chk:.2e}.")
for a, b in [("rmpl_seq_emp", "rmpl_full_emp"), ("rmpl_kll_512", "rmpl_seq_emp"), ("rmpl_seq_emp_t0_2000", "rmpl_seq_emp"), ("rmpl_seq_mid", "rmpl_seq_emp"), ("rmpl_seq_mid", "cm_default"),
             ("rmpl_kll_512_t0_2000", "rmpl_kll_512"), ("rmpl_seq_emp", "cm_default"), ("rmpl_full_emp", "cm_default"),
             ("rmpl_oracle", "cm_default")]:
    for w in ("rmse_all", "rmse_late"):
        dd = W[w][a] - W[w][b]
        M.append(f"- {a} − {b} ({w}): {dd.mean():+.3f} ± {dd.std(ddof=1) / 10:.3f}")
infl = {k: (W["rmse_all"][f"{k}_seq_emp"].mean() / W["rmse_all"][f"{k}_oracle"].mean()) for k in ("rmpl", "rankmom", "cm", "hs20", "hs60")}
M.append("\nInflation of RMSE, sequential empirical margins / true margins: " + ", ".join(f"{k} {v:.3f}" for k, v in infl.items()))
infl2 = {k: (W["rmse_all"][f"{k}_full_emp"].mean() / W["rmse_all"][f"{k}_oracle"].mean()) for k in ("rmpl", "rankmom", "cm", "hs20", "hs60")}
M.append("Inflation, full-sample empirical / true margins: " + ", ".join(f"{k} {v:.3f}" for k, v in infl2.items()))
open("results/margins_2026-11-16.md", "w").write("\n".join(M) + "\n")
print("\n".join(M))
