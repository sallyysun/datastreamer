"""Fixed-rho_S shape switches: Spearman RMSE, full-copula excess log loss and finite-q tail RMSE on the same
streams, by window, from existing probe_family_bank outputs (results/family_bank_42[_TAG]_<regime>.csv).
Paired family bank minus comparator, mean +- 2 SE over seeds, x1e-3. Usage: python3 density_audit.py [TAG]"""
import sys, math
import pandas as pd
tag = ("_" + sys.argv[1]) if len(sys.argv) > 1 else ""
regs = sys.argv[2].split(",") if len(sys.argv) > 2 else ["gauss2t", "clayton2gumbel", "gauss2clayton"]
WIN = ["pre", "rec_0_100", "rec_100_500", "rec_500_1500", "post"]
def ci(x): x = x.dropna(); return f"{1000*x.mean():+7.1f} ± {2000*x.std(ddof=1)/math.sqrt(len(x)):4.1f}"
for r in regs:
    d = pd.read_csv(f"results/family_bank_42{tag}_{r}.csv")
    p = d.pivot_table(index=["seed", "window"], columns=["method", "metric"], values="value")
    print(f"\n### {r} (switch at t = 2500, rho_S constant)")
    rows = []
    for w in WIN:
        g = p.xs(w, level="window")
        fb = lambda m: g[("family_bank", m)]
        rows.append({"window": w,
            "FB rho_S RMSE": f"{1000*fb('rmse').mean():6.1f}",
            "FB - MPL log bank, rho_S RMSE": ci(fb("rmse") - g[("mle_bank_log", "rmse")]),
            "FB - best Gauss arm (M=399), rho_S RMSE": ci(fb("rmse") - g[("arm_gaussian_399", "rmse")]),
            "FB excess log loss": f"{fb('excess_x1000').mean()/1000*1000:6.1f}",
            "FB - MPL log bank (Gaussian point density), excess log loss": ci((fb("excess_x1000") - g[("mle_bank_log", "excess_x1000")]) / 1000),
            "FB - EW ratio M=399, chi_L RMSE": ci(fb("chiL_rmse") - g[("ratio_399", "chiL_rmse")]),
            "FB - EW ratio M=399, chi_U RMSE": ci(fb("chiU_rmse") - g[("ratio_399", "chiU_rmse")])})
    print(pd.DataFrame(rows).to_string(index=False))
