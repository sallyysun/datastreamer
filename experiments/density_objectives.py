"""Three objectives on the same streams for the fixed-rho_S shape switches (probe_family_bank, DENSITY_AUDIT=1,
TAG=density): Spearman RMSE, full-copula excess log loss at the true PIT, and finite-q tail RMSE (chi(0.05)).
Predictive densities compared (all valid normalised copula densities):
  family_bank        mixture of 24 parametric arms (8 shapes x 3 memories), eta = 1, delta*
  gauss_bank         the same machinery restricted to the Gaussian shape (3 memories), eta = 1, delta* (K = 3)
  mle_bank_log_mix   the MPL log bank's weighted mixture of its arms' Gaussian densities
  mle_bank_log_point the MPL log bank's Gaussian density at its combined rho_hat (what the log bank reports)
Paired differences A - B, mean +- 2 SE over 60 seeds, x1e-3 (excess log loss in 1e-3 nats per observation).
Usage (from experiments/): python3 density_objectives.py [regimes]"""
import sys, math
import pandas as pd
regs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["gauss2t", "gauss2t3", "gauss2t8", "clayton2gumbel", "gauss2clayton"]
WIN = ["pre", "rec_0_100", "rec_100_500", "post"]
def ci(x):
    x = x.dropna(); m, h = x.mean(), 2 * x.std(ddof=1) / math.sqrt(len(x))
    flag = "*" if abs(m) > h else " "
    return f"{m:+7.1f}±{h:4.1f}{flag}"
for r in regs:
    d = pd.read_csv(f"results/family_bank_42_density_{r}.csv")
    p = d.pivot_table(index=["seed", "window"], columns=["method", "metric"], values="value")
    print(f"\n### {r}   (* = |difference| > 2 paired SE)")
    rows = []
    for w in WIN:
        g = p.xs(w, level="window")
        v = lambda m, k, s=1000: s * g[(m, k)]
        rows.append({"window": w,
            "rhoS: FB": f"{v('family_bank','rmse').mean():5.1f}",
            "rhoS: FB-Gauss bank": ci(v("family_bank", "rmse") - v("gauss_bank", "rmse")),
            "rhoS: FB-MPL log bank": ci(v("family_bank", "rmse") - v("mle_bank_log_mix", "rmse")),
            "EL: FB": f"{v('family_bank','excess_x1000',1).mean():5.1f}",
            "EL: FB-Gauss bank": ci(v("family_bank", "excess_x1000", 1) - v("gauss_bank", "excess_x1000", 1)),
            "EL: FB-MPL mix": ci(v("family_bank", "excess_x1000", 1) - v("mle_bank_log_mix", "excess_x1000", 1)),
            "EL: MPL mix-point": ci(v("mle_bank_log_mix", "excess_x1000", 1) - v("mle_bank_log_point", "excess_x1000", 1)),
            "chiL: FB-Gauss bank": ci(v("family_bank", "chiL_rmse") - v("gauss_bank", "chiL_rmse")),
            "chiU: FB-Gauss bank": ci(v("family_bank", "chiU_rmse") - v("gauss_bank", "chiU_rmse")),
            "chiL: FB-ratio399": ci(v("family_bank", "chiL_rmse") - v("ratio_399", "chiL_rmse"))})
    print(pd.DataFrame(rows).to_string(index=False))
