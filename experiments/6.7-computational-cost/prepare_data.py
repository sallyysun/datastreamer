"""Section 6.7: assemble the n = 15,000 budget sweep (probe_budget, one run per scenario x family) into
results/budget_{sweep,seeds}_n15k.csv, and average the five repeated timing runs into results/budget_time_table_n15k.csv
(retained KiB, mean us per observation and its SD across runs)."""
import sys
import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402


def main():
    for kind in ("sweep", "seeds"):
        d = pd.concat([pd.read_csv(f"results/budget_{kind}_n15k_{s}_{f}.csv") for s in C.SCENARIOS for f in C.FAMILIES_T31],
                      ignore_index=True)
        d.to_csv(f"results/budget_{kind}_n15k.csv", index=False)
    T = pd.concat([pd.read_csv(f"results/budget_timing_n15k_rep{r}.csv").assign(rep=r) for r in range(1, 6)])
    g = T.groupby(["estimator", "approx"]).agg(kib=("peak_bytes", lambda x: x.max() / 1024), us=("mean_ns", lambda x: x.mean() / 1e3),
                                               us_sd=("mean_ns", lambda x: x.std() / 1e3)).reset_index()
    g["cv"] = g.us_sd / g.us
    g.to_csv("results/budget_time_table_n15k.csv", index=False)
    print("wrote results/budget_sweep_n15k.csv, results/budget_seeds_n15k.csv, results/budget_time_table_n15k.csv")


if __name__ == "__main__":
    main()
