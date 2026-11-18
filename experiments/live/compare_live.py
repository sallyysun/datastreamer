"""Compare the live consumer's rows (results/live_seeds[_fam].csv, run_live.sh) with the in-process runner's stored rows
(../6.4-copula-density/results/strategies_seeds_mix[_fam].csv), matched on copula, scenario, estimator, strategy, seed
and window. Usage: python3 compare_live.py -> results/live_vs_probe.txt"""
import os
import sys

import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402

KEY = ["copula", "scenario", "estimator", "strategy", "seed", "window"]


def main():
    sfx = [C.sfx(f) for f in C.FAMILIES if os.path.exists(f"results/live_seeds{C.sfx(f)}.csv")]
    L = pd.concat([pd.read_csv(f"results/live_seeds{s}.csv") for s in sfx])
    P = pd.concat([pd.read_csv(f"../6.4-copula-density/results/strategies_seeds_mix{s}.csv") for s in sfx])
    m = L.merge(P, on=KEY, suffixes=("_live", "_probe"), how="left")
    out = [f"{len(L)} live rows ({L.seed.nunique()} seeds; {', '.join(sorted(L.copula.unique()))}); "
           f"{m.rmse_probe.notna().sum()} matched to stored runner rows."]
    for c in ["rmse", "excess_x1000", "h2_x1000"]:
        d = (m[c + "_live"] - m[c + "_probe"]).abs()
        out.append(f"{c:13s} identical {int((d == 0).sum())}/{len(d)}  max |live - runner| {d.max():.3g}")
    open("results/live_vs_probe.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
