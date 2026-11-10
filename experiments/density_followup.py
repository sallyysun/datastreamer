"""Density follow-up to ew_marginals.md §32 (A1-A3, R1b-c, R2; §33). Paired per-seed differences over the 60
seeds 42-101, mean (SE), all x1e-3 (rho_S RMSE in rho units, excess log loss in nats per tick at the true PIT).
  A1: family bank minus the single Gaussian (mle_bank_log) and the Gaussian own-weight mixture (mle_bank_own),
      rho_S RMSE and excess, per window, with the per-seed correlation of the two differences.
      Base families: the existing matched outputs results/family_bank_42_matched[_fam].csv. Switches: the density2
      run (matched log-bank delta 1e-4; the earlier TAG=density switch runs used 3e-4, see `check`).
  A2: shape-switch summary from results/family_bank_switch_* and family_bank_path_*.
  R1/R2: results/family_bank_<first>_density2[_regime].csv (DENSITY_AUDIT=2, seeds in chunks of 15).
hermite_bank_own is excluded from every density table (its own score integrates to about 1.07, not a density).
Usage (from experiments/): python3 density_followup.py a1|a2|r1|r2|check"""
import sys, math, glob
import numpy as np
import pandas as pd

BASE = ["gaussian", "clayton", "gumbel", "t"]
SWITCH = ["gauss2t", "clayton2gumbel", "gauss2clayton"]
WIN = ["whole", "pre", "rec_0_100", "rec_100_500", "rec_500_1500", "post"]
EXCLUDE = {"hermite_bank_own"}
LAB = {"family_bank": "family-bank mixture",
       "mle_bank_log_point": "single Gaussian at the combined rho (MPL log bank)",
       "mle_bank_log": "single Gaussian at the combined rho (MPL log bank)",
       "mle_bank_log_mix": "Gaussian mixture, log-bank weights (MPL log bank)",
       "mle_bank_own": "Gaussian mixture, own-bank weights (eta 0.25, delta 3e-3)",
       "gauss_bank": "Gaussian-only family bank (3 memories)",
       "best_arm": "best fixed family-and-memory arm (hindsight, whole window)",
       "gauss_floor": "Gaussian floor, realised: log c_true - log c_G(r*)",
       "gauss_floor_expected": "Gaussian floor D(t) = KL(c_true || c_G(r*))"}


def sfx(c): return "" if c == "gaussian" else "_" + c


def load_existing(cell):
    f = f"results/family_bank_42_matched{sfx(cell)}.csv" if cell in BASE else f"results/family_bank_42_density_{cell}.csv"
    return pd.read_csv(f)


def load_d2(regime):
    fs = sorted(glob.glob(f"results/family_bank_[0-9]*_density2{sfx(regime)}.csv"))
    fs = [f for f in fs if f.endswith(f"_density2{sfx(regime)}.csv")]
    d = pd.concat([pd.read_csv(f) for f in fs])
    d = d[~d.method.isin(EXCLUDE)]
    assert d.seed.nunique() == 60 and d.seed.min() == 42 and d.seed.max() == 101, (regime, d.seed.nunique())
    return d


def piv(d):
    return d.pivot_table(index=["scenario", "window", "seed"], columns=["method", "metric"], values="value")


def ms(x):
    x = np.asarray(x, float)
    return 1000 * x.mean(), 1000 * x.std(ddof=1) / math.sqrt(len(x))


def fmt(x, signed=True):
    m, s = ms(x)
    flag = "*" if abs(m) > 2 * s else " "
    return (f"{m:+7.1f} ({s:4.1f}){flag}" if signed else f"{m:7.1f} ({s:4.1f}) ")


def cells():
    for c in BASE:
        for sc in ["constant", "drift", "jump", "oscillate"]:
            yield c, sc
    for c in SWITCH:
        yield c, "shape"


def a1():
    print("## A1. Family bank minus Gaussian comparators: rho_S RMSE and excess log loss (x1e-3, paired mean (SE), "
          "60 seeds; * = |mean| > 2 SE). r = per-seed correlation of the two differences.\n")
    for comp in ["mle_bank_log", "mle_bank_own"]:
        print(f"### Comparator: {LAB[comp]}\n")
        print("| copula | scenario | window | d rho_S RMSE | d excess | r |")
        print("| --- | --- | --- | --- | --- | --- |")
        for c, sc in cells():
            p = piv(load_existing(c) if c in BASE else load_d2(c))
            for w in WIN:
                if (sc, w) not in p.index.droplevel(2): continue
                g = p.loc[(sc, w)]
                dr = g[("family_bank", "rmse")] - g[(comp, "rmse")]
                de = (g[("family_bank", "excess_x1000")] - g[(comp, "excess_x1000")]) / 1000
                r = np.corrcoef(dr, de)[0, 1]
                print(f"| {c} | {sc} | {w} | {fmt(dr)} | {fmt(de)} | {r:+.2f} |")
        print()


def a2():
    print("## A2. Shape switches: weights, crossing delays and false switches\n")
    print("Seed-mean summed weight of each shape (pre = ticks [500, 2500), post = [2500, 5000)); delays in ticks after "
          "2500 for the new shape's summed weight to reach 50% / 90%, median [IQR] over the 60 seeds.\n")
    print("| switch | new shape | weight old/new pre | weight old/new post | t50 | t90 | reached 50% / 90% |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    old = {"gauss2t": "gaussian", "clayton2gumbel": "clayton", "gauss2clayton": "gaussian"}
    for c in SWITCH:
        s = pd.read_csv(f"results/family_bank_switch_42_density_{c}.csv")
        pth = pd.read_csv(f"results/family_bank_path_42_density_{c}.csv")
        new = s.new_shape.iloc[0]
        def wmean(shape, a, b):
            x = pth[(pth.series == "w_" + shape) & (pth.t >= a) & (pth.t < b)]
            return x.value.mean()
        def q(col):
            x = s[col][s[col] >= 0]
            return f"{x.median():.0f} [{x.quantile(.25):.0f}, {x.quantile(.75):.0f}]"
        print(f"| {c} | {new} | {wmean(old[c], 500, 2500):.2f} / {wmean(new, 500, 2500):.2f} | "
              f"{wmean(old[c], 2500, 5000):.2f} / {wmean(new, 2500, 5000):.2f} | {q('t50')} | {q('t90')} | "
              f"{(s.t50 >= 0).sum()}/60 / {(s.t90 >= 0).sum()}/60 |")
    print("\nNull paths (constant rho_S, no switch): seeds in which ANY wrong shape's summed weight reached 50% / 90% "
          "after tick 2500 (a seed counts once, whichever shape).\n")
    print("| DGP copula | seeds with a false 50% crossing | seeds with a false 90% crossing | shapes involved (50%) |")
    print("| --- | --- | --- | --- |")
    for c in BASE:
        s = pd.read_csv(f"results/family_bank_switch_42_matched{sfx(c)}.csv")
        a50 = s[s.t50 >= 0]; a90 = s[s.t90 >= 0]
        shapes = ", ".join(f"{k} {v}" for k, v in a50.new_shape.value_counts().items()) or "none"
        print(f"| {c} | {a50.seed.nunique()}/60 | {a90.seed.nunique()}/60 | {shapes} |")
    print()


DENS = ["family_bank", "mle_bank_log_point", "mle_bank_log_mix", "mle_bank_own", "gauss_bank", "best_arm"]


def best_arm(p, sc):
    g = p.loc[(sc, "whole")]
    arms = [m for m in g.columns.get_level_values(0).unique() if m.startswith("arm_")
            and (m, "excess_x1000") in g.columns]
    return min(arms, key=lambda m: g[(m, "excess_x1000")].mean())


def r1(regimes, title):
    print(title)
    for c in regimes:
        d = load_d2(c)
        p = piv(d)
        for sc in d.scenario.unique():
            ba = best_arm(p, sc)
            print(f"\n### {c} / {sc}  (best fixed arm in hindsight: {ba})\n")
            print("Excess log loss (x1e-3 nats/tick). Level per method, then family-bank mixture minus each comparator "
                  "(paired).\n")
            print("| window | floor D(t) | floor realised | family mix | single Gauss | Gauss mix (log w) | Gauss mix (own w) "
                  "| Gauss-only FB | best arm | FB - single | FB - Gmix(log) | FB - Gmix(own) | FB - best arm |")
            print("| --- " * 14 + "|")
            for w in WIN:
                if (sc, w) not in p.index.droplevel(2): continue
                g = p.loc[(sc, w)]
                e = lambda m: g[(m, "excess_x1000")] / 1000
                fb = e("family_bank")
                lv = [fmt(e(m), False) for m in ["gauss_floor_expected", "gauss_floor", "family_bank", "mle_bank_log_point",
                                                 "mle_bank_log_mix", "mle_bank_own", "gauss_bank", ba]]
                dv = [fmt(fb - e(m)) for m in ["mle_bank_log_point", "mle_bank_log_mix", "mle_bank_own", ba]]
                print(f"| {w} | " + " | ".join(lv + dv) + " |")
            print("\nDecomposition of the Gaussian forecasts' excess: excess = floor (realised, same observations) + "
                  "estimation; estimation shown (x1e-3, paired). Negative means the forecast beat the KL-best fixed "
                  "Gaussian on these observations.\n")
            print("| window | single Gauss - floor | Gauss mix (log w) - floor | Gauss mix (own w) - floor | Gauss-only FB - floor |")
            print("| --- | --- | --- | --- | --- |")
            for w in WIN:
                if (sc, w) not in p.index.droplevel(2): continue
                g = p.loc[(sc, w)]
                e = lambda m: g[(m, "excess_x1000")] / 1000
                print(f"| {w} | " + " | ".join(fmt(e(m) - e("gauss_floor")) for m in
                                               ["mle_bank_log_point", "mle_bank_log_mix", "mle_bank_own", "gauss_bank"]) + " |")
            print("\nrho_S RMSE (x1e-3): level, and family bank minus comparator (paired).\n")
            print("| window | family bank | single Gauss / Gauss mix (log w) | Gauss mix (own w) | Gauss-only FB | best-excess arm "
                  "| FB - MPL log bank | FB - MPL own bank | FB - Gauss-only FB | FB - best-excess arm |")
            print("| --- " * 10 + "|")
            for w in WIN:
                if (sc, w) not in p.index.droplevel(2): continue
                g = p.loc[(sc, w)]
                r = lambda m: g[(m, "rmse")]
                fb = r("family_bank")
                lv = [fmt(r(m), False) for m in ["family_bank", "mle_bank_log", "mle_bank_own", "gauss_bank", ba]]
                dv = [fmt(fb - r(m)) for m in ["mle_bank_log", "mle_bank_own", "gauss_bank", ba]]
                print(f"| {w} | " + " | ".join(lv + dv) + " |")
            if c == "claygum_mix" or c in SWITCH:
                print("\nTail RMSE (x1e-3): family bank and EW ratio counters, chi_L and chi_U at q = 0.05 and 0.10.\n")
                print("| window | metric | family bank | ratio 99 | ratio 399 | ratio 1999 | FB - r99 | FB - r399 | FB - r1999 |")
                print("| --- " * 9 + "|")
                for w in WIN:
                    if (sc, w) not in p.index.droplevel(2): continue
                    g = p.loc[(sc, w)]
                    for met in ["chiL_rmse", "chiU_rmse", "chiL10_rmse", "chiU10_rmse"]:
                        r = lambda m: g[(m, met)]
                        fb = r("family_bank")
                        print(f"| {w} | {met} | " + " | ".join([fmt(r(m), False) for m in ["family_bank", "ratio_99", "ratio_399", "ratio_1999"]]
                                                         + [fmt(fb - r(m)) for m in ["ratio_99", "ratio_399", "ratio_1999"]]) + " |")
                if c == "claygum_mix":
                    print("\nTail RMSE of the Gaussian forecasts and the Gaussian-only bank at q = 0.05 (x1e-3).\n")
                    print("| window | metric | family bank | single Gauss | Gauss mix (log w) | Gauss-only FB |")
                    print("| --- " * 6 + "|")
                    for w in WIN:
                        if (sc, w) not in p.index.droplevel(2): continue
                        g = p.loc[(sc, w)]
                        for met in ["chiL_rmse", "chiU_rmse"]:
                            print(f"| {w} | {met} | " + " | ".join(fmt(g[(m, met)], False) for m in
                                  ["family_bank", "mle_bank_log_point", "mle_bank_log_mix", "gauss_bank"]) + " |")
        print()


def weights(regime):
    fs = [f for f in sorted(glob.glob(f"results/family_bank_path_[0-9]*_density2_{regime}.csv"))]
    d = pd.concat([pd.read_csv(f).assign(chunk=f) for f in fs]).groupby(["t", "series"]).value.mean().reset_index()
    print(f"\nSeed-mean summed shape weights, {regime} (chunks averaged with equal weight; 15 seeds each).\n")
    print("| span | " + " | ".join(s[2:] for s in d.series.unique() if s.startswith("w_")) + " |")
    print("| --- " * (1 + sum(s.startswith("w_") for s in d.series.unique())) + "|")
    for name, a, b in [("pre [500, 2500)", 500, 2500), ("rec [2500, 2600)", 2500, 2600), ("rec [2600, 3000)", 2600, 3000),
                       ("post [2500, 5000)", 2500, 5000), ("end [4500, 5000)", 4500, 5000)]:
        x = d[(d.t >= a) & (d.t < b)].groupby("series").value.mean()
        print(f"| {name} | " + " | ".join(f"{x['w_' + s[2:]]:.2f}" for s in d.series.unique() if s.startswith("w_")) + " |")
    s = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"results/family_bank_switch_[0-9]*_density2_{regime}.csv"))])
    x = s[s.t50 >= 0]
    print(f"\nGumbel summed weight reaching 50% after tick 2500: {len(x)}/60 seeds, median {x.t50.median():.0f} "
          f"[{x.t50.quantile(.25):.0f}, {x.t50.quantile(.75):.0f}]; 90%: {(s.t90 >= 0).sum()}/60.")


def check():
    """Every row the earlier runs wrote must be reproduced exactly by the density2 run (same streams, same code).
    The switch runs of §32 (TAG=density) used the log-bank delta 3e-4, density2 the matched 1e-4: there only the
    *_bank_log rows (and mle_bank_log_point / _mix) may differ, and their shift is reported."""
    k = ["copula", "scenario", "seed", "window", "method", "metric"]
    logm = {"xiao_bank_log", "hermite_bank_log", "mle_bank_log", "mle_bank_log_point", "mle_bank_log_mix"}
    for c in BASE + SWITCH:
        old = load_existing(c)
        old = old[~old.method.isin(EXCLUDE)]
        new = load_d2(c)
        m = old.merge(new, on=k, how="left", suffixes=("_old", "_new"))
        m["d"] = (m.value_old - m.value_new).abs()
        other = m[~m.method.isin(logm)]
        print(f"{c:15s} old rows {len(old):6d}  missing {m.value_new.isna().sum():3d}  "
              f"max |diff| non-log-bank rows {other.d.max():.2e}  log-bank rows {m[m.method.isin(logm)].d.max():.2e}")
        if c in SWITCH:
            lb = m[m.method.isin(logm) & (m.metric.isin(["rmse", "excess_x1000"]))]
            t = lb.groupby(["window", "method", "metric"])[["value_old", "value_new"]].mean()
            t["shift"] = t.value_new - t.value_old
            t = t.reset_index()
            t = t[t.window.isin(["pre", "rec_0_100", "post"]) & t.method.isin(["mle_bank_log_point", "mle_bank_log_mix"])]
            print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "a1"
    if what == "a1": a1()
    if what == "a2": a2()
    if what == "check": check()
    if what == "r1":
        r1(BASE + SWITCH, "## R1. Density forecasts on identical observations (true PIT), per regime\n")
    if what == "r2":
        r1(["claygum_mix"], "## R2. Clayton-Gumbel mixture, pi(Clayton) 0.75 -> 0.25 at tick 2500, rho_S constant\n")
        weights("claygum_mix")
