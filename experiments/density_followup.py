"""Density follow-up to ew_marginals.md §32 (A1-A3, R1b-c, R2; §33). Paired per-seed differences over the 60
seeds 42-101, mean (SE), all x1e-3 (rho_S RMSE in rho units, excess log loss in nats per tick at the true PIT).
  A1: family bank minus the single Gaussian (mle_bank_log) and the Gaussian own-weight mixture (mle_bank_own),
      rho_S RMSE and excess, per window, with the per-seed correlation of the two differences.
      Base families: the existing matched outputs results/family_bank_42_matched[_fam].csv. Switches: the density2
      run (matched log-bank delta 1e-4; the earlier TAG=density switch runs used 3e-4, see `check`).
  A2: shape-switch summary from results/family_bank_switch_* and family_bank_path_*.
  R1/R2: results/family_bank_<first>_density2[_regime].csv (DENSITY_AUDIT=2, seeds in chunks of 15).
hermite_bank_own is excluded from every density table (its own score integrates to about 1.07, not a density).
  H2 (Step 2, chapter6_metrics.md §13): results/family_bank_<first>_density3[_regime].csv (DENSITY_AUDIT=3; oscillation
      not run): squared Hellinger distance beside rho_S RMSE and excess log loss, same forecasts and windows.
  FB (Step 3, chapter6_metrics.md §14): results/thesis_bank_density_density4_s<first>[_fam].csv (probe_thesis_bank DENSITY_EVAL=1,
      mle component): the MPL forgetting banks, fixed arms, no-forgetting MPL and VFF on rho_S RMSE, excess log loss and H2.
  HEDGE (Step 4, chapter6_metrics.md §16): the same run with BANK_LOG_DELTA=0 (TAG hedge0_s<first>): plain Hedge vs the
      fixed-share log bank (delta = 1e-4, Step 3's density4), everything else identical.
Usage (from experiments/): python3 density_followup.py a1|a2|r1|r2|check|check3|h2|check4|fb|hedge"""
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


def load_d2(regime, tag="density2"):
    fs = sorted(glob.glob(f"results/family_bank_[0-9]*_{tag}{sfx(regime)}.csv"))
    fs = [f for f in fs if f.endswith(f"_{tag}{sfx(regime)}.csv")]
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


FC = [("family_bank", "family-bank mixture"), ("mle_bank_log_point", "single Gaussian"),
      ("mle_bank_log_mix", "Gaussian mixture (log w)")]
MET = [("rmse", "rho_S RMSE"), ("excess_x1000", "excess log loss"), ("hellinger2_x1000", "squared Hellinger")]
WIN3 = ["whole", "pre", "rec_0_100", "rec_100_500", "post"]


def best_arm_by(p, sc, metric):
    g = p.loc[(sc, "whole")]
    arms = [m for m in g.columns.get_level_values(0).unique() if m.startswith("arm_") and (m, metric) in g.columns]
    return min(arms, key=lambda m: g[(m, metric)].mean())


def load_h(c):
    """Step 2 outputs. density3 (constant, drift) plus density3b (jump, and every switch / mixture regime): density3b
    reran the regimes with a change after correcting the window weighting of H2 (the 10-tick checkpoints now count
    only in rec_0_100). The density3 jump and shape H2 rows are superseded and not used."""
    if c in BASE:
        a = load_d2(c, "density3")
        return pd.concat([a[a.scenario.isin(["constant", "drift"])], load_d2(c, "density3b")])
    return load_d2(c, "density3b")


def check3():
    """density3 must reproduce every density2 row of the scenarios it ran (oscillation is not run)."""
    k = ["copula", "scenario", "seed", "window", "method", "metric"]
    for c in BASE + SWITCH + ["claygum_mix"]:
        old = load_d2(c)
        new = load_h(c)
        old = old[old.scenario.isin(new.scenario.unique())]
        m = old.merge(new, on=k, how="left", suffixes=("_old", "_new"))
        h = new[new.metric == "hellinger2_x1000"]
        print(f"{c:15s} density2 rows {len(old):6d}  missing {m.value_new.isna().sum():3d}  max |diff| {np.nanmax(np.abs(m.value_old - m.value_new)):.1e}"
              f"  | H2 rows {len(h)}, NaN {h.value.isna().sum()}, range [{h.value.min():.4f}, {h.value.max():.2f}] x1e-3")


def cells3():
    for c in BASE:
        for sc in ["constant", "drift", "jump"]:
            yield c, sc
    for c in SWITCH + ["claygum_mix"]:
        yield c, "shape"


def h2():
    print("## Step 2. Squared Hellinger beside rho_S RMSE and excess log loss (x1e-3; mean (SE) over 60 seeds; * = |mean| > 2 SE)\n")
    print("Forecasts: family-bank mixture (FB), single Gaussian at the combined rho (MPL log bank), Gaussian mixture under the")
    print("log bank's weights, and the best fixed arm. 'best arm (excess)' is selected in hindsight by whole-window excess log loss")
    print("and used for every metric; 'best arm (H2)' is selected separately by whole-window H2 (a metric-specific oracle).\n")
    agree = {"excess": [], "h2": []}
    rows_dis = []
    for c, sc in cells3():
        p = piv(load_h(c))
        bx, bh = best_arm_by(p, sc, "excess_x1000"), best_arm_by(p, sc, "hellinger2_x1000")
        print(f"### {c} / {sc}   best arm (excess) = {bx[4:]};  best arm (H2) = {bh[4:]}\n")
        print("| window | metric | FB | single Gauss | Gauss mix | best arm (excess) | FB - single | FB - Gauss mix | FB - best (excess) | FB - best (H2) | r(d excess, d H2) vs single |")
        print("| --- " * 11 + "|")
        for w in WIN3:
            if (sc, w) not in p.index.droplevel(2): continue
            g = p.loc[(sc, w)]
            for met, lab in MET:
                # rho_S RMSE of the single Gaussian is the MPL log bank's (the same rho_hat): stored as mle_bank_log.
                col = lambda m: (m, met) if (m, met) in g.columns else ("mle_bank_log", met)
                v = lambda m: g[col(m)] if met == "rmse" else g[col(m)] / 1000
                fb = v("family_bank")
                cells = [fmt(v(m), False) for m in ["family_bank", "mle_bank_log_point", "mle_bank_log_mix", bx]]
                diffs = [fmt(fb - v(m)) for m in ["mle_bank_log_point", "mle_bank_log_mix", bx, bh]]
                r = ""
                if met == "hellinger2_x1000":
                    de = g[("family_bank", "excess_x1000")] - g[("mle_bank_log_point", "excess_x1000")]
                    dh = g[("family_bank", met)] - g[("mle_bank_log_point", met)]
                    r = f"{np.corrcoef(de, dh)[0, 1]:+.2f}"
                print(f"| {w} | {lab} | " + " | ".join(cells + diffs) + f" | {r} |")
            # ranking agreement between excess and H2 for FB vs each comparator
            for comp in ["mle_bank_log_point", "mle_bank_log_mix", bx]:
                res = {}
                for met, key in [("excess_x1000", "excess"), ("hellinger2_x1000", "h2")]:
                    x = (g[("family_bank", met)] - g[(comp, met)]).values
                    m_, s_ = x.mean(), x.std(ddof=1) / math.sqrt(len(x))
                    res[key] = "W" if m_ < -2 * s_ else "L" if m_ > 2 * s_ else "T"
                    agree[key].append(res[key])
                if res["excess"] != res["h2"]:
                    rows_dis.append((c, sc, w, comp if comp != bx else "best arm (" + bx[4:] + ")", res["excess"], res["h2"]))
        print()
    n = len(agree["excess"])
    same = sum(a == b for a, b in zip(agree["excess"], agree["h2"]))
    print(f"## Agreement of excess log loss and H2 on FB vs comparator (W/L/T at 2 SE), over {n} cell-window-comparator triples: "
          f"{same}/{n} agree.\n")
    print("| copula | scenario | window | comparator | excess | H2 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for r in rows_dis: print("| " + " | ".join(r) + " |")


FIX = ["fixed_39", "fixed_99", "fixed_199", "fixed_399", "fixed_1999"]
FBM = [("rmse", "rho_S RMSE"), ("excess_x1000", "excess log loss"), ("hellinger2_x1000", "squared Hellinger")]


def load4(fam):
    fs = [f"results/thesis_bank_density_density4_s{b}{sfx(fam)}.csv" for b in (42, 57, 72, 87)]
    d = pd.concat([pd.read_csv(f) for f in fs])
    assert d.seed.nunique() == 60 and d.seed.min() == 42 and d.seed.max() == 101, fam
    return d


def check4():
    """The Step 3 run must reproduce the matched500 mle rows of the main CSV, and its density file must agree with them."""
    k = ["copula", "scenario", "seed", "component", "window", "method"]
    for fam in BASE:
        old = pd.read_csv(f"results/thesis_bank_matched500{sfx(fam)}.csv")
        old = old[(old.component == "mle") & (old.scenario != "oscillate")]
        new = pd.concat([pd.read_csv(f"results/thesis_bank_density4_s{b}{sfx(fam)}.csv") for b in (42, 57, 72, 87)])
        m = old.merge(new, on=k, how="left", suffixes=("_o", "_n"))
        d = load4(fam).pivot_table(index=k, columns="metric", values="value").reset_index()
        mm = new.merge(d, on=k, suffixes=("_main", "_dens"))
        print(f"{fam:9s} matched500 mle rows {len(old):5d} missing {m.rmse_n.isna().sum()} max|d| rmse {np.abs(m.rmse_o - m.rmse_n).max():.1e}"
              f" excess {np.abs(m.excess_x1000_o - m.excess_x1000_n).max():.1e} | density file vs main: rows {len(mm)}"
              f" max|d| {max(np.abs(mm.rmse_main - mm.rmse_dens).max(), np.abs(mm.excess_x1000_main - mm.excess_x1000_dens).max()):.1e}"
              f" | H2 NaN {d.hellinger2_x1000.isna().sum()}")


def wlt(x):
    x = np.asarray(x, float); m_, s_ = x.mean(), x.std(ddof=1) / math.sqrt(len(x))
    return "W" if m_ < -2 * s_ else "L" if m_ > 2 * s_ else "T"


def fb():
    print("## Step 3. MPL forgetting-factor banks: rho_S RMSE, excess log loss, squared Hellinger (x1e-3; mean (SE) over 60 seeds)\n")
    print("Forecasts held before each tick, scored at the true PIT (excess) or against the true copula (H2). 'single' = Gaussian at the")
    print("bank's combined rho; 'mix' = sum_k w_k c_{r_k} with the same weights. Best fixed arm chosen per cell and per metric in")
    print("hindsight (lowest whole-window mean); W/L/T = paired difference beyond 2 SE. Lower is better for every metric.\n")
    cells = [(f, sc) for f in BASE for sc in ["constant", "drift", "jump"]]
    comps = [("C1 log vs squared (single)", "bank_log", "bank"), ("C1 log vs squared (mix)", "bank_log_mix", "bank_mix"),
             ("C2 log bank vs best fixed arm (single)", "bank_log", "BEST"), ("C2 log bank vs best fixed arm (mix)", "bank_log_mix", "BEST"),
             ("C2 squared bank vs best fixed arm", "bank", "BEST"), ("C2 log bank vs no forgetting", "bank_log", "nocum"),
             ("C3 log bank vs VFF slow", "bank_log", "vff_slow"), ("C3 log bank vs VFF fast", "bank_log", "vff_fast"),
             ("C3 log bank (mix) vs VFF slow", "bank_log_mix", "vff_slow"),
             ("C4 mix vs single, log bank", "bank_log_mix", "bank_log"), ("C4 mix vs single, squared bank", "bank_mix", "bank"),
             ("C4 mix vs single, uniform", "uniform_mix", "uniform"),
             ("uniform vs log bank (single)", "uniform", "bank_log")]
    tally = {}
    for fam, sc in cells:
        p = piv(load4(fam))
        best = {met: min(FIX, key=lambda m: p.loc[(sc, "whole")][(m, met)].mean()) for met, _ in FBM}
        print(f"### {fam} / {sc}   best fixed arm: RMSE {best['rmse']}, excess {best['excess_x1000']}, H2 {best['hellinger2_x1000']}\n")
        wins = ["whole"] + (["pre", "rec_0_100", "rec_100_500", "post"] if sc == "jump" else [])
        meths = FIX + ["nocum", "bank", "bank_log", "uniform", "bank_mix", "bank_log_mix", "uniform_mix", "vff_slow", "vff_fast"]
        for met, lab in FBM:
            div = 1 if met == "rmse" else 1000
            print(f"**{lab}** (levels)\n")
            print("| window | " + " | ".join(meths) + " |")
            print("| --- " * (len(meths) + 1) + "|")
            for w in wins:
                g = p.loc[(sc, w)]
                print(f"| {w} | " + " | ".join(f"{1000 * (g[(m, met)] / div).mean():.1f}" for m in meths) + " |")
            print()
        print("Paired differences A − B (mean (SE)):\n")
        print("| comparison | window | " + " | ".join(l for _, l in FBM) + " |")
        print("| --- | --- | --- | --- | --- |")
        for name, a, b in comps:
            for w in wins:
                g = p.loc[(sc, w)]
                row = []
                for met, _ in FBM:
                    bb = best[met] if b == "BEST" else b
                    div = 1 if met == "rmse" else 1000
                    x = (g[(a, met)] - g[(bb, met)]) / div
                    row.append(fmt(x))
                    tally.setdefault((name, w, met), []).append((f"{fam}/{sc}", wlt(x)))
                print(f"| {name} | {w} | " + " | ".join(row) + " |")
        print()
    print("## Summary: W/L/T of A vs B (2 SE), counted over cells; 'whole' over all 12 cells, 'rec_0_100' / 'post' over the 4 jump cells\n")
    print("| comparison | window | " + " | ".join(l for _, l in FBM) + " |")
    print("| --- | --- | --- | --- | --- |")
    for name, _, _ in comps:
        for w in ("whole", "rec_0_100", "post"):
            row = []
            for met, _ in FBM:
                v = [r for _, r in tally.get((name, w, met), [])]
                row.append(f"{v.count('W')}/{v.count('L')}/{v.count('T')}")
            print(f"| {name} | {w} | " + " | ".join(row) + " |")
    agree = tot = 0
    dis = []
    for (name, w, met), v in tally.items():
        if met != "excess_x1000": continue
        h = tally[(name, w, "hellinger2_x1000")]
        for (cell, a_), (_, b_) in zip(v, h):
            tot += 1; agree += a_ == b_
            if a_ != b_: dis.append((name, w, cell, a_, b_))
    print(f"\nExcess vs H2 verdict agreement over all comparison x cell x window entries: {agree}/{tot}.")
    print("Disagreements (comparison, window, cell, excess verdict, H2 verdict):")
    for d_ in dis: print("  ", d_)


def load_tag(fam, tag):
    d = pd.concat([pd.read_csv(f"results/thesis_bank_density_{tag}_s{b}{sfx(fam)}.csv") for b in (42, 57, 72, 87)])
    assert d.seed.nunique() == 60, (fam, tag)
    return d


def hedge():
    k = ["copula", "scenario", "seed", "component", "window", "method", "metric"]
    print("## Step 4. Fixed share (delta = 1e-4) vs plain Hedge (delta = 0), MPL log bank, eta = 1; everything else identical\n")
    print("### Verification: every method other than the log bank is unchanged by delta\n")
    for fam in BASE:
        a, b = load_tag(fam, "density4"), load_tag(fam, "hedge0")
        m = a.merge(b, on=k, suffixes=("_fs", "_h"))
        oth = m[~m.method.isin(["bank_log", "bank_log_mix"])]
        lg = m[m.method.isin(["bank_log", "bank_log_mix"])]
        ma = pd.concat([pd.read_csv(f"results/thesis_bank_density4_s{s_}{sfx(fam)}.csv") for s_ in (42, 57, 72, 87)])
        mb = pd.concat([pd.read_csv(f"results/thesis_bank_hedge0_s{s_}{sfx(fam)}.csv") for s_ in (42, 57, 72, 87)])
        mm = ma.merge(mb, on=["copula", "scenario", "seed", "component", "window", "method"], suffixes=("_fs", "_h"))
        mo = mm[mm.method != "bank_log"]
        print(f"{fam:9s} density rows {len(m)}: other methods max|d| {np.abs(oth.value_fs - oth.value_h).max():.1e}; log-bank rows differing "
              f"{(np.abs(lg.value_fs - lg.value_h) > 0).sum()}/{len(lg)} | main CSV other methods max|d| "
              f"{max(np.abs(mo.rmse_fs - mo.rmse_h).max(), np.abs(mo.excess_x1000_fs - mo.excess_x1000_h).max()):.1e}")
    print("\nPaired difference fixed share − Hedge (x1e-3, mean (SE), 60 seeds; * = beyond 2 SE; negative favours fixed share).")
    print("Principal forecast: the Gaussian mixture (bank_log_mix); rho_S RMSE is the bank's combined estimate either way.")
    print("Secondary: the single Gaussian at the combined rho (bank_log).\n")
    tally = {}
    for fam in BASE:
        A = piv(load_tag(fam, "density4")); B = piv(load_tag(fam, "hedge0"))
        for sc in ["constant", "drift", "jump"]:
            wins = ["whole"] + (["pre", "rec_0_100", "rec_100_500", "rec_500_1500", "post"] if sc == "jump" else [])
            print(f"### {fam} / {sc}\n")
            print("| window | Hedge: RMSE / excess (mix) / H2 (mix) | FS − Hedge: rho_S RMSE | excess (mix) | H2 (mix) | excess (single) | H2 (single) |")
            print("| --- | --- | --- | --- | --- | --- | --- |")
            for w in wins:
                ga, gb = A.loc[(sc, w)], B.loc[(sc, w)]
                lev = f"{1000 * gb[('bank_log', 'rmse')].mean():.1f} / {gb[('bank_log_mix', 'excess_x1000')].mean():.1f} / {gb[('bank_log_mix', 'hellinger2_x1000')].mean():.2f}"
                cells = []
                for m_, met, div in [("bank_log", "rmse", 1), ("bank_log_mix", "excess_x1000", 1000), ("bank_log_mix", "hellinger2_x1000", 1000),
                                      ("bank_log", "excess_x1000", 1000), ("bank_log", "hellinger2_x1000", 1000)]:
                    x = (ga[(m_, met)] - gb[(m_, met)]) / div
                    cells.append(fmt(x))
                    tally.setdefault((sc if sc != "jump" else w, m_ + ":" + met), []).append((fam, wlt(x)))
                print(f"| {w} | {lev} | " + " | ".join(cells) + " |")
            print()
    print("## Summary: W/L/T of fixed share vs Hedge (4 families per row)\n")
    print("| scenario / window | rho_S RMSE | excess (mix) | H2 (mix) | excess (single) | H2 (single) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for key in ["constant", "drift", "whole", "pre", "rec_0_100", "rec_100_500", "rec_500_1500", "post"]:
        row = []
        for mm_ in ["bank_log:rmse", "bank_log_mix:excess_x1000", "bank_log_mix:hellinger2_x1000", "bank_log:excess_x1000", "bank_log:hellinger2_x1000"]:
            v = [r for _, r in tally.get((key, mm_), [])]
            row.append(f"{v.count('W')}/{v.count('L')}/{v.count('T')}")
        lab = key if key in ("constant", "drift") else "jump: " + key
        print(f"| {lab} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "a1"
    if what == "a1": a1()
    if what == "a2": a2()
    if what == "check": check()
    if what == "check3": check3()
    if what == "h2": h2()
    if what == "check4": check4()
    if what == "fb": fb()
    if what == "hedge": hedge()
    if what == "r1":
        r1(BASE + SWITCH, "## R1. Density forecasts on identical observations (true PIT), per regime\n")
    if what == "r2":
        r1(["claygum_mix"], "## R2. Clayton-Gumbel mixture, pi(Clayton) 0.75 -> 0.25 at tick 2500, rho_S constant\n")
        weights("claygum_mix")
