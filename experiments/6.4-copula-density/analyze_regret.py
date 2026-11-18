"""Summarize MFF regret for the Gaussian and cross-family studies."""
import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import config as C  # noqa: E402
from report_output import write_report  # noqa: E402
from summary_stats import standard_error  # noqa: E402

R = Path("results")
FAMILIES = C.FAMILIES_T31
WORKING_LABEL = {family: "RMPL " + C.FAMILY_TEXT[family] for family in FAMILIES}
ORACLES = ("best0", "best1", "best2")
K = len(C.MEMORIES)


def bound(switches, delta, ticks):
    if delta == 0:
        return math.log(K) if switches == 0 else float("nan")
    return (math.log(K) + (ticks - 1 - switches) * math.log(1 / (1 - delta + delta / K))
            + switches * math.log(K / delta))


def regret(rows, oracle, forecast="mix"):
    return rows[f"loss_{forecast}"] - rows[oracle]


def bound_summary(loss, ticks, switches, delta):
    limits = np.array([bound(switches, delta, count) for count in ticks])
    if np.isnan(limits).all():
        return None
    loss = loss.to_numpy()
    return loss.max(), limits.max(), int((loss > limits + 1e-9).sum()), len(loss)


def rows_for_delta(rows, delta):
    return rows[np.isclose(rows.delta, delta)]


def paired_updates(rows):
    fixed_share = rows_for_delta(rows, 1e-4).set_index("seed")
    hedge = rows_for_delta(rows, 0.0).set_index("seed")
    return fixed_share, hedge


def switch_counts(runs):
    return runs.groupby(["sw_from", "sw_to"]).size().sort_values(ascending=False)


def gaussian_summary(x):
    return f"{x.mean():7.2f} ± {2 * standard_error(x):5.2f} (med {x.median():6.2f})"


def regret_lines(g):
    out = []
    for d, dl in [(0.0, "Hedge      "), (1e-4, "fixed share")]:
        h = rows_for_delta(g, d)
        for m, oracle in enumerate(ORACLES):
            mix = regret(h, oracle)
            point = regret(h, oracle, "point")
            summary = bound_summary(mix, h["T"], m, d)
            violations = ""
            if summary is not None:
                _, limit, count, runs = summary
                violations = f" viol {count}/{runs} (bound {limit:.2f})"
            cells = [f"mix {gaussian_summary(mix)}{violations}", f"point {gaussian_summary(point)}"]
            out.append(f"   {dl} m={m}: " + " | ".join(cells))
    return out


def switch_line(g):
    j = g[g.delta == 0]
    sw = j[j.sw_t > 0]
    arms = ", ".join(f"{a:g}->{b:g} x{n}" for (a, b), n in switch_counts(sw).items())
    return (f"   best one-switch sequence: {len(sw)}/{len(j)} runs switch; switch time median {sw.sw_t.median():.0f} "
            f"(IQR {sw.sw_t.quantile(.25):.0f}-{sw.sw_t.quantile(.75):.0f}; true change 2500); arms: " + arms)


def gaussian_report(r):
    out = ["MFF regret (nats, cumulative over the scored ticks). Log loss, eta = 1. Gaussian copula, KLL margins, 60 seeds, Linux/libstdc++.",
           "mean ± 2 SE (median). 'viol' = runs above the Chapter 5 bound (only defined for the mixture forecaster; delta = 0 has no switching bound)."]
    for e in C.ESTIMATORS:
        out += ["", f"=== {C.EST_LABEL[e]} (T = {sorted(r[r.estimator == e]['T'].unique())}) ==="]
        for sc in C.SCENARIOS:
            g = r[(r.estimator == e) & (r.scenario == sc)]
            gain = g[g.delta == 0].best0 - g[g.delta == 0].best1
            out.append(f"-- {sc}: best fixed arm (mode) {g.arm0.mode().iloc[0]:g}; best fixed - best one-switch loss {gaussian_summary(gain)}")
            out += regret_lines(g)
            a, b = paired_updates(g)
            out.append(f"   fixed share - Hedge (paired, = difference in cumulative MFF loss): mix {gaussian_summary(a.loss_mix - b.loss_mix)} | point {gaussian_summary(a.loss_point - b.loss_point)}")
            if sc == "jump":
                out.append(switch_line(g))
    return out


def family_summary(x):
    return f"{x.mean():6.2f}±{2 * standard_error(x):4.2f}"


def switch_desc(h):
    sw = h[h.sw_t > 0]
    if not len(sw):
        return "no switch"
    top = switch_counts(sw).head(2)
    return f"{len(sw)}/60 switch, t~{sw.sw_t.median():.0f}, " + ",".join(f"{a:g}>{b:g}x{n}" for (a, b), n in top.items())


def family_table(r):
    out = ["Regret robustness (nats, cumulative over T = 4969-4970 scored ticks), mean ± 2 SE over 60 seeds, (median) where shown.",
           "FS = fixed share delta = 1e-4, H = Hedge delta = 0. stat FS = static regret of FS; sw1 = one-switch regret; FS-H = paired",
           "difference in cumulative mixture log loss; gain1 = best fixed - best one-switch comparator loss. [M] = correctly specified."]
    for sc in C.SCENARIOS:
        out += ["", f"=== {sc} ===",
                f"{'gen':9s} {'working':13s} {'stat FS':>11s} {'stat H':>11s} {'sw1 H':>11s} {'sw1 FS':>11s} {'FS - H':>12s} {'gain1':>11s}  best fixed / one-switch"]
        for gfam in FAMILIES:
            for w in FAMILIES:
                g = r[(r.gen == gfam) & (r.scenario == sc) & (r.working == w)]
                fs, h = paired_updates(g)
                wl = WORKING_LABEL[w] + (' [M]' if gfam == w else '')
                out.append(f"{C.FAMILY_TEXT[gfam]:9s} {wl:13s} {family_summary(regret(fs, 'best0')):>11s} {family_summary(regret(h, 'best0')):>11s} "
                           f"{family_summary(regret(h, 'best1')):>11s} {family_summary(regret(fs, 'best1')):>11s} {family_summary(fs.loss_mix - h.loss_mix):>12s} "
                           f"{family_summary(h.best0 - h.best1):>11s}  {h.arm0.mode().iloc[0]:g} / {switch_desc(h)}")
    return out


def bound_check(r):
    out = ["", "=== Chapter 5 bound check (every run): max regret / bound, violations ==="]
    rows = []
    for (gfam, w, d), g in r.groupby(["gen", "working", "delta"]):
        for m, oracle in enumerate(ORACLES):
            summary = bound_summary(regret(g, oracle), g["T"], m, d)
            if summary is not None:
                rows.append((gfam, w, d, m, *summary))
    t = pd.DataFrame(rows, columns=["gen", "working", "delta", "m", "max", "bound", "viol", "n"])
    for (d, m), g in t.groupby(["delta", "m"]):
        i = g["max"].idxmax()
        out.append(f"delta={d:g} m={m}: bound {g.bound.max():.2f}; max over all combos {g['max'].max():.2f} ({g.loc[i, 'gen']} x {g.loc[i, 'working']}); "
                   f"violations {g.viol.sum()}/{g.n.sum()}; max/bound by combo min {(g['max']/g.bound).min():.2f} max {(g['max']/g.bound).max():.2f}")
    return out


def numerics(r):
    c = r[r.n_clamp > 0].groupby(["gen", "working", "scenario"]).n_clamp.agg(["count", "max"])
    return ["", "=== numerics ===", f"non-finite arm log densities (guard used): {int(r.n_nonfinite.sum())} over {len(r)} run-rows",
            "scored ticks with an arm at its parameter clamp (runs, max ticks per run):", c.to_string() if len(c) else "none"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("gaussian", "families", "all"), default="gaussian")
    mode = parser.parse_args().mode

    if mode in ("gaussian", "all"):
        rows = pd.read_csv(R / "regret_seeds.csv")
        write_report(R / "regret_2026-11-14.txt", gaussian_report(rows))
    if mode in ("families", "all"):
        rows = pd.concat([pd.read_csv(R / f"regretfam_seeds{C.sfx(family)}.csv") for family in FAMILIES])
        write_report(R / "regretfam_2026-11-14.txt", family_table(rows) + bound_check(rows) + numerics(rows))


if __name__ == "__main__":
    main()
