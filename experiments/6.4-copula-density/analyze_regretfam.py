"""MFF regret over generating x working copula families (Section 6.4 appendix, tab:app-mff-regret-families), from
results/regretfam_seeds[_fam].csv (strategy_experiments REGRETFAM=1). Working model W: the five RMPL memories with
each arm's rho_S inverted into W's parameter (as in 6.5; W = Gaussian is the REGRET loss). The Chapter 5 bound is
pathwise, so it holds whatever the specification. Writes results/regretfam_2026-11-14.txt."""
import math
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "../common")
import config as C  # noqa: E402

R = "results/"
GEN = C.FAMILIES_T31
WK = C.FAMILIES_T31
WL = {f: "RMPL " + C.FAMILY_TEXT[f] for f in WK}
K = len(C.MEMORIES)


def bound(m, d, T):
    if d == 0:
        return math.log(K) if m == 0 else np.nan
    return math.log(K) + (T - 1 - m) * math.log(1 / (1 - d + d / K)) + m * math.log(K / d)


def ms(x):
    return f"{x.mean():6.2f}±{2 * x.std(ddof=1) / math.sqrt(len(x)):4.2f}"


def piv(g, d):
    return g[np.isclose(g.delta, d)].set_index("seed")


def switch_desc(h):
    sw = h[h.sw_t > 0]
    if not len(sw):
        return "no switch"
    top = sw.groupby(["sw_from", "sw_to"]).size().sort_values(ascending=False).head(2)
    return f"{len(sw)}/60 switch, t~{sw.sw_t.median():.0f}, " + ",".join(f"{a:g}>{b:g}x{n}" for (a, b), n in top.items())


def table(r):
    out = ["Regret robustness (nats, cumulative over T = 4969-4970 scored ticks), mean ± 2 SE over 60 seeds, (median) where shown.",
           "FS = fixed share delta = 1e-4, H = Hedge delta = 0. stat FS = static regret of FS; sw1 = one-switch regret; FS-H = paired",
           "difference in cumulative mixture log loss; gain1 = best fixed - best one-switch comparator loss. [M] = correctly specified."]
    for sc in C.SCENARIOS:
        out += ["", f"=== {sc} ===",
                f"{'gen':9s} {'working':13s} {'stat FS':>11s} {'stat H':>11s} {'sw1 H':>11s} {'sw1 FS':>11s} {'FS - H':>12s} {'gain1':>11s}  best fixed / one-switch"]
        for gfam in GEN:
            for w in WK:
                g = r[(r.gen == gfam) & (r.scenario == sc) & (r.working == w)]
                fs, h = piv(g, 1e-4), piv(g, 0.0)
                wl = WL[w] + (' [M]' if gfam == w else '')
                out.append(f"{C.FAMILY_TEXT[gfam]:9s} {wl:13s} {ms(fs.loss_mix - fs.best0):>11s} {ms(h.loss_mix - h.best0):>11s} "
                           f"{ms(h.loss_mix - h.best1):>11s} {ms(fs.loss_mix - fs.best1):>11s} {ms(fs.loss_mix - h.loss_mix):>12s} "
                           f"{ms(h.best0 - h.best1):>11s}  {h.arm0.mode().iloc[0]:g} / {switch_desc(h)}")
    return out


def bound_check(r):
    out = ["", "=== Chapter 5 bound check (every run): max regret / bound, violations ==="]
    rows = []
    for (gfam, w, d), g in r.groupby(["gen", "working", "delta"]):
        for m, c in [(0, "best0"), (1, "best1"), (2, "best2")]:
            b = np.array([bound(m, d, T) for T in g["T"]])
            if np.isnan(b).all():
                continue
            x = (g.loss_mix - g[c]).values
            rows.append((gfam, w, d, m, x.max(), b.max(), int((x > b + 1e-9).sum()), len(x)))
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
    r = pd.concat([pd.read_csv(R + f"regretfam_seeds{s}.csv") for s in ["", "_t", "_clayton", "_gumbel"]])
    lines = table(r) + bound_check(r) + numerics(r)
    open(R + "regretfam_2026-11-14.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
