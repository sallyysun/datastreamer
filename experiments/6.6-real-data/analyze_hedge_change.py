"""Analyze how Hedge weights and losses change on large dependence shifts."""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from real_common import EST, EVAL_FROM, EVAL_ROW, FIX, outputs, read, result, write_report

LAG, BLOCK, BOOT = 21, 21, 2000
GROUPS = ("full", "change", "other")


def evaldays(x):
    x = x.iloc[EVAL_ROW:]
    return x[x.date >= EVAL_FROM].set_index("date")


class Days:
    """Pair x calendar-day cubes on the union of the pairs' evaluation days."""

    def __init__(self, roll):
        self.roll = roll
        self.pairs = list(roll)
        self.idx = pd.DatetimeIndex(sorted(set().union(*[evaldays(x).index for x in roll.values()])))

    def change(self, W, lag=LAG):
        return np.stack([evaldays(x.assign(D=(x[f"rho_roll_{W}"] - x[f"rho_roll_{W}"].shift(lag)).abs())).D.reindex(self.idx).values
                         for x in self.roll.values()])

    def cube(self, P, c):
        return np.stack([P[p][c].reindex(self.idx).values for p in self.pairs])


def classify(D, q=0.90, per_pair=True):
    present = ~np.isnan(D)
    thr = np.nanquantile(D, q, axis=1, keepdims=True) if per_pair else np.nanquantile(D, q)
    return (D >= thr) & present, present


def stat(Dm, M):
    """Pair-equal mean of Dm over mask M (x 1e3), its moving-block bootstrap interval, and the pairs with a negative mean."""
    X = np.where(M, Dm, np.nan)
    ok = ~np.isnan(X)
    n_pairs, Tn = X.shape
    Cs = np.concatenate([np.zeros((n_pairs, 1)), np.cumsum(np.nan_to_num(X), 1)], 1)
    Cn = np.concatenate([np.zeros((n_pairs, 1)), np.cumsum(ok, 1)], 1)
    has = Cn[:, -1] > 0
    est = 1e3 * np.mean(Cs[has, -1] / Cn[has, -1])
    rng = np.random.default_rng(0)
    nb = int(np.ceil(Tn / BLOCK))
    last = Tn - (nb - 1) * BLOCK
    bs = []
    for _ in range(BOOT):
        st = rng.integers(0, Tn - BLOCK + 1, nb)
        en = st + BLOCK
        en[-1] = st[-1] + last
        n = (Cn[:, en] - Cn[:, st]).sum(1)
        k = n > 0
        bs.append(1e3 * np.mean((Cs[k][:, en] - Cs[k][:, st]).sum(1) / n[k]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi, int(((Cs[has, -1] / Cn[has, -1]) < 0).sum())


def fmt(r):
    return f"{r[0]:+6.2f} [{r[1]:+6.2f},{r[2]:+6.2f}]{'*' if r[1] > 0 or r[2] < 0 else ' '} ({r[3]:2d}/32)"


def indexing_check(S, days, D99, CHG, PRESENT):
    x, t = days.roll["JPM_BAC"], 1000
    xin = pd.read_csv(f"{S}/rerun/in/JPM_BAC.csv")
    thr = np.nanquantile(D99, 0.9, axis=1)
    return ["CHECK indexing (JPM_BAC, row t=1000): rho_roll_99 on row t is the Spearman of rows t-99..t-1 (held before day t):",
            f"   saved rho_roll_99[t] = {x.rho_roll_99[t]:.8f}; spearman(rows {t-99}..{t-1}) = {spearmanr(xin.x[t-99:t], xin.y[t-99:t]).correlation:.8f}",
            "   D_t uses rho_roll_99[t] - rho_roll_99[t-21] = rho_(t-1) - rho_(t-22) in the task's notation; shift is in the pair's own trading days",
            f"   threshold per pair: median {np.median(thr):.3f}, range {thr.min():.3f}-{thr.max():.3f} (SPY/IVV {thr[days.pairs.index('SPY_IVV')]:.4f}); "
            f"change pair-days {int(CHG.sum())} of {int(PRESENT.sum())}"]


def losses(S, days):
    data = {}
    for e, d in EST:
        P = {p: evaldays(read(f)) for p, f in outputs(S, d).items()}
        Lk = np.stack([-days.cube(P, f"lt_fixed_{k}") for k in FIX])
        data[e] = dict(Lk=Lk, W=np.stack([days.cube(P, f"wt_{k}") for k in FIX]), mff=-days.cube(P, "lt_mffmix_t"),
                       vff=-days.cube(P, "lt_vff"), eq=-np.log(np.mean(np.exp(-Lk), axis=0)))
    return data


def comparisons(data, masks, rows):
    out = []
    for n, (analysis, title, other) in enumerate([
            ("mff-vff", "MFF - VFF, paired difference in mean log loss (x1e-3 nats/day/pair; negative = MFF lower); 95% CI; pairs where MFF lower", "vff"),
            ("hedge-equal", "Hedge MFF - equal-weight mixture (same five copula densities; weights 1/5); negative = Hedge lower loss", "eq")], 1):
        out.append(f"\n{n}. {title}")
        for e in data:
            cells = []
            for lab, M in zip(GROUPS, masks):
                r = stat(data[e]["mff"] - data[e][other], M)
                cells.append(f"{lab} {fmt(r)}")
                rows.append(dict(analysis=analysis, estimator=e, days=lab, est=r[0], lo=r[1], hi=r[2]))
            out.append(f"   {e:16s} " + " | ".join(cells))
    n_pairs = data["RMPL"]["Lk"].shape[1]

    def regret(e, key):
        D = data[e]
        return 1e3 * np.nanmean([np.nanmean(D[key][i]) - min(np.nanmean(D["Lk"][k, i]) for k in range(5)) for i in range(n_pairs)])
    out.append("   context: regret vs each pair's hindsight-best single arm, full period: "
               + "; ".join(f"{e}: Hedge {regret(e, 'mff'):+.2f}, equal {regret(e, 'eq'):+.2f}" for e in data))
    return out


def concentration(data, masks):
    out = ["\n3. Hedge weight concentration (weights held before the day; pair-days pooled within each group)"]
    for e in data:
        Wt = data[e]["W"]
        neff, wmax = 1.0 / np.sum(Wt ** 2, axis=0), np.max(Wt, axis=0)
        cells = [f"{lab}: N_active median {np.nanmedian(neff[M]):.2f} (mean {np.nanmean(neff[M]):.2f}); "
                 f"max weight > 0.9 on {np.nanmean(wmax[M] > 0.9):.0%}, > 0.5 on {np.nanmean(wmax[M] > 0.5):.0%}"
                 for lab, M in zip(("all", "change", "other"), masks)]
        out += [f"   {e:16s} " + " | ".join(cells),
                f"   {'':16s} share of arm-pair-days with weight < 0.001: {np.nanmean(Wt[:, masks[0]] < 1e-3):.0%}; "
                f"smallest weight seen {np.nanmin(Wt):.1e} (fixed-share floor delta/K = 2e-5)"]
    return out


def takeovers(Lk, Wt):
    """Hedge weight paths (lags 0..120) of an arm that becomes the trailing-63-day best and stays so for 21 days."""
    tw = []
    for i in range(Lk.shape[1]):
        ok = ~np.isnan(Lk[0, i])
        Li, Wi = Lk[:, i, ok], Wt[:, i, ok]
        T = Li.shape[1]
        csum = np.concatenate([np.zeros((5, 1)), np.cumsum(Li, 1)], 1)
        trail = np.full((5, T), np.nan)
        trail[:, 63:] = (csum[:, 63:T] - csum[:, 0:T - 63]) / 63
        lead = np.where(np.isnan(trail[0]), -1, np.nanargmin(np.nan_to_num(trail, nan=np.inf), axis=0))
        for t in range(64, T - 121):
            if lead[t] != lead[t - 1] and lead[t] >= 0 and (lead[t:t + 21] == lead[t]).all():
                tw.append(Wi[lead[t], t:t + 121])
    return np.array(tw)


def reallocation(data):
    out = ["\n4. Delayed reallocation: after an arm becomes the trailing-63-day best (lowest mean loss over the last 63 days, losses known"
           "\n   before the day), its Hedge weight at lags 0..120 days; episodes = takeovers that persist >= 21 days (median over episodes)"]
    for e in data:
        tw = takeovers(data[e]["Lk"], data[e]["W"])
        days_to_half = f"{np.median([np.argmax(w > 0.5) if (w > 0.5).any() else 121 for w in tw]):.0f}" if len(tw) else "-"
        out.append(f"   {e:16s} episodes {len(tw)}; median weight of the new trailing-best arm at lag "
                   + ", ".join(f"{l}: {np.median(tw[:, l]):.2f}" for l in [0, 5, 10, 21, 42, 63, 120])
                   + f"; share of episodes where it starts below 0.05: {np.mean(tw[:, 0] < 0.05):.0%}; median days until weight > 0.5: " + days_to_half)
    return out


def sensitivity(data, days, D99, CHG, PRESENT):
    out = ["\n5. Sensitivity of the change classification: MFF - VFF and Hedge - equal on change days / other days (RMPL; counting matrix in brackets)"]
    for lab, W, q in (("W=99, top 10% (primary)", 99, 0.90), ("W=99, top 5%", 99, 0.95), ("W=99, top 20%", 99, 0.80),
                      ("W=39, top 10%", 39, 0.90), ("W=199, top 10%", 199, 0.90)):
        C, Pm = classify(days.change(W), q)
        O = (~C) & Pm
        a = [stat(data[e]["mff"] - data[e]["vff"], M)[0] for e in ("RMPL", "Counting matrix") for M in (C, O)]
        b = [stat(data[e]["mff"] - data[e]["eq"], M)[0] for e in ("RMPL", "Counting matrix") for M in (C, O)]
        out.append(f"   {lab:24s} MFF-VFF change {a[0]:+6.2f} ({a[2]:+6.2f}) other {a[1]:+6.2f} ({a[3]:+6.2f}) | Hedge-equal change {b[0]:+6.2f} "
                   f"({b[2]:+6.2f}) other {b[1]:+6.2f} ({b[3]:+6.2f}) | share of primary change days included {np.mean(C[CHG]):.0%}")
    Cg, Pg = classify(D99, 0.90, per_pair=False)
    Og = (~Cg) & Pg
    r = data["RMPL"]
    out += [f"   pooled threshold {np.nanquantile(D99, 0.9):.3f}: pairs with no change days {int((Cg.sum(1) == 0).sum())}; "
            f"change share by pair {Cg.sum(1).min() / PRESENT.sum(1).max():.0%}-{(Cg.sum(1) / PRESENT.sum(1)).max():.0%}",
            f"   {'W=99, top 10% pooled':24s} MFF-VFF change {stat(r['mff'] - r['vff'], Cg)[0]:+6.2f} other {stat(r['mff'] - r['vff'], Og)[0]:+6.2f} | "
            f"Hedge-equal change {stat(r['mff'] - r['eq'], Cg)[0]:+6.2f} other {stat(r['mff'] - r['eq'], Og)[0]:+6.2f}  (threshold across all pairs; RMPL)"]
    return out


def main():
    S = sys.argv[1]
    days = Days({p: read(f) for p, f in outputs(S, "practical_roll2").items()})
    D99 = days.change(99)
    CHG, PRESENT = classify(D99)
    out = indexing_check(S, days, D99, CHG, PRESENT)
    data = losses(S, days)
    masks = (PRESENT, CHG, (~CHG) & PRESENT)
    rows = []
    for part in (comparisons(data, masks, rows), concentration(data, masks), reallocation(data), sensitivity(data, days, D99, CHG, PRESENT)):
        out += part
    pd.DataFrame(rows).to_csv(result("hedge_change_results.csv"), index=False)
    write_report(result("hedge_change_results.txt"), out)


if __name__ == "__main__":
    main()
