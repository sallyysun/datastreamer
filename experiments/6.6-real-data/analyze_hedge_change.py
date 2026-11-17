"""Dependence changes and Hedge weighting (practical application; t4 copula). From saved per-day outputs; no reruns.
Classification: D_t = |rho99_{t-1} - rho99_{t-22}|, where rho99_{t-1} is the exact Spearman of the 99 standardised returns up
to day t-1 (saved as rho_roll_99 on row t, i.e. held before day t). On each pair's own trading days (rows), the shift is 21
rows; the series is computed on the pair's full history, then restricted to evaluation days (t >= 560, 2008 onward). The top
10% of D_t is taken separately for each pair. Equal-weight mixture: log((1/5) sum_k exp(lt_fixed_k)), i.e. the exact
equal-weight mixture of the five arms' t4 copula densities. Pair-equal weighting; 21-day moving-block bootstrap over
calendar days (blocks shared by all pairs), 2000 replicates, default_rng(0), 95% percentile intervals.
Usage: python3 analyze_hedge_change.py <scratch dir>"""
import os, sys, glob
import numpy as np, pandas as pd
S = sys.argv[1]; HERE = os.path.dirname(os.path.abspath(__file__))
FIX = [39, 99, 199, 399, 1999]; LAG = 21
EST = [("RMPL", "practical_rmpl"), ("Counting matrix", "practical_xiao"), ("Hermite series", "practical_hermite")]
out = []
def pr(s=""): print(s); out.append(s)
# ---------------------------------------------------------------- classification inputs
roll_full = {os.path.basename(f)[:-4]: pd.read_csv(f, parse_dates=["date"]) for f in sorted(glob.glob(f"{S}/practical_roll2/out/*.csv"))}
pairs = list(roll_full)
def evaldays(x): x = x.iloc[560:]; return x[x.date >= "2008-01-01"].set_index("date")
idx = pd.DatetimeIndex(sorted(set().union(*[evaldays(x).index for x in roll_full.values()])))
def change_cube(W, lag=LAG):
    return np.stack([evaldays(x.assign(D=(x[f"rho_roll_{W}"] - x[f"rho_roll_{W}"].shift(lag)).abs())).D.reindex(idx).values for x in roll_full.values()])
def classify(D, q=0.90, per_pair=True):
    present = ~np.isnan(D)
    thr = np.nanquantile(D, q, axis=1, keepdims=True) if per_pair else np.nanquantile(D, q)
    return (D >= thr) & present, present
# verification of indexing on one pair
x = roll_full["JPM_BAC"]; t = 1000
pr("CHECK indexing (JPM_BAC, row t=1000): rho_roll_99 on row t is the Spearman of rows t-99..t-1 (held before day t):")
from scipy.stats import spearmanr
xin = pd.read_csv(f"{S}/rerun/in/JPM_BAC.csv")
pr(f"   saved rho_roll_99[t] = {x.rho_roll_99[t]:.8f}; spearman(rows {t-99}..{t-1}) = {spearmanr(xin.x[t-99:t], xin.y[t-99:t]).correlation:.8f}")
pr(f"   D_t uses rho_roll_99[t] - rho_roll_99[t-21] = rho_(t-1) - rho_(t-22) in the task's notation; shift is in the pair's own trading days")
D99 = change_cube(99); CHG, PRESENT = classify(D99)
thr = np.nanquantile(D99, 0.9, axis=1)
pr(f"   threshold per pair: median {np.median(thr):.3f}, range {thr.min():.3f}-{thr.max():.3f} (SPY/IVV {thr[pairs.index('SPY_IVV')]:.4f}); "
   f"change pair-days {int(CHG.sum())} of {int(PRESENT.sum())}")
# ---------------------------------------------------------------- loss cubes
L_, B = 21, 2000
def stat(Dm, M):
    X = np.where(M, Dm, np.nan); ok = ~np.isnan(X); Tn = X.shape[1]; Xz = np.nan_to_num(X)
    Cs = np.concatenate([np.zeros((len(pairs), 1)), np.cumsum(Xz, 1)], 1); Cn = np.concatenate([np.zeros((len(pairs), 1)), np.cumsum(ok, 1)], 1)
    has = Cn[:, -1] > 0   # pairs with at least one day in the group (all pairs, except under the pooled threshold)
    est = 1e3 * np.mean(Cs[has, -1] / Cn[has, -1]); rng = np.random.default_rng(0); nb = int(np.ceil(Tn / L_)); last = Tn - (nb - 1) * L_; bs = []
    for b in range(B):
        st = rng.integers(0, Tn - L_ + 1, nb); en = st + L_; en[-1] = st[-1] + last; n = (Cn[:, en] - Cn[:, st]).sum(1); k = n > 0
        bs.append(1e3 * np.mean((Cs[k][:, en] - Cs[k][:, st]).sum(1) / n[k]))
    lo, hi = np.percentile(bs, [2.5, 97.5]); return est, lo, hi, int(((Cs[has, -1] / Cn[has, -1]) < 0).sum())
fmt = lambda r: f"{r[0]:+6.2f} [{r[1]:+6.2f},{r[2]:+6.2f}]{'*' if r[1] > 0 or r[2] < 0 else ' '} ({r[3]:2d}/32)"
data = {}
for e, d in EST:
    P = {os.path.basename(f)[:-4]: evaldays(pd.read_csv(f, parse_dates=["date"])) for f in sorted(glob.glob(f"{S}/{d}/out/*.csv"))}
    cub = lambda c: np.stack([P[p][c].reindex(idx).values for p in pairs])
    Lk = np.stack([-cub(f"lt_fixed_{k}") for k in FIX])                         # (5, pairs, T) losses
    Wt = np.stack([cub(f"wt_{k}") for k in FIX])                                # Hedge weights held before the day
    leq = -np.log(np.mean(np.exp(-Lk), axis=0))                                  # exact equal-weight mixture loss
    data[e] = dict(Lk=Lk, W=Wt, mff=-cub("lt_mffmix_t"), vff=-cub("lt_vff"), eq=leq)
ALL = PRESENT; OTH = (~CHG) & PRESENT
rows = []
pr("\n1. MFF - VFF, paired difference in mean log loss (x1e-3 nats/day/pair; negative = MFF lower); 95% CI; pairs where MFF lower")
for e in data:
    cells = []
    for lab, M in (("full", ALL), ("change", CHG), ("other", OTH)):
        r = stat(data[e]["mff"] - data[e]["vff"], M); cells.append(f"{lab} {fmt(r)}"); rows.append(dict(analysis="mff-vff", estimator=e, days=lab, est=r[0], lo=r[1], hi=r[2]))
    pr(f"   {e:16s} " + " | ".join(cells))
pr("\n2. Hedge MFF - equal-weight mixture (same five copula densities; weights 1/5); negative = Hedge lower loss")
for e in data:
    cells = []
    for lab, M in (("full", ALL), ("change", CHG), ("other", OTH)):
        r = stat(data[e]["mff"] - data[e]["eq"], M); cells.append(f"{lab} {fmt(r)}"); rows.append(dict(analysis="hedge-equal", estimator=e, days=lab, est=r[0], lo=r[1], hi=r[2]))
    pr(f"   {e:16s} " + " | ".join(cells))
pr("   context: regret vs each pair's hindsight-best single arm, full period: " + "; ".join(
    f"{e}: Hedge {1e3*np.nanmean([np.nanmean(data[e]['mff'][i]) - min(np.nanmean(data[e]['Lk'][k, i]) for k in range(5)) for i in range(len(pairs))]):+.2f}, "
    f"equal {1e3*np.nanmean([np.nanmean(data[e]['eq'][i]) - min(np.nanmean(data[e]['Lk'][k, i]) for k in range(5)) for i in range(len(pairs))]):+.2f}" for e in data))
# ---------------------------------------------------------------- weight concentration
pr("\n3. Hedge weight concentration (weights held before the day; pair-days pooled within each group)")
for e in data:
    Wt = data[e]["W"]; neff = 1.0 / np.sum(Wt ** 2, axis=0); wmax = np.max(Wt, axis=0)
    cells = []
    for lab, M in (("all", ALL), ("change", CHG), ("other", OTH)):
        cells.append(f"{lab}: N_active median {np.nanmedian(neff[M]):.2f} (mean {np.nanmean(neff[M]):.2f}); max weight > 0.9 on {np.nanmean(wmax[M] > 0.9):.0%}, > 0.5 on {np.nanmean(wmax[M] > 0.5):.0%}")
    pr(f"   {e:16s} " + " | ".join(cells))
    frac_floor = np.nanmean(Wt[:, ALL] < 1e-3)
    pr(f"   {'':16s} share of arm-pair-days with weight < 0.001: {frac_floor:.0%}; smallest weight seen {np.nanmin(Wt):.1e} (fixed-share floor delta/K = 2e-5)")
# ---------------------------------------------------------------- delayed reallocation
pr("\n4. Delayed reallocation: after an arm becomes the trailing-63-day best (lowest mean loss over the last 63 days, losses known"
   "\n   before the day), its Hedge weight at lags 0..120 days; episodes = takeovers that persist >= 21 days (median over episodes)")
for e in data:
    Lk, Wt = data[e]["Lk"], data[e]["W"]; tw = []; adv = []
    for i in range(len(pairs)):
        ok = ~np.isnan(Lk[0, i]); Li = Lk[:, i, ok]; Wi = Wt[:, i, ok]; T = Li.shape[1]
        csum = np.concatenate([np.zeros((5, 1)), np.cumsum(Li, 1)], 1)
        trail = np.full((5, T), np.nan); trail[:, 63:] = (csum[:, 63:T] - csum[:, 0:T - 63]) / 63    # mean over days t-63..t-1
        lead = np.where(np.isnan(trail[0]), -1, np.nanargmin(np.nan_to_num(trail, nan=np.inf), axis=0))
        for t in range(64, T - 121):
            if lead[t] != lead[t - 1] and lead[t] >= 0 and (lead[t:t + 21] == lead[t]).all():
                k = lead[t]; tw.append(Wi[k, t:t + 121]); adv.append(trail[lead[t - 1], t] - trail[k, t])
    tw = np.array(tw); lags = [0, 5, 10, 21, 42, 63, 120]
    pr(f"   {e:16s} episodes {len(tw)}; median weight of the new trailing-best arm at lag " + ", ".join(f"{l}: {np.median(tw[:, l]):.2f}" for l in lags)
       + f"; share of episodes where it starts below 0.05: {np.mean(tw[:, 0] < 0.05):.0%}; median days until weight > 0.5: "
       + (f"{np.median([np.argmax(w > 0.5) if (w > 0.5).any() else 121 for w in tw]):.0f}" if len(tw) else "-"))
# ---------------------------------------------------------------- sensitivity
pr("\n5. Sensitivity of the change classification: MFF - VFF and Hedge - equal on change days / other days (RMPL; counting matrix in brackets)")
for lab, W, q in (("W=99, top 10% (primary)", 99, 0.90), ("W=99, top 5%", 99, 0.95), ("W=99, top 20%", 99, 0.80), ("W=39, top 10%", 39, 0.90),
                  ("W=199, top 10%", 199, 0.90)):
    C, Pm = classify(change_cube(W), q); O = (~C) & Pm
    a = [stat(data[e]["mff"] - data[e]["vff"], M)[0] for e in ("RMPL", "Counting matrix") for M in (C, O)]
    b = [stat(data[e]["mff"] - data[e]["eq"], M)[0] for e in ("RMPL", "Counting matrix") for M in (C, O)]
    ov = np.mean(C[CHG]) if C is not CHG else 1.0
    pr(f"   {lab:24s} MFF-VFF change {a[0]:+6.2f} ({a[2]:+6.2f}) other {a[1]:+6.2f} ({a[3]:+6.2f}) | Hedge-equal change {b[0]:+6.2f} ({b[2]:+6.2f}) other {b[1]:+6.2f} ({b[3]:+6.2f}) | share of primary change days included {ov:.0%}")
Cg, Pg = classify(D99, 0.90, per_pair=False); Og = (~Cg) & Pg
pr(f"   pooled threshold {np.nanquantile(D99, 0.9):.3f}: pairs with no change days {int((Cg.sum(1) == 0).sum())}; change share by pair {Cg.sum(1).min() / PRESENT.sum(1).max():.0%}-{(Cg.sum(1) / PRESENT.sum(1)).max():.0%}")
pr(f"   {'W=99, top 10% pooled':24s} MFF-VFF change {stat(data['RMPL']['mff'] - data['RMPL']['vff'], Cg)[0]:+6.2f} other {stat(data['RMPL']['mff'] - data['RMPL']['vff'], Og)[0]:+6.2f} | "
   f"Hedge-equal change {stat(data['RMPL']['mff'] - data['RMPL']['eq'], Cg)[0]:+6.2f} other {stat(data['RMPL']['mff'] - data['RMPL']['eq'], Og)[0]:+6.2f}  (threshold across all pairs; RMPL)")
pd.DataFrame(rows).to_csv(os.path.join(HERE, "results", "hedge_change_results.csv"), index=False)
open(os.path.join(HERE, "results", "hedge_change_results.txt"), "w").write("\n".join(out) + "\n")
