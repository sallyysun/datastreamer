"""Section 6.6 tables: real-data MFF and VFF against the best fixed-arm switching paths, and how Hedge weights and losses
change on large dependence shifts. Usage: python3 analyse.py <workdir>"""
import itertools
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from real_common import EST, EVAL_FROM, EVAL_ROW, FIX, eval_days, outputs, read, result, write_report
from report_output import reports_enabled

SWITCH_LIMITS = [0, 1, 2, 5]
SWITCH_ESTIMATORS = {"RMPL": "practical_rmpl", "Counting matrix": "practical_xiao", "Hermite": "practical_hermite"}
FORECAST_COLUMNS = {"t4": ("lt_", "lt_mffmix_t", "lt_mff_t"), "Gaussian": ("lg_", "lg_mffmix_g", "lg_mff_g")}
METHODS = ("MFF", "VFF", "MFF single density")
CHANGE_WINDOW, CHANGE_LAG, CHANGE_QUANTILE = 99, 21, 0.90
BOOT_BLOCK, BOOT_DRAWS = 21, 2000
DAY_GROUPS = ("full", "change", "other")
TRAIL, HOLD, HORIZON = 63, 21, 121
REPORT_LAGS = [0, 5, 10, 21, 42, 63, 120]


def best_switching_sequence(losses, max_switches):
    """Lowest total loss over arm sequences with at most max_switches switches, and that sequence (losses: arms x days)."""
    n_arms, n_days = losses.shape
    arms = np.arange(n_arms)
    cost = np.full((max_switches + 1, n_arms), np.inf)
    cost[0] = losses[:, 0]
    back = np.zeros((n_days, max_switches + 1, n_arms, 2), dtype=np.int16)
    for t in range(1, n_days):
        new_cost = np.full((max_switches + 1, n_arms), np.inf)
        for j in range(max_switches + 1):
            stay = cost[j]
            if j > 0:
                first, second = np.argsort(cost[j - 1])[:2]
                switch_from = np.where(arms == first, second, first)
                switch = cost[j - 1][switch_from]
            else:
                switch, switch_from = np.full(n_arms, np.inf), np.zeros(n_arms, int)
            switched = switch < stay
            new_cost[j] = losses[:, t] + np.where(switched, switch, stay)
            back[t, j, :, 0] = np.where(switched, j - 1, j)
            back[t, j, :, 1] = np.where(switched, switch_from, arms)
        cost = new_cost
    j, k = np.unravel_index(np.argmin(cost), cost.shape)
    total, path = cost[j, k], np.empty(n_days, int)
    for t in range(n_days - 1, -1, -1):
        path[t] = k
        if t > 0:
            j, k = back[t, j, k]
    return total, path


def check_against_brute_force():
    rng = np.random.default_rng(1)
    n_days = 7
    for _ in range(200):
        losses = rng.normal(size=(3, n_days))
        max_switches = rng.integers(0, 3)
        best = min(sum(losses[s[t], t] for t in range(n_days)) for s in itertools.product(range(3), repeat=n_days)
                   if sum(s[t] != s[t - 1] for t in range(1, n_days)) <= max_switches)
        total, path = best_switching_sequence(losses, max_switches)
        assert abs(total - best) < 1e-9
        assert abs(losses[path, np.arange(n_days)].sum() - total) < 1e-9
        assert (np.diff(path) != 0).sum() <= max_switches


def switching_regrets(workdir):
    """Per pair, each method's regret against the best sequence with at most m switches; the oracle paths for m = 1, 2."""
    regrets, oracle_paths = [], []
    for estimator, folder in SWITCH_ESTIMATORS.items():
        for pair, path_csv in outputs(workdir, folder).items():
            days = eval_days(read(path_csv)).reset_index(drop=True)
            n_days = len(days)
            for copula, (prefix, mixture, single) in FORECAST_COLUMNS.items():
                arm_losses = -days[[f"{prefix}fixed_{k}" for k in FIX]].values.T
                static = arm_losses.sum(1).min()
                for m in SWITCH_LIMITS:
                    total, path = best_switching_sequence(arm_losses, m)
                    assert (np.diff(path) != 0).sum() <= m and abs(arm_losses[path, np.arange(n_days)].sum() - total) < 1e-6
                    assert m > 0 or abs(total - static) < 1e-6
                    for method, column in zip(METHODS, (mixture, f"{prefix}vff", single)):
                        regret = 1e3 * (-days[column].values.sum() - total) / n_days
                        regrets.append(dict(estimator=estimator, copula=copula, m=m, method=method, pair=pair, regret=regret))
                    if m in (1, 2):
                        switch_days = np.flatnonzero(np.diff(path) != 0) + 1
                        oracle_paths.append(dict(
                            estimator=estimator, copula=copula, m=m, pair=pair, switches=len(switch_days),
                            switch_dates=";".join(str(days.date.iloc[i].date()) for i in switch_days),
                            segments="->".join(str(FIX[s[0]]) for s in np.split(path, switch_days)),
                            gain_vs_static=1e3 * (static - total) / n_days))
    return pd.DataFrame(regrets), pd.DataFrame(oracle_paths)


def switching_report(regrets, oracle_paths):
    columns = f"{'Estimator':18s} {'Method':19s}" + "".join(f"{f'm={m}':>9s}" for m in SWITCH_LIMITS) + "   Negative at m=1"
    out = ["Switching regret vs the best fixed-arm sequence with ≤ m switches (×10⁻³ nats/day/pair, mean over 32 pairs)", "", columns]
    for copula in ("t4", "Gaussian"):
        out += ["", copula]
        for estimator, method in itertools.product(SWITCH_ESTIMATORS, METHODS):
            g = regrets[(regrets.estimator == estimator) & (regrets.copula == copula) & (regrets.method == method)]
            means = "".join(f" {g[g.m == m].regret.mean():+8.2f}" for m in SWITCH_LIMITS)
            negative = int((g[g.m == 1].regret < 0).sum())
            out.append(f"  {estimator:16s} {method:19s}" + means + f"   {negative}/32")
    out.append("\nStudent t oracle sequences")
    t4 = oracle_paths[oracle_paths.copula == "t4"]
    for estimator in SWITCH_ESTIMATORS:
        for m in (1, 2):
            g = t4[(t4.estimator == estimator) & (t4.m == m)]
            common = g.segments.value_counts().head(4).to_dict()
            out.append(f"  {estimator:16s} m={m}: pairs using all {m} switch(es) {int((g.switches == m).sum())}/32; oracle gain over static "
                       f"{g.gain_vs_static.mean():.2f} (median {g.gain_vs_static.median():.2f}); most common sequences {common}")
        one = t4[(t4.estimator == estimator) & (t4.m == 1) & (t4.switches == 1)]
        years = pd.to_datetime(one.switch_dates).dt.year.value_counts().sort_index().to_dict()
        out.append(f"  {'':16s} m=1 switch years: {years}")
    return out


def eval_days_by_date(x):
    x = x.iloc[EVAL_ROW:]
    return x[x.date >= EVAL_FROM].set_index("date")


class PairDays:
    """Pair x calendar-day arrays on the union of the pairs' evaluation days."""

    def __init__(self, rolling):
        self.rolling = rolling
        self.pairs = list(rolling)
        self.dates = pd.DatetimeIndex(sorted(set().union(*[eval_days_by_date(x).index for x in rolling.values()])))

    def change(self, window, lag=CHANGE_LAG):
        """|rho_roll_window[t] - rho_roll_window[t - lag]|, lagged in each pair's own trading days."""
        def size(x):
            rho = x[f"rho_roll_{window}"]
            return eval_days_by_date(x.assign(D=(rho - rho.shift(lag)).abs())).D.reindex(self.dates).values
        return np.stack([size(x) for x in self.rolling.values()])

    def stack(self, by_pair, column):
        return np.stack([by_pair[p][column].reindex(self.dates).values for p in self.pairs])


def change_days(change, quantile=CHANGE_QUANTILE, per_pair=True):
    present = ~np.isnan(change)
    threshold = np.nanquantile(change, quantile, axis=1, keepdims=True) if per_pair else np.nanquantile(change, quantile)
    return (change >= threshold) & present, present


def pair_mean_with_ci(difference, mask):
    """Pair-equal mean of difference over mask (x 1e3), its moving-block bootstrap 95% interval, and pairs with a negative mean."""
    values = np.where(mask, difference, np.nan)
    present = ~np.isnan(values)
    n_pairs, n_days = values.shape
    sums = np.concatenate([np.zeros((n_pairs, 1)), np.cumsum(np.nan_to_num(values), 1)], 1)
    counts = np.concatenate([np.zeros((n_pairs, 1)), np.cumsum(present, 1)], 1)
    has_days = counts[:, -1] > 0
    pair_means = sums[has_days, -1] / counts[has_days, -1]
    rng = np.random.default_rng(0)
    n_blocks = int(np.ceil(n_days / BOOT_BLOCK))
    last_block = n_days - (n_blocks - 1) * BOOT_BLOCK
    draws = []
    for _ in range(BOOT_DRAWS):
        starts = rng.integers(0, n_days - BOOT_BLOCK + 1, n_blocks)
        ends = starts + BOOT_BLOCK
        ends[-1] = starts[-1] + last_block
        n = (counts[:, ends] - counts[:, starts]).sum(1)
        k = n > 0
        draws.append(1e3 * np.mean((sums[k][:, ends] - sums[k][:, starts]).sum(1) / n[k]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return 1e3 * np.mean(pair_means), lo, hi, int((pair_means < 0).sum())


def format_ci(r):
    estimate, lo, hi, negative = r
    star = "*" if lo > 0 or hi < 0 else " "
    return f"{estimate:+6.2f} [{lo:+6.2f},{hi:+6.2f}]{star} ({negative:2d}/32)"


def indexing_check(workdir, days, change, is_change, present):
    x, t = days.rolling["JPM_BAC"], 1000
    inputs = pd.read_csv(f"{workdir}/rerun/in/JPM_BAC.csv")
    spearman = spearmanr(inputs.x[t - 99:t], inputs.y[t - 99:t]).correlation
    threshold = np.nanquantile(change, CHANGE_QUANTILE, axis=1)
    spy = threshold[days.pairs.index("SPY_IVV")]
    return [f"CHECK JPM_BAC t=1000: rho_roll_99[t] = {x.rho_roll_99[t]:.8f}, Spearman of rows {t-99}..{t-1} = "
            f"{spearman:.8f}; D_t lags 21 of the pair's own trading days",
            f"   threshold per pair: median {np.median(threshold):.3f}, range {threshold.min():.3f}-{threshold.max():.3f} (SPY/IVV {spy:.4f}); "
            f"change pair-days {int(is_change.sum())} of {int(present.sum())}"]


def pair_day_losses(workdir, days):
    """Per estimator: fixed-arm losses and Hedge weights (arm x pair x day), and the MFF, VFF and equal-weight losses."""
    data = {}
    for label, folder in EST:
        by_pair = {p: eval_days_by_date(read(f)) for p, f in outputs(workdir, folder).items()}
        arms = np.stack([-days.stack(by_pair, f"lt_fixed_{k}") for k in FIX])
        data[label] = dict(arms=arms, weights=np.stack([days.stack(by_pair, f"wt_{k}") for k in FIX]),
                           mff=-days.stack(by_pair, "lt_mffmix_t"), vff=-days.stack(by_pair, "lt_vff"),
                           equal=-np.log(np.mean(np.exp(-arms), axis=0)))
    return data


def comparisons(data, masks, rows):
    out = ["", "Paired log-loss differences (×10⁻³ nats/day/pair, 95% CI, pairs with lower MFF loss)"]
    for analysis, title, other in [("mff-vff", "MFF − VFF", "vff"),
                                   ("hedge-equal", "Hedge MFF − equal weights (same five densities)", "equal")]:
        out.append(f"\n{title}")
        for label, d in data.items():
            cells = []
            for group, mask in zip(DAY_GROUPS, masks):
                r = pair_mean_with_ci(d["mff"] - d[other], mask)
                cells.append(f"{group} {format_ci(r)}")
                rows.append(dict(analysis=analysis, estimator=label, days=group, est=r[0], lo=r[1], hi=r[2]))
            out.append(f"   {label:16s} " + " | ".join(cells))
    n_pairs = data["RMPL"]["arms"].shape[1]

    def regret(d, key):
        best_arm = [min(np.nanmean(d["arms"][k, i]) for k in range(len(FIX))) for i in range(n_pairs)]
        return 1e3 * np.nanmean([np.nanmean(d[key][i]) - best_arm[i] for i in range(n_pairs)])
    out.append("   context: regret vs each pair's hindsight-best single arm, full period: "
               + "; ".join(f"{label}: Hedge {regret(d, 'mff'):+.2f}, equal {regret(d, 'equal'):+.2f}" for label, d in data.items()))
    return out


def concentration(data, masks):
    out = ["\nHedge weight concentration (pooled pair-days, weights held before the day)"]
    for label, d in data.items():
        weights = d["weights"]
        active, top = 1.0 / np.sum(weights ** 2, axis=0), np.max(weights, axis=0)
        cells = [f"{group}: N_active median {np.nanmedian(active[m]):.2f} (mean {np.nanmean(active[m]):.2f}); "
                 f"max weight > 0.9 on {np.nanmean(top[m] > 0.9):.0%}, > 0.5 on {np.nanmean(top[m] > 0.5):.0%}"
                 for group, m in zip(("all", "change", "other"), masks)]
        out += [f"   {label:16s} " + " | ".join(cells),
                f"   {'':16s} share of arm-pair-days with weight < 0.001: {np.nanmean(weights[:, masks[0]] < 1e-3):.0%}; "
                f"smallest weight seen {np.nanmin(weights):.1e} (fixed-share floor delta/K = 2e-5)"]
    return out


def takeovers(arms, weights):
    """Weight paths (lags 0..HORIZON-1) of an arm that becomes the trailing-TRAIL-day best and stays best for HOLD days."""
    paths = []
    n_arms = len(FIX)
    for pair in range(arms.shape[1]):
        present = ~np.isnan(arms[0, pair])
        losses, w = arms[:, pair, present], weights[:, pair, present]
        n_days = losses.shape[1]
        cumulative = np.concatenate([np.zeros((n_arms, 1)), np.cumsum(losses, 1)], 1)
        trailing = np.full((n_arms, n_days), np.nan)
        trailing[:, TRAIL:] = (cumulative[:, TRAIL:n_days] - cumulative[:, 0:n_days - TRAIL]) / TRAIL
        leader = np.where(np.isnan(trailing[0]), -1, np.nanargmin(np.nan_to_num(trailing, nan=np.inf), axis=0))
        for t in range(TRAIL + 1, n_days - HORIZON):
            new = leader[t] != leader[t - 1] and leader[t] >= 0
            if new and (leader[t:t + HOLD] == leader[t]).all():
                paths.append(w[leader[t], t:t + HORIZON])
    return np.array(paths)


def reallocation(data):
    out = ["\nHedge weights after a change in the trailing-63-day leader (new leader best for ≥ 21 days; lags in trading days)"]
    for label, d in data.items():
        paths = takeovers(d["arms"], d["weights"])
        half = [np.argmax(w > 0.5) if (w > 0.5).any() else HORIZON for w in paths]
        days_to_half = f"{np.median(half):.0f}" if len(paths) else "-"
        at_lags = ", ".join(f"{lag}: {np.median(paths[:, lag]):.2f}" for lag in REPORT_LAGS)
        out.append(f"   {label:16s} episodes {len(paths)}; median weight of the new trailing-best arm at lag " + at_lags
                   + f"; share of episodes where it starts below 0.05: {np.mean(paths[:, 0] < 0.05):.0%}; median days until weight > 0.5: " + days_to_half)
    return out


def change_vs_other(data, estimators, other, is_change, rest):
    return [pair_mean_with_ci(data[e]["mff"] - data[e][other], m)[0] for e in estimators for m in (is_change, rest)]


def sensitivity(data, days, change, is_change, present):
    out = ["\nChange-day sensitivity (RMPL, counting matrix in brackets)"]
    variants = (("W=99, top 10% (primary)", 99, 0.90), ("W=99, top 5%", 99, 0.95), ("W=99, top 20%", 99, 0.80),
                ("W=39, top 10%", 39, 0.90), ("W=199, top 10%", 199, 0.90))
    pair = ("RMPL", "Counting matrix")
    for label, window, quantile in variants:
        chg, pres = change_days(days.change(window), quantile)
        rest = (~chg) & pres
        a = change_vs_other(data, pair, "vff", chg, rest)
        b = change_vs_other(data, pair, "equal", chg, rest)
        out.append(f"   {label:24s} MFF-VFF change {a[0]:+6.2f} ({a[2]:+6.2f}) other {a[1]:+6.2f} ({a[3]:+6.2f}) | Hedge-equal change {b[0]:+6.2f} "
                   f"({b[2]:+6.2f}) other {b[1]:+6.2f} ({b[3]:+6.2f}) | share of primary change days included {np.mean(chg[is_change]):.0%}")
    pooled, pooled_present = change_days(change, CHANGE_QUANTILE, per_pair=False)
    a = change_vs_other(data, ("RMPL",), "vff", pooled, (~pooled) & pooled_present)
    b = change_vs_other(data, ("RMPL",), "equal", pooled, (~pooled) & pooled_present)
    counts = pooled.sum(1)
    out += [f"   pooled threshold {np.nanquantile(change, CHANGE_QUANTILE):.3f}: pairs with no change days {int((counts == 0).sum())}; "
            f"change share by pair {counts.min() / present.sum(1).max():.0%}-{(counts / present.sum(1)).max():.0%}",
            f"   {'W=99, top 10% pooled':24s} MFF-VFF change {a[0]:+6.2f} other {a[1]:+6.2f} | "
            f"Hedge-equal change {b[0]:+6.2f} other {b[1]:+6.2f}  (threshold across all pairs; RMPL)"]
    return out


def switching(workdir):
    check_against_brute_force()
    regrets, oracle_paths = switching_regrets(workdir)
    regrets.to_csv(result("switching_per_pair.csv"), index=False)
    oracle_paths.to_csv(result("switching_paths.csv"), index=False)
    if reports_enabled():
        write_report(result("switching_results.txt"), switching_report(regrets, oracle_paths))


def hedge_change(workdir):
    days = PairDays({p: read(f) for p, f in outputs(workdir, "practical_roll2").items()})
    change = days.change(CHANGE_WINDOW)
    is_change, present = change_days(change)
    data = pair_day_losses(workdir, days)
    masks = (present, is_change, (~is_change) & present)
    rows = []
    comparison_lines = comparisons(data, masks, rows)
    pd.DataFrame(rows).to_csv(result("hedge_change_results.csv"), index=False)
    if reports_enabled():
        lines = indexing_check(workdir, days, change, is_change, present) + comparison_lines
        lines += concentration(data, masks) + reallocation(data) + sensitivity(data, days, change, is_change, present)
        write_report(result("hedge_change_results.txt"), lines)


def main():
    switching(sys.argv[1])
    hedge_change(sys.argv[1])


if __name__ == "__main__":
    main()
