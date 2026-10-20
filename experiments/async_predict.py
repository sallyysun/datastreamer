# Heuristic M* predictions for the async memory study, constant vs state-dependent noise.
# See notes/studies/async_memory.md ("State-dependent noise").
#
# Needs results/async_noise.csv and results/async_memory_<scenario>.csv. For each drift,
# arrival setting, estimator and arm (sync, corr), minimises the path-averaged heuristic
#   MSE(m) = mean_t [ bias_t(m)^2 + var_t(m) ]
# over the estimator's own memory kernel (EW, or flat for windowed_rank), where the kernel
# gives both the lag bias and the local average of the measured noise v_j(rho).
# Model "const" uses one v per estimator; model "state" uses v_j(rho) along the path.
#
# Usage: python3 async_predict.py

import csv, math, collections

import numpy as np

T, BURN = 20000, 2000
PATHS = {"drift": (-0.5, 0.8), "away_pos": (0.0, 0.8), "away_neg": (0.0, -0.8),
         "toward_pos": (0.8, 0.0), "toward_neg": (-0.8, 0.0)}
GROUP = {"drift": "original", "away_pos": "away", "away_neg": "away",
         "toward_pos": "toward", "toward_neg": "toward"}
ESTS = ["xiao_ew", "hermite_ew", "gaussian_mle_ew", "windowed_rank", "kalman"]
KALMAN_BLOCK = 20
DT = 10
SCORE_T = list(range(BURN, T, 250))
M_GRID = [10 * 1.04 ** i for i in range(190)]  # pairs, 10 .. ~17k


def g(r):  # Pearson -> Spearman (Gaussian)
    return 6 / math.pi * math.asin(max(-1.999, min(1.999, r)) / 2)


def gprime(r):
    return 6 / math.pi / math.sqrt(4 - r * r)


def r_of(rs):
    return 2 * math.sin(math.pi * rs / 6)


def load_noise():
    tab = collections.defaultdict(list)
    for row in csv.DictReader(open("results/async_noise.csv")):
        if float(row["m"]) == 320:
            tab[row["estimator"]].append((float(row["rho_s"]), float(row["v"])))
    return {e: sorted(v) for e, v in tab.items()}


def interp(tab, x):
    x = max(tab[0][0], min(tab[-1][0], x))
    for (x0, y0), (x1, y1) in zip(tab, tab[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return tab[-1][1]


def mstar_obs(Ms, ys):
    k = min(range(len(ys)), key=lambda i: ys[i])
    if k in (0, len(ys) - 1):
        return Ms[k]
    x = [math.log2(Ms[i]) for i in (k - 1, k, k + 1)]
    y = [ys[i] for i in (k - 1, k, k + 1)]
    den = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
    A = (x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2]) + x[0] * (y[2] - y[1])) / den
    B = (x[2] ** 2 * (y[0] - y[1]) + x[1] ** 2 * (y[2] - y[0]) + x[0] ** 2 * (y[1] - y[2])) / den
    return 2 ** min(max(-B / (2 * A), x[0]), x[2])


def predict(est, scenario, spp, cbar, arm, vfun):
    # Clock time binned at DT ticks; a bin holds DT/spp pairs, all at the bin's kernel weight.
    th0, th1 = PATHS[scenario]
    s_grid = np.arange(0, T, DT, dtype=float)
    rho_s = np.array([g(th0 + (th1 - th0) * s / T) for s in s_grid])
    v_s = np.array([vfun(r) for r in rho_s])
    t_idx = np.array([t // DT for t in SCORE_T])
    delay = (KALMAN_BLOCK / 2) * spp if est == "kalman" else 0.0
    ages = t_idx[:, None] * DT - s_grid[None, :] - delay
    valid = ages >= 0
    truth = rho_s[t_idx]
    n_per_bin = DT / spp
    best, best_m = np.inf, None
    for m in M_GRID:
        if est == "windowed_rank":
            w = (ages < m * spp).astype(float)
        else:
            w = (1 - 2 / (m + 1)) ** (np.clip(ages, 0, None) / spp)
        w = np.where(valid, w, 0.0)
        sw = (w * n_per_bin).sum(1)
        mean = (w * n_per_bin * rho_s).sum(1) / sw
        var = (w * w * n_per_bin * v_s).sum(1) / sw ** 2
        mse = np.mean((mean - truth) ** 2 + var)
        if mse < best:
            best, best_m = mse, m
    return best_m * spp  # clock ticks


def main():
    noise = load_noise()
    obs = {}
    meta = {}
    for sc in PATHS:
        cell = collections.defaultdict(list)
        for row in csv.DictReader(open(f"results/async_memory_{sc}.csv")):
            liq = (float(row["px"]), float(row["py"]))
            meta[liq] = (float(row["steps_per_pair"]), float(row["mean_c"]))
            cell[(liq, row["arm"], row["estimator"])].append((float(row["M_pairs"]), float(row["rmse_mean"])))
        for (liq, arm, e), pts in cell.items():
            if arm in ("sync", "corr"):
                obs[(sc, liq, arm, e)] = mstar_obs(*zip(*pts)) * meta[liq][0]

    out = open("results/async_predict.csv", "w", newline="")
    wr = csv.writer(out)
    wr.writerow(["scenario", "px", "py", "arm", "estimator", "obs", "pred_const", "pred_state",
                 "anch_const", "anch_state"])
    ratios = collections.defaultdict(list)
    for sc in PATHS:
        th0, th1 = PATHS[sc]
        for e in ESTS:
            tab = noise[e]
            path_v = [interp(tab, g(th0 + (th1 - th0) * t / T)) for t in SCORE_T]
            vconst = sum(path_v) / len(path_v)
            preds = {}
            for liq, (spp, cbar) in meta.items():
                for arm in ("sync", "corr"):
                    if arm == "sync":
                        vs = lambda rs: interp(tab, rs)
                        vc = lambda rs: vconst
                    else:
                        def vs(rs, cbar=cbar, tab=tab):
                            r = r_of(rs)
                            j = gprime(r) / (cbar * gprime(cbar * r))
                            return interp(tab, g(cbar * r)) * j * j
                        vc = lambda rs, cbar=cbar: vconst / cbar ** 2
                    preds[(liq, arm)] = (predict(e, sc, spp, cbar, arm, vc),
                                         predict(e, sc, spp, cbar, arm, vs))
            base_obs = obs[(sc, (1.0, 1.0), "sync", e)]
            bc, bs = preds[((1.0, 1.0), "sync")]
            for (liq, arm), (pc, ps) in preds.items():
                o = obs[(sc, liq, arm, e)]
                ac, as_ = o / (base_obs * pc / bc), o / (base_obs * ps / bs)
                wr.writerow([sc, liq[0], liq[1], arm, e, round(o), round(pc), round(ps),
                             round(ac, 3), round(as_, 3)])
                if liq != (1.0, 1.0):
                    ratios[(GROUP[sc], arm, "const")].append(ac)
                    ratios[(GROUP[sc], arm, "state")].append(as_)
                ratios[(GROUP[sc], arm, "abs_const")].append(o / pc)
                ratios[(GROUP[sc], arm, "abs_state")].append(o / ps)
    out.close()

    def summ(v):
        lg = [abs(math.log(x)) for x in v]
        return f"{min(v):.2f}-{max(v):.2f} (mean |log err| {sum(lg) / len(lg):.3f})"

    print("observed / predicted M*_clock, range over estimators x arrival settings")
    for grp in ("original", "away", "toward"):
        for arm in ("sync", "corr"):
            print(f"{grp:9s} {arm:5s} anchored  const {summ(ratios[(grp, arm, 'const')])}   "
                  f"state {summ(ratios[(grp, arm, 'state')])}")
            print(f"{'':9s} {'':5s} absolute  const {summ(ratios[(grp, arm, 'abs_const')])}   "
                  f"state {summ(ratios[(grp, arm, 'abs_state')])}")
    print("wrote results/async_predict.csv")


if __name__ == "__main__":
    main()
