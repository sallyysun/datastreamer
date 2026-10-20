# Code changes — 2026-09-07/08

A long session working down the three layers of `studies/kalman_changepoint_results.md` §0.1.
Everything below is in that document with numbers; this is the map.

## Where it lives

Five branches, each stacked on the last, all pushed:

```text
main
 └── multiseed-coverage    EWMA chart + Shiryaev-Roberts detectors, inline citations
      └── combined-doc     primer merged in, layers first, sections reordered
           └── vff         variable forgetting: Kalman, Hermite, MLE
                └── rank-pit          streaming rank-PIT marginals
                     └── nonlinear-filters   particle / EKF / UKF, Fisher-scoring
                          │                  ablation, harness guards, calibration
                          └── soft-reset     no-op and covariance-inflation resets
```

`soft-reset` has everything. `primer` and `hermite-coefficients` branch off separately.

## Layer 1 — what we assume

- **`RankPit`** (`dependence/marginal_rank_pit.hpp`) — bounded sorted window, $O(\log W)$ to
  rank, probit back to the real line. Wired into Hermite (`hermite_pit`) and the copula MLEs
  (`gaussian_mle_pit`).
- The decisive test is a **monotone marginal transform**, which leaves the copula exactly
  unchanged. `hermite_pit` and `gaussian_mle_pit` are exactly invariant; `hermite_ew`
  degrades 3.7×.
- **The Kalman channel never had the problem** — `block_spearman()` ranks within each block.
- Xiao is still unconverted.

## Layer 2 — how we estimate

- **VFF** for Kalman, Hermite and the MLEs. Gains 15.6% on the filter, ~1% elsewhere:
  insurance against bad tuning, not a free upgrade.
- **Particle filter** with a jump-diffusion prior. Wins `oscillate` only.
- **EKF / UKF** on the raw block Spearman, dropping the Fisher transform. The transform wins
  3 of 4 — variance stabilization beats handling the nonlinearity.
- **Fisher scoring ablation**: expected information (adopt, for variance), damped step
  (optional), Polyak–Ruppert (reject — it assumes convergence to a fixed point).

## Layer 3 — how we handle change

- **Calibrated** `kalman_ewma_chart` to $L = 2.5$ (was the textbook 3.0). Detection went
  12/20 → 18/20, delay 74 → 47.9.
- At matched ARL₀ the four hard-reset detectors are **statistically indistinguishable**.
- **The reset is 100% of the cost.** A no-op `reset()` returns RMSE to exactly
  `kalman_fast`'s 0.0880 with detection unchanged. **Detection is free.**
- **Covariance inflation** (`P_ *= 20`, keep $\hat z$) is free everywhere and beats having
  no detector at all on `oscillate` (3.0σ).

## Harness

- **`check_registry()`** at the top of both harnesses: refuses to start if a key in
  `all_estimator_keys()` has no `make_estimator` branch. **`make check-registry`** catches
  the other direction. Both directions had bitten — `xiao_win`, the three `_vff` keys,
  `particle`.
- **`multiseed` prints distance from the best row in pooled SEs**, `~` under 2σ. It caught a
  misread on its first run: §5.1's bolded `hermite` on `constant` is a three-way tie at 0.4σ.
- **`run_multiseed_grid_parallel.sh`** — the grid is embarrassingly parallel, ~1 MB per job.
  12 min serial becomes ~80 s across 8 cores.

## Five claims that did not survive a control

| Claim | What it actually was |
| --- | --- |
| §5.4 `oscillate` win | tuning, not method |
| §12.3 EKF win on `drift` | $\sigma_\omega^2$, not the filter |
| §13.2 expected-info win ×4 | noise (0.9σ, 1.3σ, 1.0σ); the real effect is variance |
| §5.1 `hermite` best on `constant` | a three-way tie at 0.4σ |
| §14.1 "detection costs 18–30%" | the reset costs it; detection is free |

None was found by reading code. Every one came from a matched baseline, a standalone probe,
or an error bar.
