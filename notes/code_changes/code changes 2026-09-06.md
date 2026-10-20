# Code changes — 2026-09-06

## Two new Kalman-innovation detectors (committed `222274a`)

Added `KalmanEwmaChartDetector` (`kalman_ewma_chart`) and
`KalmanShiryaevRobertsDetector` (`kalman_sr`). Both wrap `KalmanLocalLevel`,
monitor its standardized innovations, and call `reset()` on alarm — the same
detector slot as `kalman_cusum` and `kalman_glr`, so all four are directly
comparable.

**EWMA chart** (Roberts 1959): $z_n=(1-r)z_{n-1}+r\nu_n$, alarm when
$|z_n| > L\sqrt{\tfrac{r}{2-r}\left(1-(1-r)^{2n}\right)}$. The exact
time-varying variance matters — using the asymptote $r/(2-r)$ alone makes the
first few blocks far too wide to alarm, which is the same burn-in trap that
produced the BOCPD bug.

**Shiryaev-Roberts**: $R_t=(1+R_{t-1})\Lambda_t$, alarm on $\log R_t > h$.
Same latent chain as CUSUM, *summed* over the change time instead of maximised.
Carried in logs for numerical range.

Registered in both `make_estimator` and `all_estimator_keys()` — the second is
the one that actually gets them measured (cf. `xiao_win`, unmeasured until
2026-09).

`experiments/calibrate_arl0.cpp` extended to cover both keys.

## Status

- ✅ Compiles: `experiment`, `multiseed`, `calibrate_arl0`
- ⬜ **Thresholds are placeholders** — `L = 3.0` and `h = 5.0` are textbook
  defaults, NOT calibrated. Until `calibrate_arl0` is run at the same
  1/1000-block hazard target used for `kalman_cusum`/`kalman_glr`, any
  comparison against those two is unfair and must not be quoted as a result.
- ⬜ Multi-seed grid across all scenarios × families — started 2026-09-06 22:30.

## Next

1. Run `calibrate_arl0` for `kalman_ewma_chart` and `kalman_sr`; write the
   calibrated `L`/`h` back into `registry.hpp` **and** the live consumer config.
2. Re-run the grid with calibrated thresholds — the pre-calibration grid is a
   smoke test, not a thesis number.
3. Compare the four detectors on detection delay at matched ARL_0.
