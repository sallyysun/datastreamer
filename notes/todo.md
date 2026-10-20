# TODO

## Client vision — subscription-driven thin viewer

**Goal:** one web page anyone can open; each connection subscribes to the events/topics
it wants, and the display adapts to that subscription. The browser stays a **strict
viewer** — it renders what it receives and runs no estimators (those live in the C++
consumer devices).

**Why WebSocket (not raw TCP like the C++ consumers):** browsers cannot open raw TCP
sockets — only HTTP / WebSocket / WebRTC. So the browser must use WS. WS and the TCP
broker are just two transports carrying the *same* topic-based pub/sub, so the
subscribe-and-display model works identically on both.

### What to build
1. **Topic-aware WS on the server.** Today the WS side dumbly broadcasts everything to a
   single `ticks` topic. Give it the same per-connection subscription the TCP broker has:
   - client → server control message, e.g. `{"subscribe": ["trades", "quotes"]}`;
   - server parses it and calls `ws->subscribe(topic)` per topic (uWS pub/sub);
   - producer publishes each event to its topic (`app.publish(topic, msg)`), so a
     connection receives only the topics it asked for.
2. **React subscription picker + adaptive views.** A control to choose topics; the page
   renders different panels depending on what is subscribed (e.g. `trades` → price chart;
   a specific pair → that pair's view; later, a consumer's re-published estimates → a
   dependence panel). Encode the subscription in the URL so a link reopens the same view.
3. **Strict viewer.** Remove the client-side rolling-correlation calc added during the
   prettification — the browser should display only what it receives, no computation.

### Open questions
- **Topic taxonomy:** event type (`trades` / `quotes`) only, or also per-symbol / per-pair?
  And should consumers be able to **re-publish** their estimates to a topic the browser can
  subscribe to (so the display can show live dependence without the browser computing it)?
- Whether the WS and TCP broker should share one topic-routing implementation (currently
  the `Broker` is TCP-only; uWS has its own pub/sub).

---

## Backlog (consolidated from notes/code_changes 2026-08-19)

Status key: ✅ done · ⬜ open · ➖ deliberate non-goal.

### Broker / topics / sources
- ✅ **Consolidate the two "brokers".** `EventBus` removed from the tree; the network `Broker`
  is the only bus. (Was listed as "still missing" in the 2026-08-19 note; now complete.)
- ✅ **Richer topics — publish per-symbol, pair on the consumer.** Producers now publish one
  single-symbol `SymbolTick` per leg to `trades.equity.<SYMBOL>` (symbols set via `SYMBOLS=X,Y`,
  default `AAPL,MSFT`; equity-only for now). The consumer subscribes to two symbols and
  **FIFO-pairs** the legs into the bivariate sample its estimators consume — the device stack
  stays bivariate and untouched. Broker matching is dot-boundary prefix: `trades` /
  `trades.equity` get every symbol, `trades.equity.AAPL` gets one, `all` gets everything.
  - *Assumes synchronous producers* (one tick per symbol per step, same order) so arrival-order
    pairing is exact. **Async / multi-rate legs deferred** — will need timestamp/sequence
    alignment instead of FIFO, and `--rate` throttling can desync the legs today.
  - ✅ **Offline async study** (`experiments/async_memory.cpp`, `notes/studies/async_memory.md`):
    refresh-time pairing (Chakrabarti & Sen 2022) on a latent Gaussian stream, 5 memory-knob
    estimators × memory grid × 5 arrival settings, 20 seeds. Matched-rate sync obeys
    $M^*_\text{clock}\propto f^{-1/3}$ (±10%); naive async pairing is a 1.3–2× RMSE bias problem
    no memory choice fixes; the overlap correction removes the bias and shifts
    $M^*\propto f^{-1/3}\bar c^{-2/3}$. ⬜ Phase 2: liquidity-change scenario (spurious dependence
    change / detector false alarms); ✅ drift toward/away from zero (sign argument confirmed in both mirror pairs);
    ✅ state-dependent noise diagnostic (measured $v_j(\rho)$ halves corrected-arm error; the
    toward-zero "breakdown" was a `hermite_ew` startup bug); ⬜ fix `HermiteEstimator` EW startup
    (weight $\max(\lambda,1/n)$); ⬜ then timestamp pairing in the consumer.
  - *Next: a multi-symbol DGP* — routing + pairing are in place, but the synthetic source still
    emits a single correlated pair (published as its two legs).
- ✅ **Throttled subscriptions.** `SUBSCRIBE <topic> [max_hz]` (consumer flag `--rate N`)
  rate-limits a connection to at most N messages/second broker-side; excess is dropped
  (consistent with the best-effort, buffer-type-device model).
- ⬜ **Wire `FileSource` replay + live feeds as selectable sources.** `file_source.{hpp,cpp}`
  exists but the consumer only reads the broker over TCP — no source selection yet.
- ➖ **Broker retention / offsets** — deliberately omitted (minimal buffer-type devices);
  revisit only if replay / at-least-once is needed.

### Client (see "Client vision" above)
- ✅ **React topic-picker viewer** — landed in "Client vision" (#6): topic-aware WS +
  subscription picker with adaptive panels, browser stays a strict viewer.

### Estimators
- ✅ **Validate C++ estimators against the reference submodules** (`external/hermiter` in R,
  `external/onlineNPCORR` in Python) on shared inputs — cross-validation harness landed (#5).
- ✅ **Hermite scope vs `external/hermiter`:** we port only the *online bivariate Spearman's rho*
  path. Not ported (not needed for this thesis): univariate pdf/cdf/quantile, Kendall's tau,
  consistent merging of sub-estimators, batch mode. The old quadrature-vs-closed-form `W`
  difference is closed — `integral_matrix` is analytic (IBP symmetry + column recurrence).
- ✅ **Change detection + restart, step 1: innovation CUSUM** (`dependence/kalman_cusum.hpp`,
  key `kalman_cusum`; build order step 3 in
  [change_detection.md](studies/change_detection.md)). Two one-sided CUSUMs on
  `KalmanLocalLevel`'s standardized innovations $e_t/\sqrt{F_t}$ (change_detection.md Part III
  Level 1); on alarm, calls the filter's `reset()` — the hook already in the interface
  (`estimator.hpp:16`). BOCPD is still open, tracked separately below.
  **Verified on the real pipeline, not synthetic $N(0,1)$ draws:** the classical
  Page/Hawkins-Olwell default ($\delta=1,h=5$) produced zero false alarms in ~80,000 simulated
  blocks (8 seeds × 10000 blocks, `constant` scenario) — a safe lower bound on
  $\mathrm{ARL}_0$, not yet a calibrated number. That gap is now closed (below).
  On `jump` (0.2→0.8 at $t=2500$): 1 alarm, 0 false alarms before the jump, detection delay 59
  ticks (3 blocks). On `oscillate`: 21 alarms over 5000 ticks — plausible, since each half-period
  really is a large genuine shift, not noise.
  **Non-obvious finding — hard reset has a real RMSE cost, not just a detection-delay benefit:**
  `kalman_cusum`'s RMSE is *worse* than plain `kalman_fast` on both `jump` (0.118 vs 0.092) and
  `oscillate` (0.340 vs 0.316), despite (or rather because of) detecting the jump correctly. By
  the time the alarm fires, `kalman_fast` has already been adapting toward the new value for
  several blocks — the hard reset snaps the state back to $z=0$ and forces the filter to re-climb
  from a flat prior, adding transient error exactly where the smoothly-adapting filter was already
  closing in. This is the concrete instance of the Part V point that "VFF and detect-and-restart
  are not rivals, they are one dial at different decision hardness" — a soft posterior-weighted
  mix (Harrison-Stevens / IMM) would plausibly avoid this discontinuity; worth a direct comparison
  once BOCPD (softer than hard-reset CUSUM) exists.
  **The efficient-score / marginal-invariant CUSUM refinement (change_detection.md Part IV.2) is
  NOT needed here, verified directly rather than assumed:** Part IV.2 exists to fix a
  marginal-vs-dependence confound in the Fisher-scoring MLE branch (`gaussian_mle.hpp`, on the
  other branch), where a pure marginal change can masquerade as a dependence change because the
  score is built from raw normal-score products. This detector's observation is different by
  construction — a windowed *Spearman's rho*, a rank statistic, invariant to any monotone
  transform of either marginal on its own. Checked directly: a huge, purely marginal
  location+scale jump on both legs ($x\mapsto5x{+}100$, $y\mapsto0.1y{-}50$, both monotone
  increasing) with the copula (and hence true Spearman $\rho$) held *exactly* constant produced
  **zero alarms**, before or after, and the CUSUM statistic never left the noise floor (peak 3.19
  vs threshold 3.5). This structural invariance is a property of the shared block-Spearman
  observation, not of the CUSUM statistic specifically, so it carries over to `kalman_glr` below
  without a separate check.
- ✅ **Change detection + restart, step 2: window-limited GLR** (`dependence/kalman_glr.hpp`, key
  `kalman_glr`; build order step 5 in [change_detection.md](studies/change_detection.md), Part III
  Level 2). `kalman_cusum` assumes a *known* design shift size $\delta$; GLR is the honest version
  for an *unknown* jump size, built as a matched filter on the filter's own step response, and
  window-limited (Lai 1995) to $O(\text{window})$ per block — the unbounded sup-over-$\tau$ GLR is
  explicitly not streaming-admissible.
  **Derived and verified the step response, not assumed it:** for this scalar steady-state local-level
  filter, a unit level jump at $\tau$ contributes $\varrho_s(\tau)=(1-K)^{s-\tau}$ to the
  standardized innovation ($K$ = steady-state Kalman gain, Bridge 1) — derived from the bias
  recursion $b_s=(1-K)b_{s-1}+K\nu$. Monte Carlo (2000 seeds, real block-Spearman noise, not
  synthetic): empirical decay ratio between consecutive post-jump blocks averaged 0.65–0.74 across
  8 blocks vs the theoretical $(1-K)=0.672$ — matches within Monte Carlo noise at every lag.
  Matched-filter accumulators updated via the shift-register recurrence
  $A_t(j)=e_t+(1-K)A_{t-1}(j-1)$, statistic $g(j)=A(j)^2/N(j)$ maximized over lag $j$, $N(j)$
  precomputed once in closed form.
  **Result, single-seed=42:** on `jump`, `kalman_glr` matches `kalman_cusum` exactly (59-tick
  delay, 0 false alarms, identical RMSE 0.1177) — for a jump this large, "known size" vs "unknown
  size, inferred" makes no practical difference. On `oscillate`, GLR alarms slightly less often
  than CUSUM (20 vs 25) and its RMSE is worse (0.369 vs 0.340) — plausibly because GLR's own
  matched-filter response to the reset-induced re-climb transient (a second, smaller apparent
  "jump" as the filter relearns from a flat prior) doesn't line up as cleanly with the true
  oscillation shape as CUSUM's simpler running sum does; not yet investigated further.
  **Multi-seed (20 seeds, mean±SE, see the replication item below) refines this:** on `jump`,
  `kalman_glr` RMSE = 0.1110±0.0032 vs `kalman_cusum`'s 0.1146±0.0027 — GLR is marginally better on
  average, but the gap is under 1 combined SE, i.e. not clearly distinguishable from noise at this
  seed count, consistent with (not contradicting) the single-seed "makes no practical difference"
  reading above. On `oscillate` the gap is real: `kalman_cusum` 0.348±0.003 vs `kalman_glr`
  0.356±0.003, several SEs apart.
  Also exposes `estimated_jump_size()` / `estimated_lag_blocks()` — a real cost/benefit over
  CUSUM: on the synthetic verification jump (0.2→0.85 Spearman) it flagged the changepoint 1 block
  after `kalman_cusum` would need to, with an initial size estimate order-of-magnitude-correct but
  noisy (2.87 vs true 1.05 in atanh units, single realization) — useful as a diagnostic, not yet
  validated as a point estimate.
  Both detectors share a new common `ChangeDetector` interface (`dependence/change_detector.hpp`)
  so the harness logs alarms without a dynamic_cast per detector type.
- ✅ **Change detection + restart, step 3: BOCPD** (`dependence/kalman_bocpd.hpp`, key
  `kalman_bocpd`; build order step 6, the Bayesian gold standard) — Adams & MacKay (2007) run-length
  posterior on the same blockwise Fisher-z'd Spearman observation, with a conjugate Normal model
  (KNOWN $\sigma_v^2$, unknown segment mean $\mu=\operatorname{atanh}\rho$) in place of the paper's
  general NIW, since $\sigma_v^2$ already has a closed form throughout this Kalman/CUSUM/GLR line.
  Unlike `kalman_cusum`/`kalman_glr`, **no `reset()` is ever called** — `estimate()` is the
  run-length-weighted posterior mean, a soft Bayesian model average, which is exactly the mechanism
  Part V predicts should avoid the hard-reset RMSE cost documented above. Truncated at
  `max_run_length=150` blocks for $O(\text{max\_run\_length})$ per block, same bounded-window trade
  as GLR.
  **Found and fixed a real bug via direct verification, not a design choice:** $P(r_t{=}0\mid
  \text{data})$ — the textbook "changepoint probability" — is **mathematically pinned to exactly
  the hazard rate $H$ on every tick, independent of the data**, under a truly constant hazard: the
  one-step marginal likelihood $S=\sum_r P(r_{t-1}{=}r)L(r)$ is a common factor in both the reset
  branch ($H\cdot S$) and the total growth mass ($(1{-}H)\cdot S$), so it cancels in the
  normalization. Derived this from the recursion algebra, then confirmed it empirically (flat at
  exactly $0.0010=1/1000$ across a real synthetic jump). The genuine signal instead lives in how
  $(1{-}H)$ redistributes *within* the growth branch — confirmed the MAP run length correctly
  collapses to 1 at a true changepoint — so the alarm rule now thresholds $P(r_t\le
  \text{short\_run\_cutoff}\mid\text{data})$ (cutoff=2) instead, which is genuinely data-dependent
  (spiked 0.53→0.9999 across the two blocks after a synthetic 0.2→0.85 jump, back to baseline
  ~0.001–0.002 within one more block).
  **$\mathrm{ARL}_0$-calibrated** the same way and against the same target as `kalman_cusum`/
  `kalman_glr` (`experiments/calibrate_arl0.cpp`, `hazard_rate` fixed at $1/1000$ blocks to match
  their target directly; sweeps `alarm_prob_threshold` instead of $h$). Calibrated threshold 0.15
  (smallest sweep point clearing the target was 0.10, worst case ARL0 exactly 1000 with no margin;
  0.15 gives a comfortable 1.875× margin, worst case ARL0=1875 blocks at $\rho=0.0$ — same
  margin-over-bare-minimum trade as `kalman_cusum`/`kalman_glr`).
  **A second real bug, this time caught by multi-seed replication, not single-seed inspection:**
  the first calibration pass (before the fix described next) reported only a 7% ARL0 margin and,
  consistently, the single-seed harness run showed 2 false alarms before the `jump` scenario's true
  jump — plausible as single-seed variance on its own, EXCEPT `experiments/multiseed.cpp` (20
  seeds) showed the SAME ~2.1 false alarms per run on every seed, not scattered noise — a
  systematic effect, not variance. Root cause: `alarmed_this_block()` fired
  **deterministically on blocks 0–1 of every run, regardless of data** (verified directly: 200
  stationary seeds, 100% alarm rate at blocks 0 and 1, 0% from block 2 on) — with fewer than
  `short_run_cutoff`+2 blocks of history, EVERY active hypothesis trivially satisfies
  $r\le\text{short\_run\_cutoff}$ (there has not been time for a longer-run alternative to exist),
  so $P(\text{short run})\approx1$ independent of any real evidence. Fixed by suppressing alarms
  until enough history has accumulated for a genuine "short vs long" comparison to be meaningful;
  reconfirmed at 0 spurious alarms in the same 200-seed check, then recalibrated (the number above)
  and reverified on the harness: false-alarms-before-jump dropped from 2.10±0.07 to 0.10±0.07
  across 20 seeds, now in the same near-zero range as `kalman_cusum` (0.05±0.05) and `kalman_glr`
  (0.00±0.00). This is the concrete payoff of building the multi-seed tool now rather than trusting
  single-seed calibration margins.
  **Result, multi-seed harness (20 seeds, mean±SE) — wins decisively on its home scenario, loses
  where its prior is mismatched:** on `jump`, `kalman_bocpd` RMSE = 0.0598±0.0022 vs
  `kalman_cusum`'s 0.1146±0.0027 and `kalman_glr`'s 0.1110±0.0032 — a ~48% reduction, more than 15
  standard errors apart, not a single-seed artifact — plus a faster mean detection delay (31.9±3.4
  vs 46.4±4.9 / 49.6±6.9 ticks). This is the concrete, now statistically solid confirmation of the
  Part V prediction: soft model-averaging really does avoid the hard reset's forced re-climb from a
  flat prior. Same story on `constant` (RMSE 0.0189±0.0021, second only to the cumulative
  estimators and better than every other Kalman variant). But on `oscillate` (RMSE 0.368±0.003,
  worst of the Kalman-family detectors — `kalman_fast` 0.317±0.002 < `kalman_cusum` 0.348±0.003 <
  `kalman_glr` 0.356±0.003 < `kalman_bocpd`), `drift` (0.121±0.002 vs `kalman_cusum`/`kalman_glr`'s
  0.093), and `stochastic` (0.086±0.006 vs 0.078–0.079) it loses, every gap several SEs wide — the
  hazard prior assumes segments last ~1000 blocks on average, badly mismatched to `oscillate`'s
  true regime length (~10 blocks, half the period) and to `drift`/`stochastic`'s complete absence
  of any discrete segment at all. Net: BOCPD is not a uniform upgrade, it is a different bet that
  pays off exactly when its modeling assumption (rare, discrete, long-lived regimes) holds — and
  multi-seed replication is what turned that from a single-seed impression into a confirmed result.
  **This win DOES survive fair per-family tuning, unlike `kalman_fast`'s apparent `oscillate` win
  (see the ⚠️ correction under the local-level Kalman item).** Sweeping each competing family's own
  forgetting parameter on `jump` (20 seeds each): best Xiao is $\lambda=0.99\to0.0798$, best Hermite
  is $\lambda=0.01\to0.0815$, best plain Kalman is $\sigma_\omega^2=10^{-3}\to0.0840$ — all three
  already at or near their defaults, and all beaten by `kalman_bocpd`'s **0.0598** by ~25%.
  **That gap is the one genuinely structural result in this whole line of work:** no single fixed
  forgetting rate can be simultaneously long-memory (low variance while the regime is stable) and
  short-memory (fast adaptation right after it breaks) — every fixed-$\lambda$ estimator has to
  pick one compromise and live with it. BOCPD does not have to: its run-length posterior effectively
  *switches* memory length based on evidence. That is a capability a fixed $\lambda$ cannot
  replicate at any tuning, which is exactly why this gap survives tuning when the `oscillate` one
  did not.
- ✅ **`kalman_bocpd_fast`** — same model, `hazard_rate` retuned from $1/1000$ to $1/10$ blocks,
  the Bayesian analog of `kalman`/`kalman_fast`'s $\sigma_\omega^2$ contrast: hazard rate IS BOCPD's
  own "how fast do I expect things to change" knob, so a single fixed value cannot be optimal
  across regimes with different timescales any more than a single $\sigma_\omega^2$ can — this was
  the direct, testable form of the "hazard prior mismatched to `oscillate`" diagnosis above, checked
  rather than left as a plausible-sounding excuse.
  **Swept hazard against `oscillate` first** (20 seeds each): RMSE fell monotonically from 0.368
  ($1/1000$) to ~0.228 near hazard $\approx1/1.4$–$1/2$, but that regime is so short-memory it stops
  modeling "regimes" at all and degenerates into an aggressive smoother. $1/10$ (matching
  `oscillate`'s actual half-period of 10 blocks) was chosen instead of the numerically-best value
  specifically because it ties to a real DGP timescale rather than being tuned to minimize one
  scenario's RMSE — the difference between a principled setting and overfitting the test scenario.
  $\mathrm{ARL}_0$-recalibrated at this hazard (same 1000-block target, same `calibrate_arl0.cpp`):
  smallest threshold clearing it is 0.90 (1.25× margin); 0.95 used for a 2.1× margin, matching the
  convention everywhere else in this file.
  **Result, multi-seed harness, full comparison across all six Kalman-family entries (RMSE,
  mean±SE):**

  | scenario | `kalman` | `kalman_fast` | `kalman_cusum` | `kalman_glr` | `kalman_bocpd` | `kalman_bocpd_fast` |
  |---|---|---|---|---|---|---|
  | constant | 0.0208 | 0.0712 | 0.0721 | 0.0721 | **0.0189** | 0.0613 |
  | drift | 0.1274 | 0.0921 | 0.0930 | 0.0928 | 0.1214 | **0.0961** |
  | jump | 0.1288 | 0.0880 | 0.1146 | 0.1110 | **0.0598** | 0.0760 |
  | stochastic | 0.0956 | **0.0735** | 0.0792 | 0.0776 | 0.0856 | 0.0800 |
  | oscillate | 0.4153 | 0.3166 | 0.3477 | 0.3556 | 0.3683 | **0.2695** |

  The hazard retune turned out **less of a trade-off than the oscillate-only sweep suggested**:
  `kalman_bocpd_fast` isn't just better on `oscillate`, it's competitive almost everywhere except
  `constant` — it beats `kalman_bocpd` outright on `drift` and `stochastic` (both are gradual,
  continuous change, where shorter memory helps the same way `kalman_fast` beats `kalman`), loses
  only moderately on `jump` (0.076 vs 0.060, still far ahead of `kalman_cusum`/`kalman_glr`'s
  0.11+), and only really loses badly on `constant` (0.061 vs 0.019 — "expect frequent change" is
  the one prior that's actively wrong when nothing ever changes).
  **One real cost the RMSE table doesn't show:** on `jump`, `kalman_bocpd_fast` only detected the
  changepoint in 9/20 seeds (vs `kalman_bocpd`'s 17/20) despite a *faster* mean delay when it did
  (25.7±3.3 vs 31.9±3.4 ticks) — a fast hazard keeps the run-length posterior diffuse even under
  stationarity, so a genuine large jump's signature is less cleanly distinguishable from the
  model's constant background expectation of change. Detection reliability and detection speed are
  not the same axis, and this hazard trades one for the other.
- ✅ **Local-level Kalman estimator** (`dependence/kalman_local_level.hpp`, keys `kalman`/
  `kalman_fast`). Emits $y_t=\operatorname{atanh}(\hat\rho^S_t)$ every `block` ticks from a
  windowed Spearman, with **known** observation variance $\sigma_v^2\approx 1.06/(\text{block}-3)$,
  so the filter is exact linear-Gaussian (no EKF) — the free knob is $\sigma_\omega^2$ (process
  noise), not a hand-picked $\lambda$.
  **Verified — Bridge 1 identity is exact, not approximate:** on a stationary stream the filter's
  steady-state Kalman gain matches the closed form $K=(\sqrt{q^2+4q}-q)/2$, $q=\sigma_\omega^2/\sigma_v^2$,
  to numerical precision across $q\in[4\times10^{-4}, 4.4]$ — confirms every fixed-$\lambda$
  estimator in this directory really is this filter's steady-state special case, not just an
  analogy.
  **Found and fixed a real bug via the harness, not guesswork:** the block observation model
  assumes $\rho$ is ~constant *within* a block; at the default `block=50` this broke down on
  `oscillate` (period 400 — a block spans 1/8 of a cycle, and near a zero-crossing $\rho$ moves by
  nearly its full range within one block), making `kalman`/`kalman_fast` **worse than every other
  estimator including non-adaptive ones** (RMSE 0.42-0.43 vs `windowed_rank`'s 0.36). Swept block
  size against all 5 scenarios and confirmed the mechanism (RMSE fell monotonically as block
  shrank on `oscillate`, rose slightly on `constant` from noisier per-block Fisher-z reads);
  `block=20` is the harness-wide sweet spot and is now the default.
  **Result, full harness (`experiments/results/experiment_*.csv`):** `kalman` (slow,
  $\sigma_\omega^2=10^{-4}$) is competitive with the cumulative estimators on `constant` (RMSE
  0.017 vs `gaussian_mle`'s 0.015); `kalman_fast` ($\sigma_\omega^2=10^{-2}$) is competitive with
  the best `_ew` estimators on `drift`/`jump`/`stochastic` and beats them on `oscillate`
  (RMSE 0.316 vs `windowed_rank`'s 0.361 and `hermite_ew`'s 0.345).
  ⚠️ **CORRECTED 2026-09-06 — that `oscillate` comparison was NOT like-for-like, and the earlier
  wording ("beats every other estimator in the suite, including all `_ew` variants") was an
  overclaim.** It compared `kalman_fast` against `xiao_ew`/`hermite_ew` **at their default
  $\lambda$** (0.99 / 0.01), which are far too slow-forgetting for a period-400 oscillation — not
  against those families tuned for the scenario. Sweeping each family's own forgetting parameter on
  `oscillate` (20 seeds each) reverses the conclusion outright:

  | family | best setting | `oscillate` RMSE |
  | --- | --- | --- |
  | Xiao | $\lambda=0.95$ | **0.1826** |
  | Hermite | $\lambda=0.05$ | 0.1879 |
  | Kalman | $\sigma_\omega^2=0.3$ | 0.2256 |

  Properly tuned, Xiao and Hermite beat the best achievable Kalman filter on `oscillate` by ~19%,
  and the Kalman curve **plateaus** near 0.226 as $\sigma_\omega^2\to\infty$ — it cannot be tuned
  past that, because at $K\to1$ it is reading its block observation as-is, and that observation
  still carries the $m/2$ aggregation lag and the $\sigma_v^2\approx1.06/(m-3)\approx0.062$ noise
  floor derived above. The block-aggregation cost is a hard ceiling, not a tuning choice. So the
  local-level filter's value here is **not** raw tracking accuracy — see the corrected summary
  under the BOCPD item for what it actually buys.
  Exposes `innovation()` / `innovation_variance()` / `standardized_innovation()` /
  `kalman_gain()` — the exact quantities an innovation-CUSUM (`change_detection.md` Part III
  Level 1) needs, and the next thing to build on top of this.
- ✅ **Other parametric copula families** — Clayton / Gumbel / Student-t online MLE
  (`dependence/parametric_copula.hpp`, family math in `copula_family.hpp`; keys
  `clayton_mle`, `gumbel_mle`, `t_mle` + `_ew`). Fisher-preconditioned natural gradient
  (empirical Fisher, since Clayton/Gumbel have no closed-form Fisher); each reports Spearman's
  rho via `spearman_from_param` (`rho_S = 12 E[UV] - 3` by quadrature on the density).
  Clayton/Gumbel are positive-dependence only; the t copula's nu is a fixed hyper-parameter.
- ✅ **Parametric-vs-nonparametric goodness-of-fit selection** — `CopulaSelector`
  (`dependence/copula_selector.hpp`, keys `copula_select` + `_ew`). Scores Gaussian / Clayton /
  Gumbel / t + a binned nonparametric density by a forgetting-weighted cumulative prequential
  log score (shared, Gaussian-standardised marginals, so the marginal term is common, though
  marginal errors can still affect candidates differently) and reports the winner's Spearman rho.
  ⚠️ Only partly prequential: the marginal moments are updated with (x_t, y_t) before (u_t, v_t)
  is computed (`thesis_audit.md` §1.6). Fixing the order means rerunning the selector numbers. Verified: picks the generating family on
  matched data; on constant Gaussian data selects Gaussian and lands on the true rho_S.
  ⚠️ **Fixed 2026-10-20: two bugs in the nonparametric candidate (`copula::NonparamCopula`).**
  (1) Under forgetting, its counts were weighted by 1−λ, so a total mass of ~1 faced a Laplace
  prior of mass 100, leaving the density nearly flat. Counts are now kept in observation units.
  (2) Its rho_S readout was uncentred; it now uses mid-rank Pearson, as Xiao does. Before the fix,
  the flat density *won* selection near independence (22–23% of ticks on Clayton/Gumbel `drift`)
  and reported through the noisy readout. 20-seed re-run, all four families:
  `copula_select_ew` improves in every non-`constant` cell, e.g. Clayton `drift` 0.1039→0.0606 and
  Gumbel `drift` 0.1083→0.0625. The "churns under forgetting" finding was mostly this bug.
  Cumulative `copula_select` is bit-identical, so the claim below is unaffected. Table and
  platform caveat: [kalman_changepoint_results.md](studies/kalman_changepoint_results.md) §8.5.
- ✅ **Non-Gaussian copula DGP** — `dgp::Copula` (`src/streamgenerator/datasource/copula.hpp`,
  generator-side counterpart to the estimator-side `copula_family.hpp`): Gaussian / Clayton /
  Gumbel / Student-t samplers, each calibrated to the SAME target Spearman rho_S per step via
  `param_from_spearman` (bisection on the estimator-side `spearman_from_param`, so generator and
  estimator agree by construction — samplers verified: empirical rho_S matches the reported
  ground truth to ≤0.006 across families/targets, incl. negative-target clamping for the
  positive-dependence-only Archimedean families). Wired into both the offline harness
  (`COPULA=gaussian|clayton|gumbel|t ./experiment <scenario>`, family-suffixed CSV output) and
  the live server (`COPULA` env var alongside `DEPENDENCE`).
  **Result (`experiments/analysis.ipynb`, "DGP family comparison"):** the misspecification
  signature is now measurable — each parametric MLE is strongest on its OWN generating family and
  weak elsewhere (e.g. `clayton_mle` worst on `gaussian`/`gumbel` data, best on `clayton`), and
  `copula_select` is the best-or-tied cumulative estimator on every family without knowing which
  generated the data. Multi-seed refines "best-or-tied": on `constant` it is within ~1.5 SE of the
  best cumulative estimator on every family, and best only on $t$. On Gaussian it trails
  `gaussian_mle` (0.0236±0.0045 vs 0.0169±0.0021, Linux run) because it picks Gumbel on ~12% of
  ticks. The bug fix above leaves this unchanged. Caveat: Spearman rho is a rank statistic blind
  to tail dependence, so this RMSE gap likely understates the real family separation — a tail-dependence metric is the
  natural follow-up (see below).
- ⬜ **Tail-dependence metric** (e.g. empirical upper/lower tail-dependence coefficients) to
  complement the Spearman-rho comparison above — rho_S can't see the very thing (tail behaviour)
  that actually distinguishes Clayton/Gumbel/t from Gaussian, so the misspecification RMSE gaps
  measured are likely a lower bound on the real difference.
- ⬜ **Online adaptive memory for $\chi$ — the thesis deliverable**
  ([oracle_memory.md](studies/oracle_memory.md)). Known-changepoint soft restart is the *oracle*
  ($\gamma^*=1/(1+Z^2)$, derived, checked numerically); the method is an online rule that
  estimates the evidence $Z$ from the stream and maps it to retention continuously. Build order:
  1. Streaming $\hat\chi(q)$ estimator with an online $SE$ (depends on the tail-dependence item
     above). **Contract decision needed first:** `estimate()` is defined to return Spearman $\rho$
     (CLAUDE.md) — $\chi$ needs either a second method or a separate interface, not a silent
     redefinition.
  2. $\chi$-changing scenarios (Clayton/Gumbel/$t$; Gaussian has $\chi\equiv0$) with true $\chi_t$
     in the CSV.
  3. Oracle $\gamma^*$ baseline (knows $\tau$, $\Delta\chi$) — the ceiling. Port the existing
     out-of-repo oracle code here; check measured optimum against $1/(1+Z^2)$.
  4. Online $\hat\gamma_t$ rule with re-anchoring + null calibration (oracle_memory.md §4.1 traps
     1–2: naïve map loses ~⅓ of memory per quiet tick; per-tick discounts compound).
  5. Baselines: fixed λ (swept), VFF, one detect-and-hard-restart. All through `./multiseed`.
  ➖ More changepoint detectors — out of scope; CUSUM/GLR/BOCPD on $\rho_S$ stay as prior work.
- ⬜ **Same-rho_S, different-shape changepoint scenario** — now unlocked by the `Copula`
  abstraction (switch family mid-stream at constant target rho_S): the roadmap distinction #1
  case a scalar rho detector structurally cannot see. Also unlocks the tail-asymmetry
  pairs-trading angle (roadmap `application_2.md`).
- ➖ **rank-PIT** — built for Hermite (`hermite_pit`) and the copula MLEs (`gaussian_mle_pit`)
  2026-09-08, exactly invariant to monotone marginal change; the Kalman channel never needed
  it. **Xiao converted 2026-09-13** (`xiao_pit`, optional `rank_pit_window` on the config).
  Quantile binning was *already* exactly invariant to a static monotone map, so the win is
  elsewhere: the frozen cutpoints are corrupted **permanently** by a marginal shift arriving
  after calibration — rho-hat locks at −0.95 against a truth of +0.58 — and rank-PIT recovers
  within one window. Free on RMSE (≤0.2σ on every scenario tried), 5.2× the store.
  §14.19 of `studies/kalman_changepoint_results.md`.
  Still ➖ overall: the `_pit` variants are additions, not replacements — §5–§8 and the live
  consumer still run standardization / frozen cutpoints; the 500 window is inherited, not
  tuned; no detector wraps a `_pit` estimator.
- ➖ Estimating the Student-t **degrees of freedom** nu online — left as a fixed hyper-parameter;
  online nu estimation is a separate problem, not needed for the misspecification comparison.

### Experiments / device
- ✅ **Memory trade-off: discounted bank vs fixed share vs VFF vs fixed memory, for copula
  prediction** (`experiments/probe_memory_tradeoff.cpp`,
  [memory_tradeoff.md](studies/memory_tradeoff.md)). One 8×8 forgetting-histogram density; 12-config
  tuning budget per adaptive method on jump + drift; memory range 25–8000; 60 fresh seeds; excess
  log loss vs the true copula at the true PIT. Whole-run `jump`: discounted beats fixed (−6.8
  ×10⁻³ nats/tick), fixed share (−0.6) and VFF (−3.6). It is *worst* in the first 100 ticks after
  the jump and wins at 100–1500 ticks. `drift` costs +1.0 vs fixed (≈ fixed share). `constant` is
  within ~0.6 of the hindsight-best fixed memory. The Rosenblatt-driven VFF fails because its
  $e^2$ *falls* after a jump to stronger dependence (a too-diffuse forecast reads as calibrated)
  and its null level moves with ρ (1.06 near independence, 0.93 at ρ≈.58).
- ⬜ **A two-sided copula-level VFF signal** (conditional-PIT uniformity, or the log score against
  its own predicted expectation), and a `jump` toward *weaker* dependence, where $e^2>1$ should
  make the current VFF react. Both from memory_tradeoff.md §4–5.
- ✅ **Online tail dependence, step 1: stationary validation** (`experiments/tail_validate.cpp`,
  `notes/studies/tail_dependence.md`). Gaussian / Clayton / Gumbel / $t_4$ at matched $\rho_S=0.58$,
  copula space. Finite-threshold $\chi(q)-\lambda$ dominates (Gaussian reads 0.19 at $q=.01$ with
  $\lambda=0$); estimators unbiased vs own target, ratio form least noisy; noise is exactly
  binomial thinning, $\mathrm{Var}\approx\chi(1-\chi)/(qm)$; Gaussian → $t$ needs memory ≈ 5000
  for SNR 3, which explains `tail_cusum`'s 0/20 in §14.13. ✅ Step 2 (`experiments/tail_change.cpp`):
  oracle restart beats best fixed memory 1.6–6× on Clayton/Gumbel switches but *loses* to it on
  Gauss → $t$ (0.77–0.99); the bank sits near best fixed; $M^*=\sqrt{4V/(q\Delta(q)^2)}$ fits all 40
  cells to ±9 %, pure $q^{-1/2}$ falsified. ✅ Step 3a: Clayton–Gumbel mixture
  feasibility (`experiments/tail_mixture.cpp`) — $\rho_S$ and $\chi$ exactly linear, $\tau$ not;
  $\lambda_L+\lambda_U$ pinned to ≈ 0.50–0.61 at $\rho_S=.58$; equal-component path (only $\pi$ moves)
  recommended. ✅ Step 3b: soft-restart oracle
  (`experiments/tail_soft.cpp`) — best soft restart beats best fixed memory in all 48 cells (5–20 % on
  the smallest change, up to 5× on the largest) and beats hard restart on every non-extreme change;
  optimal retained old evidence follows $n_0^*=v_\text{post}/\Delta\chi^2$ events (slope 1.03,
  $R^2=0.98$; post hoc, checked on fresh seeds). ⬜ Detection + estimated $\Delta\chi$ to approach the
  soft oracle; rank-based margins; tail-structure drift.
- ✅ **Multi-seed replication + error bars**, for the Kalman/CUSUM/GLR/BOCPD line
  (`experiments/multiseed.cpp`, run: `make multiseed && ./multiseed <scenario> [n_seeds] [base_seed]`,
  default 10 seeds; run at 20 for the numbers cited in this file). Same keys/scenarios/$N$/burn-in
  as `./experiment` so results are directly comparable; reports RMSE and, for every `ChangeDetector`,
  alarm count / false-alarms-before-jump / detection delay as mean±SE, written to
  `results/multiseed_<scenario>[_<family>].csv`.
  ⚠️ **Correction:** this entry originally claimed multi-seed covered "the Kalman/CUSUM/GLR/BOCPD
  line" only, with the Xiao/Hermite/`gaussian_mle` families left single-seed. That was wrong —
  `multiseed.cpp` always shared its `keys` list with `main.cpp`, so those families were multi-seeded
  from the start. The genuinely uncovered cases were `xiao_win` (registered but in neither harness's
  `keys`) and the parametric-copula branch; both are now covered — see the grid item below.
  **Not just a formality — it caught a real bug the single-seed check missed:** the first
  `kalman_bocpd` ARL0 calibration and harness run both looked plausible in isolation (7% ARL0
  margin, 2 false alarms in one `jump` run), but running 20 seeds showed the SAME ~2.1 false alarms
  on nearly every seed — a systematic bug (alarms firing deterministically in the first 2 blocks of
  every run, independent of data), not sampling variance. See the BOCPD item above for the fix and
  the corrected, now-verified numbers. This is the concrete case for why this item was worth doing
  now rather than deferring it: a single well-chosen seed's plausible-looking numbers do not
  substitute for it.
- ✅ **Full multi-seed grid across DGP families** (`experiments/run_multiseed_grid.sh`, run:
  `./run_multiseed_grid.sh 20 42`). Both harnesses now read `COPULA=gaussian|clayton|gumbel|t`, so
  the 5 scenarios × 4 families × 22 estimators are all covered at 20 seeds with error bars. Closes
  the two genuine coverage gaps: `xiao_win`, which was registered in `registry.hpp` but absent from
  both `keys` lists and so had never been run at all, and the 8 parametric-copula keys
  (`clayton_mle`/`gumbel_mle`/`t_mle`/`copula_select` + `_ew`), which were single-seed and
  Gaussian-DGP only. Results and the misspecification reading in
  [kalman_changepoint_results.md](studies/kalman_changepoint_results.md) §10.
  `multiseed.cpp` also rejects an unknown `COPULA` up front, because `dgp::make_copula` silently
  falls back to Gaussian on a typo — which in a scripted grid would mislabel a whole family's
  results rather than fail.
- ✅ **$\mathrm{ARL}_0$ calibration** for `kalman_cusum`, `kalman_glr`, and `kalman_bocpd`
  (`experiments/calibrate_arl0.cpp`, run:
  `make calibrate_arl0 && ./calibrate_arl0 [target_blocks]` — calibrates all three detectors in one
  run). Bootstrapped directly on the real
  pipeline's block-Spearman innovations — not synthetic $N(0,1)$ draws, not a formula, per
  [change_detection.md](studies/change_detection.md) Part VI. Sweeps $h$, measures empirical
  $\mathrm{ARL}_0$ over 6 seeds × 5000 blocks at each of $\rho\in\{0.0,0.3,0.6,0.8\}$ (the
  stationary levels this project's DGP suite actually uses), and picks the smallest $h$ whose
  **worst case over $\rho$** still clears the target. $\mathrm{ARL}_0(h)$ rises roughly
  exponentially, as CUSUM theory predicts: $h{=}1\to$ ARL0 $\approx7$–10 blocks; $h{=}3.5\to$
  worst case (at $\rho=0.6$) 2143 blocks; $h{=}5\to\;\geq\!30{,}000$ blocks (a lower bound — zero
  alarms observed even at the largest simulated horizon here). At a target of 1000 blocks,
  $h{=}3.5$ is calibrated (2× margin) and is now `kalman_cusum`'s registry default, replacing the
  earlier uncalibrated $h=5$ Page/Hawkins-Olwell default, which turned out to be needlessly
  conservative — its own measured $\mathrm{ARL}_0$ was 6–15× the target. Re-running the harness at
  $h=3.5$: `jump` detection delay is unchanged (59 ticks — the 0.2→0.8 jump is large enough to
  clear either threshold about equally fast), `oscillate` alarms rose 21→25 and RMSE improved
  0.373→0.340 (closer to, but still short of, `kalman_fast`'s un-reset 0.316 — consistent with
  the hard-reset-cost finding above).
  `kalman_glr`'s own sweep at the same target: $h{=}5\to$ worst case (at $\rho=0.8$) 92 blocks;
  $h{=}10\to$ 2727 blocks; $h{=}15\to\;\geq\!30{,}000$ blocks. $h=10$ is calibrated (2.7× margin)
  — and happens to match the value picked as an initial guess before calibrating, which is a
  coincidence worth noting rather than trusting next time a config parameter changes (block,
  window, or $\sigma_\omega^2$ all shift the calibration and must be re-run).
  `kalman_bocpd` and `kalman_bocpd_fast` sweep `alarm_prob_threshold` instead of $h$ (`hazard_rate`
  fixed per-key, $1/1000$ and $1/10$ respectively). Post burn-in-bug-fix (see the BOCPD item
  above): `kalman_bocpd` threshold$=0.05\to$ worst case 417 blocks; $0.10\to$ 1000 (bare minimum,
  no margin); $0.15\to$ 1875 (calibrated, 1.875x margin). `kalman_bocpd_fast` needs a much higher
  threshold for the same target since its hazard alone makes "short run" far more often true:
  $0.80\to$ worst case 400 blocks; $0.90\to$ 1250 (bare minimum); $0.95\to$ 2143 (calibrated, 2.1x
  margin). Full results for both, and the wider hazard-as-tuning-knob finding, are under the
  BOCPD items above.
  **Still open:** detection-delay metrics proper (this item
  only calibrates the false-alarm side); multi-seed error bars on the calibration itself (next
  item) and on any resulting delay number.
- ⬜ Decide whether to keep the `StateManager` moments (now inside `DeviceState`) exposed, or
  slim the device to buffer + copula inference only.

### References to acquire (sequential analysis)
- ⬜ **Acquire and cite the primary sequential-analysis sources.** Every detector in
  `registry.hpp` — CUSUM, GLR, SR, EWMA chart, BOCPD — currently traces to no entry in
  `sallysun-honours/bibliography.bib`, and the reference list in §12 of
  [kalman_changepoint_results.md](studies/kalman_changepoint_results.md) is explicitly from
  memory. Priority order, most load-bearing first:
  - **Lorden (1971)**, *Procedures for reacting to a change in distribution*, Ann. Math. Statist.
    — the minimax criterion CUSUM is optimal for: worst case over change time **and** over
    pre-change history, $\sup_\nu \operatorname{ess\,sup} E_\nu[(T-\nu+1)^+ \mid \mathcal F_{\nu-1}]$
    subject to $E_\infty[T]\ge\gamma$. Cite for `kalman_cusum`.
  - **Pollak (1985)**, *Optimal detection of a change in distribution*, Ann. Statist. — replaces
    Lorden's inner ess-sup with a conditional average, $\sup_\nu E_\nu[T-\nu \mid T\ge\nu]$;
    Shiryaev–Roberts is asymptotically optimal for it. Cite for `kalman_sr`.
    **These two are why `kalman_cusum` vs `kalman_sr` is a comparison of optimality criteria,
    not of algorithms** — same likelihood-ratio recursion, max over $\nu$ vs sum over $\nu$.
  - **Page (1954)**, *Continuous inspection schemes*, Biometrika — CUSUM itself.
  - **Moustakides (1986)**, Ann. Statist. — CUSUM is *exactly*, not just asymptotically,
    Lorden-optimal.
  - **Shiryaev (1963)** — the SR statistic's Bayesian origin.
  - **Roberts (2000/1959)**, *Control Chart Tests Based on Geometric Moving Averages* — the EWMA
    chart implemented in `kalman_ewma_chart`. **Already in Zotero**, uncited.
  - **Adams & MacKay (2007)** — BOCPD; the one reference already cited from the paper itself.
  ⚠️ Volumes, pages and years above are from memory and must be checked against a bibliographic
  database before entering `bibliography.bib`.
- ⬜ **Adaptive-forgetting / VFF literature**, promoted by the §11 reset-cost finding: the fix
  candidate (inflate state covariance on alarm rather than clearing state) is standard there.
  Haykin, *Adaptive Filter Theory*; the Paleologu et al. variable-forgetting-factor RLS line.

### Fisher scoring for the copula MLEs — efficiency
Both are $O(1)$ and neither changes the estimator's interface. Current update in
`parametric_copula.hpp`:
```cpp
info_ = decay * info_ + score_phi * score_phi;   // empirical Fisher
double step = score_phi / info_;
step = std::clamp(step, -kMaxStep, kMaxStep);
```
- ⬜ **Use the expected Fisher information, not the squared score.** `score_phi * score_phi`
  is the *observed* information from a single observation — a very noisy curvature estimate.
  The Gaussian copula's expected information is closed-form; deriving it for Clayton/Gumbel is
  the work. Better-conditioned steps at no runtime cost. This is exactly what
  **Pereira, Costa & Gonçalves (2026)** ✓ do — they replace observed information with a
  bootstrap approximation of the expected one and report better standard errors and numerical
  stability in small samples. Their method is batch, but the *motivation* transfers directly.
- ⬜ **Polyak–Ruppert averaging.** Keep a running mean of $\phi$ alongside $\phi$ itself and
  report the mean. One extra accumulator, recovers the MLE's asymptotic efficiency, and permits
  a larger step size than the raw iterate would tolerate.
- ⬜ **Replace the `kMaxStep` clamp with a damped or trust-region step.** The clamp is a safety
  hack that works; a trust region does the same job with a justification, and would not need
  re-tuning per family.
- ➖ Not worth doing: the update is already $O(1)$ per tick, so there is no *computational*
  efficiency to recover — these are all statistical-efficiency changes.

### Done 2026-09-07/08 — see [code changes 2026-09-08](code_changes/code%20changes%202026-09-08.md)
- ✅ **VFF** for Kalman, Hermite and the copula MLEs.
- ✅ **Particle filter**, **EKF**, **UKF** — layer 2 complete except Xiao's marginals.
- ✅ **ARL₀ calibration** for all six detectors; `kalman_ewma_chart` moved to $L=2.5$.
- ✅ **The reset is the whole cost.** No-op `reset()` returns RMSE to exactly `kalman_fast`;
  covariance inflation is free everywhere and beats no detector on `oscillate`.
- ✅ **Registry invariant guarded both ways**, and `multiseed` prints σ-from-best.

### Done 2026-09-08 (second pass) — the controls
- ✅ **Inflation factor swept**, and ⚠️ **the `oscillate` win retracted**. Monotone and
  saturating (×2 → ×50: 0.3130 → 0.3055), but a blind clock (0.2580) and a plain filter at
  a larger `sigma_omega2` (0.2256) both beat it. §14.8.
- ✅ **`sigma_omega2` swept** — never done before, and it moves RMSE ~4× more than any
  detector. Best value differs per scenario over 2.5 orders of magnitude; `kalman_vff` beats
  the best *fixed* value on `drift` with no oracle. §14.8b.
- ✅ **MLE score-CUSUM** — 20/20 on `jump` at matched ARL₀ against the Kalman detectors'
  17–19/20, same delay, half the SE. Paired with information deflation it beats the best
  fixed λ by 5.0σ **and passes both controls**. The first detector that earns its keep. §14.10.
- ✅ **`hermite_cusum` baseline fixed** (frozen Phase-I). RMSE damage gone, detection power
  still zero (0/20). The two-timescale replacement is 32× slower than the score-CUSUM. §14.11.

### Done 2026-09-08 (third pass) — off-family, shape, and the nonparametric detectors
- ✅ **Score-CUSUM off-family.** Detection survives (20/20 on all four DGPs), calibration does
  not — 2.7× too many alarms on t. Cause measured: Var[score]/I goes 0.93 → 2.20. Fixed with
  the sandwich correction (one EWMA); one threshold now holds on all four. §14.12.
- ✅ **Shape-change DGP** (`shape`: Gaussian → t(4), ρ flat to 5 dp, tail 0 → ~0.3). Nothing
  detects it. `hermite_cusum2` **0/20 on the change it was built for**. The only arm with power
  is `gaussian_mle_cusum` (17/20) and it works by misspecification sensitivity — which §14.12's
  robust fix cuts to 10/20. §14.13.
- ✅ **Score-CUSUM for Xiao and Hermite** via a per-tick rank residual — no likelihood needed.
  `xiao_cusum` 20/20 with **zero** false alarms. CUSUM column of §0.3's matrix closed. §14.14.

### Open, in priority order
- ⬜ **Throughput vs arrival rate: which estimators drop data, and what it costs them.**
  Everything measured so far assumes every tick is processed. The live pipeline
  (`src/consumer/`) has no such guarantee — if events arrive faster than `update()` runs, the
  ring buffer overwrites and the estimator silently sees a *thinned* stream. No result in
  `kalman_changepoint_results.md` accounts for that, and the throughput spread is **74×**:

  | | bytes | ticks/sec |
  | --- | --- | --- |
  | `gaussian_mle_ew` | 144 | 5,958,854 |
  | `kalman_fast` | 512 | 4,571,357 |
  | `kalman_bank_sticky` | 736 | 3,581,713 |
  | `xiao_ew` | 3,824 | 1,897,512 |
  | `hermite_ew` | 1,192 | 1,007,929 |
  | `mle_bank` | 17,208 | 548,548 |
  | `hermite_bank` | 22,448 | 69,139 |
  | `xiao_bank` | 35,608 | 48,380 |

  **Why it matters more than it looks.** A single liquid pair ticks at 10–1000/s, so even the
  slowest row copes. The budget binds through the number of *pairs*: $d$ assets is $d(d-1)/2$
  pairs sharing one core, so 100 assets is 4950 pairs and `xiao_bank` gets **10 ticks/sec each**.
  That is the constraint the `application_2.md` vine track runs straight into.

  **The experiment.** Add a `--max-rate` / drop-on-overflow mode to the harness, sweep the
  arrival rate against each estimator's measured throughput, and plot RMSE vs rate. Hypotheses
  worth separating:
    1. Thinning is *benign* for rank-based estimators — a thinned sample of an i.i.d. stream is
       still i.i.d., so Xiao/Hermite should degrade only through reduced effective sample size.
    2. Thinning is *not* benign for anything with a block or a memory in TICKS rather than in
       time — `block = 20` becomes 20 *surviving* ticks spanning a much longer window, so the
       Fisher-transform variance $1.06/(m-3)$ stays right but the lag doubles, and every
       forgetting rate silently lengthens. That would hit the Kalman line and the banks.
    3. Dropping is not random — it is bursty and correlated with volatility, i.e. exactly when
       dependence is changing. That breaks the i.i.d. assumption under 1 and is the interesting
       case.
  **This reframes the whole efficiency↔adaptation question as a budget constraint:** an
  estimator 20% more accurate but 74× slower is *worse* in production if it sees 1/74 of the
  data. None of the RMSE tables in the results doc price that.

- ⬜ **Monitor tail dependence directly.** §14.13 is the one change nothing detects, and the
  statistic that projects onto it has never been tried — $\hat\lambda_L$ over a window, the
  §14.10 lesson applied to a different direction. Highest value: it is an open gap with a
  known-good method for closing it.
- ⬜ **Decide the misspecification question.** §14.13 shows robustness to a wrong family and
  detection of a changing family are one statistic with opposite signs. Running both
  accumulators at different rates would give both; untested.
- ⬜ **Why `hermite_scusum` is erratic** (185 ± 92 vs Xiao's 71 ± 5 on the same statistic).
- ✅ **Memory sweep: bias²/variance against effective memory in ticks**
  (`experiments/probe_memory.cpp`, `notes/studies/memory_sweep.md`, 40 seeds). Puts `xiao_ew`,
  `hermite_ew` and `kalman` on one memory axis (ESS = (2−α)/α ticks, ×20 for the Kalman's block)
  and decomposes RMSE. **Memory is worth ×2–×17 in RMSE; the estimator choice at matched memory
  is worth 3–15 %.** The optimum moves from 36 ticks (`oscillate`) to past 30 000 (`constant`);
  on `jump` it is 53 ticks in the recovery window and 495 in the settled one — a factor of 9.4,
  costing 2.3–3.1× whichever way a single fixed λ is set. Three findings worth acting on:
  **(a)** the Xiao–Hermite gap is pure variance (Hermite's is 1.11–1.63× Xiao's at matched
  memory), so the representation costs efficiency, not bias; **(b)** `hermite_ew` has **no EWMA
  initialisation debiasing**, so any λ ≲ 0.002 at N = 5000 measures warm-up, not memory —
  `hermite_bank`'s longest arm (λ = 0.001) is already at that edge; **(c)** the Kalman's memory
  is floored at one block, so `σ_ω²` above ~0.03 at `block = 20` is a dead knob.
- ✅ **Does adaptive memory generalise past the Kalman?** (`experiments/probe_adaptive.cpp`,
  `notes/studies/adaptive_memory.md`, 40 seeds, memory-matched arm grids in ticks). **The
  mechanism generalises completely** — the Xiao, Hermite and Kalman banks all collapse memory
  13–20× at a jump and relax back, in near-lockstep. **The RMSE win against the per-scenario
  oracle generalises only on `jump`** (Xiao −3.3σ, Hermite −2.7σ, Kalman −16.6σ); all three lose
  on `constant`, `drift` and `oscillate`. Against a single fixed setting used everywhere the
  banks win **12 of 12, by 25–65 % RMSE**, which is the claim that needs no oracle.
  ⚠️ **§14.16's 8.2σ `drift` win is retracted**: §14.8b's `σ_ω²` grid stopped at 1e-3, so its
  "oracle" was 30 % too weak; against a properly resolved one the bank loses by 7.7σ. Cost:
  `xiao_bank` is 52 KB against `kalman_bank`'s 736 B — the device generalises, its cheapness
  does not.
- ✅ **Debias `hermite_ew`'s EWMA start** — divide by `1 − (1−λ)^t`, as Xiao's scale-invariant
  ratio does implicitly. Added as `HermiteEstimator::Config::debias`, **off by default** so no
  registered number moves. It changes the *oracle*: raw Hermite's `constant` optimum is a
  warm-up artifact at 341 ticks (0.0416); debiased it is 30 000 ticks (0.0195), matching Xiao.
  ⬜ Still open: turning it on by default, which needs its own multiseed run.
- ⬜ **Learn the forgetting rate properly.** §14.8b makes it the highest-value knob and
  `kalman_vff` is one heuristic. Ties to the roadmap's "ultimate estimator". `memory_sweep.md`
  §6.1 now quantifies the prize: 2.3–3.1× on the `jump` sub-windows. `xiao_vff` now exists
  (rank residual over its known null scale, `adaptive_memory.md` §5) and confirms the shape of
  the answer: a VFF *integrates* surprise where a bank *re-weights* on it, so the VFF barely dips
  at a jump and the bank drops 15×. Banks, not VFFs.
- ⬜ **Re-specify `xiao_bank` / `hermite_bank` / `mle_bank` in ticks.** They carry λ grids
  (memory 39 → 1999), which is neither matched to `kalman_bank` (22 → 999) nor warm for Hermite
  at the long end. Nothing has been measured on them; don't quote them until they are fixed.
- ⬜ **Per-arm predictive variance in `EstimatorBank`.** It scores arms with a variance shared
  across arms, where `KalmanBank` uses `F_k` per arm and gets the log-determinant term. Candidate
  explanation for the Kalman bank's `jump` win being 5× the Xiao bank's. `adaptive_memory.md` §11.
- ⬜ **Rank-PIT for Xiao**, the last layer-1 gap.
- ⬜ **Why expected information reduces variance 5–9×.** Three hypotheses tested, all failed.
