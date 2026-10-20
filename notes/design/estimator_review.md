# Estimator code review: design proposal

2026-10-20. Living copy: the "Estimator code review: design proposal" doc.

The review's fixes are split into four stages, and only stage 3 is allowed to change what an
estimator computes. Bug fixes and refactors must reproduce existing outputs exactly; any change
that can move a thesis number ships as a new, named configuration beside the old one, and is rerun
before any number is replaced.

## Principles

Startup behaviour, block bias and configuration differences are part of the statistical
experiment, not code maintenance. Four rules follow from that.

1. **Refactors and bug fixes reproduce existing outputs exactly.** Acceptance is an identical
   `dump_trajectories` manifest before and after, on the same platform: a bit-exact hash of every
   input stream and every per-tick estimate for every key. Summary CSVs are printed at limited
   precision and can hide last-bit differences, so a matching `multiseed` CSV is supporting
   evidence, not the gate. Probes that build their own configurations (memory sweeps, adaptive,
   VFF-norm, async and tail studies) are gated the same way with
   `./check_probe_trajectories.sh <dir>`, which runs them with `TRAJ_MANIFEST` set (about 160 s).
2. **No silent replacement.** A change to what an estimator computes gets a new registry key or an
   explicit config flag. The old behaviour keeps its own name (`*_legacy`), so a thesis number can
   always be traced to the code that produced it.
3. **Rerun before replacing.** A thesis number moves only after the new configuration has been run
   through `./multiseed` (20+ seeds, paired intervals) beside the legacy one.
4. **One name, one method.** The registry key a thesis table cites must be the configuration the
   probes ran. Two commands must never evaluate different methods under the same name.

## Changes by stage

Only stage 3 rows can move a thesis number, and each of those keeps a legacy configuration.

| Change | Stage | Can move a thesis number? | Acceptance check |
| --- | --- | --- | --- |
| Complete `ParametricCopulaMLE` / `CopulaFit` reset | 1 | No (verified) | `check-estimators` passes; `multiseed` CSV byte-identical |
| `make check-estimators` contract test | 1 | No | 94 keys, 0 failing |
| Contract test: reset at five points, NaN on either side, negative ρ, Kendall | 1 | No | 94 keys, 0 failing |
| `windowed_rank::kendall()` unsigned subtraction; Hermite `kendall()` formula | 1 | No (no caller; manifest identical) | Kendall sign and range checks pass |
| `make dump_trajectories` bit-exact manifest | 1 | No | Used as the gate for every later stage |
| Shared memory helpers (`memory.hpp`: α ↔ ESS, Kalman memory, construct-at-memory) | 2 | No | Probe and `multiseed` outputs byte-identical |
| Shared rank code (`ranks.hpp`) | 2 | No | Outputs byte-identical |
| Shared scenario definitions (`experiments/scenarios.hpp`) | 2 | No | Outputs byte-identical |
| Fixed-parameter draw on `dgp::Copula`, used by the tail studies | 2 | No | Tail study outputs byte-identical |
| Hermite warm-up w = max(λ, 1/m), m = coefficient updates | 3 (on `bank-vff-study`) | Yes, under a new key only | Opt-in `cumulative_start` flag; `hermite_ew` unchanged; new key `hermite_ew_cs`; rerun Hermite comparisons |
| Banks registered on the tick grid used in the thesis | 3 | Yes | Legacy λ-grid keys kept; `multiseed` matches `probe_adaptive` |
| One `store_bytes` definition (object + heap capacity) for every estimator | 3 | Yes (memory metric) | Cost table regenerated |
| Registry split into run list and fixed comparator set | 4 | No, only if the comparator set is fixed | Manifest identical; "vs best" and oracle rows unchanged |
| Comment trim, Makefile paths, warnings | 4 | No | Manifest identical |

## Stage 1: reset fixes and contract tests

Done on `massive-refactor` (e798c17, plus a follow-up from external review); no result changes. `ParametricCopulaMLE::reset()` now clears its rank-PIT windows,
and `CopulaFit::reset()` restores the VFF λ, the score mean and variance, and the Polyak–Ruppert
state. Before the fix, `gaussian_mle_pit`, `_fisher`, `_pr` and `_vff` differed from a fresh
estimator by up to 0.26 after a reset.

Impact was checked, not assumed:

- No experiment, the consumer or the server calls `reset()`, and every harness builds estimators
  per seed.
- No detector or bank wraps `ParametricCopulaMLE`; `CopulaSelector`, the only other `CopulaFit`
  user, runs with VFF and Polyak–Ruppert off.
- `./multiseed jump 5` and `./multiseed constant 5` are byte-identical before and after for all 94
  keys.

`make check-estimators` now enforces three properties on every registered key: `reset()` equals
fresh construction, `store_bytes()` does not grow with stream length, and estimates are finite and
in [−1, 1]. Deliberate soft or no-op resets are exempted by name with the reason. The particle
filter is exempted until its reset contract is decided (see Open decisions).

Follow-up after external review:

- **Reset check had a NaN hole.** `std::max(0.0, NaN)` is 0, so a reset estimator returning NaN
  passed, and finiteness was checked only on the fresh copy. Both copies are now checked, a
  NaN on one side is a failure, and reset is tried after 1, 7, 29, 47 and 3007 observations
  (inside rank-PIT calibration, inside burn-in, mid-block for the 20-tick block estimators, and
  after a long run). A ρ = −0.7 stream checks range under negative dependence. All 94 keys pass.
- **Two Kendall bugs, neither reaching a result.** `windowed_rank::kendall()` subtracted unsigned
  counts before converting, so perfectly negative data gave about 1.5 × 10¹⁶ instead of −1.
  `HermiteEstimator::kendall()` used 4 tr(Wᵀ Aᵀ W) − 1, which is linear in A; τ = 4∫∫F dF − 1 is
  quadratic, 4 tr(Aᵀ W A Wᵀ) − 1, with W_kj = ∫ h_k H_j. The old formula clamped to +1 at every
  ρ; the new one gives −0.714, −0.341, 0.330, 0.716 against true τ −0.713, −0.333, 0.333, 0.713
  (Gaussian, n = 20000). No code calls either method; the manifest is identical before and after.
- **`dump_trajectories`.** The bit-exact gate described in Principles. 3 seeds × 5 scenarios × 94
  keys runs in about 25 s.

## Stage 2: shared helpers that reproduce existing outputs

Stage 2 removes duplicated code without changing any estimate. Every step is accepted only if
`multiseed` and the affected probes produce byte-identical CSVs before and after.

1. **`src/estimators/memory.hpp`.** One home for memory conversions and construction:
   α ↔ ESS = (2−α)/α, Kalman steady-state memory from σ²_ω, and `make_at_memory(family, ticks)`
   mapping a memory to each family's own λ. It replaces copies in `memory_grid.hpp`,
   `probe_memory`, `probe_adaptive`, `probe_vff_norm` and the async studies. This is also the single
   place the mirror-image λ convention lives, so it is tested once (Xiao retention 1−α, Hermite
   weight α, Kalman via the gain).
2. **`src/estimators/dependence/ranks.hpp`.** `average_ranks` and `block_spearman` are copied in six
   headers (three Kalman files, `nonlinear_kalman`, `particle_filter`, `windowed_rank`). Same
   arithmetic, one copy.
3. **`experiments/scenarios.hpp`.** `make_dependence` is redefined in at least six experiment
   files. One definition keeps every harness on the same five scenarios. Nothing new enters `src/`.
4. **Fixed-parameter draws on `dgp::Copula`.** `tail_common.hpp` re-implements the DGP samplers
   with the parameter held fixed. Adding that draw to the DGP classes lets the tail studies use the
   same code the main harness samples from.

Risk: floating-point order. If moving code changes the order of operations, outputs can differ in
the last bits. Any such difference fails the byte-identical check and is investigated before
merging, never waved through.

## Stage 3: versioned statistical changes, then reruns

Stage 3 changes what estimators compute, so each change adds a configuration beside the old one
and is rerun before any thesis number is replaced.

### Hermite warm-up

Decided 2026-10-20: implemented on `bank-vff-study` as the opt-in `Config::cumulative_start` flag,
not on this branch. `hermite_ew` is unchanged; the flag-on configuration is registered as
`hermite_ew_cs`.

- **Change.** Coefficient weight w_m = max(λ, 1/m), where m counts *coefficient updates*, not
  incoming observations, so the first update has weight 1 even under rank-PIT (where it lands at
  n = 31). Marginal moments are Welford while 1/n > λ, then exponentially weighted.
- **Why a separate clock.** `update()` skips the coefficient update at n = 1, and with rank-PIT
  until the window holds 30 points. On the observation count, accumulated weight at tick 50 with
  λ = 0.01 would be 0.98, or 0.40 with rank-PIT, not 1.
- **Checks reported by that session.** The accumulated weight is exactly 1 through and after
  warm-up, with and without rank-PIT. A hard reset equals fresh construction. A soft reset during
  warm-up keeps counting updates with weight min(1, max(λ, 1/m)·boost). `debias` is a no-op under
  the flag for both `estimate()` and `kendall()`. Per-tick scale invariance holds to 2e-14 for
  (x, y) → (3+5x, −2+0.1y). Under the old start the difference reaches 0.067: the initial variance
  of 1.0 acts as an artificial observation, and with a true variance of 0.01 the y-variance is 36×
  too high at n = 1350, which biases ρ by −0.21.
- **Reruns.** Every comparison that includes a Hermite estimator, with `hermite_ew_cs` beside
  `hermite_ew`: the `multiseed` grid, `probe_memory`, `probe_adaptive` and the async study.

### Bank configurations

- **Change.** Register the banks the thesis reports: 7 arms log-spaced 25–1000 ticks, fixed-share
  π = 0.002, built through `make_at_memory`.
- **Versioning.** The current λ-grid banks become `xiao_bank_lambda`, `hermite_bank_lambda` and
  `mle_bank_lambda`; the plain names point to the tick-grid banks.
- **Acceptance.** First, identical per-tick estimates from the registry bank and the probe's bank
  on the same input stream. Then the summary, on a named quantity: `multiseed` reports the mean of
  per-seed RMSEs, `probe_adaptive` also reports pooled RMSE (√ of the mean squared error over all
  seeds and ticks). These differ, so compare mean per-seed RMSE with mean per-seed RMSE.

### Memory metric

- **Known undercounts.** `gaussian_mle_pit` returns `sizeof(*this)` and omits its two rank-PIT
  windows. Hermite omits the coupling matrix `W_` and the vector `z_`, and counts `A_fast_` only
  when fast-surprise tracking is on, although all three are always allocated.
- **One definition.** `store_bytes()` = `sizeof(*this)` + allocated heap capacity (`capacity()`,
  not `size()`, for vectors; rows × cols for Eigen), measured after warm-up. Embedded members are
  already inside `sizeof(*this)`, so only their heap blocks are added, never their `sizeof` again.
  Shared constant tables (one copy per process) are reported separately rather than per instance.
- **Check.** Add to `check-estimators` a test that `store_bytes()` is at least the sum of the
  heap capacities it can see, plus a per-family audit by hand. Then regenerate the cost table;
  ranks between families may move.

### Block-Spearman bias

- A 20-tick block's Spearman has expectation ((n−2)ρ_S + 3τ)/(n+1); by simulation 0.574 against a
  true 0.600 at n = 20. Short-memory Kalman configurations inherit it (−0.023 to −0.029 on
  `constant`). Averaging in Fisher-z happened to offset it in one configuration, but that is
  configuration-dependent and is not a correction.
- **Next.** Measure bias and its share of MSE on the `jump` window itself, per configuration,
  before saying how much it matters; the constant-stream figure does not carry over. Only then
  decide between larger blocks, an unbiased block statistic, or reporting it as a limitation.

## Stage 4: cosmetic cleanup

Stage 4 changes no estimate and comes last, so its diffs never hide a statistical change. Each item
is accepted on byte-identical outputs.

- **Registry split.** 94 keys, many one-off ablations (`kalman_q*`, `_inf2/5/10/50`, detector
  variants). Keep one core list for `multiseed` by default and an archive list for reproducing old
  notes.
- **Run list ≠ comparator set.** `multiseed`'s "vs best" column and `probe_adaptive`'s
  `known_best` / `one_setting` baselines take an argmin over whatever was run. Shrinking the run
  list can therefore move a benchmark, and any regret against it, with every estimate unchanged.
  So the split defines two explicit lists: `run_keys()` (what executes) and a fixed
  `comparator_keys()` (what "best" and the oracles range over), with a check that the comparator
  set is a subset of the run list. This item is cosmetic only if `comparator_keys()` equals
  today's set.
- **Comments.** Several headers carry long derivations and section histories, against
  `CLAUDE.md`'s nearly-comment-free rule; move them into `notes/`.
- **Build.** Makefile Homebrew paths become `EIGEN ?=` and `JSON ?=`, so Linux builds need no
  edits; fix the two initialiser-order warnings.
- **`CLAUDE.md`.** Update stale lines: rank-PIT is implemented, `multiseed` exists, and
  `thesis_audit.md` predates the branch consolidation.
- **Per-tick costs.** `RankPit` erases from the front of a vector, Xiao's `estimate()` allocates
  per call, and `windowed_rank` re-sorts its window each estimate. Fixing these changes timing
  figures, so re-time before quoting any.

## Merge plan with `bank-vff-study`

Decided 2026-10-20:

1. `massive-refactor` merges into `main` first.
2. `bank-vff-study` then merges `main` in with a merge commit, not a rebase.
3. The gate after that merge compares trajectory hashes key by key against `main`, using
   `dump_trajectories` plus the probe manifests. Exactly two exemptions are allowed, and they are
   listed by name:
   - `copula_select_ew` changes on purpose. Commit 9d16c53 fixes the `NonparamCopula` counts and
     switches the ρ readout to mid-rank Pearson. It is a bug fix with no legacy mode, and the
     older `copula_select_ew` numbers are marked superseded.
   - `hermite_ew_cs` is a new key.

   Any other difference fails the gate.

## Open decisions

- [ ] **Particle filter reset contract.** Should `reset()` mean "identical to a fresh filter with
  the same seed" (reseed the RNG) or "restart the state, keep the random sequence"? Neither is
  statistically wrong; it only sets what `check-estimators` can promise.
- [x] **Hermite warm-up form.** Opt-in `cumulative_start` flag on `bank-vff-study`, registered as
  `hermite_ew_cs`; `hermite_ew` unchanged (decided 2026-10-20).
- [ ] **Hermite default.** After the reruns, does the thesis cite `hermite_ew_cs` in place of
  `hermite_ew`, and is the debias flag then kept only to reproduce old notes?
- [ ] **Block-Spearman bias.** After measuring it on `jump`: correct it (larger blocks or an
  unbiased statistic) or report it as a limitation?
- [ ] **Registry archive.** Do the detector keys stay registered, now that detection is out of the
  thesis, or move to the archive list?
