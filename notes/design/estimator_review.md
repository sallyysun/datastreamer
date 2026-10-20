# Estimator code review: design proposal

2026-10-20. Living copy: the "Estimator code review: design proposal" doc.

The review's fixes are split into four stages, and only stage 3 is allowed to change what an
estimator computes. Bug fixes and refactors must reproduce existing outputs exactly; any change
that can move a thesis number ships as a new, named configuration beside the old one, and is rerun
before any number is replaced.

## Principles

Startup behaviour, block bias and configuration differences are part of the statistical
experiment, not code maintenance. Four rules follow from that.

1. **Refactors and bug fixes reproduce existing outputs exactly.** Acceptance is a byte-identical
   `multiseed` CSV (and probe CSV where relevant) before and after, run on the same platform.
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
| Shared memory helpers (`memory.hpp`: α ↔ ESS, Kalman memory, construct-at-memory) | 2 | No | Probe and `multiseed` outputs byte-identical |
| Shared rank code (`ranks.hpp`) | 2 | No | Outputs byte-identical |
| Shared scenario definitions (`experiments/scenarios.hpp`) | 2 | No | Outputs byte-identical |
| Fixed-parameter draw on `dgp::Copula`, used by the tail studies | 2 | No | Tail study outputs byte-identical |
| Hermite warm-up w = max(λ, 1/n) | 3 | Yes | Legacy key kept; rerun all Hermite comparisons |
| Banks registered on the tick grid used in the thesis | 3 | Yes | Legacy λ-grid keys kept; `multiseed` matches `probe_adaptive` |
| `gaussian_mle_pit` counts its rank windows in `store_bytes` | 3 | Yes (memory metric) | Cost table regenerated |
| Registry split, comment trim, Makefile paths, warnings | 4 | No | Outputs byte-identical |

## Stage 1: reset fixes and contract tests

Done in PR #11; no result changes. `ParametricCopulaMLE::reset()` now clears its rank-PIT windows,
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

- **Change.** Weight each update by w_n = max(λ_n, 1/n) in both the coefficient sums and the
  marginal mean and variance. The estimator then runs as a cumulative average until 1/n falls
  below λ, so its weights always sum to one. The existing opt-in debias corrects only the
  coefficient sums.
- **Versioning.** `hermite_ew_legacy` keeps today's behaviour exactly; the warm-up becomes a config
  flag, and `hermite_ew` switches to it only after the reruns. The live consumer follows the
  registry.
- **Checks before rerunning.** That debias is a no-op under the new warm-up must be verified after
  a soft reset and during a forgetting boost, not assumed: both change the weight sequence
  mid-stream.
- **Reruns.** Every comparison that includes a Hermite estimator: the `multiseed` grid,
  `probe_memory`, `probe_adaptive`, and the async study. Long-memory configurations matter most,
  because warm-up lasts about 1/λ ticks.

### Bank configurations

- **Change.** Register the banks the thesis reports: 7 arms log-spaced 25–1000 ticks, fixed-share
  π = 0.002, built through `make_at_memory`.
- **Versioning.** The current λ-grid banks become `xiao_bank_lambda`, `hermite_bank_lambda` and
  `mle_bank_lambda`; the plain names point to the tick-grid banks.
- **Acceptance.** `./multiseed` on the plain key matches `probe_adaptive`'s bank row exactly.

### Memory metric

- `gaussian_mle_pit` returns `sizeof(*this)` and omits its two rank-PIT windows. Count them, and
  define `store_bytes` once for all estimators (object plus heap, measured after warm-up), then
  regenerate the cost table.

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
- **Comments.** Several headers carry long derivations and section histories, against
  `CLAUDE.md`'s nearly-comment-free rule; move them into `notes/`.
- **Build.** Makefile Homebrew paths become `EIGEN ?=` and `JSON ?=`, so Linux builds need no
  edits; fix the two initialiser-order warnings.
- **`CLAUDE.md`.** Update stale lines: rank-PIT is implemented, `multiseed` exists, and
  `thesis_audit.md` predates the branch consolidation.
- **Per-tick costs.** `RankPit` erases from the front of a vector, Xiao's `estimate()` allocates
  per call, and `windowed_rank` re-sorts its window each estimate. Fixing these changes timing
  figures, so re-time before quoting any.

## Open decisions

- [ ] **Particle filter reset contract.** Should `reset()` mean "identical to a fresh filter with
  the same seed" (reseed the RNG) or "restart the state, keep the random sequence"? Neither is
  statistically wrong; it only sets what `check-estimators` can promise.
- [ ] **Hermite default.** After the reruns, does `hermite_ew` switch to the max(λ, 1/n) warm-up,
  with the debias flag kept only to reproduce old notes, or deleted?
- [ ] **Block-Spearman bias.** After measuring it on `jump`: correct it (larger blocks or an
  unbiased statistic) or report it as a limitation?
- [ ] **Registry archive.** Do the detector keys stay registered, now that detection is out of the
  thesis, or move to the archive list?
