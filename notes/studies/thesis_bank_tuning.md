# Tuning η and δ for the Algorithm 1 bank (2026-11-01)

Probe: `experiments/probe_bank_tune.cpp`; tables: `tune_bank_table.py` (grid) and
`tune_bank_eval.py` (held out). Platform: Linux / libstdc++. N = 5000, scored over [500, 5000).
Follows on from `thesis_bank.md`, where η and δ were untuned.

## Protocol

- **Scenarios: `drift` and `jump` only.** These are the cases the thesis cares about. `constant`
  and `oscillate` are not used for tuning or for the verdict.
- **Tuning seeds 1–30, held-out seeds 42–101.** These are the 60 seeds used in `thesis_bank.md`,
  so the VFF numbers from that probe pair with these.
- **Grid:**
  - squared loss: η ∈ {0.1, 0.25, 0.5, 1, 2, 4, 8, 16};
  - log loss: η ∈ {0.25, …, 16};
  - δ ∈ {0, 1e-5, 3e-5, 1e-4, 3e-4, 4e-4, 1e-3, 3e-3, 1e-2}.
- **Objective:** whole-run RMSE minus the hindsight-best fixed memory of each cell, averaged over
  all 24 cells (4 families × 2 scenarios × 3 components) and seeds. **One (η, δ) per loss**,
  shared by every family and component, because a practitioner does not know the family.
- **How it is computed.** The arms and the shared rank PIT do not depend on (η, δ). Each seed runs
  them once and replays the Hedge and fixed-share recursion for every grid point. At the old
  values the replay matches `probe_thesis_bank` exactly, to every printed digit, for RMSE and
  excess log loss, for both losses.

## Tuning result (seeds 1–30)

Mean regret against the hindsight-best fixed memory, RMSE ×10⁻³:

| | old (η, δ) | tuned (η, δ) |
| --- | --- | --- |
| squared | (0.5, 1e-4): −1.2 | **(1, 1e-4): −3.3** |
| log | (1, 4e-4): −6.6 | **(1, 3e-4): −6.8** |

- **The surface is flat near the optimum.** For the log bank, every point with η ∈ [0.5, 2] and
  δ ∈ [1e-5, 1e-3] lies within about 2 of the best.
- **The per-component optima agree** (η = 1 everywhere; δ = 1e-4 or 3e-4). The tuned values
  therefore do not depend on the component.
- **A large η needs δ = 0 or it breaks.** With η ≥ 4 and δ > 0, single-tick noise flips the
  weights each tick (+20 to +70). With δ = 0, the weights lock onto one arm (+4).
- **On excess log loss**, the log bank's RMSE optimum is also its log-loss optimum (−0.8 nats ×10⁻³).
  The squared bank's is not: at η = 1 it gains 2 in RMSE but loses about 2 ×10⁻³ nats/tick
  against η = 0.5, because it weights the noisy short arm more.

## Held-out result (seeds 42–101, drift + jump)

Paired differences, RMSE ×10⁻³, with a CI of ±2 SE. Negative means the bank is better.

**Against each single fixed memory:** win / loss / tie over the 24 cells.

| | M = 39 | M = 99 | M = 199 | M = 399 | M = 1999 | better VFF rate* |
| --- | --- | --- | --- | --- | --- | --- |
| log bank | 24/0/0 | 24/0/0 | 18/6/0 | 12/12/0 | 24/0/0 | **21/3/0** |
| squared bank | 24/0/0 | 24/0/0 | 14/5/5 | 12/12/0 | 24/0/0 | 21/3/0 |

\* The VFF rate (slow 0.05 or fast 0.5) is picked per cell in hindsight, which favours the VFF.

- **Where it loses to M = 399:** all 12 losses are on `drift`.
- **Where it loses to M = 199:** the 6 losses are Gaussian and t `drift`.
- **Where it loses to the VFF:** the 3 losses are Gaussian Xiao and Hermite `drift` and t Hermite
  `drift`, each by +1.2 to +3.0.

**Against the one memory you would commit to in advance.** This is the fixed M with the best mean
over drift and jump together, in hindsight. It is M = 199 for every family and component.

| | `jump` | `drift` |
| --- | --- | --- |
| log bank | **−14.9 to −20.7** (12/12 better) | −3.8 to −6.6 on Clayton and Gumbel (6/6 better); +1.2 to +3.6 on Gaussian and t (6/6 worse) |
| squared bank | −10.1 to −13.9 (12/12 better) | −0.6 to −2.6 on Clayton and Gumbel (ties); +1.3 to +5.1 on Gaussian and t |

Averaged over the two scenarios, both banks beat M = 199 in every family × component pair. For
the log bank, the jump gain is 4–17 times the drift cost.

**Against the hindsight-best memory per cell:** the log bank wins all 12 `jump` cells (−14.9 to
−20.7) and loses all 12 `drift` cells (+2.2 to +7.4).

**Tuned against old:** the log bank is unchanged (±0.8). The squared bank gains −6.2 to −7.9 on
`jump` and loses +2.2 to +3.2 on `drift`.

**Jump recovery** (Gaussian, log bank, against the best fixed memory in each window):

- first 100 ticks: +108 to +117;
- ticks 100–500: +3 to +14;
- ticks 500–1500: +7 to +9.

The bank never matches the best arm within one window. It wins the whole run because no single
arm is best in every window.

## Reading

- **Verdict, on the criterion "beat any single fixed memory or VFF":**
  - **On `jump`, met in full.** The tuned log bank beats every fixed memory and both VFF rates in
    all 12 cells.
  - **On `drift`, met against M = 39, 99 and 1999 and against the VFF in 9/12 cells.** It is not
    met against the slow memories M = 199 and 399 on Gaussian and t, by +1 to +7.
  - **Why the drift cost is expected.** Drift favours one long memory throughout, and fixed share
    keeps paying to watch the short arms.
  - **Pooled over jump and drift, no single memory beats it.**
- **Use the log bank as the thesis's main bank.** On RMSE it beats the squared bank in 21/24 cells
  (−2.0 to −7.7) and ties in the 3 t `drift` cells. It is also better on log loss. Tuning matters
  for the squared bank (η 0.5 → 1) but hardly at all for the log bank. The log bank's original
  values were already close to optimal.
- **Caveat:** the VFF comparison is fair only for Xiao. The Hermite VFF is inert and the MPL VFF
  at the fast rate collapses (`thesis_bank.md`). The large −12 to −26 margins against the VFF on
  `drift` are MPL cells, plus t Xiao (−15).

## Code

`registry.hpp`: squared `*_bank` η 0.5 → 1 (δ 1e-4 unchanged); `*_bank_log` δ 4e-4 → 3e-4
(η 1 unchanged). `probe_thesis_bank` was rerun with these values, so `results/thesis_bank*.csv`
are post-tuning.
