# Thesis Algorithm 1 bank: first evaluation (2026-11-01)

Probe: `experiments/probe_thesis_bank.cpp`. Code: `ForgettingBank`
(`notes/code_changes/code changes 2026-11-01.md`). Platform: Linux / libstdc++. Seeds 42–101
(60), N = 5000, scored over [500, 5000). **η and δ are untuned** (squared η = 0.5, δ = 1e-4; log
η = 1, δ = 4e-4), so these numbers are provisional until η and δ are tuned on separate seeds.

## Setup

- **Components:** Xiao, Hermite and online MPL. Five fixed memories each, M = 39, 99, 199, 399,
  1999 ticks. All see the same shared rank-PIT inputs.
- **Methods:** the five fixed arms, the squared-loss bank, the log-loss bank (Gaussian
  method-of-moments density), and uniform averaging of the arms.
- **DGP families:** Gaussian, Clayton, Gumbel and t (ν = 4). Clayton and Gumbel cannot go
  negative, which distorts their `oscillate` rows.

## Results

All differences are RMSE in ×10⁻³ (×0.001 in ρ_S), with paired 95% CIs.

**Against the hindsight-best fixed memory**, chosen per scenario × component after seeing the result:

| scenario | squared bank | log bank | uniform |
| --- | --- | --- | --- |
| `jump` | −3.5 to −7.3, CI ±1.2–1.7 (all 12 cells better) | **−15 to −21**, CI ±1.2–1.9 (all 12 better) | +9.5 to +14.7 |
| `drift` | +2.4 to +4.2 | +2.5 to +8.2 | about +21 to +25 |
| `constant` | +7.2 to +10.4 | +6.4 to +15.2 | +25 to +40 |
| `oscillate` (Gaussian, t) | +0.4 to +0.7 | +2.2 to +18.4 | about +100 |

**Against one fixed memory chosen in advance** (the best mean over all four scenarios, which is
M = 99 in every case): both banks win in every scenario × component × family checked (Gaussian
and t). The gains are 9–93 ×10⁻³, with all CIs excluding 0.

**Excess log loss** ranks the methods the same way. The log bank beats the hindsight best on
`jump` (−1.0 to −3.8 ×10⁻³ nats/tick), and the squared bank does not (+0.5 to +21). The squared
Hermite bank is the noisy case (Clayton +20.6 ± 12.2), because it weights the noisy short arm.

**Jump recovery** (Gaussian, Xiao):

| window | best fixed | log bank |
| --- | --- | --- |
| first 100 ticks | 208 (M = 39) | 315 |
| 100–500 ticks | 48 (M = 99) | 59 |
| 500–1500 ticks | 26 (M = 399) | 33 |
| whole post-change | 75 | **73** |

No single arm is good in all three windows. The bank is close to the best arm in each, and that
is where its whole-run gain comes from.

## Reading

- **Good:** the bank removes the need to know the memory in advance. It always beats any single
  pre-chosen memory by a wide margin, and on a jump it beats even the per-scenario hindsight
  choice. The log bank is clearly better on the jump.
- **Cost:** when the right memory is fixed (`constant`, `drift`), the bank pays a few ×10⁻³ RMSE
  against the hindsight best.
- **Log loss when the copula family is wrong:** on t data the log bank's cost grows (`constant`
  +14, `oscillate` +11 to +18), because the Gaussian method-of-moments density is misspecified there.
- **Uniform averaging is much worse than either bank.** Choosing among the arms is what does the work.
- **Next:** tune η and δ on separate seeds, then rerun.

## Against variable forgetting factors (same probe, same seeds, same inputs)

Each component's existing VFF rule was added to the probe at two rates, slow 0.05 and fast 0.5.
Each is fed the same probit rank-PIT inputs as the banks, and starts at M = 199 (the bank's
middle arm):

- **Xiao:** rank residual over a known null variance, multiplicative per 20-tick block, clamped
  to the bank's span 39–1999 (`adaptive_memory.md` §5.1);
- **Hermite:** ratio of fast to slow coefficient surprise, clamp [0.2, 20]·λ;
- **MPL:** score drift (`parametric_copula.hpp`), clamp 0.90–0.9999.

Tables come from `python3 thesis_bank_table.py [rmse|excess_x1000] [window]`. The VFF rate is
picked per cell in hindsight, which favours the VFF.

**Count over 48 cells** (4 families × 4 scenarios × 3 components), whole run, against the
better VFF rate in each cell. A win or loss means ±2 SE excludes 0.

| | RMSE: bank better / worse / tie | excess log loss: better / worse / tie |
| --- | --- | --- |
| squared bank | 41 / 5 / 2 | 29 / 15 / 4 |
| log bank | 41 / 6 / 1 | 37 / 6 / 5 |

RMSE ×10⁻³, the method minus the per-cell hindsight-best fixed memory:

| scenario | Xiao VFF (better rate) | log bank (Xiao) | squared bank (Xiao) |
| --- | --- | --- | --- |
| `constant` | +0.7 to +4.4 (fast; on t, +22.8) | +6.5 to +13.8 | +7.9 to +10.1 |
| `drift` | −0.6 (Gaussian) to +20.5 (t) (slow) | +2.5 to +6.0 | +2.6 to +3.4 |
| `jump` | −2.8 to 0.0 (slow) | **−15.0 to −20.1** | −4.2 to −5.6 |
| `oscillate` | +11.9 to +88.3 (fast) | +1.7 to +13.7 | +0.2 to +0.5 |

**Reading**
- **Where the VFF wins.** It is competitive exactly where the right memory is long and fixed, or
  changes slowly. Xiao's VFF grows to its 1999 cap on `constant`, so it sits at the best arm,
  while the bank pays its fixed-share and mixing cost (+5 to +7 against the VFF). On Gaussian
  `drift`, Xiao's slow VFF is the best method overall.
- **Where the bank wins.** The bank wins on abrupt change and on fast change. On a jump, the slow
  VFF barely beats the best fixed memory (−3 to 0), and the log bank gains a further 12–20. On
  `oscillate` no VFF rate tracks the period of 400, and the bank sits on the M = 39 arm.
- **Two VFF rules are effectively inert or mis-tuned, so the Xiao comparison is the fair one.**
  The Hermite VFF scores within 0.1 of fixed M = 199 everywhere: its fast/slow ratio barely
  moves λ. The MPL VFF at rate 0.5 collapses to its 0.90 floor (+74 to +145). Any claim beyond
  Xiao must say it concerns these specific rules (`chapter6_plan.md` Part B, "one VFF rule").
- This matches `adaptive_memory.md` §5.2 and `memory_tradeoff.md` §4, which used the older block
  bank and other inputs. The VFF integrates surprise into one memory; the bank re-weights K
  memories. A jump is a short spike followed by a quiet new regime, which favours re-weighting.
- **Still to run:** a jump toward *weaker* dependence (0.8 → 0.2), the predicted case where the
  Rosenblatt-type VFF signal points the right way (`memory_tradeoff.md` §4).
