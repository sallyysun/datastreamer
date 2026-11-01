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
