# The η = 1/32 squared bank (2026-11-01)

Keys: `xiao_bank_sq32`, `hermite_bank_sq32`, `mle_bank_sq32` (`registry.hpp`). This is Algorithm 1 with
squared loss at η = 1/32, the largest learning rate at which (p̂ − ρ)² is exp-concave on
p̂ ∈ [−3, 3], ρ ∈ [−1, 1] (thesis §5.3.4). δ = 1e-4, as `*_bank`, so η is the only change.
Probes: `probe_bank_regret` (banks `sq32` and `sq32_dstar`) and `probe_thesis_bank` (method `bank_sq32`).
Table script: `experiments/sq32_table.py`. Platform: Linux / libstdc++. Seeds 42–101, 4 families ×
3 estimators. Rerunning both probes reproduced every existing row bit for bit.

## Why δ = 1e-4

δ = 1e-4 keeps the comparison with the η = 1 bank one-knob. The theory's δ* = Km/((K−1)(T−1))
for m = 1 and T = 4970 is 2.5e-4. It is run in the regret probe as `sq32_dstar`, and it moves the
mean regret by less than 1 at every m.

## Regret against 32·B_{T,m}(δ) (720 runs per scenario)

| bank | scenario | m | mean | max | bound | max/bound | viol |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sq (η = 1) | drift | 0 | 4.84 | 11.22 | 2.0 (B/1) | 5.59 | 688 |
| sq (η = 1) | jump | 0 | −7.60 | 5.98 | 2.0 (B/1) | 2.98 | 21 |
| sq32 | drift | 0 | 7.79 | 15.88 | 64.2 | 0.25 | 0 |
| sq32 | drift | 1 | 12.33 | 22.16 | 410.5 | 0.05 | 0 |
| sq32 | drift | 5 | 28.72 | 41.72 | 1795 | 0.02 | 0 |
| sq32 | jump | 0 | 3.94 | 9.76 | 64.2 | 0.15 | 0 |
| sq32 | jump | 1 | 19.44 | 36.29 | 410.5 | 0.09 | 0 |
| sq32 | jump | 5 | 41.58 | 62.57 | 1795 | 0.03 | 0 |
| sq32_dstar | drift | 0 | 8.49 | 17.68 | 83.5 | 0.21 | 0 |
| sq32_dstar | jump | 1 | 18.80 | 34.72 | 400.2 | 0.09 | 0 |

The bound holds in every run for every m ∈ {0, 1, 2, 3, 5, 10, 20, 50}. It is loose by a factor of 4
or more, and the η = 1/32 bank's actual regret is **higher** than the η = 1 bank's at every m.
The guarantee comes from the 32× larger bound, not from better tracking.

## ρ_S RMSE (×10⁻³, paired mean ± 2 SE, whole window [500, 5000))

Ranges over 4 families × 3 estimators:

| scenario | sq32 − bank (η = 1) | sq32 − bank_log | sq32 − uniform | sq32 − best fixed |
| --- | --- | --- | --- | --- |
| drift | +3.0 to +5.3 | +3.7 to +8.8 | −10.0 to −14.4 | +8.7 to +11.5 |
| jump | +14.9 to +17.9 | +20.1 to +25.1 | −4.8 to −8.8 | +4.0 to +5.9 |
| constant | +13 to +18 | +13 to +22 | −5 to −11 | +20 to +30 |
| oscillate | −13 to +8 | −12 to +9 | −7 to −107 | −9 to +10 |

On drift and jump, out of 24 cells, sq32 against:

- bank: 0 wins / 24 losses;
- bank_log: 0 / 24;
- uniform: 24 / 0.

After the jump:

- ticks 0–100: +67 to +88 behind the η = 1 bank;
- ticks 100–500: +37 to +55 behind the η = 1 bank, and 18–34 ahead of uniform.

## Reading

- At η = 1/32 the weights move about 32× more slowly. The bank sits between the uniform average and
  the η = 1 bank, closer to uniform right after a change.
- Of the banks run, only the MoM log bank (η = 1) is both covered by its bound at the learning rate
  it is run at and competitive in ρ_S. The η = 1/32 bank is the covered squared bank, kept as a
  labelled comparator. Whether the η = 1 squared bank stays the headline is the author's call.
