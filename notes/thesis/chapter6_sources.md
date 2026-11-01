# Chapter 6 draft: where each number comes from (2026-11-01)

Draft: `chapter6_draft.tex`. Platform for everything new: Linux / libstdc++. `[pending]` in the
draft marks text that waits on a run.

| Draft section | Source | Status |
| --- | --- | --- |
| 6.1 design | `experiments/probe_thesis_bank.cpp`, `copula_source.hpp`, `src/streamgenerator/datasource/{copula,dependence}.hpp`, `forgetting_bank.hpp`, `registry.hpp` | current code |
| 6.2 fixed memory, Table 6.2 | `notes/studies/memory_sweep.md` §4–5 (`probe_memory.cpp`) | **macOS, 40 seeds: rerun on Linux** |
| 6.3 Tables 6.3–6.5, Fig. 6.1 | `results/thesis_bank[_clayton\|_gumbel\|_t].csv`, `thesis_bank_weights.csv`; `thesis_bank_table.py`; `plot_thesis_bank.py` → `figures/ch6_bank_weights.png`; `notes/studies/thesis_bank.md` | 60 seeds, η and δ **untuned** |
| 6.4 Table 6.6 | same CSVs (log bank minus squared bank, paired) | η and δ differ between the banks: confounded until tuned |
| 6.5 shape switches | `notes/studies/memory_tradeoff.md` §3.3 (block bank, histogram density) | **earlier algorithm: rerun with Algorithm 1** |
| 6.5 tail dependence | `notes/studies/tail_dependence.md` results 1, 3, 4; `oracle_memory.md` §2 | stationary validation only; family bank and χ bank not built |
| 6.6 cost, Table 6.7 | one-off timing of registry keys over 200 000 ticks (`store_bytes()`, wall clock) | indicative; MPL `store_bytes` fixed this day to count its rank windows |

## Corrections made while drafting

- **Pre-chosen memory** is M = 99 for Gaussian and t but M = 199 for Clayton and Gumbel (all
  components). Against it both banks win 48/48 cells by 3–93 ×10⁻³. `thesis_bank.md` had said
  "M = 99 in every case", checked on Gaussian and t only.
- **Uniform averaging** is worse than both banks in 46/48 cells, not all: Hermite on Clayton and
  Gumbel `oscillate` are ties.
- **`ParametricCopulaMLE::store_bytes()`** returned `sizeof(*this)` and so omitted its two rank-PIT
  windows (~16 kB at W = 500). `mpl_*` now reports 16 400 bytes instead of 368.

## Is the thesis's current §6.1 accurate?

No, for two reasons.

1. **It describes the wrong process.** §6.1 is the live server's generator. No result in
   Chapter 6 uses the live server; every number comes from the offline harness. Move it to §2.3.
2. **It is also wrong about the live server.** Checked against
   `src/streamgenerator/datasource/synthetic_source.cpp` and `main.cpp`:

| §6.1 says | the code does |
| --- | --- |
| a 4-variate Gaussian copula over (r_x, r_y, V_x, V_y) with covariance Σ | a **bivariate** copula on the two returns only |
| volumes log-normal, `V = exp(μ_V + σ_V Z)` | volumes are **constant** at 1000; nothing draws them |
| Gaussian copula | family chosen by `COPULA=gaussian\|clayton\|gumbel\|t` (plus `gauss2t`, `frechet`, `mixgauss`) |
| fixed dependence Σ | time-varying `d_t` chosen by `DEPENDENCE=` (default `oscillate`, period 400) |
| GBM returns `r = μ + σZ` | ✓ μ = 1e-4, σ = 0.01, P₀ = 100 |

The draft's §6.1 replaces it with the harness design and keeps one paragraph saying the live
server uses the same copula sampler, followed by a monotone GBM transform of each margin.
