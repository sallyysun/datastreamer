# Chapter 6 draft: where each number comes from (2026-11-01)

Draft: `chapter6_draft.tex`. Platform for everything new: Linux / libstdc++. `[pending]` in the
draft marks text that waits on a run.

| Draft section | Source | Status |
| --- | --- | --- |
| 6.1 design | `experiments/probe_thesis_bank.cpp`, `copula_source.hpp`, `src/streamgenerator/datasource/{copula,dependence}.hpp`, `forgetting_bank.hpp`, `registry.hpp` | current code |
| 6.2 fixed memory, Table 6.2 | `notes/studies/memory_sweep.md` §4–5 (`probe_memory.cpp`) | **macOS, 40 seeds: rerun on Linux** |
| 6.3 Tables 6.3–6.5, Fig. 6.1 | `results/thesis_bank[_clayton\|_gumbel\|_t].csv`, `thesis_bank_weights.csv`; `ch6_results_md.py`, `thesis_bank_table.py`; `plot_thesis_bank.py` → `figures/ch6_bank_weights.png`; all numbers in `docs/chapter6_results.md` | 60 seeds, **tuned** η, δ (`notes/studies/thesis_bank_tuning.md`). Refreshed 2026-11-01; drift + jump are the focus, constant + oscillate one paragraph |
| 6.4 Table 6.6 | same CSVs (log bank minus squared bank, paired) | refreshed: both banks tuned (η = 1; δ = 1e-4 squared, 3e-4 log); log bank better in 21/24 drift+jump cells, ties t drift |
| 6.4.1 regret vs §5.3.4 bounds, Table 6.7 | `experiments/probe_bank_regret.cpp` → `results/bank_regret_42[_fam].csv`; `notes/studies/thesis_bank_regret.md` | 60 seeds × 4 families × 3 estimators, drift + jump; bound holds in all 1440 runs (mixture log bank) |
| 6.4.1 η = 1/32 squared bank (prose) | `probe_bank_regret` banks `sq32`, `sq32_dstar`; `probe_thesis_bank` method `bank_sq32`; `experiments/sq32_table.py`; `notes/studies/thesis_bank_sq32.md` | within 32·B in all 1440 runs; loses all 24 drift/jump cells in ρ_S RMSE to both η = 1 banks |
| 6.4.2 own-density bank, Table 6.8 | `probe_bank_tune.cpp` (Loss::Own replay, `excess_own_x1000`, `mem`) → `results/bank_tune_{1,42}[_fam].csv`; `tune_own_table.py`; `notes/studies/thesis_bank_own_density.md` | tuned η = 0.25, δ = 3e-3 on seeds 1–30; held out 42–101 |
| 6.5 shape switches | `probe_bank_tune.cpp` with `COPULA=gauss2t\|clayton2gumbel` → `results/bank_tune_42_{gauss2t,clayton2gumbel}.csv`; `tune_own_table.py` §3 | Algorithm 1, own-density bank, 60 seeds (replaces the block-bank result of `memory_tradeoff.md` §3.3) |
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
