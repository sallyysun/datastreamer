# Reference review — the Zotero library against the open items

Pass over all 56 unique items in the Zotero library (2026-09-06), read at abstract level, tagged
against the open items in [todo.md](../todo.md), [roadmap.md](roadmap.md) and
[change_detection.md](change_detection.md).

**Two findings up front:**

1. **The changepoint literature this thesis is built on is not in the library.** Adams & MacKay,
   Page, Lorden, Pollak, Shiryaev, Lai, Basseville & Nikiforov, Tartakovsky — every reference in
   [kalman_changepoint_results.md](kalman_changepoint_results.md) §11 — is absent. The library is a
   *copula/finance* library; the sequential-analysis half is missing entirely.
2. **VFF has no citations at all.** `roadmap.md` §336 recommends the variable forgetting factor as
   *the* contribution, and nothing in the library supports it. VFF proper lives in adaptive
   filtering / RLS (Paleologu, Haykin), which is not represented. Novelty cannot be claimed against
   a literature that has not been read.

---

## Tier A — act on these

| Paper | Why | Open item |
| --- | --- | --- |
| **Gregory (2019)**, *A streaming algorithm for bivariate empirical copulas* | Read in full — see [§Gregory](#gregory-2019-read-in-full) below. Distribution-free streaming empirical copula with error bounds, but **cumulative: no forgetting**. | rank-PIT, `store_bytes`, contribution |
| **Roberts (2000)**, *Control Chart Tests Based on Geometric Moving Averages* | The EWMA control chart: the canonical detector built on exactly the recursion our fixed-$\lambda$ estimators already are. Uncited (the "Roberts" in the notes is Shiryaev–Roberts). | detector ladder |
| **Fantazzini (2009)**, *Effects of misspecified marginals and copulas on VaR* | Finds that with skewness in the data and **symmetric marginals**, estimated elliptical copula correlations are negatively biased by **up to 70% of true value**. That is marginal misspecification corrupting the *copula* parameter — precisely the risk our Welford standardization runs. Strongest single argument for rank-PIT. Uncited. | rank-PIT |
| **Pereira et al. (2026)**, *Bootstrap-enhanced Fisher scoring for state-space models* | Bootstrap approximation of the Fisher information for SSM parameter MLE; targets small samples and misspecification. Directly the machinery for estimating $\sigma_\omega^2$ online. Uncited. | VFF / §8 |
| **Segers (2012)** + **Bücher & Volgushev (2013)** | Weak convergence of empirical *and sequential* empirical copula processes under serial dependence, with multiplier-bootstrap resampling. This is the theory for the non-i.i.d. null that §3.7's bootstrapped $\mathrm{ARL}_0$ currently sidesteps empirically. Bücher cited; Segers not. | $\mathrm{ARL}_0$ validity |
| **Kojadinovic & Yan (2010)**, *the* `copula` *R package* | Rank-based GoF testing framework for copulas — the established batch counterpart to `CopulaSelector`. Both a validation target and the standard our online selection should be positioned against. Uncited. | `copula_select` |
| **Pérez-Cambriles et al. (2026)**, *Time-varying copulas in R* | Dynamic-copula estimation package. Offline/batch, so it belongs in the `external/` cross-validation slot with `hermiter` and `onlineNPCORR`, not as a competitor. Uncited. | validation |
| **Cormode (2011, 2011)**, sketches / synopses | Greenwald–Khanna (which Gregory adapts) is a quantile sketch; this is the background for it and the basis for defending `store_bytes` as a streaming-admissibility claim rather than a measurement. Uncited. | `store_bytes` |

## Tier B — framing and positioning

The **GAS / observation-driven** cluster is the significant one: **Salvatierra & Patton**
(*GRAS*), **Opschoor et al. (2021)**, **Ausin & Lopes (2010)**, **Tófoli et al. (2019)**. These are
the mainstream approach to time-varying copulas — parameters driven by the *score* of the observed
likelihood — and they are the direct alternative to the parameter-driven state-space view
`change_detection.md` develops. None are cited. **The thesis needs to position against GAS
explicitly**; "observation-driven vs parameter-driven" is a standard dichotomy and a reader will
expect it.

Also: **Patton (2006)** conditional copulas, **Oh & Patton (2017)** factor copulas,
**Dias & Embrechts** and **Münnix & Schäfer (2011)** (non-constancy of the conditional copula in
intraday data — empirical motivation that dependence really does move), **Genest & Favre (2007)**
and **Embrechts (2009)** as rank-inference and survey background, **Nelsen (2006)** / **Sklar** as
the standard citations.

## Tier C — application track ([application_2.md](application_2.md))

Tófoli (D-vine VaR), Jeong & Dey (vine reserving), Okhrin & Tetereva (realized HAC),
Garcia-Jorcano (Bitcoin hedging), Reboredo (gold/oil), Li et al. (COVID banking systemic risk),
Wu & Liang (stock–bond covariance), Geenens & Dunn (nonparametric conditional VaR),
Virbickaitė (density pooling), Fiszeder (OHLC DCC), Dai et al. (large covariance, async data —
already cited), Chakrabarti & Sen (nonsynchronous data — bears on the live path, where `CLAUDE.md`
warns throttling can desync the two legs).

## Tier D — not relevant to this thesis

Hyndman (*Forecasting*), Wang & Cook (tsibble, *Conversations in Time*), Nisan et al. (regret
minimization), Roughgarden et al. (parameterized algorithms), Vaze (online algorithms — the
competitive-ratio sense, not streaming estimation), Lambert & Vandenhende (longitudinal medical),
Khan (decision trees), Erdemir (pseudo-copula regression), Bermúdez & Karlis (insurance counts),
Li et al. (hospital service quality), Ahn et al. (SGFS — MCMC posterior sampling, not our problem),
Wang (2010) Fisher scoring interpolation (relevant only if the empirical-Fisher choice in
`ParametricCopulaMLE` is ever questioned), and the philosophy items.

---

## Gregory (2019) — read in full

**What it is.** $L+1$ coupled $\varepsilon$-approximate quantile summaries: one $S_{(1)}$ over the
first margin, with a sub-summary $S^i_{(2)}$ hanging off each of its tuples. `Insert` places into
both; `Combine` runs every $\lfloor 1/(2\varepsilon)\rfloor$ elements and merges sub-summaries
alongside their parent tuples. Queries return $\varepsilon'$-accurate empirical copula values.
Worst-case space is quadratic in $1/\varepsilon$ and log-squared in $n$ — worse than 1-D
Greenwald–Khanna's $O(\varepsilon^{-1}\log \varepsilon n)$. Higher dimensions only via pair-copula /
vine constructions.

**It does not forget.** No occurrence of *forget*, *decay*, *sliding*, *window*, *exponential*,
*time-varying* or *drift* in the body. The summary is explicitly over the whole stream
($n = \sum_i g^i_{(1)}$ counts from inception) and `Combine` only merges tuples — it never ages or
down-weights. The one appearance of "non-stationary" is a **bibliography entry he does not act on**:
Arandjelović, Pham & Venkatesh (2015), *The adaptable buffer algorithm for high quantile estimation
in non-stationary data streams* (IJCNN).

**Consequences for this thesis:**

1. **It is the $\lambda = 0$ column.** Structurally cumulative, so it would behave like `xiao` /
   `hermite`: best-in-class on `constant`, poor on everything that moves. Not a flaw in the
   algorithm — it has no memory dial at all, and the memory dial is this thesis.
2. **It gives two things nothing here has**: a distribution-free marginal transform (the rank-PIT
   open item) and a *proven error bound*.
3. **He names Xiao as the weaker alternative.** From his introduction: Xiao (2017) "doesn't provide
   any theoretical error bounds on the summarized statistical approximations." That is a direct
   criticism of the estimator this project benchmarks against, and belongs in related work.
4. **The gap is the contribution.** Streaming empirical copula **+ forgetting** does not exist.
   Implementing Gregory as-is reproduces published work (one more baseline row); adding an
   exponentially-weighted or adaptable-buffer variant of his structure is a methodological
   contribution, and it is the same efficiency↔adaptation question the rest of the thesis asks,
   relocated into the marginal transform.
5. ⚠️ **Cost to the streaming-admissibility claim.** $O(\varepsilon^{-2}\log^2 n)$ is sublinear but
   *growing*, where every current estimator is flat. `store_bytes` would show it climbing. The trade
   — distribution-freeness and error bounds for non-constant memory — must be stated, not buried.

**Next**: Arandjelović et al. (2015), the forgetting-capable quantile sketch Gregory points at and
declines to use. Not currently in the library.

## What to read next, in order

1. **Gregory (2019)** — decides whether a streaming empirical copula joins the estimator bank.
2. **Fantazzini (2009)** — decides how urgent rank-PIT is. The 70% bias figure suggests: urgent.
3. **The GAS cluster** (Opschoor, Salvatierra & Patton) — needed to position the state-space framing.
4. **Segers (2012)** — needed before $\mathrm{ARL}_0$ can be defended rather than bootstrapped.

## What to acquire

- **Sequential analysis**: Page (1954), Lorden (1971), Pollak (1985), Shiryaev, Roberts (1966),
  Lai (1995), Basseville & Nikiforov (1993), Tartakovsky et al. (2014), Adams & MacKay (2007).
  All cited in §11 of the results document, none in the library.
- **VFF / adaptive filtering**: the variable-forgetting-factor RLS line (Paleologu et al.; Haykin,
  *Adaptive Filter Theory*).
- **Conformal / e-values**: Vovk, Ramdas — `change_detection.md` Part IV.1 calls this the strongest
  novelty candidate in the whole document, and none of it is here.
