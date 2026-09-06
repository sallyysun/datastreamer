# Primer — the plain-language version

Everything here is said informally on purpose. The precise versions live in
[roadmap.md](roadmap.md), [change_detection.md](change_detection.md) and
[kalman_changepoint_results.md](kalman_changepoint_results.md); this file is the
"what does that actually mean" companion.

---

## 1. What a copula actually is

Two stocks, AAPL and MSFT. There are **two separate questions** you can ask:

1. *How does AAPL behave on its own?* (big days, small days, how often it crashes)
2. *When AAPL has a big day, what does MSFT do?*

Question 1 is about the **marginals** — one variable at a time. Question 2 is about
the **dependence**. A copula is the answer to question 2 *with question 1 stripped out*.

**How the stripping works.** Don't record "AAPL returned +2.3%". Record **"today was
AAPL's 87th-percentile day"** — i.e. 87% of AAPL days were worse. Do the same for
MSFT. Now every observation is a pair like $(0.87, 0.91)$, both numbers in $[0,1]$.

Once you do that, each variable is uniform on $[0,1]$ *by construction* — every
percentile happens 1% of the time, that's what "percentile" means. So all the
information about "how big are AAPL's moves" has been **deleted on purpose**. What's
left in the pairs is only how the two percentiles move together.

**That joint distribution of percentile-pairs is the copula.** Written $C(u,v)$ — "the
probability that AAPL is below its $u$-th percentile *and* MSFT is below its $v$-th."

**Sklar's theorem** says this split is always possible and unique (for continuous data):

$$F(x,y) = C\big(F_X(x),\, F_Y(y)\big)$$

> joint distribution = marginals + copula

**Why bother.** Because the two halves can change independently. AAPL's volatility can
triple while the copula is untouched. Or volatility can hold steady while the two
stocks start crashing together. **Those are different events and you want to tell them
apart.** The copula is the tool that lets you.

---

## 2. How ρ relates to the copula

Spearman's ρ is **one number computed from the copula**. Literally:

$$\rho_s = 12\int_0^1\!\!\int_0^1 C(u,v)\,du\,dv - 3$$

Don't worry about the formula. The point is the *direction*: $C$ goes in, one number
comes out. **The copula is a whole surface; ρ is a single summary of it.**

> **The analogy to keep.** The copula is a **photograph**. Spearman's ρ is the
> photo's **average brightness**. One number, computed from the whole image, and
> wildly lossy.

**Why that matters.** Two completely different copulas can have *identical* ρ:

| | Gaussian copula, ρ = 0.5 | Clayton copula, ρ = 0.5 |
| --- | --- | --- |
| Typical days | move together moderately | move together moderately |
| **Both crash together** | rarely | **often** |

Same ρ. Same average brightness. **Utterly different photograph** — and the difference
is exactly the thing a risk manager cares about. A detector watching ρ sees nothing
when one becomes the other.

---

## 3. What we monitor now, and what O(k) would change

**Now:** every detector in `registry.hpp` watches **one number**, $\hat\rho_t$. The
estimator contract enforces it — `estimate()` returns Spearman's ρ.

In the analogy: we are watching the average brightness of the photo and alarming when
it changes. **We would completely miss the subject of the photo being replaced**, as
long as the brightness stayed put.

**O(1) vs O(k).** "O(1) per tick" means the work and memory per new observation don't
grow with how much data you've seen — one number in, fixed work, done. "O(k)" means
you carry $k$ numbers instead of 1, where $k$ is a fixed budget you choose (say 20).
Still bounded, still streaming, just $k$ times bigger.

**What you'd get for that.** The Hermite estimator already computes a whole table of
coefficients $a_{jk}$ internally — those coefficients *are* a compressed description
of the copula surface — and then **collapses them into ρ and throws the rest away**.

> Monitoring the coefficient vector instead of the scalar means watching the
> **photograph** instead of the **brightness**. You are already computing it. You are
> just discarding it before the detector gets to look.

That's the cheapest available upgrade: no new estimator, no new storage, just stop
collapsing before you detect.

---

## 4. Rank-PIT, and what "marginal invariance" means

**PIT = Probability Integral Transform.** Take a value, replace it with "what fraction
of the distribution is below me". +2.3% becomes 0.87. That's the percentile trick from
§1, named.

**Rank-PIT** = do it using **ranks**, because you don't actually know the true
distribution. If you've seen 100 days and today is the 87th-smallest, call it 0.87.
No assumption about the shape of the distribution at all.

**Marginal invariance** = the property this buys you:

> If AAPL's volatility doubles tomorrow, **the ranks don't change**. The biggest day is
> still the biggest day; the median day is still the median day. So a copula estimated
> from ranks **does not move** when the marginals change.

That's the whole point. We want a detector that fires on *dependence* changes and
**stays silent** on volatility changes. Ranks give that for free.

**What we do instead right now.** `datastreamer` uses **standardization** — subtract
the mean, divide by the standard deviation (Welford's algorithm, running). That's
exact *only if the data is Gaussian*. Real returns are skewed and heavy-tailed, so it
leaves marginal information contaminating the copula estimate. Rank-PIT is open in
[todo.md](../todo.md).

---

## 5. Fantazzini (2009), in one sentence

He measured how bad that contamination gets:

> With skewness in the data and symmetric marginals assumed, estimated elliptical
> copula correlations were biased **downward by up to 70% of the true value**.

Your ρ could read 0.3 when the truth is 1.0 — **not because the dependence estimator
is bad, but because the marginals were handled wrong.** That is the single strongest
argument for doing rank-PIT, which is why it sits in Tier A of
[reference_review.md](reference_review.md).

---

## 6. What "distinction 3" means

Nothing technical — it's a **cross-reference**. [roadmap.md](roadmap.md) has a numbered
list of five ways a copula changepoint is harder than an ordinary changepoint. Other
notes say "distinction 3" instead of restating the paragraph. The five, in brief:

| # | Short name | The problem |
| --- | --- | --- |
| 1 | Joint, not scalar | The thing changing is a surface; ρ is one number off it |
| 2 | Marginal confound | Must ignore volatility changes, detect dependence changes |
| 3 | Rank feedback | Ranks shift as new data arrives → data isn't i.i.d. |
| 4 | Weak signal | Dependence is harder to estimate → delay is fundamental |
| 5 | Drift, not jumps | Real dependence mostly drifts; jumps are the rare case |

---

## 7. Gregory's route vs the Hermite route

**Empirical copula** = the copula estimated straight from data with no parametric
assumption — essentially a 2-D histogram of the percentile-pairs. Accurate, assumption-
free, but the exact version needs **every past observation** stored, because a new point
can change everyone's rank.

**What Gregory (2019) does.** Uses a **quantile sketch** (Greenwald–Khanna): a
compressed summary that can answer "what's the 87th percentile?" to within a
*guaranteed* error, using far less memory than storing everything. That makes a
streaming empirical copula possible, with error bounds.

**The catch, already recorded in [reference_review.md](reference_review.md):** it is
**cumulative — no forgetting**. It weights a point from 10,000 ticks ago exactly like
one from this tick. For a *changepoint* problem that's fatal: the whole task is to
notice that old data no longer applies.

**So "Gregory's route" means:** adopt the sketch, then add forgetting to it. Real work
— you'd have to decide how a sketch ages, and the error bounds would need redoing.

**"The Hermite route" is the alternative:** the Hermite estimator is already in your
codebase, already forgets (exponential decay on the coefficients), and already holds an
O(k) description of the copula. Less theoretically clean than Gregory's bounds, but
it's *already built*.

---

## 8. "A detector on a functional of that object"

**Functional** = a function that eats a whole object and returns a number.
"Average brightness" is a functional of a photograph.

Right now, the detector watches ρ — which *is* a functional of the copula, just a
terrible one for this purpose. The phrase means:

> **You get to choose what number the detector watches. ρ is a choice, and a poor one.**

Other choices:

| Functional | Watches for |
| --- | --- |
| Spearman's ρ *(current)* | overall strength of co-movement |
| Tail-dependence coefficient | "do they crash together?" appearing |
| Distance between recent copula and a reference copula | *any* change of shape |
| The coefficient vector, monitored jointly | any change, with a multiple-testing cost |

The third is the interesting one: instead of summarising the copula to a number and
watching the number, you **compare the whole recent surface against a reference
surface** and watch the size of the difference.

---

## 9. "Null calibrated via multiplier bootstrap"

Three separate ideas jammed together.

**"Null"** = the null hypothesis, *"nothing has changed."* To set an alarm threshold
you must know **how big your statistic normally gets when nothing is happening**.
Without that, you can't tell "big" from "ordinary".

**"Calibrated"** = choosing the threshold so false alarms occur at a rate you picked
(here: one per 1000 blocks). That is exactly what `calibrate_arl0.cpp` does, and why
the thresholds `L = 3.0` and `h = 5.0` currently mean nothing — they were never
calibrated.

**Normally** you'd use a formula for the null distribution. **You can't**, because of
distinction 3: your ranks shift as data arrives, so the data isn't i.i.d. and every
textbook formula assumes it is.

**"Bootstrap"** = when you have no formula, **simulate**. Resample your own data many
times, recompute the statistic each time, and read the threshold off the resulting
spread.

**"Multiplier bootstrap"** = a fast trick for doing that. Instead of resampling data
points, you **multiply the existing terms by random weights** (mean 0, variance 1) and
recompute. Much cheaper, and it's the standard tool for empirical copula processes —
which is what Segers (2012) and Bücher & Volgushev (2013) provide the theory for.

> **All together:** *"get the alarm threshold by simulating 'nothing changed' via random
> reweighting, because the formula you'd normally use assumes i.i.d. data and yours
> isn't."*

---

## 10. The e-value sentence, unpacked

The sentence was:

> *e-values give anytime-valid guarantees without the i.i.d. assumption that distinction
> 3 destroys, which is suspiciously well-matched to your problem.*

**"Anytime-valid."** A normal p-value is only trustworthy if you look **once**, at a
time fixed in advance. If you keep peeking as data streams in, you *will* eventually
see something "significant" by luck — look often enough and chance alone hands you a
false alarm. That is *the* central difficulty of sequential testing.
**Anytime-valid** means: look whenever you like, as often as you like, stop whenever
you like, and the guarantee still holds.

**"E-value."** An alternative to a p-value built for exactly that. Intuition: it's a
**betting score** — how much money you'd have made betting against "nothing changed".
Bet repeatedly; if the null is true you can't get rich, so a large accumulated fortune
*is* the evidence. They **multiply** over time, which makes continuous monitoring the
natural mode rather than a violation.

**"Without the i.i.d. assumption."** Standard ARL₀ theory (Lorden, Pollak) assumes
pre-change data are i.i.d. from a fixed law. Distinction 3 says yours aren't — ranks
re-shuffle as data arrives. E-value methods hold under much weaker conditions.

**"Suspiciously well-matched."** Your problem is (a) monitoring continuously and
(b) without i.i.d. data. That is precisely the pair of conditions e-values were
invented for. **When a tool fits a problem that neatly, it's usually either a real
opportunity or a sign someone has already published it** — hence *check before
committing*. `change_detection.md` Part IV.1 already calls this the strongest novelty
candidate in the whole document, and none of the Vovk/Ramdas literature is in the
library yet.
