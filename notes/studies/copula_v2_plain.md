# Copula V2 — plain explanations

Companion to `copula_v2.md`. Everything here is explanation, not new results. The numbers in §2 are
**illustrative arithmetic**, deliberately made up to show the mechanics; every other number is
measured and traceable to the main document.

---

## 1. The mechanism that generalises

**You run seven copies of the same estimator side by side.** Identical except for one thing: how far
back each remembers.

```text
  copy 1   remembers the last    25 ticks    (jumpy, but reacts instantly)
  copy 2                         46
  copy 3                         86
  copy 4                        158
  copy 5                        292
  copy 6                        541
  copy 7   remembers the last  1000 ticks    (smooth, but slow to react)
```

**Then one rule, applied every time new data arrives:**

1. Before you look at the new data point, ask all seven: *what do you predict?*
2. Look at the actual data point.
3. Whoever was closest gets more weight. Whoever was furthest off gets less.
4. Your answer is the weighted average of the seven.

That is the whole thing. **No detector. No threshold. Nobody decides "a change happened."**

### Why that makes the memory move on its own

- **Calm stretch.** Nothing is changing, so the long-memory copies are the accurate ones — they have
  averaged away the noise. They get the weight. Effective memory drifts out to ~800 ticks.
- **The jump.** Suddenly the long-memory copies are badly wrong — still averaging in a thousand ticks
  of the *old* regime. The short-memory copies have already forgotten it and land near the truth.
  **Weight rushes to them within a few dozen ticks.** Effective memory collapses to ~55.
- **Afterwards.** The long copies gradually flush out the stale data and become accurate again — *and*
  they are less noisy. Weight drifts back. Memory climbs toward 800 again.

**Long → short → long.** Nobody programmed that trajectory. It falls out of "give weight to whoever
has been predicting well."

### What "generalises" means here

The device was invented for the Kalman filter, which has a special property: **it tracks its own
uncertainty**, so it can say how surprised it should have been. That is what makes scoring the copies
natural.

**Xiao and Hermite have no such thing.** Xiao is a histogram. Hermite is a pile of polynomial
coefficients. Neither knows how confident it is. So the honest worry was that this is a fact about
*filters* rather than about online estimation.

It is not. All three do the same thing, within a factor of 1.6:

| | calm stretch | after the jump | collapse |
| --- | --- | --- | --- |
| Xiao (histogram) | 800 ticks | 55 | **14.7×** |
| Hermite (polynomials) | 785 | 62 | **12.6×** |
| Kalman (filter) | 923 | 46 | **20×** |

**Three unrelated ways of computing dependence, one shared behaviour.** Adaptive memory is not a
filter trick — any online estimator that can score its own predictions can do it.

---

## 2. What "gets more weight" actually means

A **weight** is a number attached to each copy; the seven add to 1. Read it as *"how much do I
currently believe this is the right memory setting."* Your reported answer is the weighted average of
the seven copies' estimates — a copy at weight 0.9 basically *is* your answer, a copy at 0.001 is
ignored.

### The update

Each copy makes a prediction before the data arrives. When it arrives, ask each copy: **"how likely
was what I just saw, if you were right?"** Call that $L_k$. Then:

$$w_k \;\leftarrow\; \frac{w_k\,L_k}{\sum_j w_j\,L_j}$$

**Multiply by how well you predicted, then renormalise.** That is the entire adaptation mechanism.

### Worked example

> ⚠️ **Numbers below are invented to show the arithmetic.** They are not measured results.

**Calm stretch.** Truth sitting at $\rho = 0.19$. All three predict roughly right, but the
long-memory copy is least noisy, so it is closest:

| | old weight | × $L$ | = | **new weight** |
| --- | --- | --- | --- | --- |
| short (25 ticks) | 0.10 | 0.8 | 0.08 | 0.057 |
| medium (158) | 0.30 | 1.2 | 0.36 | 0.257 |
| long (1000) | 0.60 | 1.6 | 0.96 | **0.686** |
| | | | *sum 1.40* | |

Long drifts up, 0.60 → 0.686. Weighted-average memory ≈ **728 ticks**.

**Now the jump.** Truth goes 0.19 → 0.78. The long copy is still averaging in a thousand ticks of the
old regime, so it says ~0.19 — catastrophically wrong. The short copy has already forgotten the old
regime and says ~0.6:

| | old weight | × $L$ | = | **new weight** |
| --- | --- | --- | --- | --- |
| short | 0.057 | 2.0 | 0.114 | **0.597** |
| medium | 0.257 | 0.3 | 0.077 | 0.403 |
| long | 0.686 | **0.0001** | 0.00007 | **0.0004** |
| | | | *sum 0.191* | |

**Long went 0.686 → 0.0004 in a single step.** Weighted-average memory ≈ **79 ticks**.

That is the cliff in Figure 3. It is not a rule saying "shorten the memory when something changes" —
being wrong by a lot makes $L$ tiny, and **multiplying by a tiny number annihilates your weight**.
The response is abrupt because the update is *multiplicative*.

### One extra piece, or it could never recover

A copy at weight 0.0004 is nearly dead, and multiplying by more small numbers keeps it dead. So after
each update, **mix a little "everyone is equally likely" back in** — about 0.2%:

$$w_k \;\leftarrow\; (1-p)\,w_k \;+\; \frac{p}{K}, \qquad p = 0.002$$

- **Too little** and a copy that fell out of favour can never come back — the memory would collapse at
  the jump and be stuck short forever.
- **Too much** and the weights are dragged back to equal every step and learn nothing.
- Measured ordering: $0.002 > 0.02 > 0.15 > \text{frozen}$.

That trickle is what produces the slow climb on the right-hand side of Figure 3.

### What gets reported

Both the estimate and the memory are weighted averages over the same weights:

$$\hat\rho_t = \sum_k w_k\,\hat\rho^{(k)}_t,
\qquad
\overline{\mathrm{mem}}_t = \sum_k w_k\,\mathrm{mem}_k$$

The second is the **posterior-mean memory** plotted in Figure 3 — the estimator's own answer to *how
far back should I be looking right now*, and the only place in this project where that quantity is
observable rather than chosen.

---

## 3. Bank against VFF

Both are adaptive. They are not equally good.

| | what it carries | what it does with surprise |
| --- | --- | --- |
| **VFF** | one scalar | **integrates** it — the memory drifts toward the new regime |
| **bank** | a distribution over memories | **re-weights** on it — one bad prediction can move the posterior immediately |

**The bank wins 10 of 12 cells.** VFF wins twice, both on `constant`, and both exceptions come from
the same mechanism:

- **VFF wins where the memory only needs to be *long*.** On `constant` the optimum is ≥30,000 ticks.
  A VFF can drift there — Xiao's reaches 945 — while the bank is capped by its longest arm (1000) and
  spreads weight below it, landing near 839. *Can go longer* beats *can switch* when nothing switches.
- **VFF loses worst where the memory needs to be *short and committed*.** On `oscillate` the optimum
  is 36 ticks; Xiao's VFF settles near 90. It integrates toward an average and undershoots. The
  bank's margin there is 30% (Xiao) and 40% (Hermite) — its widest, wider than on `jump`.

A hazard is a **prior the data can override in a single step**. An integrator cannot be overridden,
only outvoted — and by then the change has already cost you.

> **A bank's advantage is specifically the ability to *move*, not the ability to be right.**

---

## 4. What the bias/variance decomposition showed

Two statements that sound contradictory, and both are true because they are **different
comparisons**:

- across **scenarios**: *every difference is in bias², variance behaves the same*
- across **estimators**: *bias² agrees to three figures, the difference is variance*

### The structure

| what you change | bias² | variance |
| --- | --- | --- |
| **the memory** | **changes** — longer = more lag | **changes** — longer = less noise |
| **the scenario** — how fast the truth moves | **changes** | ~unchanged |
| **the estimator** — at matched memory | ~unchanged | **changes** |

### Why it comes out that way

**Bias is lag.** You report a weighted average of the past, so you always report something slightly
stale. How costly that is depends on two things:

- how far back you are averaging → **the memory**
- how much the truth has moved in that time → **the scenario**

It does *not* depend on the estimator, because at matched memory Xiao and Hermite weight the past
**identically** — same decay, same effective sample size. They are stale by the same amount.

**Variance is leftover noise.** How much observation noise survives the averaging depends on:

- how much averaging you did → **the memory**
- how efficiently the estimator squeezes information out of the data → **the representation**

It does *not* depend on the scenario, because the noise in the incoming pairs is the same whether the
truth is sitting still or oscillating.

> **Memory and the scenario set the lag. Memory and the representation set the noise.**

Memory is in both — which is exactly why it is the dominant variable, and why it is a trade-off
rather than a free choice.

### Why this is worth having

- **It is a diagnostic.** If your error is bias-dominated, shortening the memory helps and switching
  estimators will not. If it is variance-dominated, a better representation helps and shortening the
  memory makes it worse. RMSE alone cannot tell you which regime you are in.
- **It explains `constant` rather than just noting it.** Bias² is flat on the floor there, so
  MSE ≈ variance ≈ $\sigma^2/\mathrm{ESS}$, which is monotone decreasing. "Longer is always better"
  becomes a consequence of $\dot\theta = 0$ rather than an observation.
- **It reframes Hermite's 3–15% loss.** Hermite loses on *variance*, which more memory buys back. It
  would only be a real defect if it lagged worse — and it does not.
- **It caught a defect.** On `constant`, bias does not reach zero: it floors at 0.006, which is
  Xiao's 20-bin discretisation. Invisible in RMSE, obvious the moment bias² is plotted on a log axis.

---

## 5. How to read Figure 2

Figure 2 is the Finding 2 figure. You read it **vertically, two different ways**:

- **How far one curve moves as you slide along it** — the cost of choosing the *memory* wrong. In the
  `oscillate` panel that is roughly 0.18 up to 0.42.
- **How far apart the three curves sit at the same x** — the cost of choosing the *algorithm* wrong,
  with memory held equal. Typically under 10%.

**The point is that the first gap is enormous and the second is small.** Same axis, same units, so
the comparison is direct rather than rhetorical. That is the **×2–×17 against 3–15%**.

It has to be plotted against **effective memory in ticks** rather than each estimator's own $\lambda$,
because Xiao's $\lambda$ and Hermite's $\lambda$ run in opposite directions and mean different
things. On their native scales the comparison is meaningless. **The figure only exists because of the
memory conversion.**

### ⚠️ The figure does not make that point well

- **The grey regions dominate.** In several panels more than half the width is excluded, and the eye
  goes to the largest visual element — which is the part that is not a result.
- **The log y-axis compresses exactly the comparison you want.** A factor-of-20 range and a 10% gap
  look much closer in size than they are.
- **The claim is not readable off it.** You cannot see "×2–17 versus 3–15%" — you would have to
  measure two distances by eye and divide.

**A better version** would plot the two quantities directly against each other: per scenario, the
spread across estimators at matched memory, beside the range within one estimator across the sweep.
Two bars per scenario, and the claim becomes the picture rather than something inferred from it.
That is a re-plot of `memory_sweep.csv`, not a re-run.
