> **How should an online dependence estimator remember the past when the relationship it is
> estimating can change?** Xiao and Hermite, Kalman as reference. 40 seeds, $N=5000$.
> Full record: `copula_v2.md`.

## The trade

$$\mathrm{MSE}(\mathrm{ESS}) \;\approx\;
\underbrace{\frac{\sigma^2}{\mathrm{ESS}}}_{\text{variance}} +
\underbrace{\Big(\frac{\dot\theta\,\mathrm{ESS}}{2}\Big)^{\!2}}_{\text{lag bias}},
\qquad \mathrm{ESS} = \frac{2-\alpha}{\alpha}$$

## One axis

$$K = \frac{\sqrt{q^2+4q}-q}{2},\quad q = \frac{\sigma_\omega^2}{\sigma_v^2}
\;\Longrightarrow\;
\hat z_t = (1-K)\hat z_{t-1} + Ky_t,\quad \boxed{\lambda = 1-K}$$

Every fixed-memory estimator **is** a settled Kalman filter. Verified to 5 dp. Without this there is
no common axis. ⚠️ $\lambda$ is mirror-image: Xiao larger = longer, Hermite smaller = longer.

## 1 — The optimum is set by how fast the truth moves

| scenario | best memory | RMSE |
| --- | --- | --- |
| `constant` | **≥30,000** | 0.0154 |
| `jump` settled | 495 | 0.0199 |
| `drift` | 235 | 0.0659 |
| `jump` all | 162 | 0.0801 |
| `jump` recovery | 53 | 0.0934 |
| `oscillate` | **36** | 0.1827 |

Variance falls with memory everywhere; **all scenario differences are in bias²**.

**Same run, two stretches: 495 against 53 ticks — 9.4×.** Tuned for one, the other is 2.3× / 3.1×
worse. No fixed memory serves both.

## 2 — Memory ×2–17. Estimator 3–15%.

Xiao against Hermite at **matched memory**: 3.4–15.4%, $|t| = 3.9$–13.1, 38 of 47 points favour Xiao.
**Pure variance** — bias² agrees to three figures, Hermite's variance is 1.11–1.63× Xiao's.

> **Memory sets the lag. Representation sets the noise.**

⚠️ Exception — singular density (Fréchet): Hermite 0.0121 → 0.0288, **4.7σ worse**, behind a
misspecified parametric model. Xiao 0.0128 → 0.0152.

## 3 — Adaptive memory

**VFF** — one estimator, one evolving rate. $e_t$ is the prediction error and $F_t$ the variance the
model predicted for it, so their ratio has a **known** null value:

$$e_t = y_t - \hat z_{t|t-1}, \qquad F_t = P_{t|t-1} + \sigma_v^2,
\qquad \mathbb E\Big[\tfrac{e_t^2}{F_t}\Big] = 1 \;\text{ under the model}$$

$$C_t = (1-r)\,C_{t-1} + r\,\frac{e_t^2}{F_t}
\qquad
\sigma_\omega^2 \leftarrow \sigma_\omega^2\,C_t^{\,r}
\;\Longleftrightarrow\;
\log\sigma_\omega^2 \mathrel{+}= r\log C_t$$

$C_t>1$ ⇒ shorter memory; $C_t<1$ ⇒ longer; $C_t=1$ is the fixed point. **Additive in $\log$, so the
step is bounded by $r$.** ⚠️ $r$ is both the smoothing rate and the step size — one knob, two jobs.

**Bank** — $K$ copies at memories $\mathrm{mem}_1\dots\mathrm{mem}_K$, weights summing to 1, where
$L_k = p(y_t \mid \text{memory }k,\, y_{1:t-1})$ is the density copy $k$ gave $y_t$ **before** seeing it:

$$w_k \leftarrow \frac{w_k\,L_k}{\sum_j w_j\,L_j}
\qquad\text{then}\qquad
w_k \leftarrow (1-p)\,w_k + \frac{p}{K}$$

$$\hat\rho_t = \sum_k w_k\,\hat\rho^{(k)}_t,
\qquad
\overline{\mathrm{mem}}_t = \sum_k w_k\,\mathrm{mem}_k$$

**Multiplicative, so the step is unbounded**: normalisation fixes $\sum_k w_k = 1$ but the *ratio*
$w_i/w_j$ is multiplied by $L_i/L_j$, which has no bound. One bad prediction can annihilate an arm.
That is the whole adaptation mechanism — no rule anywhere says "shorten the memory when something
changes". The mixing step is what lets an annihilated arm recover; without it the memory could
collapse and never return. $\overline{\mathrm{mem}}_t$ is what is plotted below.

| | step size | consequence |
| --- | --- | --- |
| VFF | bounded by $r$ | moves a set amount per observation, whatever it sees |
| bank | unbounded in $L_i/L_j$ | one observation can overturn the belief |

**Memory collapse at the jump**, three unrelated estimators:

| Xiao | Hermite | Kalman |
| --- | --- | --- |
| 800 → 55 (**14.7×**) | 785 → 62 (12.6×) | 923 → 46 (20×) |

No detector, no threshold, no reset.

**Against one fixed memory for all scenarios: 12 of 12 cells, 25–65%.**
**Against a per-scenario oracle: `jump` only** — the one scenario whose dynamics change mid-run
(−3.3σ, −2.7σ, −16.6σ). Elsewhere it loses 8.9–21.7σ.

⚠️ **VFF is not slow — it is unpredictable.** Collapse ranges 1.00× to 74.7× across settings of one
knob, and the hardest collapse has the *worst* `jump` RMSE. It replaces "choose a memory" with
"choose a rate", and no rate suits both regimes. The bank needs no such choice.

⚠️ Bank storage: **52 KB** (Xiao) against 736 B (Kalman). The device generalises; its cheapness
does not.

## Retracted: 14

Two from this work. **R13** — the bank's 8.2σ `drift` win: the oracle's grid was too coarse to
contain the optimum, so it **loses by 7.7σ**. **R14** — a VFF "win" whose measured collapse is
**1.00×**: a fixed memory wearing an adaptive label.
