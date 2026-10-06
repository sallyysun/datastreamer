# The oracle memory policy — what adaptation *should* do if the change were known

**Status:** thesis framing + closed-form derivation + design of the online rule (§4), which is
the deliverable — the oracle (§1–3) is the theory it approximates. The oracle soft-restart experiments on the
tail-dependence coefficient $\chi$ that prompted it are **not in this repo yet** (there is no $\chi$
estimator in `src/estimators/`, and no oracle scenario in `experiments/`), so nothing below is a
harness result. See `notes/todo.md` → "Oracle memory benchmark".

## 1. What the known-changepoint experiment is, and is not

We never observe $\chi_t$ (or $\rho_t$). We observe $(U_t,V_t)$ and form $\hat\chi_t(q)$. A real
online method faces three questions, in order:

1. Has the underlying dependence changed? *(inference — detection)*
2. If so, by roughly how much? *(inference — size)*
3. How much old information should I keep? *(decision — memory)*

The simulation generated the changepoint ($C_0 \xrightarrow{\tau} C_1$), so it knows $\tau$ and
$\Delta\chi=\chi_1-\chi_0$. Handing those to the estimator removes questions 1–2 and isolates 3:

$$
\boxed{\text{If the change were known, what is the optimal memory response?}}
$$

That is an **oracle benchmark**, not a deployable estimator, and the thesis must say so explicitly.
Its value is that it gives practical methods a target. Its headline: **even with perfect knowledge
of the changepoint, hard restart is not optimal in general** — perfect detection does not imply
throwing everything away. This is the same effect the harness already measured from the other side
(`kalman_cusum`/`kalman_glr` lose RMSE to the un-reset filter they wrap;
`kalman_changepoint_results.md` §6.4), now with the detection error removed as an explanation.

## 2. The optimal soft-restart weight, derived

Soft restart at $\tau$: keep the pre-change estimate $\hat\theta_0$ (s.e. $SE_0$) but discount its
precision by $\gamma\in[0,1]$; $\gamma=0$ is hard restart, $\gamma=1$ is no restart. After the
change, combine it with the post-change estimate $\hat\theta_n$ (variance $\sigma_n^2$, $n$ ticks
since $\tau$) by precision weighting:

$$
\hat\theta=w\,\hat\theta_0+(1-w)\,\hat\theta_n,\qquad
w=\frac{\gamma\,\sigma_n^2}{\gamma\,\sigma_n^2+SE_0^2}.
$$

With $\Delta=\theta_1-\theta_0$ and independent errors,
$\mathrm{MSE}(w)=w^2(\Delta^2+SE_0^2)+(1-w)^2\sigma_n^2$, minimised at
$w^\*=\sigma_n^2/(\Delta^2+SE_0^2+\sigma_n^2)$. Solving $w(\gamma)=w^\*$ for $\gamma$:

$$
\gamma(\Delta^2+SE_0^2)=SE_0^2
\quad\Longrightarrow\quad
\boxed{\gamma^\*=\frac{1}{1+Z^2},\qquad Z=\frac{|\Delta|}{SE_0}}
$$

Exact under the Gaussian/precision-weighting approximation, not just asymptotic — and
**independent of $n$**: one $\gamma$ is optimal at every post-change horizon. Hard restart
($\gamma=0$) is optimal only as $Z\to\infty$; no restart only at $Z=0$.

## 3. Why this is the Kalman identity again

Discounting prior precision by $\gamma^\*$ is the same as inflating the prior variance:
$SE_0^2/\gamma^\* = SE_0^2+\Delta^2$. That is one Kalman predict step with a **one-shot process-noise
impulse $q_\tau=\Delta^2$**. So:

- the oracle soft restart *is* a Kalman filter that knows $q_t=\Delta^2\,\delta_{t\tau}$;
- hard restart is $q_\tau=\infty$; a fixed-$\lambda$ estimator is $q_t\equiv q$ spread over every tick
  (`change_detection.md`, "fixed-λ = steady-state Kalman, λ = 1−K");
- every practical method is a way of **estimating $q_t$ online** — which `change_detection.md`
  already identifies as the VFF problem.

## 4. The online rule — the actual deliverable

The thesis is **online adaptive memory for tail-dependence estimation**: from $(U_t,V_t)$ alone,
adapt memory as $\chi$ changes. Sections 1–3 are the oracle theory that says what such a rule should
approximate; this section is the method. Progression:

$$
\text{fixed memory}\;\to\;\text{oracle } \gamma^\*\;\to\;\boxed{\text{online approximation } \hat\gamma_t}
$$

The online problem is smaller than "solve changepoint detection": it is

$$
\boxed{\text{estimate the evidence } Z \text{ online well enough to approximate } \gamma^\*=1/(1+Z^2).}
$$

Two routes, both genuinely online:

- **A — continuous (VFF-style).** $(U_t,V_t)\to$ surprise $\to\alpha_t\to\hat\chi_t$. No changepoint
  is ever declared.
- **B — change-driven.** $(U_t,V_t)\to$ evidence $\to\hat\tau\to\hat\gamma\to\hat\chi_t$. The detector
  says *when* to intervene; $\gamma^\*$ says *how hard*.

The proposed rule sits between them: keep a reference estimate and a recent one, form
$\widehat\Delta_t=\hat\chi^{\text{recent}}_t-\hat\chi^{\text{ref}}_t$ and its evidence $\hat Z_t$, and
map evidence to retention $[0,\infty)\to[1,0]$ continuously, instead of thresholding it into
{change, no change}. That is a continuous alternative to detect-and-restart (Part V's dial,
with the setting derived rather than tuned).

### 4.1 Design traps — each one breaks the naïve $\hat\gamma_t=1/(1+\hat Z_t^2)$

1. **Null leakage.** With no change, $\hat Z_t^2\sim\chi^2_1$, so $1/(1+\hat Z_t^2)$ has mean $0.66$
   (median $0.69$): the rule would discard a third of its memory on a typical *quiet* tick.
   The bias-corrected positive-part form from §2's MSE,

   $$
   \hat\gamma_t=\frac{SE_0^2}{SE_0^2+\big(\widehat\Delta_t^2-\widehat{\mathrm{Var}}(\widehat\Delta_t)\big)_+},
   $$

   gives $\hat\gamma=1$ exactly unless $\hat Z^2>1$ — but that still fires on 32% of null ticks.
   So the quiet-period false-forgetting rate must be **measured and calibrated**, the analogue of
   $\mathrm{ARL}_0$ for detectors (change_detection.md Part VI). A higher evidence floor (e.g.
   $(\hat Z^2-c)_+$) trades that against post-change delay; $c$ is the one calibration knob.
2. **Compounding.** $\gamma^\*$ is a *one-shot* discount at $\tau$. Apply $\hat\gamma_t$ on every tick
   while $\hat Z_t$ stays elevated (it will, for the length of the recent window) and the discounts
   multiply — far harder than any oracle restart. Fix by **re-anchoring**: the reference must be the
   adaptive estimate itself, so after a discount the gap $\widehat\Delta_t$ closes and the evidence
   is spent. Equivalently, in Kalman form, inject $q_t=(\widehat\Delta_t^2-\widehat{\mathrm{Var}})_+$
   as process noise into the state the gap is measured from; the update absorbs it.
3. **Wrong denominator.** The oracle needs $\Delta^2/SE_0^2$; $\hat Z$ standardises by
   $SE(\widehat\Delta)$, which includes the recent window's noise. Use the formula above, not $\hat Z$.
4. **$SE$ of $\hat\chi$ online.** $\hat\chi(q)$ is driven by ~$n(1-q)$ joint exceedances, so its
   $SE$ is a binomial-type quantity in the effective exceedance count — needs a streaming formula,
   not a bootstrap.

**Honest positioning.** With re-anchoring, this is innovation-based adaptive estimation of $q$
(covariance matching, Mehra 1972 / Sage–Husa) — i.e. a VFF. The contribution is not "a new kind of
adaptation" but **a VFF whose evidence→memory map is derived from the oracle MSE optimum and
calibrated against a null false-forgetting rate**, applied to tail dependence, where small $Z$ makes
the gap between soft and hard response largest. Say that plainly; it is a stronger claim than
novelty of mechanism.

## 5. Evaluation and scope

Compare, multi-seed, on scenarios where **$\chi$** changes (Clayton/Gumbel/$t$ DGP — a Gaussian
copula has $\chi=0$ for every $\rho<1$):

$$
\text{fixed forgetting (swept)}\;\text{vs}\;\text{VFF}\;\text{vs}\;\textbf{online }\hat\gamma\text{ rule}
\;\;[\text{vs one detect-and-hard-restart}]\;\;\text{with oracle }\gamma^\*\text{ as the ceiling.}
$$

Metrics: RMSE of $\hat\chi_t$; post-change recovery; quiet-period false-forgetting rate; and the
fraction of the fixed-λ → oracle gap the online rule closes. **Not** in scope: building more
changepoint detectors. The existing CUSUM/GLR/BOCPD on $\rho_S$ stay as prior work;
at most one hard-restart baseline is carried over to $\chi$.

**Note for $\chi$ specifically:** $SE_0$ is large and $Z$ small at realistic window lengths, so the
oracle will usually say *keep a lot of history*. Expect soft response to beat hard restart by more
for $\chi$ than it did for $\rho_S$.
