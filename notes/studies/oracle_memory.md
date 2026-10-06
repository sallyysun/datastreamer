# The oracle memory policy — what adaptation *should* do if the change were known

**Status:** framing note + closed-form derivation. The oracle soft-restart experiments on the
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

## 4. From oracle to online (extension / future work)

In practice $\tau$ and $\Delta$ are unknown. The natural plug-in is
$\widehat\Delta=\hat\theta_{\text{recent}}-\hat\theta_{\text{old}}$ and
$\hat Z=|\widehat\Delta|/\widehat{SE}(\widehat\Delta)$, and the same statistic does both jobs:
small $\hat Z$ ⇒ weak evidence of change ⇒ retain history; large $\hat Z$ ⇒ strong evidence ⇒ retain
little. That is exactly where VFF and detect-and-soft-restart stop being separate (Part V: one dial,
different decision hardness).

Two traps to handle before using it, both of which the naïve $1/(1+\hat Z^2)$ gets wrong:

- **Wrong denominator.** The oracle needs $\Delta^2/SE_0^2$; $\hat Z$ standardises by
  $SE(\widehat\Delta)$, which also contains the recent window's noise.
- **Upward bias.** $\mathbb E[\widehat\Delta^2]=\Delta^2+\mathrm{Var}(\widehat\Delta)$, so plugging in
  $\widehat\Delta^2$ over-forgets under no change. The bias-corrected, positive-part form is

$$
\hat\gamma=\frac{SE_0^2}{SE_0^2+\big(\widehat\Delta^2-\widehat{\mathrm{Var}}(\widehat\Delta)\big)_+}
$$

  (James–Stein shape: exactly $\gamma=1$ whenever the observed shift is within noise).

Uncertainty in $\tau$ itself is a third layer; BOCPD's run-length posterior is the principled way
to integrate over it, and is already the only estimator here that wins without resetting.

## 5. Thesis framing

$$
\underbrace{\text{Oracle: what SHOULD adaptation do?}}_{\text{known-changepoint experiments}}
\;\longrightarrow\;
\underbrace{\text{Online: can we approximate it from noisy data?}}_{\text{VFF, bank, CUSUM/GLR, BOCPD}}
$$

VFF, the multi-$\lambda$ bank and explicit detection are each one practical approximation to
$\gamma^\*$ without knowing the latent path. Whether the thesis adds a new one (e.g. the $\hat\gamma$
above) is a scope decision; the existing results do not depend on it.

**Note for $\chi$ specifically:** $\hat\chi(q)$ uses only the $\approx n(1-q)$ joint exceedances, so
$SE_0$ is large and $Z$ small at realistic window lengths — the oracle will usually say *keep a lot
of history*. Expect soft restart to beat hard restart by more for $\chi$ than for $\rho_S$.
