# Bank proofs: transcription of notebook screenshots (2026-11-04)

Transcribed from 16 photos of the "time-varying dependence in data streams" notebook
(`127.0.0.1:51907`, pages *Scoring & bank proofs* and *Bank bias & variance*, `#bank-asymptotics`).
The text follows the screen as written. `[cut off]` marks text outside the photo. Bracketed
*Transcriber note* lines are mine and are not part of the notebook.

---

## 1. Squared-score regret with estimated margins (*Scoring & bank proofs*)

*(The photo starts mid-paragraph.)*

… deterministic finite bound. Use $R=1$ only if the forecasts really are restricted to $[-1,1]$; this
statement does not assume clipping is implemented. Let $U_t, V_t$ be the true continuous conditional
PITs, set $\rho_{S,t}=\mathbb E_{t-1}[12(U_t-\tfrac12)(V_t-\tfrac12)]$, and define

$$
m_t=\mathbb E_{t-1}\hat p_t,\qquad \varepsilon_t=\mathbb E\big[|\hat U_t-U_t|+|\hat V_t-V_t|\big].
$$

**Question.** If the stated pathwise regret bound holds for these observed squared scores with
deterministic penalty $B_T$, prove

$$
\boxed{\;\mathbb E\sum_{t=1}^T(\hat\rho_{B,t-1}-\rho_{S,t})^2\le\min_k\mathbb E\sum_{t=1}^T(\hat\rho_{k,t-1}-\rho_{S,t})^2+B_T+24R\sum_{t=1}^T\varepsilon_t.\;}
$$

**Proof.** For any fixed comparator $k$, conditional expected squared scores have the same conditional
variance term. Write $r_{j,t}=\mathbb E_{t-1}\ell^{\rm sq}_{j,t}$. Cancelling that term and expanding
around $\rho_{S,t}$ gives

$$
\begin{aligned}
(\hat\rho_{B,t-1}-\rho_{S,t})^2-(\hat\rho_{k,t-1}-\rho_{S,t})^2
&=r_{B,t}-r_{k,t}+2(\hat\rho_{B,t-1}-\hat\rho_{k,t-1})(m_t-\rho_{S,t})\\
&\le r_{B,t}-r_{k,t}+4R\,|m_t-\rho_{S,t}|.
\end{aligned}
$$

The marginal-error lemma gives $\mathbb E|m_t-\rho_{S,t}|\le 6\varepsilon_t$. Sum over time and take
expectations. The … `[cut off]`

---

## 2. Why does KL represent excess risk?

For any true density $c$ and forecast density $\hat c$, Jensen's inequality for $-\log$ gives

$$
\int c\log\frac{c}{\hat c}\;\ge\;-\log\int_{\{c>0\}}\hat c\;\ge\;0.
$$

If the forecast vanishes on a region of positive true probability, the divergence is infinite.
Otherwise equality requires the two normalized densities to agree almost everywhere. Expected log
loss is consequently minimized at the true density in the ideal scoring setup; finite-sample
observed scores can still rank forecasts differently.

With estimated margins, squared loss instead targets $m_t$, and the marginal-error lemma measures
distortion. Log-risk transfer needs the same common correct conditional PIT setting or an explicit
perturbation argument. Under the martingale conditions, the pathwise bound also transfers to
conditional-risk sums up to $o(T)$ almost surely for a fixed finite bank.

To see the last statement, put $A_{j,T}=\sum_{t\le T}(\ell_{j,t}-r_{j,t})$ for the bank and each
estimator. Proof 4 gives $A_{j,T}=o(T)$. The pathwise inequality implies

$$
\sum_{t=1}^T r_{B,t}\le\min_k\sum_{t=1}^T r_{k,t}+B_T+|A_{B,T}|+\max_k|A_{k,T}|=\min_k\sum_{t=1}^T r_{k,t}+B_T+o(T).
$$

Finiteness of the bank makes the maximum of these finitely many $o(T)$ terms another $o(T)$ term.
This is a probabilistic bridge added to the pathwise algorithm theorem, not part of the algebraic
definition of regret.

If $B_T/T\to0$, the asymptotic upper bound on average excess conditional risk is zero:

$$
\limsup_{T\to\infty}\frac{\sum_{t=1}^T r_{B,t}-\min_k\sum_{t=1}^T r_{k,t}}{T}\le0\quad\text{almost surely.}
$$

---

## 3. Weighted-mean estimator: exact identity and drift bound

The target is the current mean $\mu_t$. Write $A_t=\sum_s a_{t,s}(t-s)$, the weighted mean age in
observation steps.

### Exact identity

$$
\hat\mu_t-\mu_t=\sum_s a_{t,s}(Z_s-\mu_s)+\sum_s a_{t,s}(\mu_s-\mu_t).
$$

The first term has mean zero. Independence makes its cross-covariances zero. The second term is
deterministic. Squaring and taking expectations therefore gives

$$
\mathbb E[(\hat\mu_t-\mu_t)^2]=\sum_s a_{t,s}^2v_s+\Big\{\sum_s a_{t,s}(\mu_s-\mu_t)\Big\}^2.
$$

### Drift bound

Assume $|\mu_s-\mu_t|\le d(t-s)$ for $s\le t$. Then

$$
\Big|\sum_s a_{t,s}(\mu_s-\mu_t)\Big|\le\sum_s a_{t,s}\,d(t-s)=dA_t,
$$

$$
\boxed{\;\mathbb E[(\hat\mu_t-\mu_t)^2]\le\frac{v}{N_{{\rm eff},t}}+d^2A_t^2.\;}
$$

> *Transcriber note:* the step from $\sum_s a_{t,s}^2v_s$ to $v/N_{{\rm eff},t}$ needs $v_s\le v$ and
> $N_{{\rm eff},t}=1/\sum_s a_{t,s}^2$. Neither is visible in the photo; they may be stated off-screen.

---

## 4. *Bank bias & variance*, part A (NEW TO THESIS)

**A. What can we prove for any adaptive weights?** *Standard calculation · Credit & what differs →*

### A. Can arbitrary adaptive weights preserve a component MSE rate?

For a deterministic target $\theta$, define $\hat\theta_B=\sum_{k=1}^K w_k\hat\theta_k$, where
$w_k\ge0$ and $\sum_k w_k=1$ almost surely. The weights may depend on the same data as the
estimators. Assume finite second moments. We seek to prove

$$
\boxed{\;\mathbb E(\hat\theta_B-\theta)^2\le\sum_{k=1}^K\mathbb E(\hat\theta_k-\theta)^2.\;}
$$

**Proof.** Put $e_k=\hat\theta_k-\theta$. Convexity gives a pointwise inequality, before taking any
expectation:

$$
\Big(\sum_k w_ke_k\Big)^2\le\sum_k w_ke_k^2\le\sum_k e_k^2.
$$

Taking expectations proves the bound without factoring a weight out of an expectation. If $K$ is
fixed and each component has MSE $O(\Delta^{2/3})$, then

$$
{\rm MSE}(\hat\theta_B)=O(\Delta^{2/3}),\qquad|{\rm Bias}(\hat\theta_B)|=O(\Delta^{1/3}),\qquad{\rm Var}(\hat\theta_B)=O(\Delta^{2/3}).
$$

The last two statements follow from MSE = Var + Bias². This is a rate bound, not a best-estimator
oracle inequality: its right-hand side is a sum, not a minimum. It does not establish an adaptive
advantage. The same logic applies to the actual scalar bank once appropriate component MSE bounds are
supplied. ∎

---

## 5. Part B: a full expansion for fixed share with linear components

> **"Linear" describes the component estimators:** each directly averages scalar observation
> contributions with fixed geometric weights. Both this model and the mixed-type bank combine outputs
> by a weighted average. Because the combination weights depend on observed performance, the complete
> adaptive bank need not be a linear function of the data. Count-matrix ratios, Hermite coefficient
> products and likelihood updates require their own component analysis.

**Assumptions.** Fix $\tau>0$, $\Delta=\tau/n$, and a finite $K$. Within each experiment, the
observations $Y_{1,\Delta},\dots,Y_{n,\Delta}$ are independent and satisfy

$$
|Y_{i,\Delta}|\le M,\qquad\mathbb EY_{i,\Delta}=\theta(i\Delta),\qquad{\rm Var}(Y_{i,\Delta})=v(i\Delta).
$$

The bound $M$ is uniform across experiments, $\theta\in C^2([0,\tau])$, and $v$ is continuous. No
Gaussian distribution is assumed. Choose constant-within-experiment forgetting weights

$$
\lambda_{k,\Delta}\sim A_k\Delta^{2/3},\quad A_k>0,\qquad N_{k,\Delta}=\frac{2-\lambda_{k,\Delta}}{\lambda_{k,\Delta}}.
$$

Suppress $\Delta$ in the indices below. Initialise $\hat\theta_{k,1}=Y_1$ and any simplex weight vector
$w_1$ chosen without future observations. For $i\ge2$, use the pre-observation forecasts to score,
update the combination weights, and update the component states:

$$
\ell_{k,i}=(\hat\theta_{k,i-1}-Y_i)^2,\qquad\tilde w_{k,i}=\frac{w_{k,i-1}e^{-\eta\ell_{k,i}}}{\sum_j w_{j,i-1}e^{-\eta\ell_{j,i}}},
$$

$$
w_{k,i}=(1-\delta)\tilde w_{k,i}+\frac{\delta}{K},\qquad\hat\theta_{k,i}=(1-\lambda_k)\hat\theta_{k,i-1}+\lambda_kY_i.
$$

Here $0<\delta<1$ and $0\le\eta<\infty$ are fixed as $\Delta\to0$. This asymptotic theorem does not
require the separate exp-concavity learning-rate restriction. The forecast of $\theta(\tau)$ uses data
only through $n-1$:

$$
\hat\theta_{B,n-1}=\sum_k w_{k,n-1}\hat\theta_{k,n-1}.
$$

### Question: what are the leading bank bias, variance and MSE?

> **Scope of these formulas:** the linear-component model in part B. They are not yet established for
> the actual mixed-type bank. *What remains to extend them?*

Write $N_k=N_{k,\Delta}$. Under the assumptions above, we seek to prove

$$
\boxed{\;{\rm Bias}(\hat\theta_{B,n-1})=-\frac{\theta'(\tau)\Delta}{2K}\sum_k(N_k+1)+o(\Delta^{1/3}).\;}
$$

$$
\boxed{\;{\rm Var}(\hat\theta_{B,n-1})=\frac{v(\tau)}{K^2}\sum_{k,j}\frac{2}{N_k+N_j}+o(\Delta^{2/3}).\;}
$$

$$
\boxed{\;{\rm MSE}(\hat\theta_{B,n-1})=\frac{v(\tau)}{K^2}\sum_{k,j}\frac{2}{N_k+N_j}+\frac{\theta'(\tau)^2\Delta^2}{4K^2}\Big\{\sum_k(N_k+1)\Big\}^2+o(\Delta^{2/3}).\;}
$$

The double sum includes component cross-covariances; these estimators process the same … `[cut off]`

> *Transcriber note:* here $\lambda$ is the **gain** ($\hat\theta\leftarrow(1-\lambda)\hat\theta+\lambda Y$,
> small = long memory). That is the Hermite convention, the reverse of the retention
> $\lambda=1-2/(M+1)$ used for the family bank and Xiao. With gain $\lambda$, $N=(2-\lambda)/\lambda$ is
> our $M$.

### Proof in four steps

**Step 1: control component errors locally.** Let $\|Z\|_p=(\mathbb E|Z|^p)^{1/p}$. Uniformly for
observation indices in the latter half of the experiment, each component's geometric weights satisfy
$\sum_s a_{k,i,s}^2=O(\Delta^{2/3})$; initialization terms decay exponentially in $\Delta^{-1/3}$.
Bounded $\theta'$ gives bias $O(\Delta/\lambda_k)=O(\Delta^{1/3})$.

For centered observations $\varepsilon_s=Y_s-\mathbb EY_s$, independence and boundedness give

$$
\begin{aligned}
\mathbb E\Big(\sum_s a_s\varepsilon_s\Big)^4&=\sum_s a_s^4\,\mathbb E\varepsilon_s^4+6\sum_{s<r}a_s^2a_r^2\,v(s\Delta)v(r\Delta)\\
&\le C\Big(\sum_s a_s^2\Big)^2=O(\Delta^{4/3}).
\end{aligned}
$$

Combining bias and centered variation, and using fixed $K$, yields

$$
\sup_{i\ge\lfloor n/2\rfloor}\Big\|\max_k|\hat\theta_{k,i}-\theta(i\Delta)|\Big\|_4=O(\Delta^{1/3}).
$$

Replacing $\theta((n-1)\Delta)$ by $\theta(\tau)$ adds only $O(\Delta)$.

**Step 2: prove the fixed-share weights approach uniformity.** Let $u=(1/K,\dots,1/K)$, $q=1-\delta$,
and ${\rm osc}(\ell)=\max_k\ell_k-\min_k\ell_k$. For a fixed simplex vector $w$, define the exponential
update $F(w,\ell)$. Interpolating the loss by $s\ell$, $0\le s\le1$, gives

$$
\frac{d}{ds}F_k(w,s\ell)=-\eta F_k(w,s\ell)\Big\{\ell_k-\sum_jF_j(w,s\ell)\ell_j\Big\}.
$$

Sum absolute derivatives and integrate over $s$:

$$
\|F(w,\ell)-w\|_1\le\eta\,{\rm osc}(\ell).
$$

The fixed-share update therefore satisfies the pathwise inequality

$$
d_i:=\|w_i-u\|_1\le q\,d_{i-1}+q\eta\,{\rm osc}(\ell_i).
$$

The observations and component forecasts lie in $[-M,M]$, so

$$
{\rm osc}(\ell_i)\le4M\max_{k,j}|\hat\theta_{k,i-1}-\hat\theta_{j,i-1}|.
$$

Step 1 bounds its local $L^4$ norm by $O(\Delta^{1/3})$. Unroll the contraction from
$m=\lfloor n/2\rfloor$, noting $d_m\le2$, and apply Minkowski's inequality:

$$
\|d_{n-1}\|_4\le2q^{\,n-1-m}+C\Delta^{1/3}\sum_{r=0}^{n-2-m}q^r=O(\Delta^{1/3}).
$$

This proves the limiting weights; we have not assumed they are deterministic or independent of
estimator errors. Fixed positive $\delta$ and fixed $\eta$ are essential to this step.

**Step 3: control the effect of random weighting.** Define the uniform average
$\bar\theta_{n-1}=K^{-1}\sum_k\hat\theta_{k,n-1}$ and $R=\hat\theta_{B,n-1}-\bar\theta_{n-1}$. Because the
weight differences sum to zero,

$$
|R|\le d_{n-1}\max_k|\hat\theta_{k,n-1}-\theta(\tau)|.
$$

Hölder's inequality, rather than an independence assumption, gives

$$
\|R\|_2\le\|d_{n-1}\|_4\,\Big\|\max_k|\hat\theta_{k,n-1}-\theta(\tau)|\Big\|_4=O(\Delta^{2/3}).
$$

Thus $\mathbb ER=o(\Delta^{1/3})$. Also ${\rm Var}(R)=O(\Delta^{4/3})$, and Cauchy–Schwarz bounds its
covariance with the uniform average by $O(\Delta)$. Hence

$$
{\rm Var}(\hat\theta_{B,n-1})={\rm Var}(\bar\theta_{n-1})+o(\Delta^{2/3}).
$$

**Step 4: compute bias and shared-stream covariance.** For a pre-observation forecast, the limiting
mean observation age is $1/\lambda_k$. Taylor expansion of $\theta$, with negligible initialized tails,
yields

$$
\mathbb E\hat\theta_{k,n-1}-\theta(\tau)=-\theta'(\tau)\frac{\Delta}{\lambda_k}+O((\Delta/\lambda_k)^2)=-\frac{\theta'(\tau)\Delta(N_k+1)}{2}+O(\Delta^{2/3}).
$$

Write $r_k=1-\lambda_k$. The infinite geometric cross-weight sum is

$$
S_{kj}:=\sum_{r=0}^\infty\lambda_k\lambda_j(r_kr_j)^r=\frac{\lambda_k\lambda_j}{\lambda_k+\lambda_j-\lambda_k\lambda_j}=\frac{2}{N_k+N_j}.
$$

For the initialized estimates through $n-1$, the exact cross-weight sum is
$S_{kj}+(1-S_{kj})(r_kr_j)^{n-2}$. Its additional term is negligible. Independence across observation
times gives covariance as a sum of these products multiplied by $v(i\Delta)$. Continuity of $v$ and
concentration of the products near $\tau$ imply

$$
{\rm Cov}(\hat\theta_{k,n-1},\hat\theta_{j,n-1})=v(\tau)\frac{2}{N_k+N_j}+o(\Delta^{2/3}).
$$

To justify that remainder, split the sum at a fixed physical distance $\varepsilon$ from $\tau$. Nearby
variance differences are bounded by the modulus of continuity of $v$; older cross-weight mass is
exponentially small. Divide by $\Delta^{2/3}$, let $\Delta\to0$, then let $\varepsilon\downarrow0$.

Average the biases and sum all $K^2$ covariance terms for $\bar\theta$. Step 3 transfers their leading
terms to the adaptive bank. Finally use MSE = variance + squared bias. This proves all three displayed
expansions. ∎

*What this establishes—and what rema…* `[cut off]`

---

## 6. Part C: transfer from component expansions to the bank

**Status:** a proved transfer result under the assumptions below; verifying them for the actual
components is unfinished.

**Question.** If each estimator has a specified leading bias and covariance, what are the bank's
leading bias, variance and MSE?

Fix an endpoint $\tau>0$, let $\Delta=\tau/n\to0$, and keep $K$ fixed. Write $\hat\rho_k=\hat\rho_{k,n-1}$
for forecasts of the same deterministic target $\rho_S(\tau)$. Assume, for constants $b_k$ and $V_{kj}$,

$$
\mathbb E\hat\rho_k-\rho_S(\tau)=b_k\Delta^{1/3}+o(\Delta^{1/3}),
$$

$$
{\rm Cov}(\hat\rho_k,\hat\rho_j)=V_{kj}\Delta^{2/3}+o(\Delta^{2/3}).
$$

The coefficients $V_{kj}$ include cross-covariances from using the same stream. They are not assumed
zero. Let $\|X\|_p=(\mathbb E|X|^p)^{1/p}$, and also assume local fourth-moment control:

$$
\sup_{\lfloor n/2\rfloor\le i\le n}\Big\|\max_k|\hat\rho_{k,i-1}-\rho_S(i\Delta)|\Big\|_4=O(\Delta^{1/3}).
$$

This condition controls occasional large errors as well as average squared error. It is an assumption
to establish from the component updates, not a consequence of the two endpoint expansions alone.

Predictions and weights use only the past. Suppose $|\hat\rho_{k,i-1}|\le R$ and the common scoring
target satisfies $|\hat p_i|\le P$, for finite constants independent of $\Delta$. The rank proxy has
$P=3$. Start with any simplex weight vector and use

$$
\ell_{k,i}=(\hat\rho_{k,i-1}-\hat p_i)^2,\qquad\tilde w_{k,i}=\frac{w_{k,i-1}e^{-\eta\ell_{k,i}}}{\sum_jw_{j,i-1}e^{-\eta\ell_{j,i}}},
$$

$$
w_{k,i}=(1-\delta)\tilde w_{k,i}+\frac{\delta}{K},\qquad0<\delta<1,\quad0\le\eta<\infty.
$$

Both $\delta$ and $\eta$ are fixed as $\Delta\to0$. The conditional mean of the scoring target need not
equal $\rho_S$ for this particular weighting argument; the component assumptions must still hold. This
is not a theorem for discounted softmax, log-score weights, or arbitrary changing share and learning
rates.

**Result.** For $\hat\rho_B=\sum_kw_{k,n-1}\hat\rho_k$, we seek to prove

$$
\boxed{\;{\rm Bias}(\hat\rho_B)=\frac{\Delta^{1/3}}{K}\sum_kb_k+o(\Delta^{1/3}).\;}
$$

$$
\boxed{\;{\rm Var}(\hat\rho_B)=\frac{\Delta^{2/3}}{K^2}\sum_{k,j}V_{kj}+o(\Delta^{2/3}).\;}
$$

$$
\boxed{\;{\rm MSE}(\hat\rho_B)=\Delta^{2/3}\bigg[\Big(\frac1K\sum_kb_k\Big)^2+\frac1{K^2}\sum_{k,j}V_{kj}\bigg]+o(\Delta^{2/3}).\;}
$$

### Proof: control the adaptive weights, then transfer the component expansions

**Step 1. Fixed share makes the weights asymptotically uniform.** Set $u=(1/K,\dots,1/K)$, $q=1-\delta$,
and $D_i=\|w_i-u\|_1$. The exponential-update inequality proved in part B gives

$$
D_i\le qD_{i-1}+q\eta\,{\rm osc}(\ell_i),\qquad{\rm osc}(\ell_i)=\max_k\ell_{k,i}-\min_k\ell_{k,i}.
$$

Factoring the difference of two squares gives

$$
|\ell_{k,i}-\ell_{j,i}|\le2(R+P)|\hat\rho_{k,i-1}-\hat\rho_{j,i-1}|.
$$

Subtract the same target $\rho_S(i\Delta)$ from both predictions. The local fourth-moment assumption
implies $\|{\rm osc}(\ell_i)\|_4=O(\Delta^{1/3})$, uniformly in the latter half of the experiment. With
$m=\lfloor n/2\rfloor$ and $D_m\le2$, unroll the contraction:

$$
\|D_{n-1}\|_4\le2q^{\,n-1-m}+O(\Delta^{1/3})\sum_{r=0}^{n-2-m}q^r=O(\Delta^{1/3}).
$$

The geometric sum is bounded because $q<1$ is fixed; its initial-state term decays faster than a power
of $\Delta$. No independence between weights and predictions was used.

**Step 2. Random weighting contributes a smaller-order error.** Let $\bar\rho=K^{-1}\sum_k\hat\rho_k$ and
$Z_\Delta=\hat\rho_B-\bar\rho$. Since the weight differences sum to zero,

$$
Z_\Delta=\sum_k(w_{k,n-1}-1/K)(\hat\rho_k-\rho_S(\tau)).
$$

Hölder's inequality gives

$$
\|Z_\Delta\|_2\le\|D_{n-1}\|_4\,\Big\|\max_k|\hat\rho_k-\rho_S(\tau)|\Big\|_4=O(\Delta^{2/3}).
$$

Hence $|\mathbb EZ_\Delta|=O(\Delta^{2/3})$ and ${\rm Var}(Z_\Delta)=O(\Delta^{4/3})$. Averaging the assumed
component biases proves the bank bias formula. Summing the assumed cross-covariances gives

$$
{\rm Var}(\bar\rho)=\frac{\Delta^{2/3}}{K^2}\sum_{k,j}V_{kj}+o(\Delta^{2/3}).
$$

Finally, Cauchy–Schwarz implies

$$
\begin{aligned}
|{\rm Var}(\hat\rho_B)-{\rm Var}(\bar\rho)|&\le{\rm Var}(Z_\Delta)+2\sqrt{{\rm Var}(Z_\Delta){\rm Var}(\bar\rho)}\\
&=O(\Delta^{4/3})+O(\Delta)=o(\Delta^{2/3}).
\end{aligned}
$$

This proves the variance formula. The identity MSE = variance + squared bias proves the final
expansion. ∎

> **What is now proved:** different estimator formulas are allowed, and dependence between weights and
> estimator errors is accounted for. **What is still assumed:** the component bias/covariance
> expansions and local fourth-moment control. Persistent grid, truncation or model bias can violate
> them; see *G for a separate theorem allowing different persistent limits*. The theorem here gives
> uniform limiting weights in this regime, not adaptive superiority or pointwise optimal selection.
> Fixed forgetting at fixed sampling frequency is a different limiting problem.

---

## 7. Part D (NEW TO THESIS): derive the component bias, variance and MSE

*Delaigle connection* · *Derived here* · *Delaigle connection & credit →*

**Purpose:** derive the component error terms that part C assumes. This covers a finite vector of
exponentially weighted features followed by a smooth output map, including fixed-order Hermite
estimates under the conditions below. "Derived here" identifies this worked calculation; it does not
establish research novelty.

### Question: what are a nonlinear component's bias and variance?

Write $\rho_k^\circ(s)$ for the component's population output at time $s$, and
$b_k^{\rm app}=\rho_k^\circ(\tau)-\rho_S(\tau)$ for its approximation bias. We seek to prove

$$
\mathbb E\hat\rho_{k,n-1}-\rho_S(\tau)=b_k^{\rm app}-(\rho_k^\circ)'(\tau)\frac{\Delta}{\lambda_k}+O(\Delta^{2/3}),
$$

$$
{\rm Var}(\hat\rho_{k,n-1})=\frac{v_k}{N_k}+o(\Delta^{2/3}).
$$

The first term is approximation error, the second is lag behind a changing target, and the variance
term decreases with effective sample size. No Gaussian observation model is assumed.

### Setup and assumptions

Fix $\tau>0$, $\Delta=\tau/n\to0$, a finite number of estimators, and fixed finite-dimensional bounded
feature maps $\psi_k$. Observations $Z_{i,\Delta}$ are independent within each experiment. Define

$$
\hat s_{k,1}=\psi_k(Z_1),\qquad\hat s_{k,i}=(1-\lambda_k)\hat s_{k,i-1}+\lambda_k\psi_k(Z_i),
$$

$$
\hat\rho_{k,i}=g_k(\hat s_{k,i}),\qquad\lambda_k\sim A_k\Delta^{2/3},\quad A_k>0,\qquad N_k=\frac{2-\lambda_k}{\lambda_k}.
$$

For observations at physical time $s$, let

$$
m_k(s)=\mathbb E\psi_k(Z_s),\qquad\Sigma_{kj}(s)={\rm Cov}(\psi_k(Z_s),\psi_j(Z_s)),\qquad\rho_k^\circ(s)=g_k(m_k(s)).
$$

Assume $m_k\in C^2([0,\tau])$, the cross-covariances are continuous, and $g_k$ has bounded derivatives
through order three on a neighbourhood of the bounded state region. Features and their dimensions stay
fixed as $\Delta\to0$. Forecasts formed through $n-1$ target time $\tau$. Subscripts $\Delta$ are
suppressed.

All quantities below without a time argument are evaluated at $\tau$. Define the gradient, Hessian and
projected covariance by

$$
G_k=\nabla g_k(m_k),\qquad J_k=\nabla^2g_k(m_k),\qquad\Gamma_{kj}=G_k^\top\Sigma_{kj}G_j,\qquad v_k=\Gamma_{kk}.
$$

$N_k$ is the long-history effective sample size. The exact initialized finite weights have exponentially
small corrections in this asymptotic regime.

### Proof: weighted feature moments, then Taylor expansion

**1. Express the state as a weighted sum.** At time $n-1$, write
$\hat s_{k,n-1}=\sum_{i=1}^{n-1}a_{k,i}\psi_k(Z_i)$. Its weights are

$$
a_{k,1}=(1-\lambda_k)^{n-2},\qquad a_{k,i}=\lambda_k(1-\lambda_k)^{n-1-i}\quad(2\le i\le n-1).
$$

Initialization contributes only exponentially small errors because $n\lambda_k\asymp\Delta^{-1/3}\to\infty$.
For the corresponding infinite geometric weights, mean age and squared age are

$$
\sum_{r=1}^\infty r\lambda_k(1-\lambda_k)^{r-1}=\lambda_k^{-1},\qquad\sum_{r=1}^\infty r^2\lambda_k(1-\lambda_k)^{r-1}=\frac{2-\lambda_k}{\lambda_k^2}.
$$

Set $d_k=\hat s_{k,n-1}-m_k(\tau)$. Taylor expansion of $m_k(\tau-r\Delta)$, with geometric decay
controlling distant observations, gives

$$
\mathbb Ed_k=-\frac{\Delta}{\lambda_k}m_k'+\frac{\Delta^2(2-\lambda_k)}{2\lambda_k^2}m_k''+o(\Delta^{2/3}).
$$

**2. Retain covariance between estimators using the same stream.** Independence across observation
times gives a sum of same-time cross-covariances. The geometric overlap is

$$
\sum_{r=1}^\infty\lambda_k\lambda_j(1-\lambda_k)^{r-1}(1-\lambda_j)^{r-1}=\frac{\lambda_k\lambda_j}{\lambda_k+\lambda_j-\lambda_k\lambda_j}=\frac{2}{N_k+N_j}.
$$

Continuity of $\Sigma_{kj}$ and concentration of the weights near $\tau$ therefore imply

$$
{\rm Cov}(d_k,d_j)=\frac{2\Sigma_{kj}}{N_k+N_j}+o(\Delta^{2/3}).
$$

For a fixed-dimensional vector of bounded independent centered features, expansion of the fourth moment
coordinate by coordinate yields

$$
\mathbb E\Big\|\sum_ia_{k,i}\{\psi_k(Z_i)-m_k(i\Delta)\}\Big\|^4\le C\Big(\sum_ia_{k,i}^2\Big)^2=O(\Delta^{4/3}).
$$

The mean drift term is $O(\Delta^{1/3})$, so $\|d_k\|_{L^4}=O(\Delta^{1/3})$. In particular,

$$
\mathbb E(d_kd_k^\top)=\frac{\Sigma_{kk}}{N_k}+\frac{\Delta^2}{\lambda_k^2}m_k'(m_k')^\top+o(\Delta^{2/3}).
$$

**3. Pass these moments through the output map.**

$$
g_k(m_k+d_k)=g_k(m_k)+G_k^\top d_k+\tfrac12d_k^\top J_kd_k+R_k,\qquad|R_k|\le C\|d_k\|^3.
$$

Thus $\mathbb E|R_k|=O(\Delta)=o(\Delta^{2/3})$. Taking expectations proves the displayed first-order bias
and the more detailed expansion below, since $(\rho_k^\circ)'=G_k^\top m_k'$.

For the covariance calculation, the remainder after the linear term has $L^2$ norm $O(\Delta^{2/3})$,
while the centered linear term has $L^2$ norm $O(\Delta^{1/3})$. Cauchy–Schwarz bounds their covariance
by $O(\Delta)$. Consequently,

$$
\boxed{\;{\rm Cov}(\hat\rho_{k,n-1},\hat\rho_{j,n-1})=\frac{2\Gamma_{kj}}{N_k+N_j}+o(\Delta^{2/3}).\;}
$$

Taking $j=k$ proves the variance formula. The same moment argument holds uniformly in the latter half of
the observation interval. Bounded gradients make each $g_k$ Lipschitz on the state region, giving

$$
\boxed{\;\sup_{\lfloor n/2\rfloor\le i\le n}\Big\|\max_k|\hat\rho_{k,i-1}-\rho_k^\circ(i\Delta)|\Big\|_4=O(\Delta^{1/3}).\;}
$$

This is error around each component's population output. It is not automatically error around the true
Spearman coefficient. ∎

### Full bias and MSE when approximation bias persists

A fixed nonzero approximation bias requires second-order bias terms to obtain MSE accurately through
order $\Delta^{2/3}$. Define

$$
B_{1,k}=-(\rho_k^\circ)'\frac{\Delta}{\lambda_k},
$$

$$
B_{2,k}=\frac{\Delta^2}{2\lambda_k^2}\big[(2-\lambda_k)G_k^\top m_k''+(m_k')^\top J_km_k'\big]+\frac{{\rm tr}(J_k\Sigma_{kk})}{2N_k}.
$$

The proof above gives

$$
{\rm Bias}_{\rho_S}(\hat\rho_{k,n-1})=b_k^{\rm app}+B_{1,k}+B_{2,k}+o(\Delta^{2/3}).
$$

Since $B_{1,k}=O(\Delta^{1/3})$ and $B_{2,k}=O(\Delta^{2/3})$, squaring the bias and adding the variance
yields

$$
{\rm MSE}_{\rho_S}(\hat\rho_{k,n-1})=(b_k^{\rm app})^2+2b_k^{\rm app}B_{1,k}+B_{1,k}^2+2b_k^{\rm app}B_{2,k}+\frac{v_k}{N_k}+o(\Delta^{2/3}).
$$

If $b_k^{\rm app}=0$, this reduces to squared drift bias plus variance. If it is fixed and nonzero, the
component MSE converges to $(b_k^{\rm app})^2$. Dropping $2b_k^{\rm app}B_{2,k}$ would omit a term of the
same order as the variance.

### Application: fixed-order Hermite estimates

*Adaptation of published method · Credit & what differs →*

Take $\psi_k(X,Y)$ to contain the finite collection of Hermite functions $h_r(X)$, $h_s(Y)$, and products
$h_r(X)h_s(Y)$. Each fixed-order Hermite function is bounded. Their exponentially updated coefficients
therefore form exactly the finite vector state above.

The centered Spearman plug-in expression in the thesis (printed pp.28–30) is

$$
\hat\rho_N=12\iint(\hat F_{X,N}(x)-\tfrac12)(\hat F_{Y,N}(y)-\tfrac12)\hat f_{XY,N}(x,y)\,dx\,dy.
$$

For fixed finite order, this is a polynomial of the coefficient vectors, with fixed integral constants.
Its derivatives are bounded on the bounded state region. Under the smooth-time and independence
assumptions, the component theorem thus supplies its bias, variance, shared-stream covariance and local
fourth-moment rate.

> **Scope:** fixed truncation can leave $b_N^{\rm app}\ne0$. The raw output is bounded by some finite
> constant depending on the order, not automatically by 1. Adaptive standardization or features computed
> from evolving estimated marginal transforms need an additional analysis; they are not independent
> fixed-feature observations.

*Related precedent:* Stephanou and Varughese, Section 5, Theorem 2 and Appendix A, analyse exponentially
weighted Hermite variance in an i.i.d., fixed-order setting using Taylor expansion. The infill drift and
second-order bias calculation here is our derivation under the displayed assumptions, not their theorem.

### Application: count matrices, subject to denominator control

*Conditional application · Credit & what differs →*

For fixed cell boundaries, let $\psi(X,Y)$ be the vector of cell indicators. If normalized cell masses
follow the stated exponential recursion, they satisfy the state assumptions. The final count-based
correlation is a nonlinear map of these masses.

For cell probabilities $p_{ij}$, define row and column probabilities $\pi_i=\sum_jp_{ij}$,
$\omega_j=\sum_ip_{ij}$ and midranks $r_i=\sum_{a<i}\pi_a+\pi_i/2$, $s_j=\sum_{b<j}\omega_b+\omega_j/2$. The
output map is

$$
g(p)=\frac{\sum_{i,j}p_{ij}(r_i-\frac12)(s_j-\frac12)}{\sqrt{\{\sum_i\pi_i(r_i-\frac12)^2\}\{\sum_j\omega_j(s_j-\frac12)^2\}}}.
$$

This map is smooth where both denominator factors are positive. To extend the theorem, require population
rank variances uniformly bounded away from zero and a defined, bounded output for degenerate sample
matrices. The latter is an implementation condition, not an assumed new fallback.

Under those conditions, bounded independent indicators concentrate exponentially about their means: for
fixed $\varepsilon>0$, $\Pr(\|\hat s-\mathbb E\hat s\|>\varepsilon)\le C\exp(-c/\lambda)$. A smooth extension
agreeing with $g$ near the population state can then be used as a proof device. The exceptional region
contributes less than any power of $\Delta$, so the displayed expansions apply.

Fixed grid resolution may still leave a nonzero population approximation bias. The thesis's exact
normalized forgetting update and degenerate-matrix behaviour must match these conditions before claiming
this result for its implementation.

### What can now be passed to the bank theorem?

If every component's population output equals the common true target throughout the local time interval,
$\rho_k^\circ(s)=\rho_S(s)$, then these calculations give the component constants required by part C:

$$
b_k=-\frac{\rho_S'(\tau)}{A_k},\qquad V_{kj}=\Gamma_{kj}\frac{A_kA_j}{A_k+A_j}.
$$

The local fourth-moment condition is also established in that case. Fixed-share combination then gives
the bias, variance and MSE in part C.

> **What remains:** fixed Hermite order, fixed grids or model misspecification may give different
> persistent population biases. The existing common-target bank proof does not cover that situation.
> Online Fisher scoring also needs its own recursion analysis. This is progress on the component
> calculations, not a completed theorem for every estimator in the actual bank.

*(E. "What remains for online Fisher scoring?" was collapsed in the photo.)*

---

## 8. Part F: how does this compare with Delaigle's proof?

### How does this compare with Delaigle and coauthors' published proof?

Camirand Lemyre, Delaigle and Mao (2026), Proposition 1, p.6, derive bias, variance, MSE and integrated
versions for their specified recursive kernel estimator from data, smoothness, kernel and tuning
assumptions. Their remainder control is uniform over density locations, times within the current block
and the stated candidate smoothing parameters. The proof is in Supplement Appendix B.

**Our part C proves the combination step:** starting from component error expansions, it derives the
effect of adaptive fixed-share weights. The new component result in part D now derives bias, variance and
covariance for smooth functions of finite exponentially weighted feature states. It covers fixed-order
Hermite estimates under explicit assumptions, while retaining their approximation bias. It does not yet
supply every condition for the actual mixed bank, especially its Fisher component.

Both use the bias–variance identity and control lower-order terms. Their estimator calculations separate
spatial smoothing, temporal drift and preceding-block contributions. Our weighting calculation uses
fixed-share contraction, Hölder's inequality and covariance bounds. Their separate Theorem 1, pp.8–9,
connects cross-validation to asymptotic integrated error; Proposition 1 alone is not the tuning
guarantee.

Our conditional scalar MSE order $\Delta^{2/3}$ and their kernel-density order $\Delta t^{4/7}$ concern
different estimation targets and assumptions. They are not a performance ranking. Use the published proof
as a model for explicit assumptions, leading terms and remainder control, not as a citation proving our
bank theorem. *Read the paper's theory in more detail.*

> *Transcriber note:* "$\Delta t^{4/7}$" is as printed; probably $(\Delta t)^{4/7}$.

---

## 9. Part G (NEW TO THESIS): what if the estimators retain different approximation biases?

*Derived here · Credit & what differs →*

**Purpose:** remove the common-true-target restriction from the bank's weighting argument.
**Result:** asymptotic bias, variance and MSE limits for the specified fixed-share bank. The limits can be
nonzero. This gives leading constants plus $o(1)$, not the finer rate expansion in B–C.

### Question: can we prove limits with persistent component bias?

Fix the physical endpoint $\tau=n\Delta$ and let $\Delta\to0$, so $n\to\infty$. Let $a_k=a_k(\tau)$ be
estimator $k$'s population limit. It may differ from the true target $\rho_S(\tau)$. We seek to prove

$$
{\rm Bias}(\hat\rho_{B,n-1})=b_\infty+o(1),\qquad{\rm Var}(\hat\rho_{B,n-1})=v_\infty+o(1),
$$

$$
{\rm MSE}(\hat\rho_{B,n-1})=b_\infty^2+v_\infty+o(1).
$$

Unlike C, we will allow the limiting weights to remain random. We construct their probability law below,
rather than assuming that they become fixed numbers.

*(Collapsed in the photos: "Assumptions and exact update"; "The limiting random bank and its
bias–variance constants"; "Why can the bank retain variance after its components stabilize?")*

### The separate discounted-softmax version (NEW TO THESIS)

*Derived here · Credit & what differs →*

For the alternative score recursion $L_{k,i}=\beta L_{k,i-1}+(1-\beta)\ell_{k,i}$,
$w_{k,i}\propto e^{-\eta L_{k,i}}$, fix $0\le\beta<1$, $0\le\eta<\infty$, and initial scores bounded
uniformly across the $\Delta$ experiments. Keep the same component and proxy assumptions. Define

$$
Q=(1-\beta)\sum_{r=0}^\infty\beta^rP^*_{-r},\qquad\pi_k(q)=\frac{e^{-\eta(a_k^2-2a_kq)}}{\sum_je^{-\eta(a_j^2-2a_jq)}},\qquad g(q)=\sum_ka_k\pi_k(q).
$$

Then $\hat\rho_{B,n-1}\Rightarrow g(Q)$, with convergence of the first two moments. The bias and variance
limits are $\mathbb Eg(Q)-\rho_S(\tau)$ and ${\rm Var}\{g(Q)\}$.

**Proof.** Put $m=n-1$. Unroll the scores and replace the component outputs in them by their limits:

$$
L^a_{k,m}=\beta^mL_{k,0}+(1-\beta)\sum_{r=0}^{m-1}\beta^r(a_k-P_{m-r})^2.
$$

The difference of squares bounds the $L^2$ score replacement error by

$$
2(R+P_{\max})(1-\beta)\sum_{r=0}^{m-1}\beta^r\|\hat\rho_{k,m-r-1}-a_k\|_2.
$$

Split at a fixed recent lag $h$. The finite head tends to zero by component convergence; boundedness makes
the tail $O(\beta^h)$. Thus the score error tends to zero. Let
$Q_m=(1-\beta)\sum_{r=0}^{m-1}\beta^rP_{m-r}$. Expansion of the squares shows

$$
L^a_{k,m}=\beta^mL_{k,0}+(1-\beta^m)a_k^2-2a_kQ_m+T_m,
$$

where $T_m$ is common to all estimators and cancels in softmax. Initial terms vanish; continuity on
bounded score ranges gives $\|\hat\rho_{B,m}-g(Q_m)\|_2\to0$. Finite-window convergence and the uniform
geometric tail bound imply $Q_m\Rightarrow Q$. Boundedness gives convergence of moments and the result. ∎

The derivative $g'(q)=2\eta\,{\rm Var}_{\pi(q)}(a_k)$ is positive when $\eta>0$ and the component limits
differ. Also

$$
{\rm Var}(Q)=\frac{1-\beta}{1+\beta}{\rm Var}(P_0^*).
$$

Thus this alternative also generally has positive limiting bank variance with different persistent
component limits and nondegenerate proxies.

> *Transcriber note:* the minus signs in $\pi_k$'s exponents are faint in the photo; $-\eta(a_k^2-2a_kq)$
> is the reading consistent with $w\propto e^{-\eta L}$ and with $g'(q)=2\eta{\rm Var}_\pi(a_k)$.

#### A concrete rank-proxy example

Let $U,V$ be independent uniform variables and $P=12(U-1/2)(V-1/2)$. Then the true Spearman target is
zero, $\mathbb EP=0$, and ${\rm Var}(P)=1$. For illustrative constant component limits $(-a,a)$,
$0<a\le1$,

$$
g(q)=a\tanh(2\eta aq).
$$

Symmetry makes the limiting bank bias zero, but for $\eta>0$,

$$
\lim{\rm Var}(\hat\rho_B)=\lim{\rm MSE}(\hat\rho_B)=a^2\,\mathbb E\tanh^2(2\eta aQ)>0.
$$

This is an analytical example, not a claim that the implemented estimators have these limits.

> **What is now proved:** different persistent component limits are permitted in the weighting step,
> and random limiting weights are accounted for. **Still needed for the …** `[cut off]`

---

## 10. *Bank questions* page (`#bank-questions`): what we still need to check

*(Question 3, "Can changing weights keep the bank's estimate fluctuating?", was collapsed.)*

**4. Where does the forgetting factor enter the Fisher update?**
**Unresolved.** We need the actual update showing how λ affects the score, accumulated information or
parameter step. Saying that it enters "through the copula estimate" does not specify that equation.
*What the thesis currently specifies*

**5. What exactly do we compare each prediction against?**
**Check the squared-loss proxy.** We need its exact formula and timing, including how estimated margins
or a rolling estimate enter it. Then we can check whether its behaviour matches the proof's assumptions.
The squared-loss result does not automatically cover log scoring. *Scoring assumptions*

**6. Do the actual estimators satisfy the proof's assumptions?**
**Partly established.** We have calculations for specified linear and smooth feature-based constructions.
We still need to match the actual count-matrix, Hermite and Fisher outputs to the required bounds and
limits, including any approximation bias. *Component calculations · What checking the implementation
means*

**7. Does our asymptotic setting match the claim we want to make?**
**Check the scope.** The infill results study increasingly frequent observations over a fixed time
interval, with specified tuning assumptions. They do not automatically describe a longer stream sampled
at a fixed frequency with a fixed forgetting factor. *Infill explained · The linear example's tuning
assumptions*

**8. Does the bank perform better than choosing one estimator?**
**These bias–variance proofs do not establish that.** Use the existing experiments to compare the bank
with its constituent estimators. Regret gives a separate cumulative-loss comparison under its own
assumptions. *What regret-to-risk tells us*

> **Start with questions 4–6.** They determine whether we can apply the new theorem to the implemented
> mixed bank. This is a list of checks, not a requirement to develop new methods.

*The results provide theoretical support for the bank. "New to thesis" does not establish research
novelty. Credit and contribution notes.*
