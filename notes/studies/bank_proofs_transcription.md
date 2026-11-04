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

*(Collapsed in the photo: "Setup and assumptions"; "Proof: weighted feature moments, then Taylor
expansion"; "Full bias and MSE when approximation bias persists".)*
