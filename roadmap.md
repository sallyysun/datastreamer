# Tying it together
## The central question
> **How should we estimate a time-varying copula dependence parameter when observations arrive sequentially?**

Then build around this:
$$
\boxed{\text{stationary efficiency} \quad \leftrightarrow \quad \text{adaptation to change}}
$$

## Progress

**1. Baseline: online MLE / Fisher scoring**

Start with simplest sequential estimator:

$$
\theta_t
=
\theta_{t-1}
+
I(\theta_{t-1})^{-1}s_t(\theta_{t-1})
$$

This gives the cleanest connection to **Fisher information** and lets us derive/measure estimation efficiency.

Then introduce **exponential forgetting**:

$$
\hat\theta_t
=
\text{weighted contribution of recent observations}.
$$

Study how $\lambda$ affects:

- variance in a stationary environment
- adaptation speed after a change
- bias
- recovery time.

This should probably be the **core**.


**2. Explicit change detection + restart**

Then compare forgetting with:

$$
\boxed{\text{estimate continuously} \quad vs \quad \text{detect change + restart}}
$$

For example:

$$
\text{online estimator}
\rightarrow
\text{CUSUM}
\rightarrow
\text{change detected}
\rightarrow
\text{restart estimator}.
$$

This is particularly interesting because it directly addresses the original idea:

> Is it better to continuously forget old information, or explicitly detect structural changes and reset?

That's a very clean thesis contribution.


**3. Dynamic latent-process model**

A model for the *true data-generating process* in simulations.
$$
z_t = z_{t-1}+\omega_t,
\qquad
\rho_t=\tanh(z_t).
$$

For example, generate data under:

* constant $\rho$
* smoothly changing $\rho_t$
* abrupt change in $\rho$
* stochastic latent $\rho_t$.

Then we know the true $\rho_t$, allowing us to properly evaluate estimators.

**4. Bayesian sequential estimation — optional extension**

Then ask:
> What if instead of maintaining a point estimate, we maintain a posterior?
$$
p(\theta_t\mid x_{1:t})
$$

- Implement simple **sequential Bayesian update**.
- Probably not start with full MCMC over the entire history. That's computationally heavier and starts pulling thesis toward "Bayesian dynamic copula modelling"

**5. Particle filter / SMC — only if time permits**

This is the most natural Bayesian method if we treat
$$
\theta_t
$$
as a latent stochastic process.

Particles represent possible current states:
$$
{\theta_t^{(1)},\ldots,\theta_t^{(N)}}.
$$

Then:
$$
\text{predict}
\rightarrow
\text{observe}
\rightarrow
\text{weight}
\rightarrow
\text{resample}.
$$

## Research hierarchy
$$
\boxed{
\begin{array}{c}
\textbf{Dynamic copula DGP} \\
\downarrow \\
\textbf{Online MLE / Fisher scoring} \\
\downarrow \\
\textbf{Exponential forgetting} \\
\downarrow \\
\textbf{Change detection + restart} \\
\downarrow \\
\textbf{Bayesian sequential estimation} \\
\downarrow \\
\textbf{Particle filter (optional)}
\end{array}
}
$$
And evaluate everything on the **same simulated changing-dependence environments**.

## Conclusion
**Question:** how should an estimator adapt?

→ **Frequentist baseline:** online Fisher scoring
→ **Adaptation mechanism:** forgetting
→ **Alternative adaptation:** change detection/restart
→ **Alternative paradigm:** Bayesian filtering
→ **Advanced version:** particle filtering.

And `dcvar`'s dynamic copula tells me *how dependence can evolve*, while my work asks *how an estimator should track that evolution online*.

# Non-parametric
We have **two levels**:

**1. Estimate the copula itself**
$$
C_t(u,v)
$$

**2. Or estimate a parameter describing the copula**
$$
\theta_t \quad\text{or}\quad \rho_t
$$

Xiao and Hermite are primarily relevant to **(1)**.

## Xiao: online empirical copula
The Xiao/streaming approach gives you a way to estimate the **empirical copula directly as observations arrive**.

Instead of assuming
$$
C_t = C_{\theta_t}
$$

you essentially estimate
$$
\widehat C_t(u,v)
$$
directly from the observations.

Using pseudo-observations $U_t,V_t$, followed by an empirical copula
$$
\widehat C_n(u,v)
=
\frac1n\sum_{i=1}^n
1{U_i\leq u,V_i\leq v}.
$$

Then your main contribution is to make this **streaming + exponentially weighted**:
$$
\widehat C_t^{(\lambda)}
=
\lambda\widehat C_{t-1}^{(\lambda)}
+
(1-\lambda)
1{U_t\leq u,V_t\leq v}.
$$

So:

> **Xiao = computational/online way of maintaining the empirical copula.**

## Hermite: another way to represent/estimate the copula
Hermite is useful when you don't want to store the entire empirical copula surface.

Instead of representing $C(u,v)$ on a grid, you represent it through a **basis expansion**:
$$
C(u,v)
\approx
\sum_{j,k} a_{jk} H_j(u)H_k(v)
$$
(or the corresponding Hermite representation).

Then you only need to maintain/update the coefficients $a_{jk}$.

So conceptually:

- **Xiao → online empirical/checkerboard representation of (C_t)**
- **Hermite → compact basis representation of (C_t)**

Both are ways of estimating the **full dependence structure without imposing a Gaussian/t/Clayton/etc. copula family**.

## Exponential forgetting

This is actually the really interesting part.

You can apply exponential forgetting to **either representation**.

For Xiao:
$$
\widehat C_t
=
\lambda\widehat C_{t-1}
+
(1-\lambda)C_t^{\text{new}}.
$$

For Hermite coefficients:
$$
\hat a_{jk,t}
=
\lambda\hat a_{jk,t-1}
+
(1-\lambda),h_{jk}(U_t,V_t).
$$

So the **core research question** becomes:

> How should a nonparametric online copula estimator trade off statistical efficiency against adaptation to changing dependence?

That's exactly aligned with your existing motivation around exponential forgetting and effective sample size. 

## And this is different from the dynamic parametric copula paper

This distinction is important for your thesis.

|                    | Your Xiao/Hermite approach     | Dynamic copula paper                |
| ------------------ | ------------------------------ | ----------------------------------- |
| Copula             | **Nonparametric**              | Parametric                          |
| Estimate           | $C_t(u,v)$                     | $\theta_t$                          |
| Dependence family  | Not assumed                    | Gaussian/t/etc.                     |
| Data processing    | **Online/sequential**          | Uses full sample likelihood         |
| Forgetting         | **Yes**                        | Generally no                        |
| Main question      | How should estimator adapt?    | How does dependence evolve?         |
| Representation     | Empirical/basis                | Parametric copula                   |
| Fisher information | Can study estimator efficiency | Used for likelihood-based inference |
| Change detection   | Central to your project        | Not the main objective              |

So **Hermite/Xiao aren't competing with the dynamic copula model**. They're providing your **nonparametric estimator**.

And that gives you a very clean thesis structure:
$$
\boxed{
\text{Dynamic DGP}
\rightarrow
\begin{cases}
    \text{Xiao: online empirical copula}\\
    \text{Hermite: online basis copula}
\end{cases}
\rightarrow
\text{forgetting}
\rightarrow
\text{change detection}
}
$$

Then, if you want, you can add **parametric Fisher-scoring/MLE** as a benchmark:
$$
\text{Dynamic DGP}
\rightarrow
\text{Parametric online MLE/Fisher scoring}
\rightarrow
\text{forgetting}
$$

That comparison is actually quite strong: **nonparametric online estimation vs parametric online estimation**, both under stationary, smoothly drifting, and abruptly changing dependence. Your current draft already frames the experiments around exactly those three regimes. 

And **Hermite/Xiao are therefore not extra random methods to bolt on**—they are the natural estimators for the nonparametric branch of what you are already doing.
