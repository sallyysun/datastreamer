# Chapter 5 — Adaptive memory: the estimator bank (sample draft)

> Transcribed 2026-10-24 from screenshots of `chapter-5-sample.html` ("Separate sample draft ·
> Chapter 5"). Gaps the screenshots did not show are marked **[not visible]**. Boxed
> "implementation notes" from the page are kept as `> **Note.**` blocks. Equation numbers are the
> page's own.

---

## 5.1 Exponential forgetting

Chapter 4 described compact estimators that can be updated as observations arrive. Their usefulness
under changing dependence also depends on which observations retain influence. An estimator using
all past data equally may respond slowly after a change, while an estimator using very little
history may fluctuate considerably. This chapter develops a bank that maintains several memory
settings simultaneously and adjusts their combination weights using observed performance.

Throughout this chapter, $t$ indexes arriving observations. For a linearly updated summary, let
$S_t$ be its state after arrival $t$ and $T_t$ the new contribution. With a constant update weight
$\lambda$,

$$
\begin{aligned}
S_t &= (1-\lambda)S_{t-1} + \lambda T_t, \qquad 0<\lambda<1,\\
S_t &= (1-\lambda)^t S_0 + \sum_{j=0}^{t-1} \lambda(1-\lambda)^j T_{t-j}.
\end{aligned}
\tag{5.1}
$$

Thus $\lambda$ is the weight of the new observation, and $1-\lambda$ retains the preceding state.
Larger $\lambda$ gives faster forgetting. This form applies to quantities such as weighted cell
masses and Hermite coefficients. A likelihood estimator instead requires its own gain or
information-discounting rule; a recursive parameter update need not have these exact geometric
observation weights.

Initialisation matters. With $S_0 = 0$, the weights attached to observed contributions sum to
$1-(1-\lambda)^t$. A weighted probability distribution must account for this missing mass.
Initialising at the first observation gives a different finite-history weighting. Neither approach
makes an arbitrary weighted empirical distribution on the unit square a copula: uniform margins
must follow from the copula construction in Chapter 4.

### 5.1.1 Interpreting the memory scale

For an infinite history, the contribution at lag $j$ has weight $a_j$. Its weight-based effective
sample size and half-life are

$$
\begin{aligned}
a_j &= \lambda(1-\lambda)^j, \qquad j = 0, 1, \dots,\\
N_{\text{eff}} &= \frac{\left(\sum_{j=0}^{\infty} a_j\right)^2}{\sum_{j=0}^{\infty} a_j^2}
= \frac{2-\lambda}{\lambda},\\
h_{1/2} &= \frac{\log(1/2)}{\log(1-\lambda)}.
\end{aligned}
\tag{5.2}
$$

The half-life is the lag at which a contribution has half its original weight; it is measured in
arrivals, not elapsed time. The effective sample size describes weight concentration. For an
average of independent observations with equal variance, it also gives a variance-equivalent
sample size. That interpretation does not automatically extend to nonlinear copula estimates or
serially dependent data. Martino, Elvira and Louzada (2017) discuss the limitations of weight-based
ESS measures in importance sampling; the geometric formulas here follow directly from summing the
weights.

With zero initialisation and the first $t$ observation weights normalised to sum to one, the
corresponding finite-history expression is

$$
N_{\text{eff},t} = \frac{2-\lambda}{\lambda}\,\frac{1-(1-\lambda)^t}{1+(1-\lambda)^t}.
\tag{5.3}
$$

It approaches the infinite-history value as the initialisation effect decays. For illustration,
$\lambda = 0.01$ gives a limiting weight-based ESS of 199, while $\lambda = 0.10$ gives 19. These
numbers describe the chosen weighting, not demonstrated information content or recommended
settings for the experiments.

### 5.1.2 Responsiveness and variability

A smaller update weight usually smooths fluctuations but can delay adjustment when dependence
changes. A larger update weight responds more quickly and usually produces a more variable
estimate. Under a stable distribution, increasing memory need not increase bias. The useful memory
scale consequently depends on both the pattern of change and the estimator's approximation error.
The bank addresses uncertainty about that scale by retaining several possibilities.

---

## 5.2 Maintaining several memory scales

One way to adapt is to change the update rate of a single estimator. However, changing its rate
does not reconstruct information already discarded. A bank maintains several states in parallel,
each having processed the same stream with its own fixed setting. Changing their combination
weights changes which retained histories influence the output, without restarting the estimators.

This general strategy has precedents. Hammer et al. (2020) maintain quantile estimators at
different tuning rates and select among them using estimated tracking mean squared error. Their
estimated-MSE "oracle" differs from a comparator supplied with simulation truth. The present bank
concerns copula forecasts and combines estimates using performance-based weights. Jacobs et al.
(1991) provide broader mixture-of-experts context, although their learned gating network is
different from weighting fixed-memory estimators by recent losses.

The methodological contribution is therefore the specified use of the Chapter 4 estimators within
a copula bank: its component states, output construction, observable criteria and sequential
updates. Parallel estimation, forgetting and exponential weighting are established ingredients.
Whether this combination improves on selecting one memory setting in advance is an empirical
question.

> **Note — Record the actual bank composition.** State which of the count-matrix, Hermite and
> likelihood estimators have banks, and whether any bank mixes estimator types. All three base
> estimators are implemented, but that alone does not establish that all three are combined in one
> bank. Record how a fixed memory setting is implemented for each type, especially likelihood.

---

## 5.3 The adaptive estimator bank

Let the bank contain $K$ estimators indexed by $k$, with fixed memory settings
$\lambda_1, \dots, \lambda_K$ where that parameterisation applies. Estimator $k$ returns
$\hat C_t^{(k)}$ after processing observation $t$. Its combination weight $w_t^{(k)}$ is
nonnegative, and the weights sum to one. Both the state and weight evolve, even though the
component's memory setting is fixed. The estimate and weights available to predict observation $t$
are those indexed $t-1$.

### 5.3.1 The combined output

Two output constructions are relevant. Starting from the component Spearman average in the
original draft, a single Gaussian copula can be constructed as follows:

$$
\begin{aligned}
\bar\rho_{S,t} &= \sum_{k=1}^{K} w_t^{(k)} \hat\rho_{S,t}^{(k)},\\
\hat\rho_t^{\text{bank}} &= 2\sin\!\left(\frac{\pi\bar\rho_{S,t}}{6}\right), \qquad
\tilde C_t^{\text{coef}} = C_{\hat\rho_t^{\text{bank}}}.
\end{aligned}
\tag{5.4}
$$

**[not visible — text between (5.4) and the line below]** … scale using
$(6/\pi)\arcsin(\hat\rho_t^{(k)}/2)$. This construction returns one Gaussian copula and keeps its
scalar dependence summary easy to interpret.

An alternative combines the component copulas directly. When they have densities,

$$
\begin{aligned}
\tilde C_t^{\text{mix}}(u,v) &= \sum_{k=1}^{K} w_t^{(k)} \hat C_t^{(k)}(u,v),\\
\tilde c_t^{\text{mix}}(u,v) &= \sum_{k=1}^{K} w_t^{(k)} \hat c_t^{(k)}(u,v).
\end{aligned}
\tag{5.5}
$$

A convex mixture of copulas is a copula: its margins remain uniform and rectangle probabilities
remain nonnegative. A mixture of Gaussian copulas need not itself be Gaussian. Its Spearman
coefficient is nevertheless the same weighted average, because

$$
\begin{aligned}
\rho_S(C) &= 12\int_0^1\!\!\int_0^1 C(u,v)\,du\,dv - 3,\\
\rho_S(\tilde C_t^{\text{mix}}) &= \sum_{k=1}^{K} w_t^{(k)} \rho_S(\hat C_t^{(k)}).
\end{aligned}
\tag{5.6}
$$

These identities follow from the Spearman integral. Equality of a coefficient does not imply
equality of the two forecast distributions. Coefficient averaging is suitable when the intended
output is a single Gaussian copula. Mixing retains the component distributions and directly
supports evaluation of their combined density. The final choice should follow the intended output
and experimental evidence; neither is declared superior here.

> **Note — Output choice remains open.** Both formulas are included for the later implementation
> decision. Select and name the construction used in each reported experiment. A shared Spearman
> trajectory cannot distinguish these two distributions, so a comparison of them needs a
> distribution-level criterion as well.

### 5.3.2 Two loss criteria

The bank uses two kinds of loss: predictive copula log loss and squared discrepancy from a
reference Spearman estimate. These are criteria for judging an estimator. Discounted softmax and
fixed share, introduced below, are rules for turning such losses into weights. The choice of
criterion and the choice of weight rule should therefore be recorded separately.

**Predictive copula log loss.** Transform the incoming observation using the marginal states
available through $t-1$, and evaluate the preceding copula forecast:

$$
\begin{aligned}
\hat U_t^{(k)} &= \hat F_{X,t-1}^{(k)}(X_t), \qquad \hat V_t^{(k)} = \hat F_{Y,t-1}^{(k)}(Y_t),\\
s_{t,\log}^{(k)} &= -\log \hat c_{t-1}^{(k)}\big(\hat U_t^{(k)}, \hat V_t^{(k)}\big).
\end{aligned}
\tag{5.7}
$$

Lower loss means that the forecast assigns higher density to the transformed observation. Density
is not point probability: it may exceed one, so log loss can be negative. For the independence
copula it is zero. The forecast–observation ordering follows the prequential approach of Dawid
(2008, slides 34–39); fitting on the observation before scoring it would answer a different
question.

The logarithmic score has a standard justification for valid densities (Gneiting and Raftery,
2007, §4.1). With correct continuous marginal transforms, suppose the true copula density is
$c_*$. Assuming the displayed expectations are finite, its expected-loss advantage over a
candidate density $c$ is

$$
\begin{aligned}
&\mathbb{E}_{c_*}[-\log c(U,V)] - \mathbb{E}_{c_*}[-\log c_*(U,V)]\\
&\quad= \int_0^1\!\!\int_0^1 c_*(u,v)\log\frac{c_*(u,v)}{c(u,v)}\,du\,dv\\
&\quad= D_{\mathrm{KL}}(c_*\,\|\,c) \ge 0.
\end{aligned}
\tag{5.8}
$$

The divergence is zero only when the densities agree almost everywhere. This gives strict
propriety in that ideal setting. Within a restricted family, minimising expected loss instead
seeks its closest available density in this divergence, when a minimiser exists.

Estimated margins require a qualification. Different marginal maps give different evaluation
points, while even a shared estimated map need not produce the true uniforms. The criterion is
therefore a *plug-in copula loss*. Fissler and Hoga (2026, §3.2 of the preprint) explain why
misspecified margins invalidate a general claim that the true copula uniquely minimises such a
score. If valid marginal densities are also available, then before probability clipping or density
flooring, the joint density and its loss satisfy

$$
\begin{aligned}
\hat f_{t-1}^{(k)}(x,y) &= \hat c_{t-1}^{(k)}\big(\hat F_{X,t-1}^{(k)}(x), \hat F_{Y,t-1}^{(k)}(y)\big)
\times \hat f_{X,t-1}^{(k)}(x)\,\hat f_{Y,t-1}^{(k)}(y),\\
-\log \hat f_{t-1}^{(k)}(X_t,Y_t) &= s_{t,\log}^{(k)} - \log \hat f_{X,t-1}^{(k)}(X_t)
- \log \hat f_{Y,t-1}^{(k)}(Y_t).
\end{aligned}
\tag{5.9}
$$

This evaluates marginal and dependence forecasts together. Shared fitted margins make the marginal
terms common across components; separate fitted margins generally do not. The identity explains
the interpretation of copula-only scoring without requiring the bank to change its implemented
criterion.

Finite log loss requires positive, finite density at the evaluated point. Here that density should
come from the valid fitted copula, rather than a raw empirical CDF or an unchecked truncated
Hermite expansion. Probability clipping, parameter constraints and any density floor must be
stated. Flooring a density pointwise does not by itself preserve its integral or the ideal
proper-score argument.

**Squared Spearman discrepancy.** Let $\tilde\rho_{S,t}$ be a rolling or online reference estimate
constructed from data available by arrival $t$. Compare each preceding component estimate with
this reference:

$$
s_{t,S}^{(k)} = \big(\hat\rho_{S,t-1}^{(k)} - \tilde\rho_{S,t}\big)^2.
\tag{5.10}
$$

The reference is observable, but it is not the population Spearman correlation. Its window or
update rate introduces another memory scale; its sampling noise, lag and shared data with the
components affect the comparison. A single incoming pair does not reveal a population rank
correlation. This criterion therefore rewards agreement with the selected reference, rather than
establishing squared error relative to unknown truth.

When both quantities lie in $[-1,1]$, the loss lies in $[0,4]$. Dividing by four changes its scale
and hence the meaning of a fixed learning rate. In simulations, the known population coefficient
can separately measure tracking error. It is not substituted for the operational reference in this
bank.

> **Note — Complete the reference and scoring details.** Specify the reference estimator, window
> length or rate, treatment of ties and warm-up. The displayed timing compares forecasts through
> $t-1$ with a reference available through $t$; check it against the code. Also record shared
> versus separate marginal maps, endpoint handling and whether each criterion runs a separate bank
> or the losses are combined. No combined-loss formula is assumed here.

### 5.3.3 Discounted softmax and fixed share

Write $s_t^{(k)}$ for the selected loss. Both weighting rules give lower-loss estimators more
influence, but use different mechanisms to avoid permanent domination by earlier performance.

**Discounted softmax.** Retain an exponentially weighted loss summary and convert it to normalised
exponential weights:

$$
\begin{aligned}
L_t^{(k)} &= \beta L_{t-1}^{(k)} + (1-\beta)s_t^{(k)}, \qquad 0 \le \beta < 1,\\
w_t^{(k)} &= \frac{\exp(-\eta L_t^{(k)})}{\sum_{j=1}^{K}\exp(-\eta L_t^{(j)})}, \qquad \eta > 0.
\end{aligned}
\tag{5.11}
$$

The parameter $\beta$ retains performance history, whereas $\lambda_k$ controls component
estimation memory. Larger $\beta$ retains losses for longer. The parameter $\eta$ controls weight
concentration for the chosen loss scale. With a common start of scoring after warm-up arrival
$t_0$,

$$
L_t^{(k)} = (1-\beta)\sum_{i=t_0+1}^{t} \beta^{t-i} s_i^{(k)}, \qquad L_{t_0}^{(k)} = 0.
\tag{5.12}
$$

The initial summaries are uncorrected exponentially weighted averages. Dividing them by
$1-\beta^{t-t_0}$ would give a different start-up weight concentration at fixed $\eta$. The
displayed convention must therefore be kept explicit.

For constant $\beta$ and $\eta$, substitution gives the equivalent weight recursion

$$
w_t^{(k)} \propto \big(w_{t-1}^{(k)}\big)^{\beta}\exp\!\big[-\eta(1-\beta)s_t^{(k)}\big].
\tag{5.13}
$$

This is an algebraic consequence of the chosen loss update. Chernov and Zhdanov (2010) study
prediction with discounted expert losses; their unnormalised loss accumulation and prediction rule
should not be identified automatically with this bank. In particular, $\beta = 1$ freezes the
displayed loss state. Ordinary cumulative-loss weighting instead uses an additive loss recursion.
There is no universal optimal identity between $\beta$ and $\eta$.

Subtract the smallest loss summary before exponentiating to obtain the same weights without
unnecessarily large exponents:

$$
m_t = \min_j L_t^{(j)}, \qquad
w_t^{(k)} = \frac{\exp[-\eta(L_t^{(k)} - m_t)]}{\sum_j \exp[-\eta(L_t^{(j)} - m_t)]}.
\tag{5.14}
$$

**Fixed share.** First apply a multiplicative update to the previous weights, then redistribute a
fixed fraction uniformly:

$$
\begin{aligned}
v_t^{(k)} &= \frac{w_{t-1}^{(k)}\exp(-\eta s_t^{(k)})}{\sum_j w_{t-1}^{(j)}\exp(-\eta s_t^{(j)})},\\
w_t^{(k)} &= (1-\alpha)v_t^{(k)} + \frac{\alpha}{K}, \qquad 0 \le \alpha \le 1.
\end{aligned}
\tag{5.15}
$$

The share parameter $\alpha$ permits previously downweighted estimators to recover; in this
convention every weight is at least $\alpha/K$. Setting $\alpha = 0$ removes sharing, while
$\alpha = 1$ gives uniform weights. Herbster and Warmuth (1998) introduced fixed share for tracking
changing expert performance. The uniform-sharing version above appears explicitly in Cesa-Bianchi
et al. (2012, §5, equation (5)).

> **Fixed-share parameter conventions** — **[not visible: collapsed section in the screenshots;
> equation (5.16) is presumably inside it]**

For stable computation, normalise $\log w_{t-1}^{(k)} - \eta s_t^{(k)}$ after subtracting its
maximum, then apply sharing. Discounted softmax reduces the influence of old loss differences;
fixed share continually restores weight through redistribution. They are separate weighting
choices here, not an assumed composition. The same numerical $\eta$ is not necessarily comparable
across their accumulation rules or across the two losses.

If all component losses are observed after each arrival, the bank has full-information feedback in
the expert-advice sense of Freund and Schapire (1997, §2). Their bounded-loss Hedge result does not
automatically cover continuous copula log loss. Fixed-share and discounted-aggregation guarantees
likewise require their precise loss, prediction function and parameter conditions. This chapter
does not claim a regret guarantee for the present bank solely from the form of its weights.

> **Note — Match the weight updates.** The equations specify a reproducible candidate version of
> each confirmed rule. Check the code's loss normalisation, softmax temperature, sharing convention
> and initial weights. List the loss–weighting combinations actually tested; their availability
> does not establish that all four combinations were run.

### 5.3.4 Sequential algorithm

Initialise the component and marginal states using the stated warm-up policy. In this sample
specification, scoring begins after a common warm-up arrival $t_0$, with uniform weights
$w_{t_0}^{(k)} = 1/K$; discounted loss summaries start at zero. At each later arrival:

1. **Retain the forecast.** Keep the component copulas, marginal maps and bank weights formed
   through $t-1$. Construct the bank's forecast from these quantities.
2. **Observe and score.** Receive $(X_t, Y_t)$. Evaluate each saved forecast using the selected
   criterion. For squared Spearman discrepancy, update the specified reference to its allowed time
   point before calculating the loss.
3. **Update the weights.** Apply discounted softmax or fixed share to the losses. The resulting
   $w_t^{(k)}$ are used for the next forecast.
4. **Update every estimator.** Incorporate the observation using each component's fixed memory
   setting and the marginal/dependence ordering from Chapter 4. Even estimators with small
   combination weights continue to learn.
5. **Form the next output.** Combine the updated estimates using $w_t^{(k)}$, with the selected
   coefficient or distribution construction, to obtain $\hat C_t$.

The bank's own forecast must also be evaluated before updating its weights. For a specified common
marginal map, its log loss is

$$
s_{t,\log}^{\text{bank}} = -\log \hat c_{t-1}^{\text{bank}}(\hat U_t, \hat V_t).
\tag{5.17}
$$

This is not generally the weighted average of component log losses. For a density mixture
evaluated at a common point, convexity of the negative logarithm gives

$$
-\log\Big[\sum_k w_{t-1}^{(k)}\hat c_{t-1}^{(k)}(u,v)\Big]
\le \sum_k w_{t-1}^{(k)}\big[-\log \hat c_{t-1}^{(k)}(u,v)\big].
\tag{5.18}
$$

This pointwise inequality does not establish that the mixture beats the best component over a
stream. It also does not apply unchanged when component losses use different transformed points,
or when the output is a single copula fitted after coefficient averaging. If components have
separate marginal maps, specify the map used to evaluate the bank; mixing full joint forecasts
would be a further, distinct construction.

### 5.3.5 Initialisation and tuning

Warm-up must provide usable component forecasts, valid marginal transforms and a reference
correlation where required. Components should start scoring on a common basis, with an explicit
policy for degenerate or nonfinite outputs. Resetting a component, excluding it or using a
fallback forecast changes the procedure and must be recorded. A poor score is not by itself
evidence of a numerical failure.

The memory grid should cover the response scales of interest; $\lambda_k$ can be reported together
with its geometric half-life where applicable. The performance parameters $\beta$ and $\eta$, or
$\alpha$ and $\eta$ for fixed share, control a different part of the system. The reference window
or rate used in squared Spearman loss adds another setting. Choose or compare these settings using
a stated tuning protocol rather than selecting them from the final test results. All values,
initialisation rules, copula constraints and loss normalisations belong in the experimental
specification.

### 5.3.6 Computational and storage cost

Let $M_k$ denote component storage, $U_k$ its update cost and $Q_k$ the cost of querying and
scoring its forecast. The bank has

$$
\begin{aligned}
\text{storage} &= \sum_{k=1}^{K} M_k + O(K),\\
\text{time per arrival} &= \sum_{k=1}^{K} (U_k + Q_k) + O(K).
\end{aligned}
\tag{5.19}
$$

The additional $O(K)$ work covers weight calculation and combination. A rolling reference and
shared marginal estimators add their own storage and processing costs if these are not already
included. For dense $m_X \times m_Y$ count matrices, the state and correlation-query work scale
with $K m_X m_Y$. Common-order Hermite banks scale with $K(N+1)^2$. Likelihood-bank costs include
their marginal estimators. Fixed dimensions and a fixed number of estimators can give bounded state
size in stream length, but actual runtime and memory use must be measured.

---

## 5.4 What the experiments should establish

The bank adapts the influence of estimators that have retained different histories. It does not
recover discarded observations, establish a change point from weight movement, or guarantee that
its preferred estimator is closest to the true copula. Its usefulness should be assessed against
each constituent fixed-rate estimator and, where available, uniform averaging using the same output
construction.

Stationary and changing-dependence runs can examine whether the bank remains stable when long
memory is useful and shifts effectively when faster response is needed. Report the loss criterion
and weighting rule for every configuration. Plot the true simulation dependence beside the
component and bank estimates, and show the combination weights. For squared-loss banks, also show
the reference correlation so that agreement with it can be distinguished from accuracy relative to
truth.

Predictive log loss, correlation tracking error and measured resource cost answer different
questions. Keep these results separate, report replication variability where available, and
identify any choices made using simulation truth as oracle comparisons. The chapter supplies the
method specification; the next chapter must determine its benefits and limitations from the
experimental evidence.

---

## References

*Author–year links lead to these entries. Page and section locators identify the versions
consulted. No theorem is claimed for the unverified combination of implementation choices.*

1. Cesa-Bianchi, N., Gaillard, P., Lugosi, G. and Stoltz, G. (2012). Mirror descent meets fixed
   share (and feels no regret). *Advances in Neural Information Processing Systems*, 25, 989–997.
   Paper.
   — §5, equation (5), for sharing uniformly, including the originating estimator.
2. Chernov, A. and Zhdanov, F. (2010). Prediction with expert advice under discounted loss.
   *Algorithmic Learning Theory*, LNCS 6331, 255–269. Extended manuscript.
   — Protocol 1 and §2.1, Algorithm 2. The bank recurrence is a separately specified construction.
3. Dawid, A. P. (2008). *Prequential Analysis.* NIPS tutorial. Slides.
   — Slides 34–35 and 39 for predictive loss and estimation before the scored observation.
4. Fissler, T. and Hoga, Y. (2026). How to compare copula forecasts? *Journal of Business &
   Economic Statistics.* DOI · 2024 preprint.
   — Preprint §3.2, Proposition 5 and Examples 6–7; §3.3 distinguishes joint assessment from
   isolated copula assessment.
5. Freund, Y. and Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning
   and an application to boosting. *Journal of Computer and System Sciences*, 55(1), 119–139.
   DOI · Manuscript.
   — §2, Figure 1, for exponential weighting and full loss-vector feedback.
6. Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and
   estimation. *Journal of the American Statistical Association*, 102(477), 359–378. DOI · Paper.
   — §4.1, p. 365, for the logarithmic score and its divergence. This chapter uses negative scores
   as losses.
7. Hammer, H. L., Yazidi, A., Riegler, M. A. and Rue, H. (2020). *Efficient Quantile Tracking
   Using an Oracle.* arXiv:2004.12588. Manuscript.
   — §3 and Figure 1 for the bank of trackers and selection using estimated MSE.
8. Herbster, M. and Warmuth, M. K. (1998). Tracking the best expert. *Machine Learning*, 32(2),
   151–178. DOI · Paper.
   — §3, Figure 1 and equation (6), pp. 156–157, use sharing to the other estimators.
9. Jacobs, R. A., Jordan, M. I., Nowlan, S. J. and Hinton, G. E. (1991). Adaptive mixtures of local
   experts. *Neural Computation*, 3(1), 79–87. DOI · Paper.
   — §1 gives architectural context; the gating mechanism differs from this bank's weighting.
10. Martino, L., Elvira, V. and Louzada, F. (2017). Effective sample size for importance sampling
    based on discrepancy measures. *Signal Processing*, 131, 386–401. DOI · Manuscript.
    — Background on weight-based ESS; the geometric memory formulas are derived directly in this
    chapter.
