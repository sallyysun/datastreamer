# The Hermite Spearman coupling matrix: closed form vs. quadrature

*Why `HermiteEstimator::integral_matrix` builds the coupling matrix $W$ from an
analytic recurrence at construction time instead of numerically integrating it
(the original approach) or shipping a precomputed table (what the reference R
package `external/hermiter` does).*

Code: `src/estimators/dependence/hermiter.hpp` (`integral_matrix`,
`basis_integrals`, `estimate`). Reference: Stephanou & Varughese (2021),
*Sequential estimation of Spearman rank correlation using Hermite series
estimators*, JMVA.

---

## 1. Background: the Hermite-series Spearman estimator

Let $\{h_k\}_{k\ge 0}$ be the **orthonormal Hermite functions**

$$
h_k(x) = \left(2^k\,k!\,\sqrt{\pi}\right)^{-1/2} H_k(x)\, e^{-x^2/2},
\qquad \int_{-\infty}^{\infty} h_k(x)\,h_l(x)\,dx = \delta_{kl},
$$

with $H_k$ the physicists' Hermite polynomials. They satisfy the three-term
recurrence used in `hermite_functions`:

$$
h_{k}(x) = x\sqrt{\tfrac{2}{k}}\,h_{k-1}(x) - \sqrt{\tfrac{k-1}{k}}\,h_{k-2}(x),
\qquad h_0(x)=\pi^{-1/4}e^{-x^2/2}.
$$

A bivariate density is approximated by its truncated Hermite expansion, whose
coefficients are exactly the quantities the estimator accumulates online (after
standardizing each margin). With $N+1$ terms,

$$
\hat f_{X}(x)=\sum_{k=0}^{N} a^X_k\,h_k(x),\quad
\hat f_{XY}(x,y)=\sum_{p,q=0}^{N} A_{pq}\,h_p(x)\,h_q(y),
$$

$$
a^X_k = \tfrac1n\sum_{i=1}^n h_k(\tilde x_i),\qquad
A_{pq}= \tfrac1n\sum_{i=1}^n h_p(\tilde x_i)\,h_q(\tilde y_i),
$$

where $\tilde x_i,\tilde y_i$ are the standardized observations. In the code
these are `ax_`, `ay_` (the coefficient vectors) and `A_` (the coefficient
matrix), updated by the running means in `update`.

**Spearman's rho** is the grade correlation

$$
\rho_S = 12\,\mathbb{E}\!\left[\left(F_X(X)-\tfrac12\right)\left(F_Y(Y)-\tfrac12\right)\right],
$$

with $F_X$ the marginal CDF. Writing the CDF as the antiderivative of the density
expansion introduces the **lower-integral functions**

$$
I_j(x) := \int_{-\infty}^{x} h_j(t)\,dt,
\qquad\text{so}\qquad
F_X(x)=\sum_k a^X_k\, I_k(x),
$$

and the whole estimator collapses onto two fixed objects: the **full-domain
integrals** $z_k$ and the **coupling matrix** $W$,

$$
\boxed{\;z_k := \int_{-\infty}^{\infty} h_k(x)\,dx,
\qquad
W_{kj} := \int_{-\infty}^{\infty} h_k(x)\, I_j(x)\,dx. \;}
$$

### Where $W$ enters

Expanding $\rho_S$ and using $\int h_p I_k = W_{pk}$ and $\int h_q\,dy = z_q$:

$$
\mathbb{E}[F_X(X)F_Y(Y)]
=\sum_{k,l,p,q} a^X_k a^Y_l A_{pq}\,W_{pk} W_{ql}
=(W a^X)^{\!\top} A\,(W a^Y)=u^\top A\,v,
$$

with $u:=W a^X$, $v:=W a^Y$. The lower-order terms are
$\mathbb{E}[F_X(X)]=u^\top A z$, $\mathbb{E}[F_Y(Y)]=z^\top A v$, and
$\mathbb{E}[1]=z^\top A z$, so

$$
\rho_S
= 12\Big(u^\top A v - \tfrac12 u^\top A z - \tfrac12 z^\top A v + \tfrac14 z^\top A z\Big)
= 12\,u^\top A v - 6\,u^\top A z - 6\,z^\top A v + 3\,z^\top A z .
$$

This is exactly `estimate()` (`hermiter.hpp:65-70`). $W$ and $z$ are **constants
of the estimator** — they do not depend on the data — so they are built once in
the constructor. Everything hinges on computing $W$ accurately.

---

## 2. The two ways to get $W$

$W_{kj}=\int h_k(x) I_j(x)\,dx$ is an integral of a product of a Hermite function
and the antiderivative of another. There are two routes.

**(a) Numerical quadrature — the original port.** Evaluate $I_j$ on a grid by
cumulative trapezoid, then $W_{kj}\approx \sum_i h_k(x_i)I_j(x_i)\,\Delta x$ over
a truncated domain $[-L,L]$. This carries two approximation errors: the
$O(\Delta x^2)$ trapezoid error and the tail truncation at $\pm L$. It was *the*
remaining numerical discrepancy from the reference (see
`notes/code_changes/code changes 2026-08-19.md`).

**(b) Closed form — the current code.** $W$ satisfies exact algebraic identities,
so it can be built to machine precision with no grid at all. Derived below.

---

## 3. Closed form for $W$

### The full-domain integrals $z_k$

$z_0=\int \pi^{-1/4}e^{-x^2/2}dx=\pi^{-1/4}\sqrt{2\pi}=\sqrt{2}\,\pi^{1/4}$.
Integrating the $h_k$ recurrence and using that odd Hermite functions are odd
(so integrate to $0$):

$$
z_k=\sqrt{\tfrac{k-1}{k}}\;z_{k-2}\ \ (k\ \text{even}),\qquad z_k=0\ \ (k\ \text{odd}).
$$

This is `basis_integrals` (`hermiter.hpp:146-154`).

### Identity 1 — integration-by-parts symmetry

Since $I_k' = h_k$, the product rule gives
$\frac{d}{dx}\big(I_k I_j\big)=h_k I_j + I_k h_j$. Integrating over $\mathbb R$
and using $I_k(-\infty)=0,\ I_k(+\infty)=z_k$:

$$
\int_{-\infty}^{\infty}\!\big(h_k I_j + h_j I_k\big)\,dx
=\big[I_kI_j\big]_{-\infty}^{\infty}=z_k z_j
\quad\Longrightarrow\quad
\boxed{\,W_{kj}+W_{jk}=z_k z_j\,.}
$$

### Identity 2 — column recurrence

The lower integral obeys (raising/lowering relations for Hermite functions)

$$
I_j(x) = -\sqrt{\tfrac{2}{j}}\,h_{j-1}(x) + \sqrt{\tfrac{j-1}{j}}\,I_{j-2}(x),
\qquad j\ge 2.
$$

Multiply by $h_k$, integrate, and use orthonormality $\int h_k h_{j-1}=\delta_{k,j-1}$:

$$
\boxed{\;W_{kj} = -\sqrt{\tfrac{2}{j}}\,\delta_{k,\,j-1}
+ \sqrt{\tfrac{j-1}{j}}\;W_{k,\,j-2}\;,\qquad j\ge 2.\;}
$$

Each column $j$ is fixed by column $j-2$ plus a single Kronecker spike — no
integration.

### Base cases

* **Column $j=1$:** $h_1(t)=\sqrt2\,t\,h_0(t)$ integrates in closed form to
  $I_1(x)=-\sqrt2\,h_0(x)$, hence $W_{k1}=-\sqrt2\,\delta_{k0}$.
* **Row $k=0$:** setting $k=0$ in Identity 2 kills the spike for $j\ge 2$
  ($\delta_{0,j-1}=0$), giving $W_{0j}=\sqrt{\tfrac{j-1}{j}}\,W_{0,j-2}$, seeded by
  $W_{00}=\tfrac12 z_0^2$ (Identity 1 at $k=j=0$) and $W_{01}=-\sqrt2$.
* **Column $j=0$:** recovered from Identity 1, $W_{k0}=z_k z_0 - W_{0k}$.

### The algorithm

`integral_matrix` (`hermiter.hpp:162-190`) assembles $W$ in $O(N^2)$:

1. row $0$ via $W_{0j}=\sqrt{(j-1)/j}\,W_{0,j-2}$, from $W_{00}=z_0^2/2$, $W_{01}=-\sqrt2$;
2. column $0$ via $W_{k0}=z_k z_0 - W_{0k}$; column $1$ via $W_{k1}=-\sqrt2\,\delta_{k0}$;
3. columns $j\ge2$ via the recurrence $W_{kj}=\sqrt{(j-1)/j}\,W_{k,j-2}-\sqrt{2/j}\,\delta_{k,j-1}$.

In exact arithmetic this is $W$ *exactly*; in `double` the only error is rounding
of the $O(N^2)$ multiply–adds, i.e. a few ulps.

---

## 4. Why the closed form is not just *equal to* the reference — it is *more accurate*

The reference package ships $W$ as a precomputed constant table
(`W_serialized` in `hermiter`'s `sysdata.rda`), itself produced by a numerical
route and rounded to `double`. Comparing the C++ closed-form $W$ against that
table (order $N=10$) shows a worst-entry gap of $3.6\times10^{-10}$ — far above
machine precision, and it is the table that is off.

$W_{00}$ makes this unambiguous: its exact value is
$\tfrac12 z_0^2 = \tfrac12\big(\sqrt2\,\pi^{1/4}\big)^2 = \sqrt{\pi}
= 1.7724538509055160\ldots$

| source | $W_{00}$ | error vs $\sqrt\pi$ |
|---|---|---|
| closed form (this code) | `1.7724538509055159` | $1.3\times10^{-16}$ |
| `hermiter` table         | `1.7724538505406728` | $3.6\times10^{-10}$ |

An **independent** high-precision check (mpmath, 40 digits, direct double
integration of $\int h_k I_j$ — *not* the recurrence above) confirms the pattern
matrix-wide, including entries that are neither special constants nor structural
zeros:

| entry | true value | closed-form err | `hermiter` table err |
|---|---|---|---|
| $W_{0,0}$   | $\sqrt\pi=1.77245385091$ | $1.3\times10^{-16}$ | $3.6\times10^{-10}$ |
| $W_{0,2}$   | $\sqrt{\pi/2}=1.25331413732$ | $4.9\times10^{-17}$ | $2.6\times10^{-10}$ |
| $W_{10,10}$ | $0.43618981487$ | $2.4\times10^{-16}$ | $1.8\times10^{-12}$ |

So the $\sim10^{-10}$ discrepancy the comparison (`experiments/reference_comparison`)
reports between our Hermite $\rho_S$ and `hermiter`'s is **not** our port drifting
from the reference — it is the reference's stored $W$ drifting from the true $W$,
and our analytic construction is the correct one. (Confirmed by injecting the
`hermiter` table into our estimator: the $\rho_S$ gap then collapses from
$4\times10^{-11}$ to $7\times10^{-15}$ — pure machine noise.)

---

## 5. Why compute it at runtime rather than hardcode a table

Once the form is analytic, building it in the constructor is the natural fit:

* **Self-contained, header-only C++** with no asset/serialization path — the only
  alternative would be pasting an opaque magic-number table into the source.
* **Any truncation order.** The recurrence works for whatever `config.order` is
  set; a table is fixed at the order it was generated.
* **Free.** $O(N^2)$, once, in the constructor ($N=10 \Rightarrow$ an $11\times11$
  build), then reused every `estimate()`. Nothing to amortize.
* **Auditable.** The recurrence *is* its own derivation; a table cannot be checked
  without recomputing it — which is exactly how its $10^{-10}$ error slipped in.

### Practical caveat

"More accurate" here means *closer to the exact real-valued $W$*. The statistic
itself is a rank correlation in $[-1,1]$; a $10^{-10}$ difference is orders of
magnitude below any estimation or statistical relevance. The point is not that
the reference is "wrong" in use — it is that we **cannot** validate our Hermite
path to machine epsilon *against* `hermiter`, because the reference package is the
limiting factor at $10^{-10}$. The honest claim for the thesis is: *agrees with
the reference to $10^{-10}$, with the residual attributable to the reference's own
table rounding, verified against a high-precision ground truth.*
