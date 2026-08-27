#!/usr/bin/env python3
"""Generate shared (x,y) input streams for the reference_parity check.

Each dataset is written as data/<name>.csv with a header `x,y` and one
observation per line at full float precision, so the C++ harness, the Python
onlineNPCORR reference, and the R hermiter reference all consume byte-identical
inputs.
"""
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT, exist_ok=True)


def bivariate_normal(n, rho, seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    x = z[:, 0]
    y = rho * z[:, 0] + np.sqrt(1.0 - rho * rho) * z[:, 1]
    return x, y


def write(name, x, y):
    path = os.path.join(OUT, name + ".csv")
    with open(path, "w") as f:
        f.write("x,y\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi!r},{yi!r}\n")
    print(f"wrote {path}  (n={len(x)})")


def main():
    # Gaussian copula, Gaussian marginals, a range of true rho.
    write("gauss_pos", *bivariate_normal(2000, 0.7, seed=1))
    write("gauss_neg", *bivariate_normal(2000, -0.4, seed=2))
    write("gauss_zero", *bivariate_normal(2000, 0.0, seed=3))

    # Non-Gaussian marginals (Gaussian copula pushed through monotone maps):
    # checks that the rank-based Xiao is marginal-invariant and that Hermite's
    # online standardization copes with skew. Same copula rho=0.6.
    x, y = bivariate_normal(2500, 0.6, seed=4)
    write("skew_marg", np.exp(x), y ** 3)


if __name__ == "__main__":
    main()
