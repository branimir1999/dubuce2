from typing import Callable, Union, Tuple

from functools import reduce

import numpy as np
import sympy as sp

NumT = Union[np.ndarray, float]
Expr = sp.Expr
Sym = sp.Symbol

def gen_f_zt(z: Sym, t: Sym) -> Expr:
    a = (-1) ** np.random.randint(2) * np.random.randint(1, 10 + 1)
    c = (-1) ** np.random.randint(2) * np.random.randint(1, 10 + 1)
    d = -np.abs(c) - np.random.randint(1, 10 + 1)
    b = np.maximum(np.abs(a), (-d + np.abs(c - a))) + np.random.randint(1, 10 + 1)

    f = t * (a * z + b) + (c * z + d)
    f = f ** 2 * (1 - z ** 2) * t
    f = f / sp.integrate(sp.integrate(f, (z, (-1, 1)))).subs({t: 1})

    return f

def gen_qz(f: Expr, z: Sym, t: Sym) -> Expr:
    p = sp.Rational(3, 2) * (1 - z ** 2)
    q = sp.integrate(f, (t, 0, 1)) * p
    q = q.expand()

    S = sp.integrate(q, (z, -1, 1))

    q /= S

    return q

def gen_inv_cdf(cdf: Callable[[NumT], NumT], low: NumT = -1, high: NumT = 1, steps: int = 18) -> Callable[[NumT], NumT]:
    def f_inv(eps):
        a, b = np.ones_like(eps) * low, np.ones_like(eps) * high
        sample = None

        for _ in range(steps):
            sample = (a + b) / 2
            cdf_sample = cdf(sample)
            cond = cdf_sample < eps
            a[cond] = sample[cond]
            b[~cond] = sample[~cond]

        return sample

    return f_inv

def gen_p(x: Sym, z: Sym, m: int = 2, n: int = 2) -> Tuple[Expr, Expr]:
    C1 = np.random.randint(-5, 5 + 1, size=[m + 1, n + 1])
    C2 = np.random.randint(-5, 5 + 1, size=[m + 1, n + 1])

    cheby = sp.polys.chebyshevt_poly
    p_1 = sp.Poly(sum(C1[i, j] * cheby(i, x) * cheby(j, z) for i in range(m + 1) for j in range(n + 1))).as_expr()
    p_2 = sp.Poly(sum(C2[i, j] * cheby(i, x) * cheby(j, z) for i in range(m + 1) for j in range(n + 1))).as_expr()

    p = (p_1 ** 2 + p_2 ** 2) * (1 - x ** 2) * (1 - z ** 2)
    p = p / sp.integrate(sp.integrate(p, (z, (-1, 1))), (x, -1, 1))

    p_mat = sp.Poly(p, x, z)

    C = [[p_mat.coeff_monomial(x ** i * z ** j) for j in range(2 * n + 3)] for i in range(2 * m + 3)]

    lcm = reduce(np.lcm, [sp.Rational(C[i][j]).q for j in range(2 * n + 3) for i in range(2 * m + 3)], 1)

    C = [[C[i][j] * lcm for j in range(2 * n + 3)] for i in range(2 * m + 3)]

    C = sp.Matrix(C)
    X = sp.Matrix([[x ** i for i in range(2 * m + 3)]])
    Z = sp.Matrix([[z ** j] for j in range(2 * m + 3)])

    return sp.UnevaluatedExpr(sp.MatMul(X, C, Z)) / lcm, p
