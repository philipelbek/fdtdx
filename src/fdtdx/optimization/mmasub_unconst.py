"""Literal Python translation of mmasub_unconst.m.

Adapted from mmasub.m (Krister Svanberg) for the UNCONSTRAINED (box-constrained only)
case, i.e. m = 0 general constraints. Because there are no general constraints f_i(x),
the MMA subproblem is separable in the x_j and its minimizer has an explicit closed
form (Svanberg 1987) -- there is no need for the primal-dual Newton solver used in
:func:`fdtdx.optimization.subsolv.subsolv` for the general case.

Ported line-for-line from the original MATLAB so that fdtdx does not depend on any
third-party MMA implementation. See :func:`fdtdx.optimization.mma.mma_unconstrained`
for the optax-compatible wrapper used in fdtdx training loops.
"""

import numpy as np


def mmasub_unconst(
    n: int,
    iter: int,
    xval: np.ndarray,
    xmin: np.ndarray,
    xmax: np.ndarray,
    xold1: np.ndarray,
    xold2: np.ndarray,
    f0val: float,
    df0dx: np.ndarray,
    low: np.ndarray,
    upp: np.ndarray,
    move: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performs one unconstrained MMA iteration, aimed at solving the box-constrained problem.

    Minimize  f_0(x)
    subject to  xmin_j <= x_j <= xmax_j,    j = 1,...,n

    Args:
        n (int): The number of variables x_j.
        iter (int): Current iteration number ( =1 the first time mmasub_unconst is
            called).
        xval (np.ndarray): Column vector with the current values of the variables x_j.
        xmin (np.ndarray): Column vector with the TRUE physical lower bounds for x_j
            (e.g., 0 for densities). Used for asymptote calculations.
        xmax (np.ndarray): Column vector with the TRUE physical upper bounds for x_j
            (e.g., 1 for densities). Used for asymptote calculations.
        xold1 (np.ndarray): xval, one iteration ago (provided that iter>1).
        xold2 (np.ndarray): xval, two iterations ago (provided that iter>2).
        f0val (float): The value of the objective function f_0 at xval. Unused -- f0val
            is a pure additive constant in the subproblem objective and provably never
            affects the resulting xmma, exactly as in the original MATLAB (which accepts
            but never references it either).
        df0dx (np.ndarray): Column vector with the derivatives of the objective function
            f_0 with respect to the variables x_j, calculated at xval.
        low (np.ndarray): Column vector with the lower asymptotes from the previous
            iteration (provided that iter>1).
        upp (np.ndarray): Column vector with the upper asymptotes from the previous
            iteration (provided that iter>1).
        move (float | np.ndarray): Scalar (or n x 1) move limit, as a FRACTION of
            (xmax-xmin), restricting how far x_j may move from xval in this MMA step.
            E.g., move = 0.15 (as opposed to move = 1.0 which imposes no additional
            restriction, as in the original mmasub.m).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: xmma, low, upp -- the column vector
            with the optimal values of the variables x_j in the current MMA subproblem,
            and the lower/upper asymptotes calculated and used in it.
    """
    albefa = 0.1
    asyinit = 0.5
    asyincr = 1.08
    asydecr = 0.5
    eeen = np.ones((n, 1))
    zeron = np.zeros((n, 1))
    feps = 0.000001

    # Calculation of the asymptotes low and upp (based on the TRUE physical bounds
    # xmin, xmax -- independent of the move limit) :
    if iter < 2.5:
        low = xval - asyinit * (xmax - xmin)
        upp = xval + asyinit * (xmax - xmin)
    else:
        zzz = (xval - xold1) * (xold1 - xold2)
        factor = eeen.copy()
        factor[zzz > 0] = asyincr
        factor[zzz < 0] = asydecr
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)
        lowmin = xval - 10 * (xmax - xmin)
        lowmax = xval - 0.01 * (xmax - xmin)
        uppmin = xval + 0.01 * (xmax - xmin)
        uppmax = xval + 10 * (xmax - xmin)
        low = np.maximum(low, lowmin)
        low = np.minimum(low, lowmax)
        upp = np.minimum(upp, uppmax)
        upp = np.maximum(upp, uppmin)

    # Calculation of the bounds alfa and beta :
    # (move limit applied here only, exactly as in the original mmasub.m)
    zzz1 = low + albefa * (xval - low)
    zzz2 = xval - move * (xmax - xmin)
    zzz = np.maximum(zzz1, zzz2)
    alfa = np.maximum(zzz, xmin)
    zzz1 = upp - albefa * (upp - xval)
    zzz2 = xval + move * (xmax - xmin)
    zzz = np.minimum(zzz1, zzz2)
    beta = np.minimum(zzz, xmax)

    # Calculations of p0 and q0.
    ux1 = upp - xval
    ux2 = ux1 * ux1
    xl1 = xval - low
    xl2 = xl1 * xl1

    p0 = zeron.copy()
    q0 = zeron.copy()
    p0 = np.maximum(df0dx, 0)
    q0 = np.maximum(-df0dx, 0)
    pq0 = 0.001 * (p0 + q0) + feps / (upp - low)
    p0 = p0 + pq0
    q0 = q0 + pq0
    p0 = p0 * ux2
    q0 = q0 * xl2

    # Solving the (separable) subproblem explicitly, since there are no general
    # constraints to dualize.
    xmma = (np.sqrt(p0) * low + np.sqrt(q0) * upp) / (np.sqrt(p0) + np.sqrt(q0))
    xmma = np.maximum(alfa, np.minimum(beta, xmma))

    return xmma, low, upp
