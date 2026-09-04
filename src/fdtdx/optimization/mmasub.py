"""Literal Python translation of mmasub.m (Krister Svanberg, September 2007).

Performs one iteration of the Method of Moving Asymptotes (MMA), Svanberg 1987/2002.
Ported line-for-line from the original MATLAB so that fdtdx does not depend on any
third-party MMA implementation. See :func:`fdtdx.optimization.mma.mma` for the
optax-compatible wrapper used in fdtdx training loops.
"""

import numpy as np

from fdtdx.optimization.subsolv import subsolv


def mmasub(
    m: int,
    n: int,
    iter: int,
    xval: np.ndarray,
    xmin: np.ndarray,
    xmax: np.ndarray,
    xold1: np.ndarray,
    xold2: np.ndarray,
    f0val: float,
    df0dx: np.ndarray,
    fval: np.ndarray,
    dfdx: np.ndarray,
    low: np.ndarray,
    upp: np.ndarray,
    a0: float,
    a: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Performs one MMA iteration, aimed at solving the nonlinear programming problem.

    Minimize  f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )
    subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m
                xmin_j <= x_j <= xmax_j,    j = 1,...,n
                z >= 0,   y_i >= 0,         i = 1,...,m

    Args:
        m (int): The number of general constraints.
        n (int): The number of variables x_j.
        iter (int): Current iteration number ( =1 the first time mmasub is called).
        xval (np.ndarray): Column vector with the current values of the variables x_j.
        xmin (np.ndarray): Column vector with the lower bounds for the variables x_j.
        xmax (np.ndarray): Column vector with the upper bounds for the variables x_j.
        xold1 (np.ndarray): xval, one iteration ago (provided that iter>1).
        xold2 (np.ndarray): xval, two iterations ago (provided that iter>2).
        f0val (float): The value of the objective function f_0 at xval.
        df0dx (np.ndarray): Column vector with the derivatives of the objective function
            f_0 with respect to the variables x_j, calculated at xval.
        fval (np.ndarray): Column vector with the values of the constraint functions f_i,
            calculated at xval.
        dfdx (np.ndarray): (m x n)-matrix with the derivatives of the constraint functions
            f_i with respect to the variables x_j, calculated at xval. dfdx[i, j] = the
            derivative of f_i with respect to x_j.
        low (np.ndarray): Column vector with the lower asymptotes from the previous
            iteration (provided that iter>1).
        upp (np.ndarray): Column vector with the upper asymptotes from the previous
            iteration (provided that iter>1).
        a0 (float): The constant a_0 in the term a_0*z.
        a (np.ndarray): Column vector with the constants a_i in the terms a_i*z.
        c (np.ndarray): Column vector with the constants c_i in the terms c_i*y_i.
        d (np.ndarray): Column vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: xmma, ymma, zmma,
            lam, xsi, eta, mu, zet, s, low, upp -- the optimal values of the variables
            x_j/y_i/z in the current MMA subproblem, the Lagrange multipliers/slacks of
            that subproblem, and the lower/upper asymptotes calculated and used in it.
    """
    epsimin = np.sqrt(m + n) * 1e-9
    raa0 = 0.00001
    move = 1.0
    albefa = 0.1
    asyinit = 0.5
    asyincr = 1.08
    asydecr = 0.5
    eeen = np.ones((n, 1))
    eeem = np.ones((m, 1))
    zeron = np.zeros((n, 1))

    # Calculation of the asymptotes low and upp :
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
    zzz1 = low + albefa * (xval - low)
    zzz2 = xval - move * (xmax - xmin)
    zzz = np.maximum(zzz1, zzz2)
    alfa = np.maximum(zzz, xmin)
    zzz1 = upp - albefa * (upp - xval)
    zzz2 = xval + move * (xmax - xmin)
    zzz = np.minimum(zzz1, zzz2)
    beta = np.minimum(zzz, xmax)

    # Calculations of p0, q0, P, Q and b.
    xmami = xmax - xmin
    xmamieps = 0.00001 * eeen
    xmami = np.maximum(xmami, xmamieps)
    xmamiinv = eeen / xmami
    ux1 = upp - xval
    ux2 = ux1 * ux1
    xl1 = xval - low
    xl2 = xl1 * xl1
    uxinv = eeen / ux1
    xlinv = eeen / xl1

    p0 = zeron.copy()
    q0 = zeron.copy()
    p0 = np.maximum(df0dx, 0)
    q0 = np.maximum(-df0dx, 0)
    pq0 = 0.001 * (p0 + q0) + raa0 * xmamiinv
    p0 = p0 + pq0
    q0 = q0 + pq0
    p0 = p0 * ux2
    q0 = q0 * xl2

    P = np.maximum(dfdx, 0)
    Q = np.maximum(-dfdx, 0)
    PQ = 0.001 * (P + Q) + raa0 * (eeem @ xmamiinv.T)
    P = P + PQ
    Q = Q + PQ
    # P = P * spdiags(ux2,0,n,n); a diagonal matrix on the right scales columns, which is
    # the same as broadcasting the diagonal as a row vector.
    P = P * ux2.T
    Q = Q * xl2.T
    b = P @ uxinv + Q @ xlinv - fval

    # Solving the subproblem by a primal-dual Newton method
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(
        m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d
    )

    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp
