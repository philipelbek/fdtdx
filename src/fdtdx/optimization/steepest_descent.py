"""Literal Python translation of SteepestDescent.m.

Simplified version of the modified steepest descent method of Da Silva et al., as used
to solve the (box-constrained) sub-problems

  Minimize  L(x)
subject to  xmin_j <= x_j <= xmax_j,    j = 1,...,n

where L is the augmented Lagrangian (here simply the objective f_0 passed in through
f0val/df0dx). The procedure is, at iteration b:

  1. Steepest descent direction        S = -dL/dx                (--)
  2. Reset the gradient contributions at the bound constraints   (D22)
  3. Normalise by the maximum absolute value, D = S/max|S|       (--)
  4. Move limits from the two previous iterations, via
     d_e = (x^(b)-x^(b-1))*(x^(b-1)-x^(b-2)):
         delta_e = 0.7*delta_e  if d_e < 0
         delta_e = 1.1*delta_e  if d_e > 0
     clipped to  0.001 <= delta_e <= 0.1, giving
         x^inf = x^(b) - delta,   x^sup = x^(b) + delta          (D23)
  5. Update with unitary step length Psi = 1, clipped to the
     move limits intersected with the physical bounds            (D24)

The optimization is started with the maximum range of move limits, delta_e = 0.1, which
are then updated from the third iteration on. The step length Psi is unitary (no
backtracking line search), which is the simplification relative to the original Da
Silva et al. procedure.

The calling sequence is identical to :func:`fdtdx.optimization.mmasub_unconst.mmasub_unconst`
so that the two are interchangeable in the optimizer switch (see
:func:`fdtdx.optimization.mma.steepest_descent` for the optax-compatible wrapper used in
fdtdx training loops). The asymptote slots low/upp are re-used to carry the move-limit
state (x^inf and x^sup) between calls; the move limits delta of the previous iteration
are recovered from them as delta = (upp-low)/2.

Ported line-for-line from the original MATLAB (SteepestDescent.m) so that fdtdx does not
depend on any third-party implementation.
"""

import numpy as np


def SteepestDescent(
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
    move: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performs one steepest-descent step, aimed at solving the box-constrained problem.

    Minimize  f_0(x)
    subject to  xmin_j <= x_j <= xmax_j,    j = 1,...,n

    Args:
        n (int): The number of variables x_j.
        iter (int): Current iteration number ( =1 the first time SteepestDescent is
            called).
        xval (np.ndarray): Column vector with the current values of the variables x_j.
        xmin (np.ndarray): Column vector with the TRUE physical lower bounds for x_j
            (the "0" of eqs. (D22) and (D24); 0 for densities).
        xmax (np.ndarray): Column vector with the TRUE physical upper bounds for x_j
            (the "1" of eqs. (D22) and (D24); 1 for densities). NOTE: unlike
            mmasub/mmasub_unconst, this method must NOT be handed bounds that have
            already been narrowed by an external move limit (e.g. x +/- opt.move). The
            move limit here is the internal, per-element, adaptive delta of eq. (D23);
            an external box would silently override it and freeze the adaptation, and
            it would also make step 2 reset gradients at the move limit instead of at
            the physical bound.
        xold1 (np.ndarray): xval, one iteration ago (provided that iter>1).
        xold2 (np.ndarray): xval, two iterations ago (provided that iter>2).
        f0val (float): The value of the objective function f_0 at xval (not used by
            this method; kept for interface compatibility).
        df0dx (np.ndarray): Column vector with the derivatives of the objective
            function f_0 with respect to the variables x_j, calculated at xval.
        low (np.ndarray): Column vector with x^inf from the previous iteration
            (provided that iter>1).
        upp (np.ndarray): Column vector with x^sup from the previous iteration
            (provided that iter>1).
        move (float): Maximum (and initial) move limit delta_e.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: xmma, low, upp -- the column vector
            with the updated values of the variables x_j, and the lower/upper move
            limits x^inf/x^sup calculated and used in the current step.
    """
    deltamax = move  # maximum move limit (start value)
    deltamin = 0.001  # minimum move limit
    deltaincr = 1.1  # move limit increase factor
    deltadecr = 0.7  # move limit decrease factor
    Psi = 1.0  # unitary step length
    eeen = np.ones((n, 1))

    # 1. Steepest descent direction.
    S = -df0dx

    # 2. Reset the gradient contributions at the bound constraints (D22), so that
    #    variables sitting on a bound are not pushed further outside it.
    S[(xval >= xmax) & (S > 0)] = 0
    S[(xval <= xmin) & (S < 0)] = 0

    # 3. Normalise S by its maximum absolute value.
    Smax = np.max(np.abs(S))
    if Smax > 0:
        D = S / Smax
    else:
        D = np.zeros((n, 1))

    # 4. Move limits based on the two previous iterations (D23). The move limits are
    #    held at their maximum range until the third iteration.
    if iter < 2.5:
        delta = deltamax * eeen
    else:
        delta = 0.5 * (upp - low)  # delta of the previous iteration
        ddd = (xval - xold1) * (xold1 - xold2)
        factor = eeen.copy()
        factor[ddd < 0] = deltadecr
        factor[ddd > 0] = deltaincr
        delta = factor * delta
        delta = np.minimum(delta, deltamax)
        delta = np.maximum(delta, deltamin)

    low = xval - delta  # x^inf
    upp = xval + delta  # x^sup

    # 5. Update with the unitary step length, clipped to the move limits intersected
    #    with the physical bounds (D24).
    alfa = np.maximum(low, xmin)
    beta = np.minimum(upp, xmax)

    xmma = xval + Psi * D
    xmma = np.maximum(alfa, np.minimum(beta, xmma))

    return xmma, low, upp
