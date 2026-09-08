from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp


def _face_conductances(kappa: jax.Array, axis: int, eps: float) -> tuple[jax.Array, jax.Array]:
    """Harmonic-mean conductance of the two faces of every voxel along `axis`.

    Padding `kappa` with zeros at the array boundary gives a zero-conductance "ghost" neighbor
    there, which the harmonic mean turns into a zero-flux (Neumann) boundary automatically --
    no separate boundary-condition code path is needed.
    """
    k = jnp.moveaxis(kappa, axis, 0)
    n = k.shape[0]
    kp = jnp.pad(k, [(1, 1)] + [(0, 0)] * (k.ndim - 1))
    g_left = 2.0 * kp[0:n] * kp[1 : n + 1] / (kp[0:n] + kp[1 : n + 1] + eps)
    g_right = 2.0 * kp[1 : n + 1] * kp[2 : n + 2] / (kp[1 : n + 1] + kp[2 : n + 2] + eps)
    return jnp.moveaxis(g_left, 0, axis), jnp.moveaxis(g_right, 0, axis)


def _shifted_neighbors(u: jax.Array, axis: int) -> tuple[jax.Array, jax.Array]:
    """Left/right neighbor values of `u` along `axis`, edge-padded at the array boundary.

    The boundary's edge-padded value is arbitrary (never inspected on its own) since it is
    always multiplied by the zero boundary conductance from `_face_conductances`.
    """
    v = jnp.moveaxis(u, axis, 0)
    n = v.shape[0]
    vp = jnp.pad(v, [(1, 1)] + [(0, 0)] * (v.ndim - 1), mode="edge")
    left = jnp.moveaxis(vp[0:n], 0, axis)
    right = jnp.moveaxis(vp[2 : n + 2], 0, axis)
    return left, right


def _diffusion_apply(u: jax.Array, kappa: jax.Array, eps: float) -> jax.Array:
    """Matrix-free application of the finite-volume operator `-div(kappa * grad(u))`."""
    total = jnp.zeros_like(u)
    for axis in range(u.ndim):
        g_left, g_right = _face_conductances(kappa, axis, eps)
        u_left, u_right = _shifted_neighbors(u, axis)
        total = total + g_left * (u - u_left) + g_right * (u - u_right)
    return total


def _diffusion_diagonal(kappa: jax.Array, eps: float) -> jax.Array:
    """Diagonal of the same operator, i.e. the total conductance out of each voxel -- used as a
    cheap Jacobi preconditioner (important given the large kappa_max/kappa_min contrast)."""
    total = jnp.zeros_like(kappa)
    for axis in range(kappa.ndim):
        g_left, g_right = _face_conductances(kappa, axis, eps)
        total = total + g_left + g_right
    return total


def solve_steady_heat(
    conductivity: jax.Array,
    source: jax.Array,
    sink_mask: jax.Array,
    *,
    tol: float = 1e-6,
    atol: float = 0.0,
    maxiter: int = 1000,
    eps: float = 1e-30,
) -> jax.Array:
    """Solve a steady-state heat-diffusion (Poisson) problem on a regular voxel grid.

    Discretizes ``-div(kappa * grad(u)) = source`` with a matrix-free, cell-centered
    finite-volume stencil (harmonic-mean face conductances, so a zero-conductivity voxel
    correctly blocks flux rather than being smeared out by an arithmetic mean) and zero-flux
    (Neumann) boundaries at the edges of the array. `sink_mask` selects voxels held at u=0
    (homogeneous Dirichlet, i.e. "heat sinks"). This is implemented by clamping those voxels to
    zero both when *reading* neighbor values and when *writing* the operator's output there,
    which keeps the resulting linear operator symmetric positive definite (required by `cg`) --
    equivalent to eliminating the Dirichlet rows/columns from the assembled stiffness matrix.

    Because `A` below is a plain JAX function closing over `conductivity`, `cg`'s reverse-mode
    rule differentiates through the solve via one implicit adjoint solve (reusing `cg` itself,
    since `A` is symmetric) rather than unrolling through the CG iterations -- the same cost
    structure as the adjoint method already used for the FDTD gradient, not backprop-through-
    iterations.

    Args:
        conductivity (jax.Array): Per-voxel thermal conductivity, shape `(nx, ny, nz)` (or any
            rank), strictly positive.
        source (jax.Array): Per-voxel heat-generation rate, same shape as `conductivity`.
        sink_mask (jax.Array): Boolean array, same shape, True at voxels held at u=0.
        tol (float): Relative tolerance passed to `cg`. Defaults to 1e-6.
        atol (float): Absolute tolerance passed to `cg`. Defaults to 0.0.
        maxiter (int): Maximum CG iterations. Defaults to 1000.
        eps (float): Floor added to conductance denominators to avoid division by zero when
            both neighboring conductivities are (numerically) zero. Defaults to 1e-30.

    Returns:
        jax.Array: Steady-state temperature field `u`, same shape as `conductivity`.
    """

    def A(u: jax.Array) -> jax.Array:
        u_bc = jnp.where(sink_mask, 0.0, u)
        diffused = _diffusion_apply(u_bc, conductivity, eps)
        return jnp.where(sink_mask, u, diffused)

    b = jnp.where(sink_mask, 0.0, source)
    diag = _diffusion_diagonal(conductivity, eps)
    diag = jnp.where(sink_mask, 1.0, jnp.clip(diag, eps, None))

    def M(u: jax.Array) -> jax.Array:
        return u / diag

    u, _ = jax.scipy.sparse.linalg.cg(A, b, tol=tol, atol=atol, maxiter=maxiter, M=M)
    return u


def connectivity_penalty(
    density: jax.Array,
    sink_mask: jax.Array,
    kappa_min: float = 1e-3,
    kappa_max: float = 1.0,
    *,
    invert: bool = False,
    **solve_kwargs,
) -> jax.Array:
    """Integrated-temperature connectivity penalty (Kuster et al., Nanophotonics 14(9):1415-1426).

    Treats the (projected, physical) `density` as a fictitious heat source and interpolates a
    per-voxel conductivity between `kappa_min` (void) and `kappa_max` (material). Fixing
    `sink_mask` voxels at zero temperature and integrating the resulting steady-state field
    measures how well the "hot" region can dissipate heat to the sinks: a design where every hot
    voxel has a continuous conductive path to a sink reaches a low integrated temperature, while
    a disconnected island heats up (kept finite rather than diverging only by the small
    always-present `kappa_min` background conductivity -- keep it strictly positive). Minimizing
    this value is a smooth, differentiable proxy for "this material is fully connected to the
    sinks" (or, with `invert=True`, "this void is fully connected to the sinks").

    Args:
        density (jax.Array): Physical (post-filter, post-projection) density in [0, 1], shape
            `(nx, ny, nz)`.
        sink_mask (jax.Array): Boolean array, same shape, True at heat-sink voxels -- for the
            material problem, typically the ports/pads the material must stay attached to; for
            the void problem (`invert=True`), typically the outer rim of the design region minus
            those same ports.
        kappa_min (float): Conductivity of the "cold" phase (void, for the material problem).
            Must be strictly positive -- it is what keeps a disconnected island's temperature
            finite instead of diverging. Defaults to 1e-3.
        kappa_max (float): Conductivity of the "hot"/source phase. Defaults to 1.0.
        invert (bool): If True, solves the void-connectivity problem by swapping the role of
            `density` (uses `1 - density` as both the source and the conductivity driver).
            Defaults to False (material connectivity).
        **solve_kwargs: Forwarded to `solve_steady_heat` (`tol`, `atol`, `maxiter`, `eps`).

    Returns:
        jax.Array: Scalar sum of the steady-state temperature field over the design region --
        proportional to the integrated temperature (a fixed voxel-volume factor is omitted since
        it is a constant rescaling of the objective).
    """
    d = 1.0 - density if invert else density
    conductivity = kappa_min + d * (kappa_max - kappa_min)
    u = solve_steady_heat(conductivity, source=d, sink_mask=sink_mask, **solve_kwargs)
    return jnp.sum(u)


def renormalize(
    raw: jax.Array,
    threshold: float | jax.Array,
    *,
    maximize: bool = False,
    max_value: float | jax.Array | None = None,
) -> jax.Array:
    """Rescale a raw sub-objective so 0 marks its threshold and negative means "good enough".

    For a minimized sub-objective (`maximize=False`, e.g. the connectivity penalties above):
    ``(raw - threshold) / threshold``. For a maximized one (`maximize=True`, e.g. an efficiency):
    ``(threshold - raw) / (max_value - threshold)``, which needs an estimated `max_value` larger
    than any achievable `raw` so the normalized value stays > -1 (Kuster et al., Eq. 8-9).

    Args:
        raw (jax.Array): Raw sub-objective value.
        threshold (float | jax.Array): Value that should map to a normalized 0.
        maximize (bool): Whether `raw` is being maximized rather than minimized. Defaults to
            False.
        max_value (float | jax.Array | None): Required when `maximize=True`: an estimate of the
            largest value `raw` could plausibly reach, used to scale the normalization.

    Returns:
        jax.Array: Normalized sub-objective; negative once past its threshold.
    """
    if maximize:
        if max_value is None:
            raise ValueError("max_value is required when maximize=True")
        return (threshold - raw) / (max_value - threshold)
    return (raw - threshold) / threshold


def softplus_combine(
    normalized_terms: Sequence[jax.Array],
    weights: Sequence[float | jax.Array] | None = None,
) -> jax.Array:
    """Combine renormalized sub-objectives into one scalar loss via softplus + l2-norm.

    Applies `softplus` to each term -- which, unlike a hard cutoff, smoothly (but sharply) lets a
    term's gradient vanish once it goes well negative (its threshold is satisfied) while still
    contributing near its threshold, avoiding both a discontinuous switch and a permanently-
    active competing objective -- then combines via an l2 norm (Kuster et al., Eq. 10-12), so the
    combined loss is dominated by whichever term is currently furthest from satisfied rather than
    by their sum.

    Args:
        normalized_terms (Sequence[jax.Array]): Scalar terms from `renormalize`, one per
            sub-objective (e.g. `[norm_em, norm_material, norm_void]`).
        weights (Sequence[float | jax.Array] | None): Optional per-term multiplier applied to
            `softplus(term)**2` before summing. `None` (default) weights every term 1, reproducing
            the plain Eq. 10-12 l2-norm; a weight of 0 fully excludes a term (and its gradient)
            from the combined loss. Useful for a term that should only join the objective partway
            through optimization -- e.g. a binarization/non-discreteness penalty deliberately
            deferred until beta continuation reaches its final stage, mirroring Kuster et al.'s own
            "we only turn on the binarization penalty once we are close to convergence" (their
            Eq. 11 term is handled exactly this way, unlike the always-on connectivity terms).

    Returns:
        jax.Array: Scalar combined loss to minimize.
    """
    softened = jnp.stack([jax.nn.softplus(t) for t in normalized_terms])
    squared = softened**2
    if weights is not None:
        squared = jnp.asarray(weights) * squared
    return jnp.sqrt(jnp.sum(squared))
