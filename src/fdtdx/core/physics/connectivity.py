from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp


def _is_periodic(periodic_axes: Sequence[bool] | None, axis: int) -> bool:
    """Whether `axis` wraps. `None` means "no axis wraps", the original behavior."""
    return False if periodic_axes is None else bool(periodic_axes[axis])


def _face_conductances(kappa: jax.Array, axis: int, eps: float, periodic: bool = False) -> tuple[jax.Array, jax.Array]:
    """Harmonic-mean conductance of the two faces of every voxel along `axis`.

    Padding `kappa` with zeros at the array boundary gives a zero-conductance "ghost" neighbor
    there, which the harmonic mean turns into a zero-flux (Neumann) boundary automatically --
    no separate boundary-condition code path is needed.

    With `periodic=True` the padding wraps instead, so the ghost neighbor is the real material at
    the opposite face and flux crosses the boundary as it physically does in a periodic unit cell.
    A Neumann wall on a periodic axis is not a harmless approximation: an island attached only
    through the cell boundary reads as isolated, and since an isolated island's temperature is
    bounded solely by `kappa_min` a handful of such voxels can dominate the integral.
    """
    k = jnp.moveaxis(kappa, axis, 0)
    n = k.shape[0]
    pad_mode = "wrap" if periodic else "constant"
    kp = jnp.pad(k, [(1, 1)] + [(0, 0)] * (k.ndim - 1), mode=pad_mode)
    g_left = 2.0 * kp[0:n] * kp[1 : n + 1] / (kp[0:n] + kp[1 : n + 1] + eps)
    g_right = 2.0 * kp[1 : n + 1] * kp[2 : n + 2] / (kp[1 : n + 1] + kp[2 : n + 2] + eps)
    return jnp.moveaxis(g_left, 0, axis), jnp.moveaxis(g_right, 0, axis)


def _shifted_neighbors(u: jax.Array, axis: int, periodic: bool = False) -> tuple[jax.Array, jax.Array]:
    """Left/right neighbor values of `u` along `axis`, edge-padded at the array boundary.

    On a bounded axis the boundary's edge-padded value is arbitrary (never inspected on its own)
    since it is always multiplied by the zero boundary conductance from `_face_conductances`. On a
    periodic axis it wraps, and then it genuinely is the neighbor's value.
    """
    v = jnp.moveaxis(u, axis, 0)
    n = v.shape[0]
    vp = jnp.pad(v, [(1, 1)] + [(0, 0)] * (v.ndim - 1), mode="wrap" if periodic else "edge")
    left = jnp.moveaxis(vp[0:n], 0, axis)
    right = jnp.moveaxis(vp[2 : n + 2], 0, axis)
    return left, right


def _diffusion_apply(
    u: jax.Array, kappa: jax.Array, eps: float, periodic_axes: Sequence[bool] | None = None
) -> jax.Array:
    """Matrix-free application of the finite-volume operator `-div(kappa * grad(u))`."""
    total = jnp.zeros_like(u)
    for axis in range(u.ndim):
        periodic = _is_periodic(periodic_axes, axis)
        g_left, g_right = _face_conductances(kappa, axis, eps, periodic)
        u_left, u_right = _shifted_neighbors(u, axis, periodic)
        total = total + g_left * (u - u_left) + g_right * (u - u_right)
    return total


def _diffusion_diagonal(kappa: jax.Array, eps: float, periodic_axes: Sequence[bool] | None = None) -> jax.Array:
    """Diagonal of the same operator, i.e. the total conductance out of each voxel -- used as a
    cheap Jacobi preconditioner (important given the large kappa_max/kappa_min contrast)."""
    total = jnp.zeros_like(kappa)
    for axis in range(kappa.ndim):
        g_left, g_right = _face_conductances(kappa, axis, eps, _is_periodic(periodic_axes, axis))
        total = total + g_left + g_right
    return total


def solve_steady_heat(
    conductivity: jax.Array,
    source: jax.Array,
    sink_mask: jax.Array,
    *,
    periodic_axes: Sequence[bool] | None = None,
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
        periodic_axes (Sequence[bool] | None): Per-axis flag; True makes that axis wrap instead of
            meeting a zero-flux wall. `None` (default) means every axis is bounded.
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
        diffused = _diffusion_apply(u_bc, conductivity, eps, periodic_axes)
        return jnp.where(sink_mask, u, diffused)

    b = jnp.where(sink_mask, 0.0, source)
    diag = _diffusion_diagonal(conductivity, eps, periodic_axes)
    diag = jnp.where(sink_mask, 1.0, jnp.clip(diag, eps, None))

    def M(u: jax.Array) -> jax.Array:
        return u / diag

    u, _ = jax.scipy.sparse.linalg.cg(A, b, tol=tol, atol=atol, maxiter=maxiter, M=M)
    return u


def _pcg(
    operator,
    rhs: jax.Array,
    precondition,
    *,
    tol: float,
    atol: float,
    maxiter: int,
) -> jax.Array:
    """Plain preconditioned conjugate gradient, as a `lax.while_loop`.

    Deliberately not `jax.scipy.sparse.linalg.cg`: that routine is built on
    `lax.custom_linear_solve`, which cannot be nested inside the `tangent_solve` of a
    `lax.custom_root` (it raises NotImplementedError). This loop carries no custom
    differentiation rule of its own, which is exactly what the two call sites need -- the Newton
    iteration is wrapped in `custom_root` and so is never differentiated through, and
    `tangent_solve` is itself the derivative rule.
    """
    x = jnp.zeros_like(rhs)
    r = rhs - operator(x)
    z = precondition(r)
    rz = jnp.sum(r * z)
    target = jnp.maximum(tol * jnp.linalg.norm(rhs), atol)

    def keep_going(state):
        i, _, r, _, _, _ = state
        return (i < maxiter) & (jnp.linalg.norm(r) > target)

    def iterate(state):
        i, x, r, z, p, rz = state
        ap = operator(p)
        denom = jnp.sum(p * ap)
        step = rz / jnp.where(denom == 0.0, 1e-30, denom)
        x = x + step * p
        r = r - step * ap
        z = precondition(r)
        rz_next = jnp.sum(r * z)
        beta = rz_next / jnp.where(rz == 0.0, 1e-30, rz)
        return (i + 1, x, r, z, z + beta * p, rz_next)

    return jax.lax.while_loop(keep_going, iterate, (0, x, r, z, z, rz))[1]


def solve_nonlinear_steady_heat(
    conductivity: jax.Array,
    source_strength: jax.Array,
    sink_mask: jax.Array,
    *,
    alpha: float,
    t_max: float,
    periodic_axes: Sequence[bool] | None = None,
    newton_steps: int = 12,
    backtrack_steps: int = 10,
    tol: float = 1e-6,
    atol: float = 0.0,
    maxiter: int = 1000,
    eps: float = 1e-30,
) -> jax.Array:
    """Nonlinear virtual-temperature solve: ``-div(kappa grad T) = Q(T)`` with a *saturating*
    source, after Luo, Sigmund, Li & Liu, Comp. Meth. Appl. Mech. Eng. 372 (2020) 113385.

    The source weakens as the voxel heats up,

        Q(T) = q / (1 + exp(alpha * (T - t_max))),

    so an isolated island stops generating heat near ``t_max`` instead of climbing until
    ``kappa_min`` leakage balances it. Two consequences, and the second is why the method exists:

    1. Island temperature is set by ``t_max`` rather than by the conductivity contrast, so the
       field saturates to a near-uniform plateau over every disconnected region whatever its size
       or depth. (``t_max`` is a soft knee, not a hard ceiling -- equilibrium sits wherever the
       collapsing source meets the leakage, which can be modestly above it.)
    2. The constraint then needs no calibrated threshold. Luo et al. take ``T < t_max / 2``:
       connected material drains well below it, isolated material sits near ``t_max``. That is what
       removes the "highly dependent on the user-defined inputs" objection to the linear VTM
       (Cool, Aage & Sigmund, Struct. Multidisc. Optim. 68 (2025) 73, section 3.2.1.2).

    Calibrating ``source_strength`` matters and is the caller's job. Luo et al. scale it so that a
    *linear* solve over an all-source domain peaks at ``alpha * t_max``; too small and the sigmoid
    never leaves its flat top, making this the linear method with extra steps, too large and even
    well-connected material saturates and the contrast is lost again.

    Solved by **damped** Newton. The Jacobian is ``J = H - dQ/dT``, and since
    ``dQ/dT = -q*alpha*s*(1-s) <= 0`` with ``s`` the sigmoid factor, ``J`` is ``H`` plus a
    non-negative diagonal -- still symmetric positive definite, so each step is a preconditioned CG.
    The damping is not optional: from a cold start the first step lands where the *linear* problem
    would, which for an isolated island is orders of magnitude above ``t_max``; the source there has
    collapsed to nothing, so the next undamped step drives it straight back to zero and the
    iteration oscillates instead of converging. A backtracking line search on the residual norm
    fixes that.

    Gradients come from `jax.lax.custom_root`: one implicit adjoint solve against the same
    Jacobian, NOT differentiation through the Newton loop. Iteration count therefore costs forward
    time only and never enters the backward pass.

    Args:
        conductivity (jax.Array): Per-voxel conductivity, strictly positive.
        source_strength (jax.Array): Per-voxel ``q``, the *unsaturated* source -- see the
            calibration note above.
        sink_mask (jax.Array): True at voxels held at T=0.
        alpha (float): Sharpness of the saturation. Luo et al. use 0.2 against ``t_max`` 100; it is
            the dimensionless product ``alpha * t_max`` that sets the knee, not either alone.
        t_max (float): Temperature an isolated region saturates towards.
        periodic_axes (Sequence[bool] | None): Per-axis wrap flags, as in `solve_steady_heat`.
        newton_steps (int): Newton iterations. Fixed rather than tolerance-based so the solve keeps
            a static shape under `jit`. Defaults to 12.
        backtrack_steps (int): Step lengths tried per Newton iteration, halving from 1. Defaults
            to 10, i.e. down to 1/512.
        tol, atol, maxiter, eps: Inner CG controls, as in `solve_steady_heat`.

    Returns:
        jax.Array: Steady-state temperature field, same shape as `conductivity`.
    """

    def saturation(u: jax.Array) -> jax.Array:
        # sigmoid(-alpha*(u - t_max)), not 1/(1+exp(...)): the explicit exponential overflows
        # float32 for a modest alpha*(u - t_max), which this problem reaches routinely.
        return jax.nn.sigmoid(-alpha * (u - t_max))

    def residual(u: jax.Array) -> jax.Array:
        u_bc = jnp.where(sink_mask, 0.0, u)
        conduction = _diffusion_apply(u_bc, conductivity, eps, periodic_axes)
        q = source_strength * saturation(u)
        return jnp.where(sink_mask, u, conduction - q)

    base_diag = _diffusion_diagonal(conductivity, eps, periodic_axes)

    def newton(f, u0: jax.Array) -> jax.Array:
        def step(u, _):
            s = saturation(u)
            # -dQ/dT >= 0, so J stays SPD and CG remains valid.
            extra = jnp.where(sink_mask, 0.0, source_strength * alpha * s * (1.0 - s))

            def jacobian(v: jax.Array) -> jax.Array:
                v_bc = jnp.where(sink_mask, 0.0, v)
                jv = _diffusion_apply(v_bc, conductivity, eps, periodic_axes) + extra * v_bc
                return jnp.where(sink_mask, v, jv)

            diag = jnp.where(sink_mask, 1.0, jnp.clip(base_diag + extra, eps, None))
            delta = _pcg(jacobian, -f(u), lambda v: v / diag, tol=tol, atol=atol, maxiter=maxiter)

            # Backtracking: take the trial length with the smallest residual norm. Evaluating all
            # of them is a handful of stencil applications, far cheaper than the CG solve above,
            # and keeps the loop statically shaped.
            lengths = 0.5 ** jnp.arange(backtrack_steps, dtype=u.dtype)
            norms = jax.vmap(lambda t: jnp.linalg.norm(f(u + t * delta)))(lengths)
            return u + lengths[jnp.argmin(norms)] * delta, None

        return jax.lax.scan(step, u0, None, length=newton_steps)[0]

    def tangent_solve(linearized, rhs: jax.Array) -> jax.Array:
        # `linearized` is the JVP of `residual` at the root, i.e. J itself. The exact Jacobi
        # diagonal is not reachable here, so precondition with the conduction part alone: a valid
        # preconditioner (it only has to be SPD), just not the sharpest one.
        #
        # Wrapped in `custom_linear_solve` rather than called directly, because reverse mode has to
        # TRANSPOSE this solve and `_pcg`'s `while_loop` is not transposable ("Reverse-mode
        # differentiation does not work for lax.while_loop"). `custom_linear_solve` supplies that
        # transpose rule itself -- it re-runs `solve` on the transposed system instead of
        # differentiating through the iteration -- and `symmetric=True` is exact here: J is the SPD
        # conduction operator plus a non-negative diagonal, so it is its own transpose.
        diag = jnp.where(sink_mask, 1.0, jnp.clip(base_diag, eps, None))

        def solve(matvec, b):
            return _pcg(matvec, b, lambda v: v / diag, tol=tol, atol=atol, maxiter=maxiter)

        return jax.lax.custom_linear_solve(linearized, rhs, solve, symmetric=True)

    return jax.lax.custom_root(residual, jnp.zeros_like(conductivity), newton, tangent_solve)


def _aggregate(u: jax.Array, aggregation: str, pnorm_p: float) -> jax.Array:
    """Reduce a temperature field to the scalar the constraint is written on.

    ``"sum"`` is the integrated temperature. ``"pnorm"`` is the smooth maximum
    ``(mean(T_i**p))**(1/p)``, which is what the VTM literature actually constrains -- the
    constraint there is ``T_max <= T_bar`` and the p-norm is its differentiable stand-in (Cool,
    Aage & Sigmund 2025 use p = 12). The difference is not cosmetic: a sum spreads one scorching
    island across every voxel in the design, so a structure with many warm-but-connected voxels can
    out-weigh a genuinely detached one.

    Computed as ``max(T) * (mean((T/max(T))**p))**(1/p)``, algebraically identical but with every
    power taken on a quantity in [0, 1]. The plain form overflows float32 immediately: a field
    peaking at 5e3 raised to p = 12 is ~1e44 against a float32 ceiling of 3.4e38.

    Negative temperatures are clipped away first. They are unphysical but do occur in intermediate
    density regions under a linear source -- a documented VTM failure mode, and one of the
    motivations for the nonlinear source above.
    """
    if aggregation == "sum":
        return jnp.sum(u)
    if aggregation != "pnorm":
        raise ValueError(f"unknown aggregation {aggregation!r}; expected 'sum' or 'pnorm'")
    positive = jnp.clip(u, 0.0, None)
    scale = jnp.max(positive)
    safe = jnp.where(scale > 0.0, scale, 1.0)
    return safe * jnp.mean((positive / safe) ** pnorm_p) ** (1.0 / pnorm_p)


def connectivity_penalty(
    density: jax.Array,
    sink_mask: jax.Array,
    kappa_min: float = 1e-3,
    kappa_max: float = 1.0,
    *,
    invert: bool = False,
    penalization: float = 1.0,
    saturation_alpha: float | None = None,
    saturation_t_max: float | None = None,
    source_scale: float = 1.0,
    aggregation: str = "sum",
    pnorm_p: float = 12.0,
    periodic_axes: Sequence[bool] | None = None,
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
        penalization (float): SIMP exponent on the conducting/source phase: conductivity becomes
            `kappa_min + d**penalization * (kappa_max - kappa_min)` and the source `d**penalization`.
            1.0 (default) is the linear interpolation.

            Raise it to stop *intermediate* density from faking a connection. With the linear law a
            density of 0.04 conducts at 0.04, which against a `kappa_min` of 1e-5 is four thousand
            times better than void -- so a greyscale haze far too faint to see acts as a thermal
            bridge and a detached island reads as attached. At `penalization` 3 that haze conducts
            at `kappa_min` and the island is isolated again.

            Note the orientation. Luo et al. write their exponent on the phase *complementary* to
            the conducting one (`1 - mu**p` with `mu` the solid density, for their void problem),
            which makes intermediate density behave like the conducting phase -- conservative when
            the question is "is this void sealed". Here the exponent is on `d` itself, which makes
            intermediate density behave like the *insulating* phase. That is the orientation that
            suppresses spurious bridging, which is the failure mode this parameter exists for; the
            two coincide only at `penalization = 1`.
        saturation_alpha (float | None): With `saturation_t_max`, switches to the nonlinear
            virtual-temperature method -- see `solve_nonlinear_steady_heat`. `None` (default)
            keeps the linear source and the original linear solve. Both must be given together.
        saturation_t_max (float | None): The plateau temperature of the saturating source.
        source_scale (float): Multiplies the source term. Meaningless for the linear problem (the
            solve is linear in it, so it only rescales the returned value), but it sets where the
            field sits relative to `saturation_t_max` in the nonlinear one, and so is the knob Luo
            et al. calibrate -- they scale it so a *linear* solve would peak at
            `saturation_alpha * saturation_t_max`. Defaults to 1.0.
        aggregation (str): `"sum"` (default) for the integrated temperature, or `"pnorm"` for the
            smooth maximum the VTM literature actually constrains. See `_aggregate`.
        pnorm_p (float): Exponent for `aggregation="pnorm"`. Defaults to 12, after Cool, Aage &
            Sigmund (2025).
        periodic_axes (Sequence[bool] | None): Per-axis wrap flags. `None` (default) keeps every
            axis bounded. Required for a periodic unit cell, where a Neumann wall would report
            material attached only through the cell boundary as floating.
        **solve_kwargs: Forwarded to the solver (`tol`, `atol`, `maxiter`, `eps`, and
            `newton_steps` for the nonlinear one).

    Returns:
        jax.Array: Scalar measure of how poorly the source phase reaches the sinks, under
        `aggregation`. A fixed voxel-volume factor is omitted since it is a constant rescaling of
        the objective.

    Raises:
        ValueError: If only one of `saturation_alpha` / `saturation_t_max` is given.
    """
    if (saturation_alpha is None) != (saturation_t_max is None):
        raise ValueError(
            "saturation_alpha and saturation_t_max must be given together: both select the "
            "nonlinear virtual-temperature method, and neither has a meaning on its own."
        )
    d = 1.0 - density if invert else density
    weight = d if penalization == 1.0 else d**penalization
    conductivity = kappa_min + weight * (kappa_max - kappa_min)
    source = source_scale * weight
    # Both-or-neither is already guaranteed above; testing both here is what lets a type checker
    # see `saturation_t_max` as non-None in the nonlinear branch.
    if saturation_alpha is None or saturation_t_max is None:
        u = solve_steady_heat(
            conductivity,
            source=source,
            sink_mask=sink_mask,
            periodic_axes=periodic_axes,
            **solve_kwargs,
        )
    else:
        u = solve_nonlinear_steady_heat(
            conductivity,
            source_strength=source,
            sink_mask=sink_mask,
            alpha=saturation_alpha,
            t_max=saturation_t_max,
            periodic_axes=periodic_axes,
            **solve_kwargs,
        )
    return _aggregate(u, aggregation, pnorm_p)


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
