"""MMA (Method of Moving Asymptotes, Svanberg 1987/2002) optimizers, wrapped as
optax.GradientTransformationExtraArgs so they are drop-in replacements for optax.adam(...)
in fdtdx training loops.

Two variants are provided, both implemented natively in fdtdx (see
fdtdx.optimization.mmasub, fdtdx.optimization.mmasub_unconst, fdtdx.optimization.subsolv
for the underlying line-for-line ports of Svanberg's original MATLAB code -- no
third-party MMA package is required):

- `mma`: The general Svanberg subproblem solved by a primal-dual Newton method
  (`subsolv`). Supports general inequality constraints (n_constraints > 0) in addition
  to the box constraints.
- `mma_unconstrained`: The box-constrained-only (m=0) variant, whose subproblem is
  separable and solved in closed form -- no Newton iterations, and an explicit `move`
  limit per step (mmasub.m's move limit is hardcoded to 1.0, i.e. unused in practice).
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree

from fdtdx.fdtd.container import ParameterContainer
from fdtdx.optimization.mmasub import mmasub
from fdtdx.optimization.mmasub_unconst import mmasub_unconst


@dataclass(frozen=True)
class MMAState:
    """Optimizer state shared by the `mma` and `mma_unconstrained` optimizers.

    Deliberately a plain Python dataclass rather than a jax pytree / fdtdx TreeClass.
    Unlike ADAM's optax state (which lives inside a jax.jit-compiled step and must be
    traceable), MMA's outer loop runs at Python level *between* jax.jit-compiled
    simulation/gradient steps -- exactly like optax's own `.update()` calls in existing
    fdtdx examples (see examples/optimize_ceviche_corner.py). It is never itself passed
    through jax.jit/vmap/grad, so there is nothing to gain from pytree registration.

    Only carries quantities that vary between calls to update(); static configuration
    (bounds, move limit) is closed over by `mma()`/`mma_unconstrained()`'s
    `init_fn`/`update_fn`, mirroring optax's own convention (e.g. `optax.scale_by_adam`
    closes over b1/b2/eps and stores only mu/nu/count in its state).

    Attributes:
        iter (int): Iteration counter. 0 at init(), incremented by 1 on every update()
            call (so the first update() call runs the subproblem with iter=1).
        xold1 (np.ndarray): Design point one update() call ago, shape (n, 1), float64.
        xold2 (np.ndarray): Design point two update() calls ago, shape (n, 1), float64.
        low (np.ndarray): Lower moving asymptote, shape (n, 1), float64.
        upp (np.ndarray): Upper moving asymptote, shape (n, 1), float64.
    """

    iter: int
    xold1: np.ndarray
    xold2: np.ndarray
    low: np.ndarray
    upp: np.ndarray


def _flatten_bound(n: int, bound: float | ParameterContainer, name: str) -> np.ndarray:
    """Flatten a scalar-or-pytree bound to shape (n, 1) float64, matching ravel_pytree order."""
    if isinstance(bound, int | float):
        return np.full((n, 1), float(bound), dtype=np.float64)
    flat_bound, _ = ravel_pytree(bound)
    if flat_bound.shape[0] != n:
        raise ValueError(
            f"{name} pytree must match the flattened structure of params "
            f"(flattened length {flat_bound.shape[0]} != {n}). Pass a scalar instead "
            f"for a uniform bound."
        )
    return np.asarray(flat_bound, dtype=np.float64).reshape(n, 1)


def mma(
    lower_bound: float | ParameterContainer = 0.0,
    upper_bound: float | ParameterContainer = 1.0,
    n_constraints: int = 0,
) -> optax.GradientTransformationExtraArgs:
    """MMA (Method of Moving Asymptotes) optimizer, drop-in compatible with optax.

    A literal replacement for optax.adam(...)/optax.nadam(...) in fdtdx training loops:
    same `.init(params)` / `.update(grads, state, params)` / `optax.apply_updates(...)`
    calling convention. Box constraints are handled natively by MMA's own xmin/xmax
    asymptote machinery -- unlike ADAM, no manual
    `jax.tree_util.tree_map(lambda p: jnp.clip(p, 0, 1), params)` is needed afterwards.

    Every MMA iteration is solved by `fdtdx.optimization.mmasub.mmasub`, a faithful port
    of Svanberg's original MATLAB (mmasub.m/subsolv.m) -- its Svanberg hyperparameters
    (asymptote init/growth/shrink factors, albefa, raa0, the move limit) are hardcoded
    exactly as in that source rather than exposed here, matching mmasub.m exactly. For a
    tunable move limit (and a faster closed-form solve), use `mma_unconstrained` instead
    for pure box-constrained (n_constraints=0) problems.

    The primary supported and tested path is n_constraints=0 (pure box-constrained,
    matching every current fdtdx ADAM usage -- no volume/other inequality constraints
    exist anywhere in fdtdx today). n_constraints > 0 is architecturally supported --
    pass pre-flattened `fval`/`dfdx` arrays to `.update()` -- but callers are responsible
    for matching the flattening order of `jax.flatten_util.ravel_pytree(params)`; this
    path is not covered by this version's test suite.

    Args:
        lower_bound (float | ParameterContainer): Lower box bound. Either a scalar
            (broadcast to every parameter) or a pytree with the exact same structure and
            leaf shapes as `params`. Defaults to 0.0.
        upper_bound (float | ParameterContainer): Upper box bound, same rules as
            `lower_bound`. Defaults to 1.0.
        n_constraints (int): Number of general inequality constraints (m in Svanberg's
            notation). 0 means pure box-constrained (the tested path). Defaults to 0.

    Returns:
        optax.GradientTransformationExtraArgs: An optax-compatible optimizer. `.update()`
            additionally accepts optional `fval`/`dfdx` kwargs for n_constraints > 0 use.
    """

    def init_fn(params: ParameterContainer) -> MMAState:
        flat_params, _ = ravel_pytree(params)
        n = flat_params.shape[0]
        xval = np.asarray(flat_params, dtype=np.float64).reshape(n, 1)
        xmin = _flatten_bound(n, lower_bound, "lower_bound")
        xmax = _flatten_bound(n, upper_bound, "upper_bound")
        return MMAState(iter=0, xold1=xval, xold2=xval, low=xmin, upp=xmax)

    def update_fn(
        grads: ParameterContainer,
        state: MMAState,
        params: ParameterContainer | None = None,
        *,
        fval: np.ndarray | None = None,
        dfdx: np.ndarray | None = None,
        **extra_args,
    ) -> tuple[ParameterContainer, MMAState]:
        del extra_args  # accepted only for the GradientTransformationExtraArgs protocol
        if params is None:
            raise ValueError(
                "mma() requires `params` at every update() call, e.g. "
                "optimizer.update(grads, opt_state, params) -- MMA's asymptote update "
                "needs the current design point, not just the gradient."
            )

        flat_params, unravel_fn = ravel_pytree(params)
        flat_grads, _ = ravel_pytree(grads)
        n = flat_params.shape[0]

        xval = np.asarray(flat_params, dtype=np.float64).reshape(n, 1)
        df0dx = np.asarray(flat_grads, dtype=np.float64).reshape(n, 1)
        xmin = _flatten_bound(n, lower_bound, "lower_bound")
        xmax = _flatten_bound(n, upper_bound, "upper_bound")

        m = n_constraints
        fval_arr = np.zeros((m, 1)) if fval is None else np.asarray(fval, dtype=np.float64).reshape(m, 1)
        dfdx_arr = np.zeros((m, n)) if dfdx is None else np.asarray(dfdx, dtype=np.float64).reshape(m, n)
        a = np.zeros((m, 1))
        c = 1000.0 * np.ones((m, 1))
        d = np.zeros((m, 1))

        # f0val is a pure additive constant in Svanberg's convex subproblem objective --
        # it provably never affects the resulting xmma (only df0dx/fval/dfdx do), so it
        # is hardcoded to 0.0 rather than requiring the caller to compute/pass the loss.
        f0val = 0.0
        new_iter = state.iter + 1

        xmma, ymma, zmma, lam, xsi, eta, mu_out, zet, s, low, upp = mmasub(
            m,
            n,
            new_iter,
            xval,
            xmin,
            xmax,
            state.xold1,
            state.xold2,
            f0val,
            df0dx,
            fval_arr,
            dfdx_arr,
            state.low,
            state.upp,
            1.0,
            a,
            c,
            d,
        )
        del ymma, zmma, lam, xsi, eta, mu_out, zet, s  # unused MMA dual/auxiliary outputs

        # Convert without an explicit dtype= -- unravel_fn restores each leaf's original
        # dtype automatically, and jnp.asarray(numpy_float64_array) silently downcasts to
        # float32 when jax x64 mode is off (fdtdx's default). Passing dtype=jnp.float64
        # explicitly instead triggers a "truncated to float32" UserWarning on every single
        # update() call.
        new_params = unravel_fn(jnp.asarray(xmma.reshape(-1)))
        updates = jax.tree_util.tree_map(lambda new, old: new - old, new_params, params)
        new_state = MMAState(iter=new_iter, xold1=xval, xold2=state.xold1, low=low, upp=upp)
        return updates, new_state

    # optax's bundled type stubs assume state is Array-like/iterable; MMAState is a plain
    # (deliberately non-pytree) dataclass, which optax's real runtime accepts as an opaque
    # state object just fine -- see MMAState's docstring for why it isn't a pytree.
    return optax.GradientTransformationExtraArgs(init_fn, update_fn)  # type: ignore


def mma_unconstrained(
    lower_bound: float | ParameterContainer = 0.0,
    upper_bound: float | ParameterContainer = 1.0,
    move: float = 0.5,
) -> optax.GradientTransformationExtraArgs:
    """Box-constrained-only MMA optimizer, drop-in compatible with optax.

    Every iteration is solved by `fdtdx.optimization.mmasub_unconst.mmasub_unconst`, a
    faithful port of Svanberg's algorithm specialized to m=0 general constraints (see
    that module's docstring): the subproblem is separable in x_j and solved explicitly
    in closed form, with no primal-dual Newton iterations -- unlike `mma`, which routes
    every step through `subsolv` regardless of n_constraints. In exchange, only box
    constraints are supported (no `fval`/`dfdx`/general inequality constraints), and the
    per-step move limit is an explicit, tunable fraction of (upper_bound - lower_bound)
    rather than the hardcoded, effectively-unused move=1.0 in `mma`/mmasub.m.

    Same `.init(params)` / `.update(grads, state, params)` / `optax.apply_updates(...)`
    calling convention as `mma`/optax.adam(...). Box constraints are handled natively via
    MMA's own xmin/xmax asymptote machinery -- no manual
    `jax.tree_util.tree_map(lambda p: jnp.clip(p, 0, 1), params)` is needed afterwards.

    Args:
        lower_bound (float | ParameterContainer): Lower box bound. Either a scalar
            (broadcast to every parameter) or a pytree with the exact same structure and
            leaf shapes as `params`. Defaults to 0.0.
        upper_bound (float | ParameterContainer): Upper box bound, same rules as
            `lower_bound`. Defaults to 1.0.
        move (float): Maximum allowed change per coordinate per iteration, as a fraction
            of (upper_bound - lower_bound). Defaults to 0.5.

    Returns:
        optax.GradientTransformationExtraArgs: An optax-compatible optimizer.
    """

    def init_fn(params: ParameterContainer) -> MMAState:
        flat_params, _ = ravel_pytree(params)
        n = flat_params.shape[0]
        xval = np.asarray(flat_params, dtype=np.float64).reshape(n, 1)
        xmin = _flatten_bound(n, lower_bound, "lower_bound")
        xmax = _flatten_bound(n, upper_bound, "upper_bound")
        return MMAState(iter=0, xold1=xval, xold2=xval, low=xmin, upp=xmax)

    def update_fn(
        grads: ParameterContainer,
        state: MMAState,
        params: ParameterContainer | None = None,
        **extra_args,
    ) -> tuple[ParameterContainer, MMAState]:
        del extra_args  # accepted only for the GradientTransformationExtraArgs protocol
        if params is None:
            raise ValueError(
                "mma_unconstrained() requires `params` at every update() call, e.g. "
                "optimizer.update(grads, opt_state, params) -- MMA's asymptote update "
                "needs the current design point, not just the gradient."
            )

        flat_params, unravel_fn = ravel_pytree(params)
        flat_grads, _ = ravel_pytree(grads)
        n = flat_params.shape[0]

        xval = np.asarray(flat_params, dtype=np.float64).reshape(n, 1)
        df0dx = np.asarray(flat_grads, dtype=np.float64).reshape(n, 1)
        xmin = _flatten_bound(n, lower_bound, "lower_bound")
        xmax = _flatten_bound(n, upper_bound, "upper_bound")

        # f0val is a pure additive constant in Svanberg's convex subproblem objective --
        # it provably never affects the resulting xmma (only df0dx does), so it is
        # hardcoded to 0.0 rather than requiring the caller to compute/pass the loss.
        f0val = 0.0
        new_iter = state.iter + 1

        xmma, low, upp = mmasub_unconst(
            n,
            new_iter,
            xval,
            xmin,
            xmax,
            state.xold1,
            state.xold2,
            f0val,
            df0dx,
            state.low,
            state.upp,
            move,
        )

        # See the equivalent conversion in mma()'s update_fn above for why this omits an
        # explicit dtype= (avoids a spurious float64-truncation UserWarning).
        new_params = unravel_fn(jnp.asarray(xmma.reshape(-1)))
        updates = jax.tree_util.tree_map(lambda new, old: new - old, new_params, params)
        new_state = MMAState(iter=new_iter, xold1=xval, xold2=state.xold1, low=low, upp=upp)
        return updates, new_state

    # optax's bundled type stubs assume state is Array-like/iterable; MMAState is a plain
    # (deliberately non-pytree) dataclass, which optax's real runtime accepts as an opaque
    # state object just fine -- see MMAState's docstring for why it isn't a pytree.
    return optax.GradientTransformationExtraArgs(init_fn, update_fn)  # type: ignore
