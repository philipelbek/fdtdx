"""MMA (Method of Moving Asymptotes, Svanberg 1987/2002) optimizer, wrapped as an
optax.GradientTransformationExtraArgs so it is a drop-in replacement for optax.adam(...)
in fdtdx training loops.

Requires the optional `mmapy` package (GPL-3.0 licensed, NOT installed by default with
`pip install fdtdx`). Install with `pip install fdtdx[mma]`. Importing this module never
imports mmapy; only calling `mma(...)` does, so `import fdtdx` is always safe.
"""

from dataclasses import dataclass

import jax
import numpy as np
import optax
from jax.flatten_util import ravel_pytree

from fdtdx.fdtd.container import ParameterContainer


@dataclass(frozen=True)
class MMAState:
    """Optimizer state for the MMA optimizer.

    Deliberately a plain Python dataclass rather than a jax pytree / fdtdx TreeClass.
    Unlike ADAM's optax state (which lives inside a jax.jit-compiled step and must be
    traceable), MMA's outer loop runs at Python level *between* jax.jit-compiled
    simulation/gradient steps -- exactly like optax's own `.update()` calls in existing
    fdtdx examples (see examples/optimize_ceviche_corner.py). It is never itself passed
    through jax.jit/vmap/grad, so there is nothing to gain from pytree registration.

    Only carries quantities that vary between calls to update(); static configuration
    (bounds, Svanberg hyperparameters) is closed over by `mma()`'s `init_fn`/`update_fn`,
    mirroring optax's own convention (e.g. `optax.scale_by_adam` closes over b1/b2/eps
    and stores only mu/nu/count in its state).

    Attributes:
        iter (int): Iteration counter. 0 at init(), incremented by 1 on every update()
            call (so the first update() call runs mmasub with iter=1).
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
    move: float = 0.5,
    asyinit: float = 0.5,
    asydecr: float = 0.7,
    asyincr: float = 1.2,
    asymin: float = 0.01,
    asymax: float = 10.0,
    raa0: float = 1e-5,
    albefa: float = 0.1,
) -> optax.GradientTransformationExtraArgs:
    """MMA (Method of Moving Asymptotes) optimizer, drop-in compatible with optax.

    A literal replacement for optax.adam(...)/optax.nadam(...) in fdtdx training loops:
    same `.init(params)` / `.update(grads, state, params)` / `optax.apply_updates(...)`
    calling convention. Box constraints are handled natively by MMA's own xmin/xmax
    asymptote machinery -- unlike ADAM, no manual
    `jax.tree_util.tree_map(lambda p: jnp.clip(p, 0, 1), params)` is needed afterwards.

    Requires the optional `mmapy` package. Install with: pip install fdtdx[mma]

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
        move (float): Maximum allowed change per coordinate per iteration, as a fraction
            of (xmax - xmin). Defaults to 0.5.
        asyinit (float): Initial distance of the moving asymptotes from xval, as a
            fraction of (xmax - xmin), used for iterations 1-2. Defaults to 0.5.
        asydecr (float): Asymptote shrink factor when oscillating. Defaults to 0.7.
        asyincr (float): Asymptote growth factor when moving steadily. Defaults to 1.2.
        asymin (float): Minimum asymptote distance, as a fraction of (xmax - xmin).
            Defaults to 0.01.
        asymax (float): Maximum asymptote distance, as a fraction of (xmax - xmin).
            Defaults to 10.0.
        raa0 (float): Small positive constant in the MMA approximation function
            enforcing strict convexity. Defaults to 1e-5.
        albefa (float): Factor controlling how close the "move limits" alfa/beta may get
            to the asymptotes low/upp. Defaults to 0.1.

    Returns:
        optax.GradientTransformationExtraArgs: An optax-compatible optimizer. `.update()`
            additionally accepts optional `fval`/`dfdx` kwargs for n_constraints > 0 use.

    Raises:
        ImportError: If mmapy is not installed.
    """
    try:
        import mmapy
    except ImportError as exc:
        raise ImportError("mmapy is required for the MMA optimizer. Install it with: pip install fdtdx[mma]") from exc

    hyperparams = dict(
        move=move,
        asyinit=asyinit,
        asydecr=asydecr,
        asyincr=asyincr,
        asymin=asymin,
        asymax=asymax,
        raa0=raa0,
        albefa=albefa,
    )

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

        # mmapy's own type stub under-declares mmasub's return arity (10 vs the real 11
        # values at runtime) -- this is a real upstream stub inaccuracy, not a bug here.
        xmma, ymma, zmma, lam, xsi, eta, mu_out, zet, s, low, upp = mmapy.mmasub(  # type: ignore
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
            **hyperparams,
        )
        del ymma, zmma, lam, xsi, eta, mu_out, zet, s  # unused MMA dual/auxiliary outputs

        # Pass the raw float64 numpy array straight into unravel_fn -- do NOT wrap it in
        # jnp.asarray(..., dtype=jnp.float64) first. unravel_fn restores each leaf's
        # original dtype automatically, whereas manually casting via jnp.asarray while
        # jax x64 mode is off (fdtdx's default) triggers a "truncated to float32"
        # UserWarning on every single update() call.
        new_params = unravel_fn(xmma.reshape(-1))
        updates = jax.tree_util.tree_map(lambda new, old: new - old, new_params, params)
        new_state = MMAState(iter=new_iter, xold1=xval, xold2=state.xold1, low=low, upp=upp)
        return updates, new_state

    # optax's bundled type stubs assume state is Array-like/iterable; MMAState is a plain
    # (deliberately non-pytree) dataclass, which optax's real runtime accepts as an opaque
    # state object just fine -- see MMAState's docstring for why it isn't a pytree.
    return optax.GradientTransformationExtraArgs(init_fn, update_fn)  # type: ignore
