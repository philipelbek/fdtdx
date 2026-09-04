"""Tests for optimization/mma.py - MMA (Method of Moving Asymptotes) optax-compatible optimizers."""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fdtdx.optimization.mma import mma, mma_unconstrained

# ── Convergence (m=0, pure box-constrained quadratic) ────────


def test_mma_converges_on_bound_constrained_quadratic():
    """Minimizes f(x) = sum((x - target)**2), target inside [0, 1]^n; converges near target."""
    target = jnp.array([0.2, 0.8, 0.5, 0.3, 0.65])
    params = {"device": 0.5 * jnp.ones((5,))}

    optimizer = mma(lower_bound=0.0, upper_bound=1.0)
    state = optimizer.init(params)

    def loss(p):
        return jnp.sum((p["device"] - target) ** 2)

    for _ in range(100):
        grads = jax.grad(loss)(params)
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)

    assert jnp.allclose(params["device"], target, atol=1e-2)


def test_mma_unconstrained_converges_on_bound_constrained_quadratic():
    """Same convergence check as above, for the closed-form box-only variant."""
    target = jnp.array([0.2, 0.8, 0.5, 0.3, 0.65])
    params = {"device": 0.5 * jnp.ones((5,))}

    optimizer = mma_unconstrained(lower_bound=0.0, upper_bound=1.0, move=0.5)
    state = optimizer.init(params)

    def loss(p):
        return jnp.sum((p["device"] - target) ** 2)

    for _ in range(100):
        grads = jax.grad(loss)(params)
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)

    assert jnp.allclose(params["device"], target, atol=1e-2)


# ── Native box-bound handling (no manual clipping needed) ────


def test_mma_respects_bounds_when_target_outside_range():
    """Target outside [0, 1] -- iterates must stay in-bounds at every step, natively."""
    target = jnp.array([-1.0, 2.0, -0.5, 1.5])
    params = {"device": 0.5 * jnp.ones((4,))}

    optimizer = mma(lower_bound=0.0, upper_bound=1.0)
    state = optimizer.init(params)

    def loss(p):
        return jnp.sum((p["device"] - target) ** 2)

    for _ in range(50):
        grads = jax.grad(loss)(params)
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        assert jnp.all(params["device"] >= -1e-8)
        assert jnp.all(params["device"] <= 1.0 + 1e-8)

    assert jnp.allclose(params["device"], jnp.array([0.0, 1.0, 0.0, 1.0]), atol=5e-2)


def test_mma_unconstrained_respects_bounds_when_target_outside_range():
    """Same in-bounds check as above, for the closed-form box-only variant."""
    target = jnp.array([-1.0, 2.0, -0.5, 1.5])
    params = {"device": 0.5 * jnp.ones((4,))}

    optimizer = mma_unconstrained(lower_bound=0.0, upper_bound=1.0, move=0.5)
    state = optimizer.init(params)

    def loss(p):
        return jnp.sum((p["device"] - target) ** 2)

    for _ in range(50):
        grads = jax.grad(loss)(params)
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        assert jnp.all(params["device"] >= -1e-8)
        assert jnp.all(params["device"] <= 1.0 + 1e-8)

    assert jnp.allclose(params["device"], jnp.array([0.0, 1.0, 0.0, 1.0]), atol=5e-2)


# ── Pytree structure round-trip (mirrors real ParameterContainer) ─


def test_mma_preserves_nested_pytree_structure_and_dtype():
    """Nested-dict ParameterContainer structure/dtype survives init() + update()."""
    params = {
        "device_a": jnp.full((3, 3), 0.5, dtype=jnp.float32),
        "device_b": {
            "params": jnp.full((2, 2), 0.5, dtype=jnp.float32),
            "eta": jnp.full((2, 2), 0.3, dtype=jnp.float32),
        },
    }
    grads = jax.tree_util.tree_map(lambda p: jnp.ones_like(p) * 0.1, params)

    optimizer = mma(lower_bound=0.0, upper_bound=1.0)
    state = optimizer.init(params)
    updates, state = optimizer.update(grads, state, params)
    new_params = optax.apply_updates(params, updates)

    assert jax.tree_util.tree_structure(new_params) == jax.tree_util.tree_structure(params)
    for leaf_new, leaf_old in zip(jax.tree_util.tree_leaves(new_params), jax.tree_util.tree_leaves(params)):
        assert leaf_new.shape == leaf_old.shape
        assert leaf_new.dtype == leaf_old.dtype


def test_mma_unconstrained_preserves_nested_pytree_structure_and_dtype():
    """Same pytree round-trip check as above, for the closed-form box-only variant."""
    params = {
        "device_a": jnp.full((3, 3), 0.5, dtype=jnp.float32),
        "device_b": {
            "params": jnp.full((2, 2), 0.5, dtype=jnp.float32),
            "eta": jnp.full((2, 2), 0.3, dtype=jnp.float32),
        },
    }
    grads = jax.tree_util.tree_map(lambda p: jnp.ones_like(p) * 0.1, params)

    optimizer = mma_unconstrained(lower_bound=0.0, upper_bound=1.0)
    state = optimizer.init(params)
    updates, state = optimizer.update(grads, state, params)
    new_params = optax.apply_updates(params, updates)

    assert jax.tree_util.tree_structure(new_params) == jax.tree_util.tree_structure(params)
    for leaf_new, leaf_old in zip(jax.tree_util.tree_leaves(new_params), jax.tree_util.tree_leaves(params)):
        assert leaf_new.shape == leaf_old.shape
        assert leaf_new.dtype == leaf_old.dtype


# ── optax.apply_updates compatibility ─────────────────────────


def test_mma_updates_compatible_with_optax_apply_updates():
    """Returned `updates` plug directly into real optax.apply_updates, additive convention."""
    params = {"device": jnp.full((4,), 0.5, dtype=jnp.float32)}
    grads = {"device": jnp.array([0.1, -0.1, 0.2, -0.2], dtype=jnp.float32)}

    optimizer = mma(lower_bound=0.0, upper_bound=1.0)
    state = optimizer.init(params)
    updates, state = optimizer.update(grads, state, params)
    new_params = optax.apply_updates(params, updates)

    assert jnp.allclose(new_params["device"], params["device"] + updates["device"])
    assert jnp.all(new_params["device"] >= 0.0)
    assert jnp.all(new_params["device"] <= 1.0)


def test_mma_unconstrained_move_limit_bounds_first_step():
    """move=0.1 caps how far the very first update can go, as a fraction of the box."""
    params = {"device": jnp.full((3,), 0.5, dtype=jnp.float32)}
    grads = {"device": jnp.array([10.0, -10.0, 10.0], dtype=jnp.float32)}

    optimizer = mma_unconstrained(lower_bound=0.0, upper_bound=1.0, move=0.1)
    state = optimizer.init(params)
    updates, state = optimizer.update(grads, state, params)
    new_params = optax.apply_updates(params, updates)

    assert jnp.all(new_params["device"] >= 0.5 - 0.1 - 1e-6)
    assert jnp.all(new_params["device"] <= 0.5 + 0.1 + 1e-6)


# ── Native Svanberg routines (raw numpy interface, no optax) ──


def test_mmasub_unconst_closed_form_at_first_iteration():
    """At iter=1 both the asymptotes and the descent direction are pinned down by
    closed-form formulas independent of subsolv -- check both directly for a 1-D case."""
    from fdtdx.optimization.mmasub_unconst import mmasub_unconst

    n = 1
    xval = np.array([[0.5]])
    xmin = np.array([[0.0]])
    xmax = np.array([[1.0]])
    df0dx = np.array([[2.0]])  # positive gradient -> minimizer moves below xval
    low = xmin.copy()
    upp = xmax.copy()

    xmma, low_out, upp_out = mmasub_unconst(n, 1, xval, xmin, xmax, xval, xval, 0.0, df0dx, low, upp, move=1.0)

    # iter<2.5 branch: low = xval - asyinit*(xmax-xmin), upp = xval + asyinit*(xmax-xmin), asyinit=0.5
    assert np.isclose(low_out.item(), 0.0)
    assert np.isclose(upp_out.item(), 1.0)
    assert 0.0 <= xmma.item() < 0.5


def test_mmasub_respects_general_constraint():
    """Minimize f0(x) = x subject to the single general constraint f1(x) = 0.3 - x <= 0
    (i.e. x >= 0.3); repeated mmasub iterations should converge to x* = 0.3. Exercises
    the m >= n branch of subsolv (m=1, n=1), complementing the m=0 (m < n) branch already
    covered by mma()'s box-only tests above."""
    from fdtdx.optimization.mmasub import mmasub

    n, m = 1, 1
    xmin = np.array([[0.0]])
    xmax = np.array([[1.0]])
    low = xmin.copy()
    upp = xmax.copy()
    x = np.array([[0.5]])
    xold1 = x.copy()
    xold2 = x.copy()
    a0 = 1.0
    a = np.zeros((m, 1))
    c = 1000.0 * np.ones((m, 1))
    d = np.zeros((m, 1))

    for it in range(1, 30):
        f0val = float(x.item())
        df0dx = np.array([[1.0]])
        fval = np.array([[0.3 - x.item()]])
        dfdx = np.array([[-1.0]])
        xmma, _ymma, _zmma, _lam, _xsi, _eta, _mu, _zet, _s, low, upp = mmasub(
            m, n, it, x, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d
        )
        xold2, xold1 = xold1, x
        x = xmma

    assert np.isclose(x.item(), 0.3, atol=1e-2)
