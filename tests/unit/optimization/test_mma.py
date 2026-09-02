"""Tests for optimization/mma.py - MMA (Method of Moving Asymptotes) optax-compatible optimizer."""

import jax
import jax.numpy as jnp
import optax
import pytest

from fdtdx.optimization.mma import mma

# ── Import guard ─────────────────────────────────────────────


def test_mma_raises_import_error_when_mmapy_unavailable():
    """Without mmapy installed, mma() raises ImportError with a helpful message at
    construction time (not deferred to the first .update() call)."""
    from unittest.mock import patch

    with patch.dict("sys.modules", {"mmapy": None}):
        with pytest.raises(ImportError, match="mmapy"):
            mma()


# ── Convergence (m=0, pure box-constrained quadratic) ────────


def test_mma_converges_on_bound_constrained_quadratic():
    """Minimizes f(x) = sum((x - target)**2), target inside [0, 1]^n; converges near target."""
    pytest.importorskip("mmapy")
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


# ── Native box-bound handling (no manual clipping needed) ────


def test_mma_respects_bounds_when_target_outside_range():
    """Target outside [0, 1] -- iterates must stay in-bounds at every step, natively."""
    pytest.importorskip("mmapy")
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


# ── Pytree structure round-trip (mirrors real ParameterContainer) ─


def test_mma_preserves_nested_pytree_structure_and_dtype():
    """Nested-dict ParameterContainer structure/dtype survives init() + update()."""
    pytest.importorskip("mmapy")
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


# ── optax.apply_updates compatibility ─────────────────────────


def test_mma_updates_compatible_with_optax_apply_updates():
    """Returned `updates` plug directly into real optax.apply_updates, additive convention."""
    pytest.importorskip("mmapy")
    params = {"device": jnp.full((4,), 0.5, dtype=jnp.float32)}
    grads = {"device": jnp.array([0.1, -0.1, 0.2, -0.2], dtype=jnp.float32)}

    optimizer = mma(lower_bound=0.0, upper_bound=1.0)
    state = optimizer.init(params)
    updates, state = optimizer.update(grads, state, params)
    new_params = optax.apply_updates(params, updates)

    assert jnp.allclose(new_params["device"], params["device"] + updates["device"])
    assert jnp.all(new_params["device"] >= 0.0)
    assert jnp.all(new_params["device"] <= 1.0)
