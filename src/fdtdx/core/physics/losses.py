from typing import Any, Sequence

import jax
import jax.numpy as jnp


def weighted_p_mean(
    values: jax.Array,
    p: float | jax.Array,
    weights: jax.Array | None = None,
    eps: float = 1e-12,
) -> jax.Array:
    """Combine per-sample objectives into a single scalar via a weighted p-mean.

    Computes ``(sum_i w_i * f_i**p) ** (1/p)``, evaluated in log-space via ``logsumexp``
    for numerical stability at large ``|p|``. Useful for aggregating the objectives of several
    stochastic samples (e.g. different random realizations of a fabrication-uncertainty field)
    into one differentiable scalar loss: with uniform weights and ``p=1`` this is a plain Monte
    Carlo average; increasing ``|p|`` smoothly pushes the aggregate towards a worst-case
    (``p -> inf``) or best-case (``p -> -inf``) objective.

    Args:
        values (jax.Array): Per-sample objective values, shape ``(K,)``. Must be strictly positive
            (values at or below ``eps`` are floored, which zeroes their gradient contribution).
        p (float | jax.Array): p-mean exponent. ``p=0`` is the weighted geometric-mean limit,
            handled separately (double-``where`` trick, evaluated everywhere so no branch produces
            NaN gradients).
        weights (jax.Array | None): Per-sample weights, shape ``(K,)``. Defaults to uniform
            (``1/K`` each), i.e. an unweighted Monte Carlo estimate.
        eps (float): Floor applied to ``values`` before taking the log, to keep the result finite.
            Defaults to 1e-12.

    Returns:
        jax.Array: Scalar combined objective.
    """
    if weights is None:
        weights = jnp.full(values.shape, 1.0 / values.shape[0])
    log_values = jnp.log(jnp.clip(values, eps, None))

    is_zero = p == 0
    safe_p = jnp.where(is_zero, 1.0, p)
    log_weighted_sum = jax.scipy.special.logsumexp(safe_p * log_values, b=weights)
    p_mean = jnp.exp(log_weighted_sum / safe_p)

    geometric_mean = jnp.exp(jnp.sum(weights * log_values))
    return jnp.where(is_zero, geometric_mean, p_mean)


def metric_efficiency(
    detector_states: dict[str, dict[str, jax.Array]],
    in_names: Sequence[str],
    out_names: Sequence[str],
    metric_name: str,
) -> tuple[jax.Array, dict[str, Any]]:
    """Calculate efficiency metrics between input and output detectors.

    Computes efficiency ratios between input and output detectors by comparing their
    metric values (e.g. energy, power). For each input-output detector pair, calculates
    the ratio of output/input metric values.

    Args:
        detector_states (dict[str, dict[str, jax.Array]]): Dictionary mapping detector names to their state dictionaries,
            which contain metric values as JAX arrays
        in_names (Sequence[str]): Names of input detectors to use as reference
        out_names (Sequence[str]): Names of output detectors to compare against inputs
        metric_name (str): Name of the metric to compare between detectors (e.g. "energy")

    Returns:
        tuple[jax.Array, dict[str, Any]]: tuple containing:
            - jax.Array: Mean efficiency across all input-output pairs
            - dict: Additional info including individual metric values and efficiencies
              with keys like:

                - "{detector}_{metric}" for raw metric values
                - "{out}_{by}_{in}_efficiency" for individual efficiency ratios
    """
    efficiencies, info = [], {}
    for in_name in in_names:
        in_value = jax.lax.stop_gradient(detector_states[in_name][metric_name].mean())
        info[f"{in_name}_{metric_name}"] = in_value
        for out_name in out_names:
            out_value = detector_states[out_name][metric_name].mean()
            eff = jnp.where(in_value == 0, 0, out_value / in_value)
            efficiencies.append(eff)
            info[f"{out_name}_{metric_name}"] = out_value
            info[f"{out_name}_by_{in_name}_efficiency"] = eff
    objective = jnp.mean(jnp.asarray(efficiencies))
    return objective, info
