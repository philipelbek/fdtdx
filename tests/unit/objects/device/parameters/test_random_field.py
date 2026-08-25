"""Tests for objects/device/parameters/random_field.py - stochastic eta field transforms."""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.materials import Material
from fdtdx.objects.device.parameters.random_field import (
    FieldTanhProjection,
    RandomEtaFieldGenerator,
    generate_gaussian_random_field,
    map_gaussian_to_uniform_range,
)


@pytest.fixture
def two_materials():
    """Two materials fixture."""
    return {
        "Air": Material(permittivity=1.0),
        "Silicon": Material(permittivity=11.7),
    }


@pytest.fixture
def dummy_config():
    """Minimal simulation config."""
    return SimulationConfig(
        time=100e-15,
        grid=UniformGrid(spacing=500e-9),
        backend="cpu",
    )


class TestRandomEtaFieldGeneratorClass:
    """Tests for RandomEtaFieldGenerator class."""

    def test_shape_and_key_propagation(self, two_materials, dummy_config):
        transform = RandomEtaFieldGenerator(correlation_length=2e-6)
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4), "eta": (4, 4, 4)},
        )
        assert transform._input_shape == {"params": (4, 4, 4)}
        assert transform._output_shape == {"params": (4, 4, 4), "eta": (4, 4, 4)}

    def test_call_missing_field_key_raises_error(self, two_materials, dummy_config):
        transform = RandomEtaFieldGenerator(correlation_length=2e-6)
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4), "eta": (4, 4, 4)},
        )
        params = {"params": jnp.ones((4, 4, 4)) * 0.5}
        with pytest.raises(Exception, match="field_key"):
            transform(params)  # Missing field_key

    def test_pass_through_untouched(self, two_materials, dummy_config):
        transform = RandomEtaFieldGenerator(correlation_length=2e-6)
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4), "eta": (4, 4, 4)},
        )
        data = jax.random.uniform(jax.random.PRNGKey(0), (4, 4, 4))
        params = {"params": data}

        result = transform(params, field_key=jax.random.PRNGKey(1))

        assert jnp.array_equal(result["params"], data)

    def test_determinism(self, two_materials, dummy_config):
        transform = RandomEtaFieldGenerator(correlation_length=2e-6)
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4), "eta": (4, 4, 4)},
        )
        params = {"params": jnp.ones((4, 4, 4)) * 0.5}

        result1 = transform(params, field_key=jax.random.PRNGKey(42))
        result2 = transform(params, field_key=jax.random.PRNGKey(42))
        result3 = transform(params, field_key=jax.random.PRNGKey(43))

        assert jnp.array_equal(result1["eta"], result2["eta"])
        assert not jnp.array_equal(result1["eta"], result3["eta"])

    def test_eta_bounds(self, two_materials, dummy_config):
        transform = RandomEtaFieldGenerator(correlation_length=2e-6, eta_mean=0.5, eta_deviation=0.2)
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(8, 8, 8),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (8, 8, 8), "eta": (8, 8, 8)},
        )
        params = {"params": jnp.ones((8, 8, 8)) * 0.5}

        result = transform(params, field_key=jax.random.PRNGKey(0))

        assert jnp.all(result["eta"] >= 0.3 - 1e-5)
        assert jnp.all(result["eta"] <= 0.7 + 1e-5)

    def test_dtype_is_float32(self, two_materials, dummy_config):
        transform = RandomEtaFieldGenerator(correlation_length=2e-6)
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4), "eta": (4, 4, 4)},
        )
        params = {"params": jnp.ones((4, 4, 4), dtype=jnp.float32) * 0.5}

        result = transform(params, field_key=jax.random.PRNGKey(0))

        assert result["eta"].dtype == jnp.float32
        assert result["params"].dtype == jnp.float32

    @pytest.mark.parametrize("correlation_length", [2e-6, (2e-6, 3e-6, 1e-6)])
    def test_scalar_and_tuple_correlation_length(self, two_materials, dummy_config, correlation_length):
        transform = RandomEtaFieldGenerator(correlation_length=correlation_length)
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4), "eta": (4, 4, 4)},
        )
        params = {"params": jnp.ones((4, 4, 4)) * 0.5}

        result = transform(params, field_key=jax.random.PRNGKey(0))

        assert result["eta"].shape == (4, 4, 4)


class TestFieldTanhProjectionClass:
    """Tests for FieldTanhProjection class."""

    def test_shape_propagation(self, two_materials, dummy_config):
        transform = FieldTanhProjection()
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4)},
        )
        assert transform._input_shape == {"params": (4, 4, 4), "eta": (4, 4, 4)}
        assert transform._output_shape == {"params": (4, 4, 4)}

    def test_call_missing_beta_raises_error(self, two_materials, dummy_config):
        transform = FieldTanhProjection()
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4)},
        )
        params = {"params": jnp.ones((4, 4, 4)) * 0.5, "eta": jnp.ones((4, 4, 4)) * 0.5}

        with pytest.raises(Exception, match="beta parameter"):
            transform(params)  # Missing beta

    def test_call_missing_eta_raises_error(self, two_materials, dummy_config):
        transform = FieldTanhProjection()
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4)},
        )
        params = {"params": jnp.ones((4, 4, 4)) * 0.5}  # Missing "eta"

        with pytest.raises(Exception, match="eta"):
            transform(params, beta=5.0)

    def test_call_projects_with_field_eta(self, two_materials, dummy_config):
        transform = FieldTanhProjection()
        transform = transform.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(2, 1, 1),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (2, 1, 1)},
        )
        x = jnp.array([0.2, 0.8]).reshape(2, 1, 1)
        eta = jnp.array([0.1, 0.9]).reshape(2, 1, 1)

        result = transform({"params": x, "eta": eta}, beta=50.0)

        # x[0]=0.2 is above its threshold eta[0]=0.1 -> projects towards 1
        assert result["params"][0, 0, 0] > 0.5
        # x[1]=0.8 is below its threshold eta[1]=0.9 -> projects towards 0
        assert result["params"][1, 0, 0] < 0.5


class TestRandomEtaFieldGeneratorGradients:
    """Gradient-flow / NaN-safety tests through the full generator -> projection chain."""

    @pytest.mark.parametrize("beta", [0.0, 5.0, jnp.inf])
    def test_gradient_no_nan(self, two_materials, dummy_config, beta):
        generator = RandomEtaFieldGenerator(correlation_length=2e-6)
        generator = generator.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4), "eta": (4, 4, 4)},
        )
        projection = FieldTanhProjection()
        projection = projection.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4)},
        )

        def loss(x):
            out = generator({"params": x}, field_key=jax.random.PRNGKey(0))
            out = projection(out, beta=beta)
            return out["params"].sum()

        x = jnp.linspace(0.0, 1.0, 64).reshape(4, 4, 4)
        grad = jax.grad(loss)(x)
        assert not jnp.any(jnp.isnan(grad)), f"NaN in gradient at beta={beta}: {grad}"

    def test_gradient_only_flows_through_data_not_eta(self, two_materials, dummy_config):
        """The eta field depends only on field_key/hyperparameters, not on the design
        parameter -- so the gradient of the projected output w.r.t. the design parameter
        must match plain tanh_projection with a constant eta at each voxel."""
        from fdtdx.objects.device.parameters.projection import tanh_projection

        generator = RandomEtaFieldGenerator(correlation_length=2e-6)
        generator = generator.init_module(
            config=dummy_config,
            materials=two_materials,
            matrix_voxel_grid_shape=(4, 4, 4),
            single_voxel_size=(1e-6, 1e-6, 1e-6),
            output_shape={"params": (4, 4, 4), "eta": (4, 4, 4)},
        )
        x = jnp.linspace(0.0, 1.0, 64).reshape(4, 4, 4)
        eta_field = generator({"params": x}, field_key=jax.random.PRNGKey(0))["eta"]

        def via_transform(v):
            out = generator({"params": v}, field_key=jax.random.PRNGKey(0))
            return tanh_projection(out["params"], beta=5.0, eta=out["eta"]).sum()

        def via_direct(v):
            return tanh_projection(v, beta=5.0, eta=eta_field).sum()

        grad_transform = jax.grad(via_transform)(x)
        grad_direct = jax.grad(via_direct)(x)
        assert jnp.allclose(grad_transform, grad_direct)


class TestGenerateGaussianRandomField:
    """Statistical correctness tests for the FFT moving-average random field generator."""

    def test_output_shape_matches_request(self):
        shape = (5, 7, 3)  # deliberately awkward (non-power-of-two) sizes
        field = generate_gaussian_random_field(jax.random.PRNGKey(0), shape, (2.0, 2.0, 2.0))
        assert field.shape == shape

    def test_deterministic_given_key(self):
        shape = (4, 4, 4)
        field1 = generate_gaussian_random_field(jax.random.PRNGKey(5), shape, (2.0, 2.0, 2.0))
        field2 = generate_gaussian_random_field(jax.random.PRNGKey(5), shape, (2.0, 2.0, 2.0))
        assert jnp.array_equal(field1, field2)

    def test_ensemble_mean_and_variance_match_configuration(self):
        """Draw many independent realizations and check the empirical mean/variance at a
        fixed grid point matches the configured mean_value/standard_deviation**2 -- this is
        the correctness-critical FFT-normalization self-consistency check."""
        shape = (6, 6, 6)
        n_samples = 300
        standard_deviation = 1.5
        mean_value = 0.3
        correlation_length_cells = (2.0, 2.0, 2.0)

        keys = jax.random.split(jax.random.PRNGKey(0), n_samples)
        samples = jnp.stack(
            [
                generate_gaussian_random_field(
                    k, shape, correlation_length_cells, standard_deviation=standard_deviation, mean_value=mean_value
                )
                for k in keys
            ]
        )
        voxel_values = samples[:, 0, 0, 0]

        assert jnp.abs(jnp.mean(voxel_values) - mean_value) < 0.15
        assert jnp.abs(jnp.std(voxel_values) - standard_deviation) < 0.25


class TestMapGaussianToUniformRange:
    """Statistical correctness tests for the inverse-CDF eta-field mapping."""

    def test_output_within_bounds(self):
        shape = (6, 6, 6)
        lower, upper = 0.3, 0.7
        gaussian_field = generate_gaussian_random_field(jax.random.PRNGKey(0), shape, (2.0, 2.0, 2.0))
        eta = map_gaussian_to_uniform_range(
            gaussian_field, mean_value=0.0, standard_deviation=1.0, lower_bound=lower, upper_bound=upper
        )
        assert jnp.all(eta >= lower - 1e-5)
        assert jnp.all(eta <= upper + 1e-5)

    def test_marginal_is_approximately_uniform(self):
        """Draw many independent realizations and check the empirical marginal
        distribution of a fixed grid point's eta value is approximately flat on its bounds
        (self-consistency between the covariance kernel's std and the CDF normalization)."""
        shape = (6, 6, 6)
        n_samples = 300
        standard_deviation = 1.0
        mean_value = 0.0
        lower, upper = 0.3, 0.7

        keys = jax.random.split(jax.random.PRNGKey(1), n_samples)
        etas = jnp.stack(
            [
                map_gaussian_to_uniform_range(
                    generate_gaussian_random_field(
                        k, shape, (1.5, 1.5, 1.5), standard_deviation=standard_deviation, mean_value=mean_value
                    ),
                    mean_value=mean_value,
                    standard_deviation=standard_deviation,
                    lower_bound=lower,
                    upper_bound=upper,
                )
                for k in keys
            ]
        )
        voxel_values = etas[:, 0, 0, 0]

        midpoint = 0.5 * (lower + upper)
        fraction_below_midpoint = jnp.mean(voxel_values < midpoint)
        assert jnp.abs(fraction_below_midpoint - 0.5) < 0.15
