import math
from typing import Sequence

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import autoinit, frozen_field, frozen_private_field
from fdtdx.objects.device.parameters.projection import tanh_projection
from fdtdx.objects.device.parameters.transform import ParameterTransformation
from fdtdx.typing import ParameterType


def _resolve_correlation_length_cells(
    correlation_length: float | tuple[float, float, float],
    single_voxel_size: tuple[float, float, float],
) -> tuple[float, float, float]:
    if isinstance(correlation_length, (int, float)):
        lengths = (float(correlation_length),) * 3
    else:
        lengths = tuple(float(v) for v in correlation_length)
        if len(lengths) != 3:
            raise Exception(f"correlation_length must be a scalar or a 3-tuple, but got: {correlation_length}")
    return (
        lengths[0] / single_voxel_size[0],
        lengths[1] / single_voxel_size[1],
        lengths[2] / single_voxel_size[2],
    )


def generate_gaussian_random_field(
    field_key: jax.Array,
    shape: tuple[int, ...],
    correlation_length_cells: tuple[float, float, float],
    standard_deviation: float = 1.0,
    mean_value: float = 0.0,
) -> jax.Array:
    """Generate a spatially-correlated 3D Gaussian random field via the moving-average
    (FFT spectral) method.

    White noise is filtered in Fourier space by the square root of the power spectral
    density of a squared-exponential covariance kernel
    ``C(dx,dy,dz) = standard_deviation**2 * exp(-((dx/Lx)**2 + (dy/Ly)**2 + (dz/Lz)**2))``,
    which is mathematically equivalent to convolving white noise with a Gaussian moving-average
    kernel but is far cheaper via the FFT. The domain is zero-padded to twice its size on each
    axis before the FFT and cropped back to ``shape`` afterwards, which pushes the FFT's implicit
    periodic wrap-around at least one domain-width away from the returned region -- negligible
    correlation leakage as long as the correlation length is not much larger than the domain.

    Args:
        field_key (jax.Array): JAX PRNG key. A fresh key produces a statistically independent
            realization of the random field.
        shape (tuple[int, ...]): Shape of the returned field. Must be 3D.
        correlation_length_cells (tuple[float, float, float]): Correlation length (1/e distance
            of the covariance kernel) per axis, in grid cells.
        standard_deviation (float): Standard deviation of the generated field. Defaults to 1.0.
        mean_value (float): Mean value of the generated field. Defaults to 0.0.

    Returns:
        jax.Array: Real-valued float32 array of shape ``shape``.
    """
    if len(shape) != 3:
        raise Exception(f"generate_gaussian_random_field expects a 3D shape, but got: {shape}")
    padded_shape = tuple(2 * n for n in shape)

    offsets = [jnp.arange(n) - n // 2 for n in padded_shape]
    grids = jnp.meshgrid(*offsets, indexing="ij")
    r2 = sum((g / length) ** 2 for g, length in zip(grids, correlation_length_cells))
    covariance_grid = standard_deviation**2 * jnp.exp(-r2)

    spectrum = jnp.fft.fftn(jnp.fft.fftshift(covariance_grid))
    white_noise = jax.random.normal(field_key, padded_shape, dtype=jnp.float32)
    filtered = jnp.fft.ifftn(jnp.sqrt(spectrum) * jnp.fft.fftn(white_noise))

    crop = tuple(slice(0, n) for n in shape)
    return jnp.real(filtered)[crop].astype(jnp.float32) + mean_value


def map_gaussian_to_uniform_range(
    gaussian_field: jax.Array,
    mean_value: float,
    standard_deviation: float,
    lower_bound: float,
    upper_bound: float,
) -> jax.Array:
    """Map a Gaussian field to a bounded field via the probability integral transform.

    Pushes the field through its own (Gaussian) CDF, giving a spatially-correlated field
    whose marginal distribution is uniform on ``[0, 1]``, then rescales to
    ``[lower_bound, upper_bound]``. Spatial correlation of the input field is preserved;
    only the marginal distribution changes.

    Args:
        gaussian_field (jax.Array): Field with (approximately) N(mean_value, standard_deviation**2)
            marginal distribution, e.g. from :func:`generate_gaussian_random_field`.
        mean_value (float): Mean of ``gaussian_field``. Must match the value used to generate it.
        standard_deviation (float): Standard deviation of ``gaussian_field``. Must match the value
            used to generate it, or the output marginal will not actually be uniform.
        lower_bound (float): Lower bound of the output range.
        upper_bound (float): Upper bound of the output range.

    Returns:
        jax.Array: Field with the same shape as ``gaussian_field``, valued in
            ``[lower_bound, upper_bound]``.
    """
    cdf = 0.5 * (1 + jax.scipy.special.erf((gaussian_field - mean_value) / (standard_deviation * math.sqrt(2))))
    return cdf * (upper_bound - lower_bound) + lower_bound


@autoinit
class RandomEtaFieldGenerator(ParameterTransformation):
    """Generates a spatially-correlated random eta (projection threshold) field.

    Draws a Gaussian random field via the moving-average (FFT spectral) method and maps it
    through the inverse Gaussian CDF into a bounded eta field, used to model spatially-varying
    fabrication uncertainty (e.g. over/under-etch) in a subsequent projection step such as
    :class:`FieldTanhProjection`.

    The input design array is passed through unchanged; the generated eta field is added to the
    parameter dict under ``eta_key`` alongside it.

    Notes:
        The call method requires a ``field_key`` (a ``jax.Array`` PRNG key) as additional keyword
        argument, so that a fresh, independent eta field can be drawn each time (e.g. once per
        stochastic sample per training iteration).
    """

    #: Correlation length (1/e distance) of the random field, in meters. A scalar applies
    #: isotropically to all three axes; a 3-tuple specifies ``(Lx, Ly, Lz)`` independently.
    correlation_length: float | tuple[float, float, float] = frozen_field()

    #: Standard deviation of the underlying Gaussian field before the inverse-CDF mapping.
    #: Defaults to 1.0. Does not affect the marginal distribution of the output eta field
    #: (which is always uniform on its bounds by construction), only used self-consistently
    #: to build the covariance kernel and to normalize the inverse-CDF step.
    standard_deviation: float = frozen_field(default=1.0)

    #: Mean value of the underlying Gaussian field before the inverse-CDF mapping. Defaults to 0.0.
    mean_value: float = frozen_field(default=0.0)

    #: Center of the output eta field's range. Defaults to 0.5.
    eta_mean: float = frozen_field(default=0.5)

    #: Half-width of the output eta field's range, i.e. the output is bounded in
    #: ``[eta_mean - eta_deviation, eta_mean + eta_deviation]``. Defaults to 0.2.
    eta_deviation: float = frozen_field(default=0.2)

    #: Key under which the generated eta field is stored in the parameter dict. Defaults to "eta".
    eta_key: str = frozen_field(default="eta")

    _fixed_input_type: ParameterType | Sequence[ParameterType] | None = frozen_private_field(
        default=ParameterType.CONTINUOUS
    )
    _check_single_array: bool = frozen_private_field(default=True)

    def _get_input_shape_impl(
        self,
        output_shape: dict[str, tuple[int, ...]],
    ) -> dict[str, tuple[int, ...]]:
        if len(output_shape) != 2 or self.eta_key not in output_shape:
            raise Exception(
                f"RandomEtaFieldGenerator expects its output to be exactly one data array plus an "
                f"'{self.eta_key}' array, but got: {output_shape}"
            )
        data_key = next(k for k in output_shape if k != self.eta_key)
        if output_shape[data_key] != output_shape[self.eta_key]:
            raise Exception(
                f"RandomEtaFieldGenerator expects the data array and the '{self.eta_key}' array to "
                f"have the same shape, but got: {output_shape}"
            )
        return {data_key: output_shape[data_key]}

    def _get_output_type_impl(
        self,
        input_type: dict[str, ParameterType],
    ) -> dict[str, ParameterType]:
        data_key = next(iter(input_type))
        return {data_key: input_type[data_key], self.eta_key: ParameterType.CONTINUOUS}

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        if "field_key" not in kwargs:
            raise Exception(
                "RandomEtaFieldGenerator needs the field_key parameter (a jax PRNG key) as additional keyword argument!"
            )
        field_key = kwargs["field_key"]
        data_key = next(iter(params.keys()))
        data = params[data_key]

        correlation_length_cells = _resolve_correlation_length_cells(self.correlation_length, self._single_voxel_size)
        gaussian_field = generate_gaussian_random_field(
            field_key=field_key,
            shape=data.shape,
            correlation_length_cells=correlation_length_cells,
            standard_deviation=self.standard_deviation,
            mean_value=self.mean_value,
        )
        eta_field = map_gaussian_to_uniform_range(
            gaussian_field,
            mean_value=self.mean_value,
            standard_deviation=self.standard_deviation,
            lower_bound=self.eta_mean - self.eta_deviation,
            upper_bound=self.eta_mean + self.eta_deviation,
        ).astype(jnp.float32)

        return {data_key: data, self.eta_key: eta_field}


@autoinit
class FieldTanhProjection(ParameterTransformation):
    """Tanh projection filter with a spatially-varying (field-valued) threshold.

    Identical to :class:`~fdtdx.objects.device.parameters.projection.TanhProjection`, except the
    threshold ``eta`` is read from an array in the parameter dict (under ``eta_key``, e.g. produced
    by :class:`RandomEtaFieldGenerator`) instead of being a fixed scalar hyperparameter -- allowing
    the threshold to vary per-voxel and per-sample.

    Notes:
        The call method requires a ``beta`` parameter as a keyword argument passed to the parameter
        transformation, and expects an ``eta_key`` array to already be present in its input.
    """

    #: Key under which the eta field is read from the parameter dict. Defaults to "eta".
    eta_key: str = frozen_field(default="eta")

    _fixed_input_type: ParameterType | Sequence[ParameterType] | None = frozen_private_field(
        default=ParameterType.CONTINUOUS
    )

    def _get_input_shape_impl(
        self,
        output_shape: dict[str, tuple[int, ...]],
    ) -> dict[str, tuple[int, ...]]:
        if len(output_shape) != 1:
            raise Exception(f"FieldTanhProjection expects its output to be a single array, but got: {output_shape}")
        data_key, shape = next(iter(output_shape.items()))
        return {data_key: shape, self.eta_key: shape}

    def _get_output_type_impl(
        self,
        input_type: dict[str, ParameterType],
    ) -> dict[str, ParameterType]:
        if len(input_type) != 2 or self.eta_key not in input_type:
            raise Exception(
                f"FieldTanhProjection expects its input to be exactly one data array plus an "
                f"'{self.eta_key}' array, but got: {input_type}"
            )
        data_key = next(k for k in input_type if k != self.eta_key)
        return {data_key: input_type[data_key]}

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        if "beta" not in kwargs:
            raise Exception("FieldTanhProjection needs the beta parameter as additional keyword argument!")
        if self.eta_key not in params:
            raise Exception(
                f"FieldTanhProjection expects an '{self.eta_key}' array in its input, but got: {list(params.keys())}"
            )
        beta = kwargs["beta"]
        data_key = next(k for k in params if k != self.eta_key)
        result = tanh_projection(params[data_key], beta, params[self.eta_key])
        return {data_key: result}
