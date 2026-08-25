"""
This script demonstrates stochastic/robust inverse design in fdtdx: instead of optimizing a
single deterministic design, the device's projection threshold ("eta") is a spatially-correlated
random field, drawn fresh from a moving-average (FFT spectral) random field generator each training
step. This models spatially-varying fabrication uncertainty (e.g. over/under-etch). K independent
realizations are drawn per training step and combined via a weighted p-mean into a single scalar
loss, so the optimizer is pushed towards a design whose performance is robust across many random
fabrication outcomes, not just the nominal one.

The geometry (a corner component connecting two silicon waveguides) is the same as
`optimize_ceviche_corner.py`; the only difference is the device's parameter-transform chain and
the loss function's per-step sampling/aggregation. See that script for a non-stochastic baseline.
"""

import sys
import time

import chex
import jax
import jax.numpy as jnp
import optax
import pytreeclass as tc
from loguru import logger

import fdtdx


def main(
    seed: int,
    evaluation: bool,
):
    logger.info(f"{seed=}")

    exp_logger = fdtdx.Logger(
        experiment_name="stochastic_eta_corner",
        name=None,
    )
    key = jax.random.PRNGKey(seed=seed)

    wavelength = 1.55e-6
    period = fdtdx.constants.wavelength_to_period(wavelength)

    config = fdtdx.SimulationConfig(
        time=50e-15,
        grid=fdtdx.UniformGrid(spacing=20e-9),
        dtype=jnp.float32,
        courant_factor=0.99,
    )

    period_steps = round(period / config.time_step_duration)
    all_time_steps = list(range(config.time_steps_total))
    logger.info(f"{config.time_steps_total=}")
    logger.info(f"{period_steps=}")

    if not evaluation:
        gradient_config = fdtdx.GradientConfig(
            recorder=fdtdx.Recorder(
                modules=[
                    fdtdx.LinearReconstructEveryK(k=5),
                    fdtdx.DtypeConversion(dtype=jnp.float8_e4m3fnuz),
                ]
            )
        )
        config = config.aset("gradient_config", gradient_config)

    placement_constraints, object_list = [], []

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(2.7e-6, 2.7e-6, 1.5e-6),
    )
    object_list.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=10)
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    placement_constraints.extend(c_list)
    object_list.extend(list(bound_dict.values()))

    substrate = fdtdx.UniformMaterialObject(
        partial_real_shape=(None, None, 0.5e-6),
        material=fdtdx.Material(permittivity=fdtdx.constants.relative_permittivity_silica),
        color=fdtdx.colors.XKCD_ORANGE,
    )
    placement_constraints.append(
        substrate.place_relative_to(
            volume,
            axes=2,
            own_positions=-1,
            other_positions=-1,
        )
    )
    object_list.append(substrate)

    height = 400e-9
    width = 400e-9
    material_config = {
        "Air": fdtdx.Material(permittivity=fdtdx.constants.relative_permittivity_air),
        "Silicon": fdtdx.Material(permittivity=fdtdx.constants.relative_permittivity_silicon),
    }
    voxel_size = 20e-9
    device = fdtdx.Device(
        name="Device",
        partial_real_shape=(1.6e-6, 1.6e-6, height),
        materials=material_config,
        param_transforms=[
            # Draws a fresh correlated random eta field from `field_key` each call, and projects
            # the raw design parameter through a tanh threshold at that spatially-varying eta.
            fdtdx.RandomEtaFieldGenerator(
                correlation_length=100e-9,
                eta_deviation=0.2,
            ),
            fdtdx.FieldTanhProjection(),
        ],
        partial_voxel_real_shape=(voxel_size, voxel_size, height),
    )
    placement_constraints.append(
        device.place_relative_to(
            substrate,
            axes=(0, 1, 2),
            own_positions=(1, -1, -1),
            other_positions=(1, -1, 1),
            grid_margins=(-bound_cfg.thickness_grid_maxx, bound_cfg.thickness_grid_miny, 0),
            margins=(-0.2e-6, 0.2e-6, 0),
        )
    )
    object_list.append(device)

    waveguide_in = fdtdx.UniformMaterialObject(
        partial_real_shape=(None, width, height),
        material=material_config["Silicon"],
        color=fdtdx.colors.XKCD_LIGHT_BLUE,
    )
    placement_constraints.extend(
        [
            waveguide_in.place_at_center(device, axes=1),
            waveguide_in.extend_to(device, axis=0, direction="+"),
            waveguide_in.place_above(substrate),
        ]
    )
    object_list.append(waveguide_in)

    source = fdtdx.ModePlaneSource(
        partial_grid_shape=(1, None, None),
        wave_character=fdtdx.WaveCharacter(wavelength=wavelength),
        direction="+",
    )
    placement_constraints.extend(
        [
            source.place_relative_to(
                waveguide_in,
                axes=(0,),
                other_positions=(-1,),
                own_positions=(1,),
                grid_margins=(bound_cfg.thickness_grid_minx + 4,),
            )
        ]
    )
    object_list.append(source)

    waveguide_out = fdtdx.UniformMaterialObject(
        partial_real_shape=(width, None, height),
        material=material_config["Silicon"],
        color=fdtdx.colors.XKCD_LIGHT_BLUE,
    )
    placement_constraints.extend(
        [
            waveguide_out.place_at_center(device, axes=0),
            waveguide_out.extend_to(device, axis=1, direction="-"),
            waveguide_out.place_above(substrate),
        ]
    )
    object_list.append(waveguide_out)

    flux_in_detector = fdtdx.PoyntingFluxDetector(
        name="in flux",
        partial_grid_shape=(1, None, None),
        direction="+",
        switch=fdtdx.OnOffSwitch(fixed_on_time_steps=all_time_steps[7 * period_steps : 8 * period_steps]),
    )
    placement_constraints.append(
        flux_in_detector.place_relative_to(
            source,
            axes=0,
            own_positions=1,
            other_positions=1,
            grid_margins=2,
        )
    )
    object_list.append(flux_in_detector)

    flux_out_detector = fdtdx.PoyntingFluxDetector(
        name="out flux",
        partial_grid_shape=(None, 1, None),
        direction="+",
        switch=fdtdx.OnOffSwitch(fixed_on_time_steps=all_time_steps[-period_steps:]),
    )
    placement_constraints.append(
        flux_out_detector.place_relative_to(
            waveguide_out,
            axes=1,
            own_positions=1,
            other_positions=1,
            grid_margins=-bound_cfg.thickness_grid_maxy - 5,
        )
    )
    object_list.append(flux_out_detector)

    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=object_list,
        config=config,
        constraints=placement_constraints,
        key=subkey,
    )
    start_idx = 0

    logger.info(tc.tree_summary(arrays, depth=2))

    epochs = 501
    # Number of independent random eta-field realizations drawn per training step. Each is a
    # fresh, statistically independent fabrication-uncertainty outcome; their objectives are
    # combined via a weighted p-mean (uniform weights, p=1 -> plain Monte Carlo average) into
    # one scalar loss, so a single jax.value_and_grad call yields the correctly-weighted
    # gradient across all K samples automatically -- no manual gradient bookkeeping needed.
    num_samples = 4
    aggregation_p = 1.0

    if not evaluation:
        schedule_finetune: optax.Schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-5,
            peak_value=0.005,
            end_value=0.0005,
            warmup_steps=10,
            decay_steps=round(0.9 * epochs),
        )
        optimizer_finetune = optax.inject_hyperparams(optax.nadam)(learning_rate=schedule_finetune)
        optimizer_finetune = optax.MultiSteps(optimizer_finetune, every_k_schedule=1)
        opt_state_finetune: optax.OptState = optimizer_finetune.init(params)

    def custom_schedule(idx: chex.Numeric) -> chex.Numeric:
        beta_schedule = optax.linear_schedule(0.1, 50, epochs)
        return jax.lax.cond(idx < epochs - 2, lambda: beta_schedule(idx), lambda: jnp.inf)

    exp_logger.savefig(
        exp_logger.cwd,
        "setup.png",
        fdtdx.plot_setup(config=config, objects=objects),
    )

    changed_voxels = exp_logger.log_params(
        iter_idx=-1,
        params=params,
        objects=objects,
        export_stl=True,
        export_figure=True,
        beta=custom_schedule(start_idx),
        field_key=jax.random.PRNGKey(0),  # fixed key: this call is only for visualization
    )

    def loss_func(
        params: fdtdx.ParameterContainer,
        arrays: fdtdx.ArrayContainer,
        key: jax.Array,
        idx: int,
    ):
        beta = custom_schedule(idx)
        key, sample_key = jax.random.split(key)
        sample_keys = jax.random.split(sample_key, num_samples)

        def run_single_sample(carry_arrays: fdtdx.ArrayContainer, field_key: jax.Array):
            # apply_params resets inv_permittivities from initial_inv_permittivities, and
            # run_fdtd resets the E/H field state internally (ArrayContainer.reset()) -- so
            # each sample starts from a clean state regardless of what the previous sample's
            # `carry_arrays` looked like; threading `arrays` through the scan carry here is only
            # to avoid re-closing over the same large pytree K times, not for state continuity.
            a, new_objects, _ = fdtdx.apply_params(carry_arrays, objects, params, key, beta=beta, field_key=field_key)
            _, a = fdtdx.run_fdtd(arrays=a, objects=new_objects, config=config, key=key)
            total_out_flux = a.detector_states[flux_out_detector.name]["poynting_flux"].sum()
            total_in_flux = a.detector_states[flux_in_detector.name]["poynting_flux"].sum()
            return a, total_out_flux / total_in_flux

        final_arrays, objectives = jax.lax.scan(run_single_sample, arrays, sample_keys)

        # weighted_p_mean requires strictly positive objectives; the flux ratio can be exactly
        # zero for a fully-dark initial random structure, which is expected early in training
        # and simply zeroes that sample's gradient contribution (see weighted_p_mean's eps floor).
        combined_objective = fdtdx.weighted_p_mean(objectives, p=aggregation_p)

        new_info = {
            "sample_objectives": objectives,
            "combined_objective": combined_objective,
        }
        return -combined_objective, (final_arrays, new_info)

    compile_start_time = time.time()
    print("Started Compilation...")
    jit_task_id = exp_logger.progress.add_task("JIT", total=None)
    idx_dummy_arr = jnp.asarray(start_idx, dtype=jnp.float32)
    if evaluation:
        jitted_loss = jax.jit(loss_func, donate_argnames=["arrays"]).lower(params, arrays, key, idx_dummy_arr).compile()
    else:
        jitted_loss = (
            jax.jit(jax.value_and_grad(loss_func, has_aux=True), donate_argnames=["arrays"])
            .lower(params, arrays, key, idx_dummy_arr)
            .compile()
        )
    compile_delta_time = time.time() - compile_start_time
    exp_logger.progress.update(jit_task_id, total=1, completed=1, refresh=True)
    print(f"Finished Compilation in {compile_delta_time} seconds")

    optim_task_id = exp_logger.progress.add_task("Optimization", total=1 if evaluation else epochs)
    for epoch in range(start_idx, start_idx + 1 if evaluation else epochs):
        run_start_time = time.time()
        key, subkey = jax.random.split(key)
        idx_arr = jnp.asarray(epoch, dtype=jnp.float32)
        if evaluation:
            loss, (arrays, info) = jitted_loss(params, arrays, subkey, idx_arr)
        else:
            (loss, (arrays, info)), grads = jitted_loss(params, arrays, subkey, idx_arr)
            updates, opt_state_finetune = optimizer_finetune.update(grads, opt_state_finetune, params)
            info["lr"] = opt_state_finetune.inner_opt_state.hyperparams["learning_rate"]
            params = optax.apply_updates(params, updates)
            params = jax.tree_util.tree_map(lambda p: jnp.clip(p, 0, 1), params)
            info["grad_norm"] = optax.global_norm(grads)
            info["update_norm"] = optax.global_norm(updates)

        runtime_delta = time.time() - run_start_time
        info["runtime"] = runtime_delta
        info["attenuation"] = 10 * jnp.log10(-loss)

        if evaluation:
            logger.info(f"{compile_delta_time=}")
            logger.info(f"{runtime_delta=}")

        changed_voxels = exp_logger.log_params(
            iter_idx=epoch,
            params=params,
            objects=objects,
            export_stl=True,
            export_figure=True,
            beta=custom_schedule(epoch),
            field_key=jax.random.PRNGKey(0),  # fixed key: this call is only for visualization
        )
        info["changed_voxels"] = changed_voxels

        exp_logger.log_detectors(iter_idx=epoch, objects=objects, detector_states=arrays.detector_states)
        exp_logger.write(info)
        exp_logger.progress.update(optim_task_id, advance=1)


if __name__ == "__main__":
    seed = 0
    evaluation = False
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        evaluation = False
    main(seed, evaluation=evaluation)
