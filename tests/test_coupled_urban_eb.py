# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import torch

from utherm.energy_balance import (
    AIR_DENSITY,
    AIR_SPECIFIC_HEAT,
    CanopyProperties,
    EBSolver,
    EnergyBalanceConfig,
    MaterialProperties,
    calculate_net_radiation,
    solve_conduction_substepped,
)
from utherm.energy_balance.coupled import (
    CANOPY,
    GROUND,
    N_FACETS,
    ROOF,
    WALL_NORTH,
    WALL_SOUTH,
    CoupledUrbanEBConfig,
    CoupledUrbanEnergyBalance,
    UrbanFacetGeometry,
    UrbanForcing,
)
from utherm.energy_balance.physics import LATENT_HEAT_VAP, STEFAN_BOLTZMANN


def _materials():
    return [MaterialProperties.soil(), MaterialProperties.roof()] + [
        MaterialProperties.brick()
    ] * 4


def _open_geometry(device, rows=1, cols=1):
    area = torch.tensor(
        [0.7, 0.3, 0.25, 0.25, 0.25, 0.25, 0.8], device=device
    ).view(N_FACETS, 1, 1).expand(-1, rows, cols).clone()
    return UrbanFacetGeometry(
        area=area,
        sky_view_area=area.clone(),
        exchange_area=torch.zeros(N_FACETS, N_FACETS, rows, cols, device=device),
    )


def _model(device, geometry=None, **config_overrides):
    geometry = geometry or _open_geometry(device)
    config_values = {
        "dt": 60.0,
        "max_coupling_iterations": 200,
        "temperature_tolerance": 0.02,
        "residual_tolerance": 1.0,
        "strict_convergence": False,
    }
    config_values.update(config_overrides)
    config = CoupledUrbanEBConfig(**config_values)
    water = torch.full_like(geometry.area, 10.0)
    return CoupledUrbanEnergyBalance(
        geometry,
        _materials(),
        CanopyProperties.deciduous(),
        water,
        config,
    )


def test_view_areas_require_reciprocity(device):
    geometry = _open_geometry(device)
    geometry.exchange_area[GROUND, ROOF] = 0.2
    with pytest.raises(ValueError, match="reciprocity"):
        geometry.validate()


def test_view_areas_require_closure(device):
    geometry = _open_geometry(device)
    geometry.sky_view_area[GROUND] = 0.5
    with pytest.raises(ValueError, match="do not close"):
        geometry.validate()


def test_geometry_tensors_require_one_dtype(device):
    base = _open_geometry(device)
    geometry = UrbanFacetGeometry(
        base.area,
        base.sky_view_area.double(),
        base.exchange_area,
    )
    with pytest.raises(ValueError, match="one floating-point dtype"):
        geometry.validate()


def test_closed_radiosity_conserves_internal_longwave(device):
    area = torch.zeros(N_FACETS, 1, 1, device=device)
    area[GROUND] = 1.0
    area[ROOF] = 1.0
    exchange = torch.zeros(N_FACETS, N_FACETS, 1, 1, device=device)
    exchange[GROUND, ROOF] = 1.0
    exchange[ROOF, GROUND] = 1.0
    geometry = UrbanFacetGeometry(area, torch.zeros_like(area), exchange)
    model = _model(device, geometry)
    temperature = torch.full_like(area, 295.0)
    temperature[GROUND] = 310.0
    shortwave = torch.zeros_like(area)
    _, longwave, _, to_sky = model._radiosity(
        temperature, shortwave, torch.full((1, 1), 400.0, device=device)
    )
    internal_sum = (area * longwave).sum()
    assert abs(float(internal_sum.item())) < 1.0e-3
    assert abs(float(to_sky.item())) < 1.0e-6


def test_closed_shortwave_is_absorbed_or_reflected_until_absorbed(device):
    area = torch.zeros(N_FACETS, 1, 1, device=device)
    area[GROUND] = 1.0
    area[ROOF] = 1.0
    exchange = torch.zeros(N_FACETS, N_FACETS, 1, 1, device=device)
    exchange[GROUND, ROOF] = 1.0
    exchange[ROOF, GROUND] = 1.0
    geometry = UrbanFacetGeometry(area, torch.zeros_like(area), exchange)
    model = _model(device, geometry)
    shortwave = torch.zeros_like(area)
    shortwave[GROUND] = 100.0
    absorbed, _, escaped, _ = model._radiosity(
        torch.full_like(area, 300.0), shortwave, torch.full((1, 1), 400.0, device=device)
    )
    assert float((absorbed * area).sum().item()) == pytest.approx(100.0, abs=1.0e-3)
    assert float(escaped.item()) == pytest.approx(0.0, abs=1.0e-6)


def test_open_radiosity_conserves_shortwave_and_longwave(device):
    model = _model(device)
    temperature = torch.full_like(model.geometry.area, 300.0)
    temperature[GROUND] = 310.0
    shortwave = torch.full_like(model.geometry.area, 200.0)
    sky_longwave = torch.full((1, 1), 380.0, device=device)
    absorbed, longwave, escaped, longwave_to_sky = model._radiosity(
        temperature, shortwave, sky_longwave
    )
    external_shortwave = (model.geometry.area * shortwave).sum()
    assert float((model.geometry.area * absorbed).sum() + escaped) == pytest.approx(
        float(external_shortwave), abs=1.0e-3
    )
    assert float((model.geometry.area * longwave).sum() + longwave_to_sky) == pytest.approx(
        0.0, abs=1.0e-3
    )


def test_coupled_step_returns_all_facets_and_energy_diagnostics(device):
    model = _model(device)
    state = model.initialize_state(298.15)
    shortwave = torch.zeros_like(model.geometry.area)
    shortwave[GROUND] = 500.0
    shortwave[ROOF] = 700.0
    shortwave[CANOPY] = 300.0
    result = model.step(
        state,
        UrbanForcing(
            air_temperature=298.15,
            vapor_pressure_kpa=2.0,
            pressure_kpa=101.3,
            wind_speed=2.0,
            sky_longwave=400.0,
            shortwave_irradiance=shortwave,
        ),
    )
    assert result.converged
    assert result.state.surface_temperature.shape == model.geometry.area.shape
    assert result.state.layer_temperature.shape[0] == 6
    assert result.max_energy_residual <= model.config.residual_tolerance
    assert torch.isfinite(result.residual).all()


def test_warm_canyon_facets_heat_canyon_air(device):
    model = _model(device)
    state = model.initialize_state(295.0)
    state.surface_temperature[GROUND] = 315.0
    h, _ = model._heat_transfer(
        state.surface_temperature,
        torch.full((1, 1), 295.0, device=device),
        state.canyon_air_temperature,
        torch.full((1, 1), 1.0, device=device),
    )
    new_air = model._canyon_temperature(
        state.canyon_air_temperature,
        torch.full((1, 1), 295.0, device=device),
        state.surface_temperature,
        h,
        torch.full((1, 1), 1.0, device=device),
        torch.zeros((1, 1), device=device),
    )
    assert float(new_air.item()) > 295.0


def test_water_budget_never_exceeds_capacity(device):
    model = _model(device)
    state = model.initialize_state(298.15, initial_water_fraction=0.0)
    result = model.step(
        state,
        UrbanForcing(
            air_temperature=298.15,
            vapor_pressure_kpa=2.0,
            pressure_kpa=101.3,
            wind_speed=1.0,
            sky_longwave=400.0,
            shortwave_irradiance=torch.zeros_like(model.geometry.area),
            precipitation_rate=1.0,
        ),
    )
    assert torch.all(result.state.water_storage <= model.water_capacity)
    assert float(result.water_drainage[GROUND].item()) > 0.0
    assert float(result.water_drainage[ROOF].item()) > 0.0
    assert float(result.water_drainage[CANOPY].item()) == 0.0


def test_precipitation_is_available_to_evaporation_in_current_step(device):
    model = _model(device)
    state = model.initialize_state(303.15, initial_water_fraction=0.0)
    shortwave = torch.zeros_like(model.geometry.area)
    shortwave[GROUND] = 600.0
    result = model.step(
        state,
        UrbanForcing(
            air_temperature=303.15,
            vapor_pressure_kpa=1.0,
            pressure_kpa=101.3,
            wind_speed=2.0,
            sky_longwave=400.0,
            shortwave_irradiance=shortwave,
            precipitation_rate=0.001,
        ),
    )
    assert float(result.latent_heat[GROUND].item()) > 0.0
    incoming = 0.001 * model.config.dt
    expected = incoming - (
        float(result.latent_heat[GROUND].item()) * model.config.dt / 2.501e6
    )
    assert float(result.state.water_storage[GROUND].item()) == pytest.approx(
        expected, rel=1.0e-5
    )


def test_rain_capture_cannot_duplicate_precipitation(device):
    model = _model(device)
    state = model.initialize_state(298.15)
    capture = torch.ones_like(model.geometry.area)
    forcing = UrbanForcing(
        298.15,
        2.0,
        101.3,
        1.0,
        400.0,
        torch.zeros_like(model.geometry.area),
        precipitation_rate=0.001,
        rain_capture_fraction=capture,
    )
    with pytest.raises(ValueError, match="duplicated"):
        model.step(state, forcing)


def test_default_rain_partition_does_not_invent_canopy_interception(device):
    model = _model(device)
    forcing = UrbanForcing(
        298.15,
        2.0,
        101.3,
        1.0,
        400.0,
        torch.zeros_like(model.geometry.area),
        precipitation_rate=0.001,
    )
    *_, capture = model._validate_forcing(forcing)
    assert torch.all(capture[GROUND] == 1.0)
    assert torch.all(capture[ROOF] == 1.0)
    assert torch.count_nonzero(capture[CANOPY]) == 0


def test_evaporation_enters_canyon_humidity_budget(device):
    model = _model(
        device,
        ventilation_coefficient=0.0,
        minimum_exchange_velocity=0.0,
    )
    previous = torch.full((1, 1), 0.01, device=device)
    latent = torch.zeros_like(model.geometry.area)
    latent[GROUND] = 100.0
    temperature = torch.full((1, 1), 300.0, device=device)
    updated = model._canyon_specific_humidity(
        previous,
        torch.full((1, 1), 1.0, device=device),
        100.0,
        latent,
        temperature,
        torch.zeros((1, 1), device=device),
    )
    density = 100000.0 / (287.05 * 300.0)
    expected_gain = (0.7 * 100.0 / 2.501e6) * model.config.dt / (
        density * model.config.canyon_height
    )
    assert float((updated - previous).item()) == pytest.approx(expected_gain, rel=1.0e-5)


def test_strict_mode_rejects_nonconvergence(device):
    model = _model(
        device,
        max_coupling_iterations=1,
        strict_convergence=True,
    )
    state = model.initialize_state(298.15)
    shortwave = torch.full_like(model.geometry.area, 1000.0)
    with pytest.raises(RuntimeError, match="did not converge"):
        model.step(
            state,
            UrbanForcing(298.15, 2.0, 101.3, 1.0, 350.0, shortwave),
        )


def test_state_dtype_must_match_geometry(device):
    model = _model(device)
    state = model.initialize_state(298.15)
    state.canyon_air_temperature = state.canyon_air_temperature.double()
    with pytest.raises(ValueError, match="wrong dtype"):
        model.step(
            state,
            UrbanForcing(
                298.15,
                2.0,
                101.3,
                1.0,
                400.0,
                torch.zeros_like(model.geometry.area),
            ),
        )


def test_spinup_reaches_periodic_thermal_and_moisture_state(device):
    model = _model(
        device,
        dt=3600.0,
        spinup_max_cycles=80,
        spinup_temperature_tolerance=0.1,
        spinup_moisture_tolerance=0.02,
    )
    forcing = UrbanForcing(
        298.15,
        2.0,
        101.3,
        2.0,
        400.0,
        torch.zeros_like(model.geometry.area),
    )
    result = model.spin_up([forcing])
    assert result.converged
    assert result.cycles >= model.config.spinup_min_cycles
    assert result.maximum_temperature_drift <= model.config.spinup_temperature_tolerance
    assert result.maximum_moisture_drift <= model.config.spinup_moisture_tolerance
    assert (
        result.maximum_specific_humidity_drift
        <= model.config.spinup_specific_humidity_tolerance
    )


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("dt", 0.0, "dt"),
        ("relaxation", 1.1, "relaxation"),
        ("max_coupling_iterations", True, "positive integer"),
        ("spinup_min_cycles", 4, "cannot exceed"),
        ("solve_wall_temperature", 1, "boolean"),
        ("insulated_deep_boundary", 1, "boolean"),
    ],
)
def test_config_rejects_invalid_controls(name, value, message):
    values = {name: value}
    if name == "spinup_min_cycles":
        values["spinup_max_cycles"] = 3
    with pytest.raises(ValueError, match=message):
        CoupledUrbanEBConfig(**values)


def test_geometry_from_view_factors_preserves_reciprocal_areas(device):
    area = torch.zeros(N_FACETS, 1, 1, device=device)
    area[GROUND] = 2.0
    area[ROOF] = 1.0
    sky = torch.zeros_like(area)
    sky[GROUND] = 0.5
    view = torch.zeros(N_FACETS, N_FACETS, 1, 1, device=device)
    view[GROUND, ROOF] = 0.5
    view[ROOF, GROUND] = 1.0
    geometry = UrbanFacetGeometry.from_view_factors(area, sky, view)
    assert float(geometry.exchange_area[GROUND, ROOF].item()) == pytest.approx(1.0)
    assert float(geometry.exchange_area[ROOF, GROUND].item()) == pytest.approx(1.0)


def test_view_factor_conversion_rejects_nonreciprocal_input(device):
    area = torch.zeros(N_FACETS, 1, 1, device=device)
    area[GROUND] = 1.0
    area[ROOF] = 1.0
    sky = torch.zeros_like(area)
    view = torch.zeros(N_FACETS, N_FACETS, 1, 1, device=device)
    view[GROUND, ROOF] = 1.0
    with pytest.raises(ValueError, match="reciprocity"):
        UrbanFacetGeometry.from_view_factors(area, sky, view)


def test_property_tensors_match_declared_materials(device):
    model = _model(device)
    assert model.thickness.shape == (6, 1, 1, 4)
    assert model.conductivity.shape == model.thickness.shape
    assert model.heat_capacity.shape == model.thickness.shape
    assert model.thickness[GROUND, 0, 0].tolist() == pytest.approx(
        MaterialProperties.soil().thickness
    )
    assert float(model.albedo[ROOF].item()) == pytest.approx(
        MaterialProperties.roof().albedo
    )
    assert float(model.emissivity[CANOPY].item()) == pytest.approx(
        CanopyProperties.deciduous().emissivity_leaf
    )


def test_per_pixel_material_ids_expand_every_solid_property(device):
    geometry = _open_geometry(device, rows=1, cols=2)
    water = torch.full_like(geometry.area, 10.0)
    table = [MaterialProperties.asphalt(), MaterialProperties.grass()]
    ids = torch.zeros((6, 1, 2), dtype=torch.long, device=device)
    ids[GROUND, 0, 1] = 1
    model = CoupledUrbanEnergyBalance(
        geometry,
        _materials(),
        CanopyProperties.deciduous(),
        water,
        CoupledUrbanEBConfig(strict_convergence=False),
        material_ids=ids,
        material_table=table,
    )
    assert model.material_ids is not None
    assert float(model.albedo[GROUND, 0, 0]) == pytest.approx(table[0].albedo)
    assert float(model.albedo[GROUND, 0, 1]) == pytest.approx(table[1].albedo)
    assert float(model.max_conductance[GROUND, 0, 0]) == 0.0
    assert float(model.max_conductance[GROUND, 0, 1]) > 0.0
    assert float(model.conductivity[GROUND, 0, 0, 0]) == pytest.approx(
        table[0].conductivity[0]
    )
    assert float(model.conductivity[GROUND, 0, 1, 0]) == pytest.approx(
        table[1].conductivity[0]
    )


def test_per_pixel_material_ids_fail_closed(device):
    geometry = _open_geometry(device)
    water = torch.full_like(geometry.area, 10.0)
    ids = torch.zeros((6, 1, 1), dtype=torch.long, device=device)
    with pytest.raises(ValueError, match="supplied together"):
        CoupledUrbanEnergyBalance(
            geometry,
            _materials(),
            CanopyProperties.deciduous(),
            water,
            material_ids=ids,
        )
    with pytest.raises(ValueError, match="between 0 and 0"):
        CoupledUrbanEnergyBalance(
            geometry,
            _materials(),
            CanopyProperties.deciduous(),
            water,
            material_ids=ids + 1,
            material_table=[MaterialProperties.asphalt()],
        )


def test_model_constructor_rejects_incomplete_materials_and_bad_water(device):
    geometry = _open_geometry(device)
    water = torch.ones_like(geometry.area)
    with pytest.raises(ValueError, match="six wall materials|ground, roof"):
        CoupledUrbanEnergyBalance(
            geometry,
            _materials()[:-1],
            CanopyProperties.deciduous(),
            water,
        )
    bad_water = water.clone()
    bad_water[GROUND] = -1.0
    with pytest.raises(ValueError, match="finite and nonnegative"):
        CoupledUrbanEnergyBalance(
            geometry,
            _materials(),
            CanopyProperties.deciduous(),
            bad_water,
        )


def test_default_coupled_initialization_does_not_invent_surface_water(device):
    model = _model(device)
    state = model.initialize_state(300.0)
    assert torch.count_nonzero(state.water_storage) == 0


def test_grid_expands_scalars_and_rejects_invalid_fields(device):
    model = _model(device, geometry=_open_geometry(device, 2, 3))
    assert model._grid(2.0, "value").shape == (2, 3)
    with pytest.raises(ValueError, match="shape"):
        model._grid(torch.ones(2, 2, device=device), "value")
    with pytest.raises(ValueError, match="non-finite"):
        model._grid(float("nan"), "value")
    with pytest.raises(ValueError, match="nonnegative"):
        model._grid(-1.0, "value", nonnegative=True)


def test_forcing_validation_normalizes_fields_and_rejects_negative_shortwave(device):
    model = _model(device)
    forcing = UrbanForcing(
        298.15,
        2.0,
        101.3,
        1.0,
        400.0,
        torch.zeros_like(model.geometry.area),
    )
    fields = model._validate_forcing(forcing)
    assert all(tuple(value.shape[-2:]) == (1, 1) for value in fields)
    bad = torch.zeros_like(model.geometry.area)
    bad[GROUND] = -1.0
    with pytest.raises(ValueError, match="finite and nonnegative"):
        model._validate_forcing(
            UrbanForcing(298.15, 2.0, 101.3, 1.0, 400.0, bad)
        )


def test_state_validation_rejects_shape_nonfinite_and_capacity_errors(device):
    model = _model(device)
    state = model.initialize_state(298.15)
    wrong = state.clone()
    wrong.surface_temperature = wrong.surface_temperature[:-1]
    with pytest.raises(ValueError, match="shape"):
        model._check_state(wrong)
    nonfinite = state.clone()
    nonfinite.layer_temperature[GROUND, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        model._check_state(nonfinite)
    overflowing = state.clone()
    overflowing.water_storage[GROUND] = model.water_capacity[GROUND] + 1.0
    with pytest.raises(ValueError, match="outside its capacity"):
        model._check_state(overflowing)


def test_specific_humidity_vapor_pressure_round_trip(device):
    vapor = torch.tensor([[0.2, 1.5, 4.0]], dtype=torch.float64, device=device)
    humidity = CoupledUrbanEnergyBalance._specific_humidity_from_vapor(vapor, 101.3)
    recovered = CoupledUrbanEnergyBalance._vapor_from_specific_humidity(
        humidity, 101.3
    )
    assert torch.allclose(recovered, vapor, atol=1.0e-12, rtol=1.0e-12)


def test_latent_heat_is_dry_zero_saturated_energy_limited_and_continuous(device):
    model = _model(device, dt=3600.0)
    temperature = torch.full_like(model.geometry.area, 303.15)
    air = torch.full_like(temperature, 298.15)
    vapor = torch.full((1, 1), 1.0, device=device)
    h = torch.full_like(temperature, 15.0)
    shortwave = torch.full_like(temperature, 500.0)
    dry = torch.zeros_like(temperature)
    latent_dry, _ = model._latent_heat(
        temperature, air, vapor, vapor, 101.3, h, dry, shortwave
    )
    assert torch.equal(latent_dry, torch.zeros_like(latent_dry))

    saturated = model.water_capacity.clone()
    latent_saturated, _ = model._latent_heat(
        temperature, air, vapor, vapor, 101.3, h, saturated, shortwave
    )
    assert float(latent_saturated[GROUND].item()) > 0.0
    available = float(saturated[GROUND].item()) * LATENT_HEAT_VAP / model.config.dt
    assert float(latent_saturated[GROUND].item()) < available

    tiny = model.water_capacity * 1.0e-6
    twice_tiny = model.water_capacity * 2.0e-6
    latent_tiny, _ = model._latent_heat(
        temperature, air, vapor, vapor, 101.3, h, tiny, shortwave
    )
    latent_twice, _ = model._latent_heat(
        temperature, air, vapor, vapor, 101.3, h, twice_tiny, shortwave
    )
    assert float(latent_tiny[GROUND].item()) > 0.0
    assert float(latent_twice[GROUND].item()) / float(
        latent_tiny[GROUND].item()
    ) == pytest.approx(2.0, rel=2.0e-3)


def test_conduction_response_is_exactly_affine_over_fifty_kelvin(device):
    base_geometry = _open_geometry(device)
    geometry = UrbanFacetGeometry(
        base_geometry.area.double(),
        base_geometry.sky_view_area.double(),
        base_geometry.exchange_area.double(),
    )
    model = _model(device, geometry=geometry)
    state = model.initialize_state(298.15)
    state.layer_temperature[GROUND, 0, 0] = torch.tensor(
        [301.0, 299.0, 296.0, 292.0], dtype=torch.float64, device=device
    )
    equivalent, conductance = model._conduction_affine(
        state, torch.full((1, 1), 298.15, device=device)
    )
    assert torch.isfinite(equivalent).all()
    assert torch.all(conductance > 0.0)
    facet = GROUND
    old = state.layer_temperature[facet].reshape(1, model.n_layers)
    surface = state.surface_temperature[facet].reshape(1)
    deep = torch.full_like(surface, model.config.ground_deep_temperature)
    args = (
        model.thickness[facet].reshape(1, model.n_layers),
        model.conductivity[facet].reshape(1, model.n_layers),
        model.heat_capacity[facet].reshape(1, model.n_layers),
        model.config.dt,
        model.config.insulated_deep_boundary,
    )
    base = solve_conduction_substepped(old, surface, deep, *args)[0, 0]
    plus_one = solve_conduction_substepped(old, surface + 1.0, deep, *args)[0, 0]
    plus_fifty = solve_conduction_substepped(old, surface + 50.0, deep, *args)[0, 0]
    assert float((plus_one - base).item()) == pytest.approx(
        float(((plus_fifty - base) / 50.0).item()), rel=2.0e-5, abs=2.0e-6
    )


def test_run_advances_every_timestep_and_chains_state(device):
    model = _model(device)
    forcing = UrbanForcing(
        298.15,
        2.0,
        101.3,
        1.0,
        400.0,
        torch.zeros_like(model.geometry.area),
    )
    assert model.run([], spin_up=False) == []
    initial = model.initialize_state(298.15)
    results = model.run([forcing, forcing], state=initial, spin_up=False)
    manual_first = model.step(initial, forcing)
    manual_second = model.step(manual_first.state, forcing)
    assert len(results) == 2
    assert torch.allclose(
        results[-1].state.layer_temperature,
        manual_second.state.layer_temperature,
    )


def test_spinup_state_is_periodic_over_one_more_cycle(device):
    model = _model(
        device,
        dt=3600.0,
        spinup_max_cycles=80,
        spinup_temperature_tolerance=0.1,
        spinup_moisture_tolerance=0.02,
    )
    forcing = UrbanForcing(
        298.15,
        2.0,
        101.3,
        2.0,
        400.0,
        torch.zeros_like(model.geometry.area),
    )
    spinup = model.spin_up([forcing])
    next_state = model.step(spinup.state, forcing).state
    assert float(
        (next_state.surface_temperature - spinup.state.surface_temperature)
        .abs()
        .max()
        .item()
    ) <= model.config.spinup_temperature_tolerance
    assert float(
        (next_state.water_storage - spinup.state.water_storage).abs().max().item()
    ) <= model.config.spinup_moisture_tolerance


def test_strict_spinup_raises_when_periodicity_is_not_reached(device):
    model = _model(
        device,
        dt=3600.0,
        spinup_min_cycles=2,
        spinup_max_cycles=2,
        spinup_temperature_tolerance=1.0e-10,
        spinup_moisture_tolerance=1.0e-10,
        spinup_specific_humidity_tolerance=1.0e-12,
        strict_convergence=True,
    )
    forcing = UrbanForcing(
        298.15,
        1.5,
        101.3,
        2.0,
        380.0,
        torch.full_like(model.geometry.area, 400.0),
    )
    with pytest.raises(RuntimeError, match="spin-up did not reach"):
        model.spin_up([forcing])


def test_closed_box_conserves_energy_across_twenty_four_steps(device):
    dtype = torch.float64
    area = torch.zeros(N_FACETS, 1, 1, dtype=dtype, device=device)
    area[GROUND] = 1.0
    area[WALL_NORTH] = 1.0
    exchange = torch.zeros(
        N_FACETS, N_FACETS, 1, 1, dtype=dtype, device=device
    )
    exchange[GROUND, WALL_NORTH] = 1.0
    exchange[WALL_NORTH, GROUND] = 1.0
    geometry = UrbanFacetGeometry(area, torch.zeros_like(area), exchange)
    config = CoupledUrbanEBConfig(
        dt=300.0,
        max_coupling_iterations=500,
        temperature_tolerance=1.0e-8,
        specific_humidity_tolerance=1.0e-12,
        residual_tolerance=1.0e-5,
        relaxation=0.8,
        ventilation_coefficient=0.0,
        minimum_exchange_velocity=0.0,
        strict_convergence=True,
        insulated_deep_boundary=True,
    )
    model = CoupledUrbanEnergyBalance(
        geometry,
        _materials(),
        CanopyProperties.deciduous(),
        torch.zeros_like(area),
        config,
    )
    state = model.initialize_state(300.0, initial_water_fraction=0.0)
    state.surface_temperature[GROUND] = 306.0
    state.surface_temperature[WALL_NORTH] = 294.0
    state.layer_temperature[GROUND] = 306.0
    state.layer_temperature[WALL_NORTH] = 294.0
    forcing = UrbanForcing(
        300.0,
        1.5,
        101.3,
        1.0,
        0.0,
        torch.zeros_like(area),
    )

    def energy(current):
        solids = (
            area[:6].unsqueeze(-1)
            * model.heat_capacity
            * model.thickness
            * current.layer_temperature
        ).sum()
        canyon = (
            AIR_DENSITY
            * AIR_SPECIFIC_HEAT
            * model.config.canyon_height
            * current.canyon_air_temperature
        ).sum()
        return solids + canyon

    initial_energy = energy(state)
    for _ in range(24):
        state = model.step(state, forcing).state
    final_energy = energy(state)
    relative_drift = abs(float((final_energy - initial_energy).item())) / abs(
        float(initial_energy.item())
    )
    assert relative_drift < 2.0e-9


def test_water_budget_closes_over_a_forcing_cycle(device):
    model = _model(device, dt=600.0)
    state = model.initialize_state(303.15, initial_water_fraction=0.25)
    initial_water = (model.geometry.area * state.water_storage).sum()
    rain_rates = [0.001, 0.0, 0.002, 0.0]
    total_input = torch.zeros((), device=device)
    total_evaporation = torch.zeros((), device=device)
    total_drainage = torch.zeros((), device=device)
    capture = torch.zeros_like(model.geometry.area)
    capture[GROUND] = 1.0
    capture[ROOF] = 1.0
    shortwave = torch.full_like(model.geometry.area, 500.0)
    for rain in rain_rates:
        forcing = UrbanForcing(
            303.15,
            1.0,
            101.3,
            2.0,
            400.0,
            shortwave,
            precipitation_rate=rain,
        )
        result = model.step(state, forcing)
        total_input += (
            model.geometry.area * capture * rain * model.config.dt
        ).sum()
        total_evaporation += (
            model.geometry.area
            * result.latent_heat
            * model.config.dt
            / LATENT_HEAT_VAP
        ).sum()
        total_drainage += (model.geometry.area * result.water_drainage).sum()
        state = result.state
    final_water = (model.geometry.area * state.water_storage).sum()
    assert float((initial_water + total_input).item()) == pytest.approx(
        float((final_water + total_evaporation + total_drainage).item()),
        rel=2.0e-6,
        abs=2.0e-6,
    )


def test_wall_humidity_and_water_additions_can_be_isolated(device):
    forcing_shortwave = torch.zeros_like(_open_geometry(device).area)
    forcing_shortwave[GROUND] = 600.0
    forcing_shortwave[WALL_NORTH] = 700.0
    forcing = UrbanForcing(
        303.15, 1.0, 101.3, 2.0, 400.0, forcing_shortwave
    )

    active_walls = _model(device, solve_wall_temperature=True)
    frozen_walls = _model(device, solve_wall_temperature=False)
    active_state = active_walls.initialize_state(303.15)
    frozen_state = frozen_walls.initialize_state(303.15)
    active_result = active_walls.step(active_state, forcing)
    frozen_result = frozen_walls.step(frozen_state, forcing)
    assert float(
        abs(active_result.state.surface_temperature[WALL_NORTH] - 303.15).item()
    ) > 0.01
    assert float(frozen_result.state.surface_temperature[WALL_NORTH].item()) == pytest.approx(
        303.15
    )

    humid = _model(
        device,
        ventilation_coefficient=0.0,
        minimum_exchange_velocity=0.0,
        solve_canyon_humidity=True,
    )
    fixed_humidity = _model(
        device,
        ventilation_coefficient=0.0,
        minimum_exchange_velocity=0.0,
        solve_canyon_humidity=False,
    )
    humid_result = humid.step(humid.initialize_state(303.15), forcing)
    fixed_result = fixed_humidity.step(
        fixed_humidity.initialize_state(303.15), forcing
    )
    q_above = fixed_humidity._specific_humidity_from_vapor(
        torch.full((1, 1), 1.0, device=device), 101.3
    )
    assert torch.allclose(
        fixed_result.state.canyon_specific_humidity, q_above, atol=1.0e-6
    )
    assert float(humid_result.state.canyon_specific_humidity.item()) > float(
        q_above.item()
    )

    limited = _model(device, water_limited_evaporation=True)
    unlimited = _model(device, water_limited_evaporation=False)
    dry_limited = limited.initialize_state(303.15, initial_water_fraction=0.0)
    dry_unlimited = unlimited.initialize_state(303.15, initial_water_fraction=0.0)
    limited_result = limited.step(dry_limited, forcing)
    unlimited_result = unlimited.step(dry_unlimited, forcing)
    assert float(limited_result.latent_heat[GROUND].item()) == pytest.approx(0.0)
    assert float(unlimited_result.latent_heat[GROUND].item()) > 0.0


def test_ground_only_coupled_solver_reduces_to_release_surface_balance(device):
    area = torch.zeros(N_FACETS, 1, 1, device=device)
    area[GROUND] = 1.0
    geometry = UrbanFacetGeometry(
        area,
        area.clone(),
        torch.zeros(N_FACETS, N_FACETS, 1, 1, device=device),
    )
    asphalt = MaterialProperties.asphalt()
    coupled = CoupledUrbanEnergyBalance(
        geometry,
        [asphalt, MaterialProperties.roof()] + [MaterialProperties.brick()] * 4,
        CanopyProperties.deciduous(),
        torch.zeros_like(area),
        CoupledUrbanEBConfig(
            dt=60.0,
            max_coupling_iterations=300,
            temperature_tolerance=0.001,
            residual_tolerance=0.1,
            strict_convergence=False,
            solve_canyon_temperature=False,
            solve_canyon_humidity=False,
            solve_wall_temperature=False,
        ),
    )
    ta_k = 298.15
    sky_lw = 380.0
    shortwave = torch.zeros_like(area)
    shortwave[GROUND] = 400.0
    coupled_state = coupled.initialize_state(ta_k, initial_water_fraction=0.0)
    coupled_result = coupled.step(
        coupled_state,
        UrbanForcing(ta_k, 2.0, 101.3, 2.0, sky_lw, shortwave),
    )

    ones = torch.ones((1, 1), device=device)
    release = EBSolver(
        EnergyBalanceConfig(
            dt=60.0,
            t_deep=288.15,
            z0=0.03,
            max_iterations=80,
            surface_residual_tolerance=0.1,
            ground_wetness=0.0,
        ),
        asphalt,
        ones,
        torch.full_like(ones, asphalt.albedo),
        torch.full_like(ones, asphalt.emissivity),
        ones,
        ones,
        ones,
        device=device,
    )
    ldown = torch.full_like(ones, sky_lw)
    kdown = torch.full_like(ones, 400.0)
    rnet = calculate_net_radiation(
        ldown,
        kdown,
        torch.full_like(ones, asphalt.albedo),
        torch.full_like(ones, asphalt.emissivity),
        ta_k,
    )
    release_result = release.solve(
        rnet,
        torch.ones_like(ones),
        kdown,
        ta_k - 273.15,
        2.0,
        20.0,
        101.3,
        0.0,
        400.0,
        prev_layer_temps=torch.full((1, 1, 4), ta_k, device=device),
        prev_tsfc=torch.full_like(ones, ta_k),
    )
    coupled_temperature = float(
        coupled_result.state.surface_temperature[GROUND].item()
    )
    release_temperature = float(release_result["tsfc_ground"].item())
    assert coupled_temperature == pytest.approx(release_temperature, abs=0.5)
    coupled_lup = (
        asphalt.emissivity
        * STEFAN_BOLTZMANN
        * coupled_temperature**4
        + (1.0 - asphalt.emissivity) * sky_lw
    )
    release_lup = (
        asphalt.emissivity * STEFAN_BOLTZMANN * release_temperature**4
        + (1.0 - asphalt.emissivity) * sky_lw
    )
    assert coupled_lup == pytest.approx(release_lup, abs=2.0)


def test_numerical_controls_do_not_change_converged_answer(device):
    slow = _model(
        device,
        relaxation=0.35,
        max_temperature_step=2.0,
        max_coupling_iterations=500,
    )
    fast = _model(
        device,
        relaxation=0.85,
        max_temperature_step=12.0,
        max_coupling_iterations=500,
    )
    shortwave = torch.full_like(slow.geometry.area, 250.0)
    forcing = UrbanForcing(298.15, 2.0, 101.3, 2.0, 390.0, shortwave)
    slow_result = slow.step(slow.initialize_state(298.15), forcing)
    fast_result = fast.step(fast.initialize_state(298.15), forcing)
    assert slow_result.converged
    assert fast_result.converged
    assert torch.allclose(
        slow_result.state.surface_temperature,
        fast_result.state.surface_temperature,
        atol=0.03,
        rtol=0.0,
    )
