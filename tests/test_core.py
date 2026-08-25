# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Public API raster validation tests."""

import numpy as np
import pytest

from utherm.core import (
    _expand_met_for_spinup,
    _load_met_file,
    _require_wind_directions,
    _validate_input_rasters,
    _validate_model_arguments,
    _validate_run_arguments,
    _validate_tile_maps,
    thermal_comfort,
)
from utherm.utci_process import (
    _daily_mean_air_temperature,
    _indices_for_current_day,
)

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin


def _write_raster(
    path, *, shape=(3, 4), crs="EPSG:32616", transform=None, count=1, data=None
):
    transform = transform or from_origin(500000.0, 4500000.0, 1.0, 1.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=count,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dataset:
        values = np.zeros((count, *shape), dtype=np.float32) if data is None else data
        dataset.write(values)


def _aligned_inputs(tmp_path):
    names = ("buildings.tif", "dem.tif", "trees.tif")
    for name in names:
        _write_raster(tmp_path / name)
    return names


def test_aligned_rasters_pass(tmp_path):
    names = _aligned_inputs(tmp_path)
    _validate_input_rasters(tmp_path, *names)


def test_missing_raster_raises_file_not_found(tmp_path):
    _write_raster(tmp_path / "buildings.tif")
    _write_raster(tmp_path / "dem.tif")
    with pytest.raises(FileNotFoundError, match="trees raster not found"):
        _validate_input_rasters(tmp_path, "buildings.tif", "dem.tif", "missing.tif")


def test_public_api_validates_before_creating_outputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        thermal_comfort(tmp_path, "2023-07-15")
    assert not (tmp_path / "processed_inputs").exists()
    assert not (tmp_path / "output_folder").exists()


def test_dimension_mismatch_raises_value_error(tmp_path):
    names = _aligned_inputs(tmp_path)
    _write_raster(tmp_path / names[2], shape=(4, 4))
    with pytest.raises(ValueError, match="dimensions do not match"):
        _validate_input_rasters(tmp_path, *names)


def test_crs_mismatch_raises_value_error(tmp_path):
    names = _aligned_inputs(tmp_path)
    _write_raster(tmp_path / names[2], crs="EPSG:4326")
    with pytest.raises(ValueError, match="CRS does not match"):
        _validate_input_rasters(tmp_path, *names)


def test_transform_mismatch_raises_value_error(tmp_path):
    names = _aligned_inputs(tmp_path)
    shifted = from_origin(500001.0, 4500000.0, 1.0, 1.0)
    _write_raster(tmp_path / names[2], transform=shifted)
    with pytest.raises(ValueError, match="transform does not match"):
        _validate_input_rasters(tmp_path, *names)


def test_multiband_input_is_rejected(tmp_path):
    names = _aligned_inputs(tmp_path)
    _write_raster(tmp_path / names[2], count=2)
    with pytest.raises(ValueError, match="must have one band"):
        _validate_input_rasters(tmp_path, *names)


def test_nonfinite_raster_is_rejected(tmp_path):
    names = _aligned_inputs(tmp_path)
    values = np.zeros((1, 3, 4), dtype=np.float32)
    values[0, 0, 0] = np.nan
    _write_raster(tmp_path / names[2], data=values)
    with pytest.raises(ValueError, match="non-finite"):
        _validate_input_rasters(tmp_path, *names)


@pytest.mark.parametrize(
    ("date", "tile_size", "overlap", "message"),
    [
        ("2023/07/15", 100, 20, "YYYY-MM-DD"),
        ("2023-07-15", 0, 0, "positive integer"),
        ("2023-07-15", 100, 100, "overlap"),
    ],
)
def test_invalid_run_arguments_raise_before_processing(date, tile_size, overlap, message):
    with pytest.raises(ValueError, match=message):
        _validate_run_arguments(date, tile_size, overlap)


@pytest.mark.parametrize("spinup_days", [-1, 1.5, True])
def test_invalid_spinup_days_are_rejected(spinup_days):
    with pytest.raises(ValueError, match="spinup_days"):
        _validate_run_arguments("2023-07-15", 100, 20, spinup_days)


def test_daily_spinup_groups_use_continuous_calendar_days():
    dectime = np.r_[738704 + np.arange(24) / 24, 738705 + np.arange(24) / 24]
    air_temperature = np.r_[np.full(24, 10.0), np.full(24, 20.0)]
    assert np.array_equal(_indices_for_current_day(dectime, 0), np.arange(24))
    assert np.array_equal(_indices_for_current_day(dectime, 47), np.arange(24, 48))
    assert _daily_mean_air_temperature(air_temperature, dectime, 5) == 10.0
    assert _daily_mean_air_temperature(air_temperature, dectime, 30) == 20.0


def test_generated_tile_sets_must_match():
    maps = {
        "building DSM": {"0_0": "building.tif", "100_0": "building2.tif"},
        "walls": {"0_0": "walls.tif"},
    }
    with pytest.raises(RuntimeError, match=r"walls.*missing=\['100_0'\]"):
        _validate_tile_maps(maps)


def test_generated_tile_sets_cannot_be_empty():
    with pytest.raises(RuntimeError, match="no generated tiles"):
        _validate_tile_maps({"building DSM": {}, "walls": {}})


def _write_met(path, values):
    header = " ".join(f"c{index}" for index in range(values.shape[1]))
    np.savetxt(path, values, header=header, comments="")


def _valid_met(rows=1, columns=23):
    values = np.zeros((rows, columns), dtype=float)
    values[:, 0:4] = (2023, 196, 12, 0)
    values[:, 9:15] = (2.0, 50.0, 25.0, 101.3, 0.0, 800.0)
    if columns > 21:
        values[:, 21] = 100.0
    if columns > 22:
        values[:, 22] = 700.0
    return values


def test_single_row_met_file_loads_as_two_dimensional(tmp_path):
    row = _valid_met()
    _write_met(tmp_path / "met.txt", row)
    loaded = _load_met_file(tmp_path / "met.txt", "2023-07-15")
    assert loaded.shape == (1, 23)


def test_spinup_repeats_forcing_on_preceding_calendar_days():
    rows = _valid_met(rows=2)
    rows[:, 2] = (0, 12)
    expanded, output_start = _expand_met_for_spinup(rows, "2023-07-15", 2)
    assert expanded.shape == (6, 23)
    assert output_start == 4
    assert expanded[:, :2].tolist() == [
        [2023.0, 194.0], [2023.0, 194.0],
        [2023.0, 195.0], [2023.0, 195.0],
        [2023.0, 196.0], [2023.0, 196.0],
    ]
    assert np.array_equal(expanded[:2, 2:], rows[:, 2:])


def test_missing_wind_direction_is_allowed_until_spatial_wind_is_requested(tmp_path):
    row = _valid_met(columns=24)
    row[0, 23] = -999.0
    _write_met(tmp_path / "met.txt", row)
    loaded = _load_met_file(tmp_path / "met.txt", "2023-07-15")
    with pytest.raises(ValueError, match="needs wind direction"):
        _require_wind_directions(loaded, tmp_path / "met.txt")


def test_valid_wind_direction_is_accepted_for_spatial_wind(tmp_path):
    row = _valid_met(columns=24)
    row[0, 23] = 270.0
    _write_met(tmp_path / "met.txt", row)
    loaded = _load_met_file(tmp_path / "met.txt", "2023-07-15")
    _require_wind_directions(loaded, tmp_path / "met.txt")


def test_met_file_date_must_match_simulation_date(tmp_path):
    row = _valid_met()
    row[0, 0] = 2022
    _write_met(tmp_path / "met.txt", row)
    with pytest.raises(ValueError, match="does not match"):
        _load_met_file(tmp_path / "met.txt", "2023-07-15")


def test_met_file_requires_all_model_columns(tmp_path):
    row = _valid_met(columns=22)
    _write_met(tmp_path / "met.txt", row)
    with pytest.raises(ValueError, match="23 columns"):
        _load_met_file(tmp_path / "met.txt", "2023-07-15")


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (0, 2023.5, "non-integer"),
        (1, 0, "day-of-year"),
        (2, 24, "hour"),
        (3, 60, "minute"),
    ],
)
def test_met_calendar_fields_are_validated(tmp_path, column, value, message):
    row = _valid_met()
    row[0, column] = value
    _write_met(tmp_path / "met.txt", row)
    with pytest.raises(ValueError, match=message):
        _load_met_file(tmp_path / "met.txt", "2023-07-15")


def test_met_timestamps_must_be_unique_and_increasing(tmp_path):
    rows = _valid_met(rows=2)
    _write_met(tmp_path / "met.txt", rows)
    with pytest.raises(ValueError, match="duplicate timestamps"):
        _load_met_file(tmp_path / "met.txt", "2023-07-15")

    rows[0, 2] = 13
    rows[1, 2] = 12
    _write_met(tmp_path / "met.txt", rows)
    with pytest.raises(ValueError, match="not increasing"):
        _load_met_file(tmp_path / "met.txt", "2023-07-15")


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (9, -0.1, "negative wind speed"),
        (10, 101.0, "relative humidity"),
        (11, 300.0, "air temperature"),
        (12, 1013.0, "pressure"),
        (13, -0.1, "negative precipitation"),
        (14, -1.0, "global radiation"),
        (21, -1.0, "diffuse radiation"),
    ],
)
def test_met_physical_ranges_are_validated(tmp_path, column, value, message):
    row = _valid_met()
    row[0, column] = value
    _write_met(tmp_path / "met.txt", row)
    with pytest.raises(ValueError, match=message):
        _load_met_file(tmp_path / "met.txt", "2023-07-15")


def _model_arguments(tmp_path):
    met = tmp_path / "met.txt"
    met.write_text("placeholder\n")
    return {
        "use_own_met": True,
        "own_met_file": met,
        "data_source_type": None,
        "data_folder": None,
        "start_time": None,
        "end_time": None,
        "use_energy_balance": False,
        "use_coupled_eb": False,
        "coupled_geometry_path": None,
        "coupled_spinup_max_cycles": 30,
        "coupled_strict_convergence": True,
        "z0": 0.01,
        "z_wind": None,
        "ground_material_type": "asphalt",
        "use_roof_eb": False,
        "roof_material_type": "roof",
        "use_canopy_eb": False,
        "canopy_type": "deciduous",
        "canopy_lai": 3.0,
        "use_canyon_air_temp": False,
        "use_wind_field": False,
        "tair_method": "conductance-blend",
        "ground_wetness": None,
        "canopy_gmax_fraction": None,
        "requested_outputs": {
            "save_qh": False,
            "save_qe": False,
            "save_tsfc": False,
            "save_tsfc_roof": False,
            "save_t_leaf": False,
            "save_canopy_qh": False,
            "save_canopy_qe": False,
            "save_tair_canyon": False,
            "save_wall_temperature": False,
            "save_wind_speed": False,
            "save_wind_direction": False,
        },
    }


def test_unknown_material_is_rejected(tmp_path):
    arguments = _model_arguments(tmp_path)
    arguments["ground_material_type"] = "mystery"
    with pytest.raises(ValueError, match="ground_material_type"):
        _validate_model_arguments(**arguments)


def test_canopy_model_requires_energy_balance(tmp_path):
    arguments = _model_arguments(tmp_path)
    arguments["use_canopy_eb"] = True
    with pytest.raises(ValueError, match="require use_energy_balance"):
        _validate_model_arguments(**arguments)


def test_disabled_diagnostic_output_is_rejected(tmp_path):
    arguments = _model_arguments(tmp_path)
    arguments["requested_outputs"]["save_wind_speed"] = True
    with pytest.raises(ValueError, match="not enabled"):
        _validate_model_arguments(**arguments)


def test_coupled_solver_requires_real_geometry(tmp_path):
    arguments = _model_arguments(tmp_path)
    arguments["use_coupled_eb"] = True
    with pytest.raises(ValueError, match="coupled_geometry_path is required"):
        _validate_model_arguments(**arguments)


def test_coupled_and_modular_energy_balances_are_mutually_exclusive(tmp_path):
    geometry = tmp_path / "geometry.npz"
    geometry.write_bytes(b"placeholder")
    arguments = _model_arguments(tmp_path)
    arguments.update(
        use_coupled_eb=True,
        use_energy_balance=True,
        coupled_geometry_path=geometry,
    )
    with pytest.raises(ValueError, match="alternative solvers"):
        _validate_model_arguments(**arguments)


def test_coupled_solver_rejects_legacy_facet_switches(tmp_path):
    geometry = tmp_path / "geometry.npz"
    geometry.write_bytes(b"placeholder")
    arguments = _model_arguments(tmp_path)
    arguments.update(
        use_coupled_eb=True,
        coupled_geometry_path=geometry,
        use_roof_eb=True,
    )
    with pytest.raises(ValueError, match="legacy facet switches"):
        _validate_model_arguments(**arguments)


def test_wall_temperature_output_requires_coupled_solver(tmp_path):
    arguments = _model_arguments(tmp_path)
    arguments["requested_outputs"]["save_wall_temperature"] = True
    with pytest.raises(ValueError, match="not enabled"):
        _validate_model_arguments(**arguments)
