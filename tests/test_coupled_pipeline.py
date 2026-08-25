# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later

import json

import numpy as np
import pytest
import torch

from utherm.coupled_pipeline import (
    CoupledRadiationBridge,
    ground_material_ids_from_trace,
    load_geometry_bundle,
    resolve_geometry_path,
)
from utherm.energy_balance import CanopyProperties, MaterialProperties
from utherm.energy_balance.coupled import N_FACETS, UrbanForcing
from utherm.utci_process import _dry_coupled_spinup_forcings


def _write_bundle(
    path,
    rows=1,
    cols=1,
    *,
    broken_body=False,
    with_solar_trace=False,
    trace_landcover=None,
    ground_origin_x=None,
):
    area = np.array(
        [0.7, 0.3, 0.25, 0.25, 0.25, 0.25, 0.8], dtype=np.float32
    )[:, None, None]
    area = np.broadcast_to(area, (N_FACETS, rows, cols)).copy()
    sky = area.copy()
    exchange = np.zeros((N_FACETS, N_FACETS, rows, cols), dtype=np.float32)
    body = np.array(
        [0.20, 0.0, 0.15, 0.15, 0.15, 0.15, 0.10], dtype=np.float32
    )[:, None, None]
    body = np.broadcast_to(body, (N_FACETS, rows, cols)).copy()
    body_sky = np.full((rows, cols), 0.10 if not broken_body else 0.20, dtype=np.float32)
    payload = dict(
        area=area,
        sky_view_area=sky,
        exchange_area=exchange,
        body_view_factor=body,
        body_sky_view_factor=body_sky,
        crs_wkt=np.array('PROJCS["Synthetic test grid"]'),
        geotransform=np.array([0.0, 1.0, 0.0, 0.0, 0.0, -1.0]),
    )
    if with_solar_trace:
        grid_y, grid_x = np.indices((rows, cols), dtype=np.float32)
        origin_x = np.broadcast_to(grid_x[None] + 0.5, (N_FACETS, rows, cols)).copy()
        origin_y = np.broadcast_to(grid_y[None] + 0.5, (N_FACETS, rows, cols)).copy()
        origin_z = np.full((N_FACETS, rows, cols), 0.02, dtype=np.float32)
        raw_sky = np.broadcast_to(
            np.array([1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)[:, None, None],
            (N_FACETS, rows, cols),
        ).copy()
        if ground_origin_x is not None:
            origin_x[0] = np.asarray(ground_origin_x, dtype=np.float32)
        if trace_landcover is None:
            trace_landcover = np.ones((rows, cols), dtype=np.int16)
        payload.update(
            raw_surface_sky_view_factor=raw_sky,
            facet_origin_x=origin_x,
            facet_origin_y=origin_y,
            facet_origin_z=origin_z,
            trace_dem=np.zeros((rows, cols), dtype=np.float32),
            trace_building_dsm=np.zeros((rows, cols), dtype=np.float32),
            trace_canopy_height=np.zeros((rows, cols), dtype=np.float32),
            trace_landcover=np.asarray(trace_landcover, dtype=np.int16),
            trace_output_window=np.array([0, 0, rows, cols], dtype=np.int64),
            config_json=np.array(
                json.dumps(
                    {
                        "pixel_size_m": 1.0,
                        "max_distance_m": 4.0,
                        "ray_step_m": 1.0,
                        "surface_ray_count": 8,
                        "body_ray_count": 16,
                    }
                )
            ),
        )
    np.savez(path, **payload)


def test_geometry_bundle_loads_and_resolves_per_tile(tmp_path, device):
    directory = tmp_path / "geometry"
    directory.mkdir()
    path = directory / "coupled_geometry_0_0.npz"
    _write_bundle(path, 2, 3)
    assert resolve_geometry_path(directory, "0_0") == path.resolve()
    bundle = load_geometry_bundle(path, 2, 3, device)
    assert bundle.geometry.area.shape == (N_FACETS, 2, 3)
    assert bundle.body_view_factor.device.type == device.type
    assert torch.all(bundle.water_capacity >= 0.0)


def test_geometry_bundle_rejects_body_view_factor_nonclosure(tmp_path, device):
    path = tmp_path / "geometry.npz"
    _write_bundle(path, broken_body=True)
    with pytest.raises(ValueError, match="body view factors do not close"):
        load_geometry_bundle(path, 1, 1, device)


def test_public_coupled_spinup_is_dry_without_changing_target_rain(device):
    forcing = UrbanForcing(
        298.15,
        2.0,
        101.3,
        2.0,
        390.0,
        torch.zeros((N_FACETS, 1, 1), device=device),
        precipitation_rate=0.001,
    )
    spinup = _dry_coupled_spinup_forcings([forcing])
    assert spinup[0].precipitation_rate == 0.0
    assert forcing.precipitation_rate == 0.001


def test_coupled_bridge_runs_cycle_and_replaces_surface_longwave(tmp_path, device):
    path = tmp_path / "geometry.npz"
    _write_bundle(path)
    bundle = load_geometry_bundle(path, 1, 1, device)
    bridge = CoupledRadiationBridge(
        bundle,
        ground_material=MaterialProperties.asphalt(),
        roof_material=MaterialProperties.roof(),
        canopy_properties=CanopyProperties.deciduous(),
        dt=60.0,
        spinup_max_cycles=2,
        strict_convergence=False,
    )
    shortwave = torch.zeros_like(bundle.geometry.area)
    forcing = UrbanForcing(
        298.15,
        2.0,
        101.3,
        2.0,
        390.0,
        shortwave,
    )
    body_shortwave = torch.full((1, 1), 100.0, device=device)
    results, radiation = bridge.solve_cycle([forcing], [body_shortwave])
    assert len(results) == len(radiation) == 1
    assert radiation[0].outgoing_longwave.shape == (N_FACETS, 1, 1)
    assert torch.isfinite(radiation[0].tmrt_celsius).all()
    assert float(radiation[0].upward_longwave.item()) > 300.0
    assert 0.0 <= float(radiation[0].relative_humidity_percent.item()) <= 100.0
    with pytest.raises(ValueError, match="spin-up and output forcing cycles"):
        bridge.solve_cycle(
            [forcing],
            [body_shortwave],
            spinup_forcings=[forcing, forcing],
        )


def test_coupled_bridge_builds_orientation_resolved_external_shortwave(tmp_path, device):
    path = tmp_path / "geometry.npz"
    _write_bundle(path, with_solar_trace=True)
    bundle = load_geometry_bundle(path, 1, 1, device)
    bridge = CoupledRadiationBridge(
        bundle,
        ground_material=MaterialProperties.asphalt(),
        roof_material=MaterialProperties.roof(),
        canopy_properties=CanopyProperties.deciduous(),
        dt=60.0,
        spinup_max_cycles=2,
        strict_convergence=False,
    )
    shortwave = bridge.facet_shortwave_irradiance(
        direct_normal_irradiance=800.0,
        diffuse_horizontal_irradiance=100.0,
        solar_altitude_degrees=30.0,
        solar_azimuth_degrees=180.0,
    )
    plan_flux = (shortwave * bundle.geometry.area).sum(dim=0)
    assert plan_flux.item() == pytest.approx(500.0, abs=1.0e-4)
    assert shortwave[4, 0, 0].item() > shortwave[2, 0, 0].item()
    assert shortwave[0, 0, 0].item() > shortwave[2, 0, 0].item()
    assert torch.all(shortwave >= 0.0)
    assert shortwave.max().item() <= 900.0
    direct_only = bridge.facet_shortwave_irradiance(
        direct_normal_irradiance=800.0,
        diffuse_horizontal_irradiance=0.0,
        solar_altitude_degrees=30.0,
        solar_azimuth_degrees=180.0,
    )
    diffuse_only = bridge.facet_shortwave_irradiance(
        direct_normal_irradiance=0.0,
        diffuse_horizontal_irradiance=100.0,
        solar_altitude_degrees=30.0,
        solar_azimuth_degrees=180.0,
    )
    assert (direct_only * bundle.geometry.area).sum().item() == pytest.approx(400.0, abs=1.0e-4)
    assert (diffuse_only * bundle.geometry.area).sum().item() == pytest.approx(100.0, abs=1.0e-4)
    assert direct_only[2, 0, 0].item() == 0.0
    assert direct_only.max().item() <= 800.0
    assert diffuse_only.max().item() <= 100.0


def test_coupled_bridge_shortwave_fails_closed_without_trace_payload(tmp_path, device):
    path = tmp_path / "geometry.npz"
    _write_bundle(path)
    bundle = load_geometry_bundle(path, 1, 1, device)
    bridge = CoupledRadiationBridge(
        bundle,
        ground_material=MaterialProperties.asphalt(),
        roof_material=MaterialProperties.roof(),
        canopy_properties=CanopyProperties.deciduous(),
        dt=60.0,
        spinup_max_cycles=2,
        strict_convergence=False,
    )
    with pytest.raises(ValueError, match="solar-trace payload"):
        bridge.facet_shortwave_irradiance(
            direct_normal_irradiance=800.0,
            diffuse_horizontal_irradiance=100.0,
            solar_altitude_degrees=30.0,
            solar_azimuth_degrees=180.0,
        )


def test_ground_material_ids_follow_representative_surface_origin(tmp_path, device):
    path = tmp_path / "geometry.npz"
    landcover = np.array([[1, 2], [5, 6]], dtype=np.int16)
    _write_bundle(
        path,
        rows=2,
        cols=2,
        with_solar_trace=True,
        trace_landcover=landcover,
        ground_origin_x=np.array([[1.5, 0.5], [1.5, 0.5]], dtype=np.float32),
    )
    bundle = load_geometry_bundle(path, 2, 2, device)
    np.testing.assert_array_equal(
        ground_material_ids_from_trace(bundle, landcover),
        [[2, 1], [6, 5]],
    )
    with pytest.raises(ValueError, match="does not match"):
        ground_material_ids_from_trace(bundle, np.ones((2, 2), dtype=np.int16))
