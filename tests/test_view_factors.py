# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

from study_code.geospatial_preprocessing.generate_coupled_geometry import main as geometry_main
from utherm.coupled_pipeline import load_geometry_bundle
from utherm.energy_balance.coupled import (
    CANOPY,
    GROUND,
    ROOF,
    WALL_SOUTH,
)
from utherm.view_factors import (
    ViewFactorConfig,
    _body_direction_sets,
    trace_direct_solar_projection,
    trace_geometry,
)


def _config(**changes):
    values = {
        "max_distance_m": 4.0,
        "ray_step_m": 1.0,
        "surface_ray_count": 8,
        "body_ray_count": 16,
    }
    values.update(changes)
    return ViewFactorConfig(**values)


def test_flat_scene_has_open_ground_and_balanced_standing_body_view():
    dem = np.zeros((11, 11), dtype=np.float64)
    result = trace_geometry(
        dem,
        dem,
        dem,
        window=(4, 4, 3, 3),
        config=_config(),
    )
    np.testing.assert_allclose(result.area[GROUND], 1.0, atol=1.0e-6)
    np.testing.assert_allclose(result.body_view_factor[GROUND], 0.5, atol=1.0e-6)
    np.testing.assert_allclose(result.body_sky_view_factor, 0.5, atol=1.0e-6)
    np.testing.assert_allclose(result.raw_surface_sky_view_factor[GROUND], 1.0)
    assert not np.any(result.area[ROOF:])


def test_body_rays_preserve_solweig_projected_area_weights():
    direction_sets = _body_direction_sets(128)
    assert sum(len(directions) for directions, _ in direction_sets) == 128
    np.testing.assert_allclose(
        [weight for _, weight in direction_sets],
        [0.06, 0.06, 0.22, 0.22, 0.22, 0.22],
    )
    assert sum(weight for _, weight in direction_sets) == pytest.approx(1.0)


def test_single_building_exposes_oriented_wall_and_reciprocal_exchange():
    dem = np.zeros((15, 15), dtype=np.float64)
    building = dem.copy()
    building[6:9, 6:9] = 10.0
    result = trace_geometry(
        dem,
        building,
        dem,
        window=(9, 7, 1, 1),
        config=_config(max_distance_m=6.0, surface_ray_count=16, body_ray_count=64),
    )
    assert result.area[ROOF, 0, 0] > 0.0
    assert result.area[WALL_SOUTH, 0, 0] > 0.0
    assert result.body_view_factor[WALL_SOUTH, 0, 0] > 0.1
    np.testing.assert_allclose(
        result.exchange_area,
        result.exchange_area.swapaxes(0, 1),
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        result.sky_view_area + result.exchange_area.sum(axis=1),
        result.area,
        atol=1.0e-7,
    )
    assert np.all(result.sky_view_area >= 0.0)


def test_canopy_lai_changes_transmission_continuously():
    dem = np.zeros((15, 15), dtype=np.float64)
    canopy = dem.copy()
    canopy[3:12, 3:12] = 8.0
    common = {
        "max_distance_m": 6.0,
        "surface_ray_count": 16,
        "body_ray_count": 64,
    }
    sparse = trace_geometry(
        dem,
        dem,
        canopy,
        window=(7, 7, 1, 1),
        config=_config(canopy_leaf_area_index=0.5, **common),
    )
    dense = trace_geometry(
        dem,
        dem,
        canopy,
        window=(7, 7, 1, 1),
        config=_config(canopy_leaf_area_index=6.0, **common),
    )
    assert dense.body_view_factor[CANOPY, 0, 0] > sparse.body_view_factor[CANOPY, 0, 0]
    assert dense.body_sky_view_factor[0, 0] < sparse.body_sky_view_factor[0, 0]


def _uniform_origins(rows, cols, *, x=0.5, y=0.5, z=0.02):
    return tuple(
        np.full((7, rows, cols), value, dtype=np.float64)
        for value in (x, y, z)
    )


def test_direct_solar_projection_uses_physical_facet_normals():
    dem = np.zeros((3, 3), dtype=np.float64)
    origins = _uniform_origins(1, 1, x=1.5, y=1.5)
    overhead = trace_direct_solar_projection(
        dem, dem, dem, *origins, 90.0, 180.0, config=_config()
    )
    np.testing.assert_allclose(overhead[[GROUND, ROOF, CANOPY]], 1.0, atol=1.0e-6)
    np.testing.assert_allclose(overhead[2:6], 0.0, atol=1.0e-6)

    from_south = trace_direct_solar_projection(
        dem, dem, dem, *origins, 30.0, 180.0, config=_config()
    )
    assert from_south[WALL_SOUTH, 0, 0] == pytest.approx(
        np.cos(np.deg2rad(30.0)), abs=1.0e-6
    )
    assert from_south[2, 0, 0] == pytest.approx(0.0, abs=1.0e-7)


def test_direct_solar_projection_resolves_building_occlusion():
    dem = np.zeros((15, 15), dtype=np.float64)
    building = dem.copy()
    building[5:8, 6:9] = 10.0
    origins = _uniform_origins(1, 1, x=7.5, y=11.5)
    clear = trace_direct_solar_projection(
        dem, dem, dem, *origins, 30.0, 0.0, config=_config(max_distance_m=10.0)
    )
    blocked = trace_direct_solar_projection(
        dem, building, dem, *origins, 30.0, 0.0, config=_config(max_distance_m=10.0)
    )
    assert clear[GROUND, 0, 0] == pytest.approx(0.5, abs=1.0e-6)
    assert blocked[GROUND, 0, 0] == pytest.approx(0.0, abs=1.0e-7)


def test_direct_solar_projection_resolves_canopy_transmission():
    dem = np.zeros((15, 15), dtype=np.float64)
    canopy = dem.copy()
    canopy[5:9, 5:10] = 8.0
    origins = _uniform_origins(1, 1, x=7.5, y=11.5)
    sparse = trace_direct_solar_projection(
        dem,
        dem,
        canopy,
        *origins,
        30.0,
        0.0,
        config=_config(max_distance_m=10.0, canopy_leaf_area_index=0.5),
    )
    dense = trace_direct_solar_projection(
        dem,
        dem,
        canopy,
        *origins,
        30.0,
        0.0,
        config=_config(max_distance_m=10.0, canopy_leaf_area_index=6.0),
    )
    assert 0.0 < dense[GROUND, 0, 0] < sparse[GROUND, 0, 0] < 0.5


def test_traced_bundle_loads_through_public_boundary(tmp_path, device):
    dem = np.zeros((9, 9), dtype=np.float64)
    result = trace_geometry(dem, dem, dem, window=(3, 3, 2, 2), config=_config())
    path = tmp_path / "coupled_geometry.npz"
    np.savez_compressed(
        path,
        area=result.area,
        sky_view_area=result.sky_view_area,
        exchange_area=result.exchange_area,
        body_view_factor=result.body_view_factor,
        body_sky_view_factor=result.body_sky_view_factor,
        crs_wkt=np.asarray('PROJCS["Test"]'),
        geotransform=np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, -1.0]),
    )
    loaded = load_geometry_bundle(path, 2, 2, device)
    assert loaded.geometry.area.shape == (7, 2, 2)


def test_geometry_input_and_configuration_fail_closed():
    dem = np.zeros((5, 5), dtype=np.float64)
    with pytest.raises(ValueError, match="same shape"):
        trace_geometry(dem, dem[:4], dem)
    bad_building = dem.copy()
    bad_building[0, 0] = -1.0
    with pytest.raises(ValueError, match="below the DEM"):
        trace_geometry(dem, bad_building, dem)
    with pytest.raises(ValueError, match="surface_ray_count must be even"):
        ViewFactorConfig(surface_ray_count=9).validate()
    with pytest.raises(ValueError, match="body_ray_count must be even"):
        ViewFactorConfig(body_ray_count=13).validate()


def test_centimetre_scale_dsm_roundoff_is_recorded_and_clamped():
    dem = np.zeros((7, 7), dtype=np.float64)
    building = dem.copy()
    building[3, 3] = -0.035
    result = trace_geometry(
        dem,
        building,
        dem,
        window=(2, 2, 3, 3),
        config=_config(),
    )
    assert result.dsm_clamped_cell_count == 1
    assert result.maximum_dsm_clamp_m == pytest.approx(0.035)


def test_physical_facet_area_does_not_depend_on_ray_count():
    dem = np.zeros((15, 15), dtype=np.float64)
    building = dem.copy()
    building[6:9, 6:9] = 10.0
    low = trace_geometry(
        dem,
        building,
        dem,
        window=(7, 7, 2, 2),
        config=_config(surface_ray_count=8, body_ray_count=16),
    )
    high = trace_geometry(
        dem,
        building,
        dem,
        window=(7, 7, 2, 2),
        config=_config(surface_ray_count=16, body_ray_count=32),
    )
    np.testing.assert_array_equal(low.area, high.area)


def test_geometry_cli_reads_only_required_halo(tmp_path, monkeypatch):
    shape = (30, 30)
    dem = np.zeros(shape, dtype=np.float32)
    building = dem.copy()
    building[14:17, 14:17] = 8.0
    canopy = dem.copy()
    canopy[11:14, 17:20] = 6.0
    landcover = np.ones(shape, dtype=np.float32)
    landcover[building > dem] = 2.0
    landcover[canopy > 0.0] = 5.0
    transform = from_origin(500_000.0, 4_000_000.0, 1.0, 1.0)
    paths = []
    for name, array in (
        ("dem", dem),
        ("building", building),
        ("canopy", canopy),
        ("landcover", landcover),
    ):
        path = tmp_path / f"{name}.tif"
        raster = array.copy()
        nodata = 0.0 if name == "canopy" else -9999.0
        if name != "canopy":
            raster[[0, -1], :] = nodata
            raster[:, [0, -1]] = nodata
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=shape[0],
            width=shape[1],
            count=1,
            dtype="float32",
            crs="EPSG:32611",
            transform=transform,
            nodata=nodata,
        ) as dataset:
            dataset.write(raster, 1)
        paths.append(path)
    output = tmp_path / "geometry.npz"
    monkeypatch.setattr(
        "sys.argv",
        [
            "utherm-geometry",
            "--dem",
            str(paths[0]),
            "--building-dsm",
            str(paths[1]),
            "--canopy-height",
            str(paths[2]),
            "--landcover",
            str(paths[3]),
            "--window",
            "14,14,2,2",
            "--max-distance",
            "4",
            "--ray-step",
            "1",
            "--surface-rays",
            "8",
            "--body-rays",
            "16",
            "--output",
            str(output),
        ],
    )
    geometry_main()
    with np.load(output, allow_pickle=False) as bundle:
        assert bundle["area"].shape == (7, 2, 2)
        np.testing.assert_array_equal(bundle["source_window"], [14, 14, 2, 2])
        np.testing.assert_array_equal(bundle["trace_output_window"], [6, 6, 2, 2])
        assert bundle["trace_dem"].shape == (14, 14)
        assert np.isfinite(bundle["trace_canopy_height"]).all()
        assert bundle["trace_canopy_height"].max() == 6.0
        assert bundle["trace_landcover"].shape == (14, 14)
        assert bundle["facet_origin_x"].shape == (7, 2, 2)
        assert rasterio.crs.CRS.from_wkt(str(bundle["crs_wkt"])).to_epsg() == 32611
    loaded = load_geometry_bundle(output, 2, 2, pytest.importorskip("torch").device("cpu"))
    assert loaded.solar_trace is not None
    assert loaded.raw_surface_sky_view_factor is not None
