# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the URock diagnostic wind field model (wind_field.py)."""

import math
import pytest
import torch

from utherm.wind_field import (
    WindFieldModel,
    WindFieldConfig,
    ZONE_OPEN,
    ZONE_WAKE,
    ZONE_DISPLACEMENT,
    ZONE_CAVITY,
    ZONE_CANYON,
    ZONE_BUILDING,
    Z_LEVELS,
    NZ,
)


# ── Helpers ──────────────────────────────────────────────────────

def _make_scene(device, rows=40, cols=40, bld_slice=None, bld_height=15.0, tree_slice=None, tree_height=8.0):
    """Create a simple urban scene with one building block and optional trees."""
    dem = torch.zeros(rows, cols, dtype=torch.float32, device=device)
    dsm = dem.clone()
    veg = torch.zeros(rows, cols, dtype=torch.float32, device=device)

    if bld_slice is None:
        # Default: 6x6 building in center
        bld_slice = (slice(17, 23), slice(17, 23))
    dsm[bld_slice] = bld_height

    if tree_slice is not None:
        veg[tree_slice] = tree_height

    return dsm, dem, veg


# ── Zone priority constants ──────────────────────────────────────

class TestZonePriorities:
    """Verify the zone priority ordering is correct (lower = higher priority)."""

    def test_open_is_lowest_priority(self):
        """ZONE_OPEN must be the highest number (lowest priority) so zones override it."""
        assert ZONE_OPEN > ZONE_WAKE
        assert ZONE_OPEN > ZONE_DISPLACEMENT
        assert ZONE_OPEN > ZONE_CAVITY
        assert ZONE_OPEN > ZONE_CANYON

    def test_canyon_is_highest_priority(self):
        assert ZONE_CANYON < ZONE_CAVITY
        assert ZONE_CANYON < ZONE_WAKE

    def test_cavity_overrides_wake(self):
        assert ZONE_CAVITY < ZONE_WAKE

    def test_building_blocks_all(self):
        """Buildings should block all non-canyon zones."""
        assert ZONE_BUILDING > ZONE_WAKE
        assert ZONE_BUILDING > ZONE_DISPLACEMENT
        assert ZONE_BUILDING > ZONE_CAVITY


# ── Wind field model init ────────────────────────────────────────

class TestWindFieldInit:

    def test_model_creates(self, device):
        dsm, dem, veg = _make_scene(device)
        cfg = WindFieldConfig()
        model = WindFieldModel(dsm, dem, veg, cfg, device)
        assert model.rows == 40
        assert model.cols == 40

    def test_building_detected(self, device):
        dsm, dem, veg = _make_scene(device, bld_height=15.0)
        cfg = WindFieldConfig()
        model = WindFieldModel(dsm, dem, veg, cfg, device)
        assert len(model._building_list) == 1
        assert model._building_list[0]['height'] == 15.0

    def test_no_buildings_when_flat(self, device):
        dem = torch.zeros(20, 20, dtype=torch.float32, device=device)
        dsm = dem.clone()
        veg = torch.zeros_like(dem)
        cfg = WindFieldConfig()
        model = WindFieldModel(dsm, dem, veg, cfg, device)
        assert len(model._building_list) == 0

    def test_small_buildings_filtered(self, device):
        """Buildings below min_building_area or min_building_height are excluded."""
        dem = torch.zeros(20, 20, dtype=torch.float32, device=device)
        dsm = dem.clone()
        # 1-pixel building (area < 4)
        dsm[10, 10] = 20.0
        veg = torch.zeros_like(dem)
        cfg = WindFieldConfig(min_building_area=4, min_building_height=2.0)
        model = WindFieldModel(dsm, dem, veg, cfg, device)
        assert len(model._building_list) == 0

    def test_input_shapes_must_match(self, device):
        dsm = torch.zeros(10, 10, device=device)
        dem = torch.zeros(9, 10, device=device)
        veg = torch.zeros_like(dsm)
        with pytest.raises(ValueError, match="input shapes differ"):
            WindFieldModel(dsm, dem, veg, WindFieldConfig(), device)

    @pytest.mark.parametrize("name", ["z_ref", "z_wind", "poisson_tol"])
    def test_positive_configuration_values_are_validated(self, name):
        values = {name: 0.0}
        with pytest.raises(ValueError, match=name):
            WindFieldConfig(**values)

    @pytest.mark.parametrize(
        "values",
        [
            {"min_wind_factor": float("nan")},
            {"max_wind_factor": float("inf")},
            {"min_wind_factor": 2.0, "max_wind_factor": 1.0},
        ],
    )
    def test_wind_factor_bounds_are_finite_and_ordered(self, values):
        with pytest.raises(ValueError, match="wind-factor bounds"):
            WindFieldConfig(**values)

    @pytest.mark.parametrize("name", ["poisson_iterations", "min_building_area"])
    def test_integer_configuration_rejects_boolean(self, name):
        with pytest.raises(ValueError, match=name):
            WindFieldConfig(**{name: True})


# ── Wind field computation ───────────────────────────────────────

class TestWindFieldCompute:

    def test_poisson_projection_reduces_divergence(self, device):
        """The mass-consistency projection must reduce, not amplify, divergence."""
        rows = cols = 20
        flat = torch.zeros(rows, cols, dtype=torch.float32, device=device)
        model = WindFieldModel(
            flat, flat, flat,
            WindFieldConfig(poisson_iterations=500, poisson_tol=1.0e-6), device,
        )
        generator = torch.Generator(device=device).manual_seed(2)
        u = torch.randn(NZ, rows, cols, generator=generator, device=device) * 0.2
        v = torch.randn(NZ, rows, cols, generator=generator, device=device) * 0.2
        w = torch.randn(NZ, rows, cols, generator=generator, device=device) * 0.05

        def divergence(u_field, v_field, w_field):
            value = torch.zeros_like(u_field)
            value[:, :, :-1] += u_field[:, :, 1:] - u_field[:, :, :-1]
            value[:, :-1, :] -= v_field[:, 1:, :] - v_field[:, :-1, :]
            for level in range(NZ - 1):
                dz = Z_LEVELS[level + 1] - Z_LEVELS[level]
                value[level] += (w_field[level + 1] - w_field[level]) / dz
            return value

        before = divergence(u, v, w)[1:-2, 2:-2, 2:-2].abs().mean()
        projected = model._solve_3d_poisson(u.clone(), v.clone(), w.clone())
        after = divergence(*projected)[1:-2, 2:-2, 2:-2].abs().mean()
        assert after < 0.05 * before
        assert model.last_poisson_iterations <= 500
        assert isinstance(model.last_poisson_converged, bool)
        assert math.isfinite(model.last_poisson_max_change)

    def test_returns_correct_shape(self, device):
        dsm, dem, veg = _make_scene(device)
        cfg = WindFieldConfig(poisson_iterations=10)
        model = WindFieldModel(dsm, dem, veg, cfg, device)
        speed, direction = model.compute_wind_field_uv(3.0, 225.0)
        assert speed.shape == (40, 40)
        assert direction.shape == (40, 40)

    def test_speed_nonnegative(self, device):
        dsm, dem, veg = _make_scene(device)
        cfg = WindFieldConfig(poisson_iterations=10)
        model = WindFieldModel(dsm, dem, veg, cfg, device)
        speed, _ = model.compute_wind_field_uv(3.0, 225.0)
        assert (speed >= 0).all()

    def test_building_pixels_zero_speed(self, device):
        dsm, dem, veg = _make_scene(device)
        cfg = WindFieldConfig(poisson_iterations=10)
        model = WindFieldModel(dsm, dem, veg, cfg, device)
        speed, direction = model.compute_wind_field_uv(3.0, 180.0)
        bld_mask = model.building_mask
        assert (speed[bld_mask] == 0.0).all()
        assert direction[bld_mask].isnan().all()

    def test_direction_in_range(self, device):
        dsm, dem, veg = _make_scene(device)
        cfg = WindFieldConfig(poisson_iterations=10)
        model = WindFieldModel(dsm, dem, veg, cfg, device)
        _, direction = model.compute_wind_field_uv(3.0, 225.0)
        valid = direction[~direction.isnan()]
        assert (valid >= 0).all()
        assert (valid < 360).all()

    def test_flat_terrain_uniform_direction(self, device):
        """No buildings → wind direction should be roughly uniform."""
        dem = torch.zeros(20, 20, dtype=torch.float32, device=device)
        dsm = dem.clone()
        veg = torch.zeros_like(dem)
        cfg = WindFieldConfig(poisson_iterations=10)
        model = WindFieldModel(dsm, dem, veg, cfg, device)
        _, direction = model.compute_wind_field_uv(3.0, 180.0)
        valid = direction[~direction.isnan()]
        # With no buildings, direction should be close to 180
        assert torch.abs(valid - 180.0).mean() < 5.0


# ── Zone assignment ──────────────────────────────────────────────

class TestZoneAssignment:

    def test_zones_actually_assigned(self, device):
        """With ZONE_OPEN=200, building zones should override open cells."""
        dsm, dem, veg = _make_scene(device, bld_height=15.0)
        cfg = WindFieldConfig(poisson_iterations=5)
        model = WindFieldModel(dsm, dem, veg, cfg, device)

        # Access internal zone assignment
        R, C = model.rows, model.cols
        u3d = torch.zeros((NZ, R, C), dtype=torch.float32, device=device)
        v3d = torch.zeros((NZ, R, C), dtype=torch.float32, device=device)
        w3d = torch.zeros((NZ, R, C), dtype=torch.float32, device=device)
        zone_priority = torch.full((NZ, R, C), ZONE_OPEN, dtype=torch.uint8, device=device)
        zone_priority[model._building_3d] = ZONE_BUILDING

        wind_dir_rad = math.radians(180.0)
        sin_wd = math.sin(wind_dir_rad)
        cos_wd = math.cos(wind_dir_rad)
        u_dir = -sin_wd
        v_dir = -cos_wd
        dr_dir = cos_wd
        dc_dir = -sin_wd

        model._compute_building_dimensions(wind_dir_rad)
        model._assign_building_zones(
            u3d, v3d, w3d, zone_priority,
            3.0, wind_dir_rad, u_dir, v_dir, dr_dir, dc_dir,
        )

        # There should be cells with zone priorities below ZONE_OPEN
        non_building = ~model._building_3d
        assigned = zone_priority[non_building] < ZONE_OPEN
        assert assigned.any(), "No zones were assigned — zone priority bug"

    def test_cavity_zone_behind_building(self, device):
        """Cavity zone should exist downwind of building."""
        dsm, dem, veg = _make_scene(device, rows=50, cols=50,
                                     bld_slice=(slice(20, 26), slice(22, 28)),
                                     bld_height=15.0)
        cfg = WindFieldConfig(poisson_iterations=5)
        model = WindFieldModel(dsm, dem, veg, cfg, device)

        R, C = model.rows, model.cols
        u3d = torch.zeros((NZ, R, C), dtype=torch.float32, device=device)
        v3d = torch.zeros((NZ, R, C), dtype=torch.float32, device=device)
        w3d = torch.zeros((NZ, R, C), dtype=torch.float32, device=device)
        zone_priority = torch.full((NZ, R, C), ZONE_OPEN, dtype=torch.uint8, device=device)
        zone_priority[model._building_3d] = ZONE_BUILDING

        # Wind from north (180° = from south in met convention... 0° = from north)
        wd = 0.0  # from north → wind travels south (row increases)
        wind_dir_rad = math.radians(wd)
        sin_wd = math.sin(wind_dir_rad)
        cos_wd = math.cos(wind_dir_rad)
        u_dir = -sin_wd
        v_dir = -cos_wd
        dr_dir = cos_wd
        dc_dir = -sin_wd

        model._compute_building_dimensions(wind_dir_rad)
        model._assign_building_zones(
            u3d, v3d, w3d, zone_priority,
            3.0, wind_dir_rad, u_dir, v_dir, dr_dir, dc_dir,
        )

        # Check downwind of building (south side, rows > 26) at ground level
        k = 1  # z=1.5m
        downwind_zones = zone_priority[k, 27:35, 22:28]
        has_cavity = (downwind_zones == ZONE_CAVITY).any()
        has_wake = (downwind_zones == ZONE_WAKE).any()
        assert has_cavity or has_wake, "No cavity or wake zone found behind building"


# ── Vegetation attenuation ───────────────────────────────────────

class TestVegetation:

    def test_trees_reduce_wind_speed(self, device):
        """Wind speed should be lower under tree canopy."""
        # Scene without trees
        dsm_no_tree, dem, _ = _make_scene(device, bld_slice=(slice(5, 8), slice(5, 8)),
                                           bld_height=10.0)
        veg_none = torch.zeros(40, 40, dtype=torch.float32, device=device)
        cfg = WindFieldConfig(poisson_iterations=10)
        model_no_tree = WindFieldModel(dsm_no_tree, dem, veg_none, cfg, device)
        speed_no_tree, _ = model_no_tree.compute_wind_field_uv(3.0, 225.0)

        # Scene with trees
        veg_trees = torch.zeros(40, 40, dtype=torch.float32, device=device)
        veg_trees[25:35, 25:35] = 8.0  # tree patch
        model_tree = WindFieldModel(dsm_no_tree.clone(), dem.clone(), veg_trees, cfg, device)
        speed_tree, _ = model_tree.compute_wind_field_uv(3.0, 225.0)

        # Mean speed in tree area should be lower
        tree_area = slice(25, 35), slice(25, 35)
        assert speed_tree[tree_area].mean() < speed_no_tree[tree_area].mean()

    def test_attenuation_stops_above_canopy(self, device):
        flat = torch.zeros(12, 12, dtype=torch.float32, device=device)
        vegetation = torch.zeros_like(flat)
        vegetation[4:8, 4:8] = 8.0
        model = WindFieldModel(flat, flat, vegetation, WindFieldConfig(), device)
        u3d = torch.ones((NZ, 12, 12), dtype=torch.float32, device=device)
        v3d = torch.ones_like(u3d)
        model._apply_vegetation_attenuation(u3d, v3d)
        assert (u3d[3, 4:8, 4:8] < 1.0).all()  # z=5 m, inside canopy
        assert (v3d[4, 4:8, 4:8] < 1.0).all()  # z=8 m, canopy top
        assert (u3d[5, 4:8, 4:8] == 1.0).all()  # z=12 m, above canopy


# ── Direction caching ────────────────────────────────────────────

class TestCaching:

    def test_cache_reuses_for_similar_direction(self, device):
        dsm, dem, veg = _make_scene(device)
        cfg = WindFieldConfig(poisson_iterations=10, direction_cache_tol=5.0)
        model = WindFieldModel(dsm, dem, veg, cfg, device)

        speed1, dir1 = model.compute_wind_field_uv(3.0, 225.0)
        # Same direction within tolerance should use cache
        speed2, dir2 = model.compute_wind_field_uv(4.0, 226.0)

        # Cached result is scaled by speed ratio
        assert model._cached_wd == 225.0  # direction didn't change in cache
        # Speed should scale proportionally
        ratio = speed2.mean() / speed1.mean()
        expected_ratio = 4.0 / 3.0
        assert abs(ratio.item() - expected_ratio) < 0.2

        # A third identical request must not fall back to the original 3 m/s field.
        speed3, _ = model.compute_wind_field_uv(4.0, 226.0)
        assert torch.allclose(speed3, speed2)

    def test_cache_direction_difference_wraps_at_north(self, device):
        dsm, dem, veg = _make_scene(device)
        model = WindFieldModel(
            dsm,
            dem,
            veg,
            WindFieldConfig(poisson_iterations=10, direction_cache_tol=5.0),
            device,
        )
        model.compute_wind_field_uv(3.0, 359.0)
        model.compute_wind_field_uv(3.0, 1.0)
        assert model._cached_wd == 359.0

    def test_cache_invalidated_for_different_direction(self, device):
        dsm, dem, veg = _make_scene(device)
        cfg = WindFieldConfig(poisson_iterations=10, direction_cache_tol=5.0)
        model = WindFieldModel(dsm, dem, veg, cfg, device)

        model.compute_wind_field_uv(3.0, 225.0)
        assert model._cached_wd == 225.0

        # Direction change > tolerance → recompute
        model.compute_wind_field_uv(3.0, 90.0)
        assert model._cached_wd == 90.0


# ── Speed clamping ───────────────────────────────────────────────

class TestSpeedClamping:

    def test_speed_within_factor_bounds(self, device):
        dsm, dem, veg = _make_scene(device)
        cfg = WindFieldConfig(
            poisson_iterations=10,
            min_wind_factor=0.05,
            max_wind_factor=1.5,
        )
        model = WindFieldModel(dsm, dem, veg, cfg, device)
        ws = 3.0
        speed, _ = model.compute_wind_field_uv(ws, 225.0)

        # Non-building pixels should be within clamped range
        non_bld = ~model.building_mask
        ws_ped = ws * model._height_ratio
        assert (speed[non_bld] >= cfg.min_wind_factor * ws_ped - 1e-5).all()
        assert (speed[non_bld] <= cfg.max_wind_factor * ws_ped + 1e-5).all()

    def test_invalid_wind_inputs_are_rejected(self, device):
        dsm, dem, veg = _make_scene(device)
        model = WindFieldModel(dsm, dem, veg, WindFieldConfig(poisson_iterations=2), device)
        with pytest.raises(ValueError, match="ws_ref"):
            model.compute_wind_field_uv(-1.0, 180.0)
        with pytest.raises(ValueError, match="wd_deg"):
            model.compute_wind_field_uv(1.0, float("nan"))
