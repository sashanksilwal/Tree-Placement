# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Geometry-derived view factors for the coupled seven-facet model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter

from .energy_balance.coupled import (
    CANOPY,
    GROUND,
    N_FACETS,
    ROOF,
    WALL_EAST,
    WALL_NORTH,
    WALL_SOUTH,
    WALL_WEST,
)


SKY = N_FACETS


@dataclass(frozen=True)
class ViewFactorConfig:
    pixel_size_m: float = 1.0
    max_distance_m: float = 60.0
    ray_step_m: float = 0.5
    surface_ray_count: int = 128
    body_ray_count: int = 256
    body_height_m: float = 1.1
    building_threshold_m: float = 2.0
    canopy_threshold_m: float = 2.0
    canopy_leaf_area_index: float = 3.0
    canopy_extinction: float = 0.5
    canopy_crown_depth_fraction: float = 0.6
    minimum_transmittance: float = 1.0e-5
    dsm_below_dem_tolerance_m: float = 0.1

    def validate(self) -> None:
        positive = {
            "pixel_size_m": self.pixel_size_m,
            "max_distance_m": self.max_distance_m,
            "ray_step_m": self.ray_step_m,
            "body_height_m": self.body_height_m,
            "canopy_leaf_area_index": self.canopy_leaf_area_index,
            "canopy_extinction": self.canopy_extinction,
            "canopy_crown_depth_fraction": self.canopy_crown_depth_fraction,
            "minimum_transmittance": self.minimum_transmittance,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.ray_step_m > self.pixel_size_m:
            raise ValueError("ray_step_m cannot exceed pixel_size_m")
        if self.max_distance_m < self.pixel_size_m:
            raise ValueError("max_distance_m must be at least one pixel")
        for name, value, minimum in (
            ("surface_ray_count", self.surface_ray_count, 8),
            ("body_ray_count", self.body_ray_count, 12),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if self.surface_ray_count % 2:
            raise ValueError("surface_ray_count must be even")
        if self.body_ray_count % 2:
            raise ValueError("body_ray_count must be even")
        if not 0.0 < self.canopy_crown_depth_fraction <= 1.0:
            raise ValueError("canopy_crown_depth_fraction must be in (0, 1]")
        if not 0.0 < self.minimum_transmittance < 1.0:
            raise ValueError("minimum_transmittance must be in (0, 1)")
        for name, value in (
            ("building_threshold_m", self.building_threshold_m),
            ("canopy_threshold_m", self.canopy_threshold_m),
            ("dsm_below_dem_tolerance_m", self.dsm_below_dem_tolerance_m),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class TracedGeometry:
    area: np.ndarray
    sky_view_area: np.ndarray
    exchange_area: np.ndarray
    body_view_factor: np.ndarray
    body_sky_view_factor: np.ndarray
    raw_surface_sky_view_factor: np.ndarray
    facet_origin_x: np.ndarray
    facet_origin_y: np.ndarray
    facet_origin_z: np.ndarray
    window: tuple[int, int, int, int]
    dsm_clamped_cell_count: int
    maximum_dsm_clamp_m: float

    def validate(self, tolerance: float = 2.0e-4) -> None:
        if self.area.ndim != 3 or self.area.shape[0] != N_FACETS:
            raise ValueError("area must have shape (7, rows, cols)")
        rows, cols = self.area.shape[1:]
        expected = {
            "sky_view_area": (N_FACETS, rows, cols),
            "exchange_area": (N_FACETS, N_FACETS, rows, cols),
            "body_view_factor": (N_FACETS, rows, cols),
            "body_sky_view_factor": (rows, cols),
            "raw_surface_sky_view_factor": (N_FACETS, rows, cols),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} does not match {shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
            if np.any(value < -tolerance):
                raise ValueError(f"{name} contains negative values")
        for name in ("facet_origin_x", "facet_origin_y", "facet_origin_z"):
            value = getattr(self, name)
            shape = (N_FACETS, rows, cols)
            if value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} does not match {shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        asymmetry = np.max(np.abs(self.exchange_area - self.exchange_area.swapaxes(0, 1)))
        if asymmetry > tolerance:
            raise ValueError(f"exchange areas are not reciprocal; error={asymmetry:.3g}")
        closure = self.sky_view_area + self.exchange_area.sum(axis=1)
        if not np.allclose(closure, self.area, atol=tolerance, rtol=tolerance):
            raise ValueError("facet exchange areas do not close")
        body_closure = self.body_view_factor.sum(axis=0) + self.body_sky_view_factor
        if not np.allclose(body_closure, 1.0, atol=tolerance, rtol=0.0):
            raise ValueError("body view factors do not close")
        if np.any((self.area <= 0.0) & (self.body_view_factor > tolerance)):
            raise ValueError("body view assigned to an inactive facet")
        if isinstance(self.dsm_clamped_cell_count, bool) or not isinstance(
            self.dsm_clamped_cell_count, (int, np.integer)
        ):
            raise ValueError("dsm_clamped_cell_count must be a nonnegative integer")
        if self.dsm_clamped_cell_count < 0:
            raise ValueError("dsm_clamped_cell_count must be a nonnegative integer")
        if not math.isfinite(self.maximum_dsm_clamp_m) or self.maximum_dsm_clamp_m < 0.0:
            raise ValueError("maximum_dsm_clamp_m must be finite and nonnegative")


def _validate_inputs(
    dem: np.ndarray,
    building_dsm: np.ndarray,
    canopy_height: np.ndarray,
    window: Sequence[int] | None,
    dsm_below_dem_tolerance_m: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[int, int, int, int],
    int,
    float,
]:
    arrays = [np.asarray(value, dtype=np.float64) for value in (dem, building_dsm, canopy_height)]
    if any(value.ndim != 2 for value in arrays):
        raise ValueError("DEM, building DSM and canopy height must be two-dimensional")
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("DEM, building DSM and canopy height must have the same shape")
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError("geometry inputs must contain only finite values")
    dem_array, building_array, canopy_array = arrays
    dsm_correction = np.maximum(dem_array - building_array, 0.0)
    maximum_dsm_clamp = float(dsm_correction.max())
    if maximum_dsm_clamp > dsm_below_dem_tolerance_m:
        raise ValueError(
            "building DSM cannot be below the DEM by more than "
            f"{dsm_below_dem_tolerance_m:g} m; maximum difference={maximum_dsm_clamp:.6g} m"
        )
    dsm_clamped_cell_count = int(np.count_nonzero(dsm_correction > 1.0e-6))
    building_array = np.maximum(building_array, dem_array)
    if np.any(canopy_array < 0.0):
        raise ValueError("canopy height cannot be negative")
    rows, cols = dem_array.shape
    if window is None:
        resolved = (0, 0, rows, cols)
    else:
        if len(window) != 4 or any(isinstance(value, bool) for value in window):
            raise ValueError("window must contain row, column, height and width")
        resolved = tuple(int(value) for value in window)
        if tuple(window) != resolved:
            raise ValueError("window values must be integers")
    row, col, height, width = resolved
    if row < 0 or col < 0 or height < 1 or width < 1:
        raise ValueError("window must have a nonnegative origin and positive size")
    if row + height > rows or col + width > cols:
        raise ValueError("window extends outside the geometry inputs")
    return (
        dem_array,
        building_array,
        canopy_array,
        resolved,
        dsm_clamped_cell_count,
        maximum_dsm_clamp,
    )


def _radical_inverse_base_two(value: int) -> float:
    result = 0.0
    scale = 0.5
    while value:
        result += scale * (value & 1)
        value >>= 1
        scale *= 0.5
    return result


def _cosine_directions(normal: Sequence[float], count: int) -> np.ndarray:
    normal_array = np.asarray(normal, dtype=np.float64)
    normal_array /= np.linalg.norm(normal_array)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(normal_array[2])) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    tangent = np.cross(helper, normal_array)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal_array, tangent)
    directions = []
    pair_count = count // 2
    for index in range(pair_count):
        radial = math.sqrt((index + 0.5) / max(pair_count, 1))
        azimuth = 2.0 * math.pi * (_radical_inverse_base_two(index) + 0.125)
        local_z = math.sqrt(max(0.0, 1.0 - radial * radial))
        for opposite in (0.0, math.pi):
            angle = azimuth + opposite
            local_x = radial * math.cos(angle)
            local_y = radial * math.sin(angle)
            direction = tangent * local_x + bitangent * local_y + normal_array * local_z
            directions.append(direction / np.linalg.norm(direction))
    if count % 2:
        directions.append(normal_array.copy())
    return np.asarray(directions)


def _body_direction_sets(count: int) -> tuple[tuple[np.ndarray, float], ...]:
    normals_and_weights = (
        ((0.0, 0.0, 1.0), 0.06),
        ((0.0, 0.0, -1.0), 0.06),
        ((0.0, -1.0, 0.0), 0.22),
        ((1.0, 0.0, 0.0), 0.22),
        ((0.0, 1.0, 0.0), 0.22),
        ((-1.0, 0.0, 0.0), 0.22),
    )
    remaining = count // 2 - len(normals_and_weights)
    desired = np.asarray([weight for _, weight in normals_and_weights]) * remaining
    extra = np.floor(desired).astype(int)
    for index in np.argsort(-(desired - extra))[: remaining - int(extra.sum())]:
        extra[index] += 1
    return tuple(
        (_cosine_directions(normal, 2 * int(ray_pairs)), weight)
        for (normal, weight), ray_pairs in zip(normals_and_weights, extra + 1)
    )


def _surface_directions(facet: int, count: int) -> np.ndarray:
    normals = {
        GROUND: (0.0, 0.0, 1.0),
        ROOF: (0.0, 0.0, 1.0),
        WALL_NORTH: (0.0, -1.0, 0.0),
        WALL_EAST: (1.0, 0.0, 0.0),
        WALL_SOUTH: (0.0, 1.0, 0.0),
        WALL_WEST: (-1.0, 0.0, 0.0),
    }
    if facet == CANOPY:
        half = count // 2
        return np.concatenate(
            (_cosine_directions((0.0, 0.0, 1.0), half), _cosine_directions((0.0, 0.0, -1.0), count - half)),
            axis=0,
        )
    return _cosine_directions(normals[facet], count)


def _wall_areas(
    dem: np.ndarray,
    building_dsm: np.ndarray,
    building: np.ndarray,
    pixel_size: float,
) -> dict[int, np.ndarray]:
    surface = np.where(building, building_dsm, dem)
    padded = np.pad(surface, 1, mode="edge")
    neighbours = {
        WALL_NORTH: padded[:-2, 1:-1],
        WALL_EAST: padded[1:-1, 2:],
        WALL_SOUTH: padded[2:, 1:-1],
        WALL_WEST: padded[1:-1, :-2],
    }
    return {
        facet: np.where(building, np.maximum(building_dsm - np.maximum(dem, neighbour), 0.0) / pixel_size, 0.0)
        for facet, neighbour in neighbours.items()
    }


def _local_surface_areas(
    dem: np.ndarray,
    building_dsm: np.ndarray,
    canopy_height: np.ndarray,
    config: ViewFactorConfig,
) -> tuple[np.ndarray, dict[int, np.ndarray], np.ndarray]:
    height = building_dsm - dem
    building = height >= config.building_threshold_m
    canopy = (canopy_height >= config.canopy_threshold_m) & ~building
    wall = _wall_areas(dem, building_dsm, building, config.pixel_size_m)
    source = {
        GROUND: ~building,
        ROOF: building,
        WALL_NORTH: wall[WALL_NORTH] > 0.0,
        WALL_EAST: wall[WALL_EAST] > 0.0,
        WALL_SOUTH: wall[WALL_SOUTH] > 0.0,
        WALL_WEST: wall[WALL_WEST] > 0.0,
        CANOPY: canopy,
    }
    physical = np.stack(
        (
            source[GROUND].astype(np.float64),
            source[ROOF].astype(np.float64),
            wall[WALL_NORTH],
            wall[WALL_EAST],
            wall[WALL_SOUTH],
            wall[WALL_WEST],
            source[CANOPY].astype(np.float64) * config.canopy_leaf_area_index,
        )
    )
    radius = max(1, int(math.ceil(config.max_distance_m / config.pixel_size_m)))
    size = 2 * radius + 1
    area = np.stack(
        [uniform_filter(value, size=size, mode="constant", cval=0.0) for value in physical]
    )
    return area, source, wall


def _nearest_indices(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not bool(mask.any()):
        shape = mask.shape
        return (
            np.full(shape, np.inf, dtype=np.float64),
            np.zeros(shape, dtype=np.int64),
            np.zeros(shape, dtype=np.int64),
        )
    distance, indices = distance_transform_edt(~mask, return_indices=True)
    return distance, indices[0], indices[1]


def _source_origins(
    facet: int,
    source_mask: np.ndarray,
    wall_area: dict[int, np.ndarray],
    dem: np.ndarray,
    building_dsm: np.ndarray,
    canopy_height: np.ndarray,
    window: tuple[int, int, int, int],
    config: ViewFactorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row, col, height, width = window
    distance, nearest_row, nearest_col = _nearest_indices(source_mask)
    source_row = nearest_row[row : row + height, col : col + width]
    source_col = nearest_col[row : row + height, col : col + width]
    source_distance = distance[row : row + height, col : col + width] * config.pixel_size_m
    x = source_col.astype(np.float64) + 0.5
    y = source_row.astype(np.float64) + 0.5
    ground = dem[source_row, source_col]
    top = building_dsm[source_row, source_col]
    epsilon = 0.02 / config.pixel_size_m
    if facet == GROUND:
        z = ground + 0.02
    elif facet == ROOF:
        z = top + 0.02
    elif facet == CANOPY:
        z = ground + canopy_height[source_row, source_col] + 0.02
    else:
        exposed_height = wall_area[facet][source_row, source_col] * config.pixel_size_m
        z = top - 0.5 * exposed_height
        if facet == WALL_NORTH:
            y = source_row.astype(np.float64) - epsilon
        elif facet == WALL_EAST:
            x = source_col.astype(np.float64) + 1.0 + epsilon
        elif facet == WALL_SOUTH:
            y = source_row.astype(np.float64) + 1.0 + epsilon
        else:
            x = source_col.astype(np.float64) - epsilon
    active = source_distance <= config.max_distance_m * (1.0 + 1.0e-9)
    return x, y, z, active


def _wall_target(direction: np.ndarray) -> int:
    dx, dy = float(direction[0]), float(direction[1])
    if abs(dx) >= abs(dy):
        return WALL_WEST if dx > 0.0 else WALL_EAST
    return WALL_NORTH if dy > 0.0 else WALL_SOUTH


def _trace_direction(
    origin_x: np.ndarray,
    origin_y: np.ndarray,
    origin_z: np.ndarray,
    direction: np.ndarray,
    dem: np.ndarray,
    building_dsm: np.ndarray,
    canopy_height: np.ndarray,
    config: ViewFactorConfig,
) -> np.ndarray:
    result = np.zeros((N_FACETS + 1, *origin_x.shape), dtype=np.float64)
    remaining = np.ones(origin_x.shape, dtype=np.float64)
    dx, dy, dz = (float(value) for value in direction)
    distance_values = np.arange(config.ray_step_m, config.max_distance_m + 0.5 * config.ray_step_m, config.ray_step_m)
    wall_target = _wall_target(direction)
    rows, cols = dem.shape
    building_height = building_dsm - dem
    for distance_m in distance_values:
        active = remaining > config.minimum_transmittance
        if not bool(active.any()):
            break
        distance_grid = distance_m / config.pixel_size_m
        x = origin_x + dx * distance_grid
        y = origin_y + dy * distance_grid
        z = origin_z + dz * distance_m
        sample_col = np.floor(x).astype(np.int64)
        sample_row = np.floor(y).astype(np.int64)
        inside = active & (sample_row >= 0) & (sample_row < rows) & (sample_col >= 0) & (sample_col < cols)
        escaped = active & ~inside
        escape_target = GROUND if dz < 0.0 else SKY
        result[escape_target, escaped] += remaining[escaped]
        remaining[escaped] = 0.0
        if not bool(inside.any()):
            continue
        flat = np.flatnonzero(inside)
        rr = sample_row.flat[flat]
        cc = sample_col.flat[flat]
        ray_z = z.flat[flat]
        weight = remaining.flat[flat]
        local_dem = dem[rr, cc]
        local_top = building_dsm[rr, cc]
        local_building = building_height[rr, cc] >= config.building_threshold_m
        solid = local_building & (ray_z <= local_top)
        if bool(solid.any()):
            solid_flat = flat[solid]
            previous_z = origin_z.flat[solid_flat] + dz * max(0.0, distance_m - config.ray_step_m)
            roof = (dz < 0.0) & (previous_z > local_top[solid])
            if np.any(roof):
                target = solid_flat[roof]
                result[ROOF].flat[target] += remaining.flat[target]
                remaining.flat[target] = 0.0
            wall = ~roof
            if np.any(wall):
                target = solid_flat[wall]
                result[wall_target].flat[target] += remaining.flat[target]
                remaining.flat[target] = 0.0
        open_mask = ~solid
        if not bool(open_mask.any()):
            continue
        open_flat = flat[open_mask]
        open_rr = rr[open_mask]
        open_cc = cc[open_mask]
        open_z = ray_z[open_mask]
        ground_hit = open_z <= dem[open_rr, open_cc]
        if np.any(ground_hit):
            target = open_flat[ground_hit]
            result[GROUND].flat[target] += remaining.flat[target]
            remaining.flat[target] = 0.0
        air = ~ground_hit
        if not np.any(air):
            continue
        air_flat = open_flat[air]
        air_rr = open_rr[air]
        air_cc = open_cc[air]
        air_z = open_z[air]
        canopy_top = dem[air_rr, air_cc] + canopy_height[air_rr, air_cc]
        crown_depth = np.maximum(
            canopy_height[air_rr, air_cc] * config.canopy_crown_depth_fraction,
            config.ray_step_m,
        )
        crown_base = canopy_top - crown_depth
        in_canopy = (
            (canopy_height[air_rr, air_cc] >= config.canopy_threshold_m)
            & (air_z >= crown_base)
            & (air_z <= canopy_top)
        )
        if np.any(in_canopy):
            target = air_flat[in_canopy]
            depth = crown_depth[in_canopy]
            absorption = 1.0 - np.exp(
                -config.canopy_extinction
                * config.canopy_leaf_area_index
                * config.ray_step_m
                / depth
            )
            absorbed = remaining.flat[target] * absorption
            result[CANOPY].flat[target] += absorbed
            remaining.flat[target] -= absorbed
    active = remaining > 0.0
    final_target = GROUND if dz < 0.0 else SKY
    result[final_target, active] += remaining[active]
    return result


def _trace_set(
    origin_x: np.ndarray,
    origin_y: np.ndarray,
    origin_z: np.ndarray,
    directions: np.ndarray,
    dem: np.ndarray,
    building_dsm: np.ndarray,
    canopy_height: np.ndarray,
    config: ViewFactorConfig,
) -> np.ndarray:
    result = np.zeros((N_FACETS + 1, *origin_x.shape), dtype=np.float64)
    for direction in directions:
        result += _trace_direction(
            origin_x,
            origin_y,
            origin_z,
            direction,
            dem,
            building_dsm,
            canopy_height,
            config,
        )
    result /= len(directions)
    result /= np.maximum(result.sum(axis=0, keepdims=True), 1.0e-12)
    return result


def _reciprocal_exchange(area: np.ndarray, raw_view: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    directed = area[:, None] * raw_view[:, :N_FACETS]
    exchange = 0.5 * (directed + directed.swapaxes(0, 1))
    target = area * np.clip(1.0 - raw_view[:, SKY], 0.0, 1.0)
    active = area > 0.0
    for _ in range(80):
        row_sum = exchange.sum(axis=1)
        ratio = np.ones_like(row_sum)
        valid = active & (row_sum > 1.0e-14)
        ratio[valid] = np.sqrt(np.clip(target[valid] / row_sum[valid], 0.0, 1.0e12))
        exchange *= ratio[:, None] * ratio[None, :]
        error = np.max(np.abs(exchange.sum(axis=1) - target))
        if error < 1.0e-8:
            break
    row_sum = exchange.sum(axis=1)
    residual = target - row_sum
    for facet in range(N_FACETS):
        exchange[facet, facet] += np.maximum(residual[facet], 0.0)
    for _ in range(80):
        row_sum = exchange.sum(axis=1)
        over = row_sum > area + 1.0e-12
        if not np.any(over):
            break
        scale = np.ones_like(area)
        scale[over] = np.sqrt(area[over] / row_sum[over])
        exchange *= scale[:, None] * scale[None, :]
    exchange = 0.5 * (exchange + exchange.swapaxes(0, 1))
    sky = np.clip(area - exchange.sum(axis=1), 0.0, None)
    return exchange, sky


def trace_geometry(
    dem: np.ndarray,
    building_dsm: np.ndarray,
    canopy_height: np.ndarray,
    *,
    window: Sequence[int] | None = None,
    config: ViewFactorConfig = ViewFactorConfig(),
) -> TracedGeometry:
    """Trace a real height-field scene into local reciprocal facet enclosures."""
    config.validate()
    (
        dem_array,
        building_array,
        canopy_array,
        resolved_window,
        dsm_clamped_cell_count,
        maximum_dsm_clamp,
    ) = _validate_inputs(
        dem,
        building_dsm,
        canopy_height,
        window,
        config.dsm_below_dem_tolerance_m,
    )
    row, col, height, width = resolved_window
    area_full, source_masks, wall_area = _local_surface_areas(
        dem_array, building_array, canopy_array, config
    )
    area = area_full[:, row : row + height, col : col + width].copy()
    grid_row, grid_col = np.indices((height, width), dtype=np.float64)
    body_x = grid_col + col + 0.5
    body_y = grid_row + row + 0.5
    base_surface = np.maximum(
        dem_array[row : row + height, col : col + width],
        building_array[row : row + height, col : col + width],
    )
    body_z = base_surface + config.body_height_m
    body_trace = np.zeros((N_FACETS + 1, height, width), dtype=np.float64)
    for directions, projected_area_weight in _body_direction_sets(config.body_ray_count):
        body_trace += projected_area_weight * _trace_set(
            body_x,
            body_y,
            body_z,
            directions,
            dem_array,
            building_array,
            canopy_array,
            config,
        )
    body_view = body_trace[:N_FACETS]
    body_sky = body_trace[SKY]
    raw = np.zeros((N_FACETS, N_FACETS + 1, height, width), dtype=np.float64)
    facet_origin_x = np.zeros((N_FACETS, height, width), dtype=np.float64)
    facet_origin_y = np.zeros_like(facet_origin_x)
    facet_origin_z = np.zeros_like(facet_origin_x)
    for facet in range(N_FACETS):
        origin_x, origin_y, origin_z, source_active = _source_origins(
            facet,
            source_masks[facet],
            wall_area,
            dem_array,
            building_array,
            canopy_array,
            resolved_window,
            config,
        )
        facet_origin_x[facet] = origin_x
        facet_origin_y[facet] = origin_y
        facet_origin_z[facet] = origin_z
        traced = _trace_set(
            origin_x,
            origin_y,
            origin_z,
            _surface_directions(facet, config.surface_ray_count),
            dem_array,
            building_array,
            canopy_array,
            config,
        )
        raw[facet] = traced
        inactive = ~source_active
        if np.any(inactive):
            area[facet, inactive] = 0.0
            body_view[facet, inactive] = 0.0
            raw[facet, :, inactive] = 0.0
            raw[facet, SKY, inactive] = 1.0
    body_total = body_view.sum(axis=0)
    body_sky = np.maximum(1.0 - body_total, 0.0)
    normalization = body_total + body_sky
    body_view /= normalization[None]
    body_sky /= normalization
    exchange, sky_area = _reciprocal_exchange(area, raw)
    result = TracedGeometry(
        area=area.astype(np.float32),
        sky_view_area=sky_area.astype(np.float32),
        exchange_area=exchange.astype(np.float32),
        body_view_factor=body_view.astype(np.float32),
        body_sky_view_factor=body_sky.astype(np.float32),
        raw_surface_sky_view_factor=raw[:, SKY].astype(np.float32),
        facet_origin_x=facet_origin_x.astype(np.float32),
        facet_origin_y=facet_origin_y.astype(np.float32),
        facet_origin_z=facet_origin_z.astype(np.float32),
        window=resolved_window,
        dsm_clamped_cell_count=dsm_clamped_cell_count,
        maximum_dsm_clamp_m=maximum_dsm_clamp,
    )
    result.validate()
    return result


def trace_direct_solar_projection(
    dem: np.ndarray,
    building_dsm: np.ndarray,
    canopy_height: np.ndarray,
    facet_origin_x: np.ndarray,
    facet_origin_y: np.ndarray,
    facet_origin_z: np.ndarray,
    solar_altitude_degrees: float,
    solar_azimuth_degrees: float,
    *,
    config: ViewFactorConfig,
) -> np.ndarray:
    """Return projected, obstruction-aware direct-beam factors for seven facets."""
    config.validate()
    if not math.isfinite(solar_altitude_degrees) or not -90.0 <= solar_altitude_degrees <= 90.0:
        raise ValueError("solar_altitude_degrees must be finite and between -90 and 90")
    if not math.isfinite(solar_azimuth_degrees):
        raise ValueError("solar_azimuth_degrees must be finite")
    origins = [
        np.asarray(value, dtype=np.float64)
        for value in (facet_origin_x, facet_origin_y, facet_origin_z)
    ]
    if any(value.ndim != 3 or value.shape[0] != N_FACETS for value in origins):
        raise ValueError("facet origins must have shape (7, rows, cols)")
    if len({value.shape for value in origins}) != 1:
        raise ValueError("facet origin arrays must have identical shapes")
    if any(not np.isfinite(value).all() for value in origins):
        raise ValueError("facet origins must contain only finite values")
    dem_array, building_array, canopy_array, _, _, _ = _validate_inputs(
        dem,
        building_dsm,
        canopy_height,
        None,
        config.dsm_below_dem_tolerance_m,
    )
    trace_rows, trace_cols = dem_array.shape
    x_origin, y_origin, _ = origins
    coordinate_tolerance = 0.1
    if np.any(
        (x_origin < -coordinate_tolerance)
        | (x_origin > trace_cols + coordinate_tolerance)
        | (y_origin < -coordinate_tolerance)
        | (y_origin > trace_rows + coordinate_tolerance)
    ):
        raise ValueError("facet origins extend outside the saved trace domain")
    result = np.zeros(origins[0].shape, dtype=np.float64)
    if solar_altitude_degrees <= 0.0:
        return result.astype(np.float32)
    altitude = math.radians(solar_altitude_degrees)
    azimuth = math.radians(solar_azimuth_degrees % 360.0)
    direction = np.asarray(
        (
            math.sin(azimuth) * math.cos(altitude),
            -math.cos(azimuth) * math.cos(altitude),
            math.sin(altitude),
        ),
        dtype=np.float64,
    )
    normals = (
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    for facet, normal in enumerate(normals):
        incidence = max(float(np.dot(direction, normal)), 0.0)
        if incidence <= 0.0:
            continue
        traced = _trace_direction(
            origins[0][facet],
            origins[1][facet],
            origins[2][facet],
            direction,
            dem_array,
            building_array,
            canopy_array,
            config,
        )
        result[facet] = incidence * traced[SKY]
    return result.astype(np.float32)


__all__ = [
    "TracedGeometry",
    "ViewFactorConfig",
    "trace_direct_solar_projection",
    "trace_geometry",
]
