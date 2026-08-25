# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Integration boundary between SOLWEIG radiation and the coupled facet solver."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .energy_balance import CanopyProperties, MaterialProperties
from .energy_balance.coupled import (
    CANOPY,
    GROUND,
    N_FACETS,
    ROOF,
    WALL_EAST,
    WALL_NORTH,
    WALL_SOUTH,
    WALL_WEST,
    CoupledUrbanEBConfig,
    CoupledUrbanEnergyBalance,
    UrbanEBResult,
    UrbanFacetGeometry,
    UrbanForcing,
)
from .energy_balance.physics import STEFAN_BOLTZMANN, e_sat_kpa
from .view_factors import ViewFactorConfig, trace_direct_solar_projection


@dataclass(frozen=True)
class SolarTraceGeometry:
    dem: np.ndarray
    building_dsm: np.ndarray
    canopy_height: np.ndarray
    landcover: np.ndarray
    facet_origin_x: np.ndarray
    facet_origin_y: np.ndarray
    facet_origin_z: np.ndarray
    output_window: tuple[int, int, int, int]
    config: ViewFactorConfig


@dataclass(frozen=True)
class CoupledGeometryBundle:
    geometry: UrbanFacetGeometry
    body_view_factor: torch.Tensor
    body_sky_view_factor: torch.Tensor
    raw_surface_sky_view_factor: torch.Tensor | None
    solar_trace: SolarTraceGeometry | None
    water_capacity: torch.Tensor
    rain_capture_fraction: torch.Tensor | None
    crs_wkt: str
    geotransform: tuple[float, float, float, float, float, float]
    source_path: Path


@dataclass(frozen=True)
class CoupledRadiationResult:
    tmrt_celsius: torch.Tensor
    outgoing_longwave: torch.Tensor
    body_longwave_irradiance: torch.Tensor
    upward_longwave: torch.Tensor
    relative_humidity_percent: torch.Tensor


def resolve_geometry_path(path: str | Path, tile_key: str) -> Path:
    """Resolve a single geometry bundle or a per-tile bundle directory."""
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"coupled geometry path not found: {candidate}")
    matches = [
        candidate / f"coupled_geometry_{tile_key}.npz",
        candidate / f"{tile_key}.npz",
    ]
    existing = [item for item in matches if item.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"expected one coupled geometry bundle for tile {tile_key!r} in {candidate}; "
            f"looked for {[item.name for item in matches]}"
        )
    return existing[0]


def _tensor_from_archive(
    archive: np.lib.npyio.NpzFile,
    name: str,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if name not in archive:
        raise ValueError(f"coupled geometry bundle is missing {name!r}")
    value = torch.as_tensor(archive[name], dtype=torch.float32, device=device)
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} does not match {shape}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    return value


def load_geometry_bundle(
    path: str | Path,
    rows: int,
    cols: int,
    device: torch.device,
) -> CoupledGeometryBundle:
    """Load and validate reciprocal facet geometry and pedestrian view factors."""
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".npz":
        raise ValueError("coupled geometry bundle must be an .npz file")
    try:
        archive_context = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read coupled geometry bundle {source}: {exc}") from exc
    with archive_context as archive:
        area = _tensor_from_archive(archive, "area", (N_FACETS, rows, cols), device)
        sky = _tensor_from_archive(
            archive, "sky_view_area", (N_FACETS, rows, cols), device
        )
        exchange = _tensor_from_archive(
            archive,
            "exchange_area",
            (N_FACETS, N_FACETS, rows, cols),
            device,
        )
        body = _tensor_from_archive(
            archive, "body_view_factor", (N_FACETS, rows, cols), device
        )
        body_sky = _tensor_from_archive(
            archive, "body_sky_view_factor", (rows, cols), device
        )
        raw_surface_sky = None
        if "raw_surface_sky_view_factor" in archive:
            raw_surface_sky = _tensor_from_archive(
                archive,
                "raw_surface_sky_view_factor",
                (N_FACETS, rows, cols),
                device,
            )
        if "water_capacity" in archive:
            water = _tensor_from_archive(
                archive, "water_capacity", (N_FACETS, rows, cols), device
            )
        else:
            defaults = torch.tensor(
                [20.0, 1.0, 0.2, 0.2, 0.2, 0.2, 1.5],
                dtype=torch.float32,
                device=device,
            ).view(N_FACETS, 1, 1)
            water = defaults.expand_as(area).clone()
            water = torch.where(area > 0.0, water, torch.zeros_like(water))
        capture = None
        if "rain_capture_fraction" in archive:
            capture = _tensor_from_archive(
                archive,
                "rain_capture_fraction",
                (N_FACETS, rows, cols),
                device,
            )
        if "crs_wkt" not in archive:
            raise ValueError("coupled geometry bundle is missing 'crs_wkt'")
        crs_wkt = str(np.asarray(archive["crs_wkt"]).item()).strip()
        if not crs_wkt:
            raise ValueError("crs_wkt cannot be empty")
        if "geotransform" not in archive:
            raise ValueError("coupled geometry bundle is missing 'geotransform'")
        geotransform_array = np.asarray(archive["geotransform"], dtype=float)
        if geotransform_array.shape != (6,) or not np.isfinite(geotransform_array).all():
            raise ValueError("geotransform must contain six finite values")
        geotransform = tuple(float(value) for value in geotransform_array)

        trace_names = (
            "trace_dem",
            "trace_building_dsm",
            "trace_canopy_height",
            "trace_landcover",
            "facet_origin_x",
            "facet_origin_y",
            "facet_origin_z",
            "trace_output_window",
        )
        trace_presence = [name in archive for name in trace_names]
        if any(trace_presence) and not all(trace_presence):
            missing = [name for name, present in zip(trace_names, trace_presence) if not present]
            raise ValueError(
                "coupled geometry bundle has an incomplete solar-trace payload; "
                f"missing {missing}"
            )
        solar_trace = None
        if all(trace_presence):
            if "config_json" not in archive:
                raise ValueError("solar-trace payload is missing 'config_json'")
            trace_arrays = [
                np.asarray(archive[name], dtype=np.float64)
                for name in ("trace_dem", "trace_building_dsm", "trace_canopy_height")
            ]
            if any(value.ndim != 2 for value in trace_arrays):
                raise ValueError("solar-trace rasters must be two-dimensional")
            if len({value.shape for value in trace_arrays}) != 1:
                raise ValueError("solar-trace rasters must have identical shapes")
            if any(not np.isfinite(value).all() for value in trace_arrays):
                raise ValueError("solar-trace rasters contain non-finite values")
            trace_dem, trace_building, trace_canopy = trace_arrays
            if np.any(trace_canopy < 0.0):
                raise ValueError("trace_canopy_height cannot contain negative values")
            trace_landcover_array = np.asarray(archive["trace_landcover"])
            if trace_landcover_array.shape != trace_dem.shape:
                raise ValueError("trace_landcover must match the solar-trace raster shape")
            if not np.issubdtype(trace_landcover_array.dtype, np.integer):
                raise ValueError("trace_landcover must use an integer dtype")
            trace_landcover = trace_landcover_array.astype(np.int16, copy=False)
            if np.any((trace_landcover < 1) | (trace_landcover > 7)):
                raise ValueError("trace_landcover must use UTherm classes 1 through 7")
            origins = [
                np.asarray(archive[name], dtype=np.float64)
                for name in ("facet_origin_x", "facet_origin_y", "facet_origin_z")
            ]
            expected_origin_shape = (N_FACETS, rows, cols)
            if any(value.shape != expected_origin_shape for value in origins):
                raise ValueError(
                    "solar-trace facet origins must have shape "
                    f"{expected_origin_shape}"
                )
            if any(not np.isfinite(value).all() for value in origins):
                raise ValueError("solar-trace facet origins contain non-finite values")
            trace_window_array = np.asarray(archive["trace_output_window"])
            if trace_window_array.shape != (4,) or not np.issubdtype(
                trace_window_array.dtype, np.integer
            ):
                raise ValueError("trace_output_window must contain four integers")
            trace_window = tuple(int(value) for value in trace_window_array)
            trace_row, trace_col, trace_height, trace_width = trace_window
            trace_rows, trace_cols = trace_dem.shape
            if (
                trace_row < 0
                or trace_col < 0
                or trace_height != rows
                or trace_width != cols
                or trace_row + rows > trace_rows
                or trace_col + cols > trace_cols
            ):
                raise ValueError("trace_output_window does not match the output grid")
            try:
                config_payload = json.loads(str(np.asarray(archive["config_json"]).item()))
                trace_config = ViewFactorConfig(**config_payload)
                trace_config.validate()
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid solar-trace config_json: {exc}") from exc
            if not (
                math.isclose(trace_config.pixel_size_m, abs(geotransform[1]), abs_tol=1.0e-9)
                and math.isclose(trace_config.pixel_size_m, abs(geotransform[5]), abs_tol=1.0e-9)
            ):
                raise ValueError("solar-trace pixel size does not match the output geotransform")
            coordinate_tolerance = 0.1
            if np.any(
                (origins[0] < -coordinate_tolerance)
                | (origins[0] > trace_cols + coordinate_tolerance)
                | (origins[1] < -coordinate_tolerance)
                | (origins[1] > trace_rows + coordinate_tolerance)
            ):
                raise ValueError("solar-trace facet origins extend outside the trace rasters")
            solar_trace = SolarTraceGeometry(
                dem=trace_dem,
                building_dsm=trace_building,
                canopy_height=trace_canopy,
                landcover=trace_landcover,
                facet_origin_x=origins[0],
                facet_origin_y=origins[1],
                facet_origin_z=origins[2],
                output_window=trace_window,
                config=trace_config,
            )

    geometry = UrbanFacetGeometry(area, sky, exchange)
    geometry.validate()
    for name, value in (
        ("body_view_factor", body),
        ("body_sky_view_factor", body_sky),
        ("water_capacity", water),
    ):
        if bool((value < 0.0).any()):
            raise ValueError(f"{name} cannot contain negative values")
    body_closure = body.sum(dim=0) + body_sky
    if not bool(torch.allclose(body_closure, torch.ones_like(body_closure), atol=2.0e-4, rtol=0.0)):
        error = float((body_closure - 1.0).abs().max().item())
        raise ValueError(f"body view factors do not close; max absolute error={error:.3g}")
    if bool(((area <= 0.0) & (body > 2.0e-4)).any()):
        raise ValueError("body_view_factor assigns view to an inactive facet")
    if raw_surface_sky is not None:
        if bool(((raw_surface_sky < 0.0) | (raw_surface_sky > 1.0 + 2.0e-4)).any()):
            raise ValueError("raw_surface_sky_view_factor must be between zero and one")
    if capture is not None:
        if bool(((capture < 0.0) | (capture > 1.0)).any()):
            raise ValueError("rain_capture_fraction must be between zero and one")
        plan_capture = (area * capture).sum(dim=0)
        if bool((plan_capture > 1.0 + 1.0e-5).any()):
            raise ValueError("rain_capture_fraction duplicates precipitation")
    return CoupledGeometryBundle(
        geometry=geometry,
        body_view_factor=body,
        body_sky_view_factor=body_sky,
        raw_surface_sky_view_factor=raw_surface_sky,
        solar_trace=solar_trace,
        water_capacity=water,
        rain_capture_fraction=capture,
        crs_wkt=crs_wkt,
        geotransform=geotransform,
        source_path=source,
    )


def ground_material_ids_from_trace(
    bundle: CoupledGeometryBundle,
    tile_landcover: np.ndarray,
) -> np.ndarray:
    """Return the land-cover class at each representative ground-facet origin."""
    if bundle.solar_trace is None:
        raise ValueError("coupled geometry bundle lacks the solar-trace payload")
    trace = bundle.solar_trace
    current = np.asarray(tile_landcover)
    rows, cols = bundle.body_sky_view_factor.shape
    if current.shape != (rows, cols):
        raise ValueError(f"tile_landcover shape {current.shape} does not match {(rows, cols)}")
    if not np.isfinite(current).all() or not np.equal(current, np.rint(current)).all():
        raise ValueError("tile_landcover must contain finite integer classes")
    current = np.rint(current).astype(np.int16)
    row, col, height, width = trace.output_window
    saved_tile = trace.landcover[row : row + height, col : col + width]
    if not np.array_equal(saved_tile, current):
        raise ValueError("coupled geometry land cover does not match the processed raster tile")
    source_row = np.floor(trace.facet_origin_y[GROUND]).astype(np.int64)
    source_col = np.floor(trace.facet_origin_x[GROUND]).astype(np.int64)
    trace_rows, trace_cols = trace.landcover.shape
    if np.any(
        (source_row < 0)
        | (source_row >= trace_rows)
        | (source_col < 0)
        | (source_col >= trace_cols)
    ):
        raise ValueError("ground facet origin falls outside trace_landcover")
    return trace.landcover[source_row, source_col].copy()


class CoupledRadiationBridge:
    """Run a spun-up seven-facet solve and convert radiosity to pedestrian Tmrt."""

    def __init__(
        self,
        bundle: CoupledGeometryBundle,
        *,
        ground_material: MaterialProperties,
        roof_material: MaterialProperties,
        canopy_properties: CanopyProperties,
        dt: float,
        spinup_max_cycles: int,
        strict_convergence: bool,
        wall_material: MaterialProperties | None = None,
        material_ids: torch.Tensor | None = None,
        material_table: Sequence[MaterialProperties] | None = None,
    ):
        wall = wall_material or MaterialProperties.brick()
        config = CoupledUrbanEBConfig(
            dt=dt,
            spinup_max_cycles=spinup_max_cycles,
            strict_convergence=strict_convergence,
        )
        materials = [ground_material, roof_material, wall, wall, wall, wall]
        self.bundle = bundle
        self.model = CoupledUrbanEnergyBalance(
            bundle.geometry,
            materials,
            canopy_properties,
            bundle.water_capacity,
            config,
            material_ids=material_ids,
            material_table=material_table,
        )
        self.last_spinup = None

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype

    @property
    def rain_capture_fraction(self) -> torch.Tensor | None:
        return self.bundle.rain_capture_fraction

    def facet_shortwave_irradiance(
        self,
        *,
        direct_normal_irradiance: float,
        diffuse_horizontal_irradiance: float,
        solar_altitude_degrees: float,
        solar_azimuth_degrees: float,
    ) -> torch.Tensor:
        """Build external direct and diffuse irradiance for every physical facet."""
        values = {
            "direct_normal_irradiance": direct_normal_irradiance,
            "diffuse_horizontal_irradiance": diffuse_horizontal_irradiance,
            "solar_altitude_degrees": solar_altitude_degrees,
            "solar_azimuth_degrees": solar_azimuth_degrees,
        }
        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if direct_normal_irradiance < 0.0 or diffuse_horizontal_irradiance < 0.0:
            raise ValueError("shortwave irradiance inputs must be nonnegative")
        if not -90.0 <= solar_altitude_degrees <= 90.0:
            raise ValueError("solar_altitude_degrees must be between -90 and 90")
        if self.bundle.solar_trace is None or self.bundle.raw_surface_sky_view_factor is None:
            raise ValueError(
                "coupled geometry bundle lacks the solar-trace payload; regenerate it "
                "with utherm-geometry"
            )
        if solar_altitude_degrees <= 0.0:
            return torch.zeros_like(self.bundle.geometry.area)
        trace = self.bundle.solar_trace
        direct_projection = trace_direct_solar_projection(
            trace.dem,
            trace.building_dsm,
            trace.canopy_height,
            trace.facet_origin_x,
            trace.facet_origin_y,
            trace.facet_origin_z,
            solar_altitude_degrees,
            solar_azimuth_degrees,
            config=trace.config,
        )
        direct = torch.as_tensor(
            direct_projection,
            dtype=self.dtype,
            device=self.device,
        )
        raw_sky = self.bundle.raw_surface_sky_view_factor.to(
            dtype=self.dtype, device=self.device
        )
        direct_shortwave = direct_normal_irradiance * direct
        diffuse_shortwave = diffuse_horizontal_irradiance * raw_sky

        leaf_area = self.bundle.geometry.area[CANOPY]
        active_canopy = leaf_area > 0.0
        safe_leaf_area = torch.where(active_canopy, leaf_area, torch.ones_like(leaf_area))
        extinction = trace.config.canopy_extinction
        sine_altitude = max(math.sin(math.radians(solar_altitude_degrees)), 1.0e-6)
        direct_interception = 1.0 - torch.exp(
            -extinction * safe_leaf_area / sine_altitude
        )
        diffuse_interception = 1.0 - torch.exp(-2.0 * extinction * safe_leaf_area)
        canopy_direct = (
            direct_normal_irradiance
            * direct[CANOPY]
            * direct_interception
            / safe_leaf_area
        )
        canopy_diffuse = (
            diffuse_horizontal_irradiance
            * torch.clamp(2.0 * raw_sky[CANOPY], max=1.0)
            * diffuse_interception
            / safe_leaf_area
        )
        direct_shortwave[CANOPY] = torch.where(
            active_canopy,
            canopy_direct,
            torch.zeros_like(canopy_direct),
        )
        diffuse_shortwave[CANOPY] = torch.where(
            active_canopy,
            canopy_diffuse,
            torch.zeros_like(canopy_diffuse),
        )
        active = self.bundle.geometry.area > 0.0
        direct_shortwave = torch.where(active, direct_shortwave, torch.zeros_like(direct_shortwave))
        diffuse_shortwave = torch.where(active, diffuse_shortwave, torch.zeros_like(diffuse_shortwave))

        def conserve_plan_flux(
            field: torch.Tensor,
            target: float,
            maximum_surface_flux: float,
            name: str,
        ) -> torch.Tensor:
            if target <= 1.0e-8:
                return torch.zeros_like(field)
            receiving = field > max(1.0e-8, maximum_surface_flux * 1.0e-7)
            field = torch.where(receiving, field, torch.zeros_like(field))
            bounded_maximum = torch.where(
                receiving,
                torch.full_like(field, maximum_surface_flux),
                torch.zeros_like(field),
            )
            available = (self.bundle.geometry.area * bounded_maximum).sum(dim=0).mean()
            if float(available.item()) < target * (1.0 - 1.0e-6):
                raise ValueError(
                    f"solar-trace geometry cannot conserve the requested {name} flux "
                    "without irradiating a shadowed facet"
                )
            target_tensor = torch.as_tensor(target, dtype=self.dtype, device=self.device)
            lower = torch.zeros((), dtype=self.dtype, device=self.device)
            upper = torch.ones((), dtype=self.dtype, device=self.device)
            for _ in range(32):
                trial = torch.clamp(field * upper, max=maximum_surface_flux)
                plan_flux = (self.bundle.geometry.area * trial).sum(dim=0).mean()
                upper = torch.where(plan_flux < target_tensor, upper * 2.0, upper)
            for _ in range(48):
                middle = 0.5 * (lower + upper)
                trial = torch.clamp(field * middle, max=maximum_surface_flux)
                plan_flux = (self.bundle.geometry.area * trial).sum(dim=0).mean()
                lower = torch.where(plan_flux < target_tensor, middle, lower)
                upper = torch.where(plan_flux < target_tensor, upper, middle)
            return torch.clamp(field * upper, max=maximum_surface_flux)

        direct_target = direct_normal_irradiance * sine_altitude
        direct_shortwave = conserve_plan_flux(
            direct_shortwave,
            direct_target,
            direct_normal_irradiance,
            "direct",
        )
        diffuse_shortwave = conserve_plan_flux(
            diffuse_shortwave,
            diffuse_horizontal_irradiance,
            diffuse_horizontal_irradiance,
            "diffuse",
        )
        return direct_shortwave + diffuse_shortwave

    def solve_cycle(
        self,
        forcings: Sequence[UrbanForcing],
        absorbed_shortwave_by_body: Sequence[torch.Tensor],
        *,
        initial_water_fraction: float = 0.0,
        spinup_forcings: Sequence[UrbanForcing] | None = None,
    ) -> tuple[list[UrbanEBResult], list[CoupledRadiationResult]]:
        if len(forcings) != len(absorbed_shortwave_by_body):
            raise ValueError("forcing and pedestrian-shortwave cycles must have equal length")
        if len(forcings) < 1:
            raise ValueError("coupled pipeline requires at least one forcing timestep")
        if spinup_forcings is None:
            spinup_forcings = forcings
        elif len(spinup_forcings) != len(forcings):
            raise ValueError("spin-up and output forcing cycles must have equal length")
        self.last_spinup = self.model.spin_up(
            spinup_forcings, initial_water_fraction=initial_water_fraction
        )
        results = self.model.run(
            forcings,
            state=self.last_spinup.state,
            spin_up=False,
            initial_water_fraction=initial_water_fraction,
        )
        radiation = [
            self.radiation_result(result, forcing, body_shortwave)
            for result, forcing, body_shortwave in zip(
                results, forcings, absorbed_shortwave_by_body
            )
        ]
        return results, radiation

    def radiation_result(
        self,
        result: UrbanEBResult,
        forcing: UrbanForcing,
        absorbed_shortwave_by_body: torch.Tensor,
    ) -> CoupledRadiationResult:
        """Replace SOLWEIG surface longwave with the coupled outgoing radiosities."""
        if tuple(absorbed_shortwave_by_body.shape) != (self.model.rows, self.model.cols):
            raise ValueError("absorbed_shortwave_by_body has the wrong shape")
        temperature = result.state.surface_temperature
        emissivity = self.model.emissivity.expand_as(temperature)
        blackbody = STEFAN_BOLTZMANN * temperature.pow(4)
        incoming = result.longwave_net / torch.clamp(emissivity, min=1.0e-6) + blackbody
        outgoing = emissivity * blackbody + (1.0 - emissivity) * incoming
        sky_longwave = self.model._grid(
            forcing.sky_longwave, "sky_longwave", nonnegative=True
        )
        body_longwave = (
            self.bundle.body_view_factor * outgoing
        ).sum(dim=0) + self.bundle.body_sky_view_factor * sky_longwave
        radiant_load = torch.clamp(
            absorbed_shortwave_by_body + 0.95 * body_longwave,
            min=0.0,
        )
        tmrt = torch.sqrt(torch.sqrt(radiant_load / (0.95 * STEFAN_BOLTZMANN))) - 273.2
        ground_active = self.bundle.geometry.area[GROUND] > 0.0
        roof_active = self.bundle.geometry.area[ROOF] > 0.0
        upward = torch.where(
            ground_active,
            outgoing[GROUND],
            torch.where(roof_active, outgoing[ROOF], torch.full_like(body_longwave, float("nan"))),
        )
        vapor = self.model._vapor_from_specific_humidity(
            result.state.canyon_specific_humidity,
            forcing.pressure_kpa,
        )
        saturation = e_sat_kpa(result.state.canyon_air_temperature)
        relative_humidity = torch.clamp(
            100.0 * vapor / torch.clamp(saturation, min=1.0e-6),
            0.0,
            100.0,
        )
        return CoupledRadiationResult(
            tmrt_celsius=tmrt,
            outgoing_longwave=outgoing,
            body_longwave_irradiance=body_longwave,
            upward_longwave=upward,
            relative_humidity_percent=relative_humidity,
        )


FACET_OUTPUT_NAMES = {
    WALL_NORTH: "TsfcWallNorth",
    WALL_EAST: "TsfcWallEast",
    WALL_SOUTH: "TsfcWallSouth",
    WALL_WEST: "TsfcWallWest",
    CANOPY: "TLeaf",
}


__all__ = [
    "CoupledGeometryBundle",
    "CoupledRadiationBridge",
    "CoupledRadiationResult",
    "FACET_OUTPUT_NAMES",
    "SolarTraceGeometry",
    "ground_material_ids_from_trace",
    "load_geometry_bundle",
    "resolve_geometry_path",
]
