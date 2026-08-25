# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Coupled urban surface, canopy and canyon-air energy balance.

Radiative exchange is solved with gray diffuse radiosity on reciprocal facet
view areas.  Ground, roof and four wall orientations retain independent
multi-layer thermal states.  A leaf facet exchanges radiation and turbulent
heat with the same canyon air volume.  Water storage constrains evaporation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import torch

from .config import CanopyProperties
from .materials import MaterialProperties
from .physics import (
    AIR_DENSITY,
    AIR_SPECIFIC_HEAT,
    EPSILON_RATIO,
    LATENT_HEAT_VAP,
    STEFAN_BOLTZMANN,
    calculate_httc_ground,
    delta_e_sat,
    e_sat_kpa,
    psychrometric_constant,
    solve_conduction_substepped,
)


GROUND = 0
ROOF = 1
WALL_NORTH = 2
WALL_EAST = 3
WALL_SOUTH = 4
WALL_WEST = 5
CANOPY = 6
N_FACETS = 7
N_SOLID_FACETS = 6
FACET_NAMES = (
    "ground",
    "roof",
    "wall_north",
    "wall_east",
    "wall_south",
    "wall_west",
    "canopy",
)


@dataclass(frozen=True)
class CoupledUrbanEBConfig:
    """Numerical and physical controls for the coupled solve."""

    dt: float = 3600.0
    max_coupling_iterations: int = 80
    temperature_tolerance: float = 0.01
    specific_humidity_tolerance: float = 1.0e-7
    residual_tolerance: float = 0.5
    relaxation: float = 0.65
    max_temperature_step: float = 8.0
    minimum_temperature: float = 213.15
    maximum_temperature: float = 373.15
    canyon_height: float = 8.0
    ventilation_coefficient: float = 0.15
    minimum_exchange_velocity: float = 0.01
    ground_deep_temperature: float = 288.15
    interior_temperature: float = 295.15
    spinup_min_cycles: int = 2
    spinup_max_cycles: int = 30
    spinup_temperature_tolerance: float = 0.05
    spinup_moisture_tolerance: float = 1.0e-3
    spinup_specific_humidity_tolerance: float = 1.0e-5
    strict_convergence: bool = True
    solve_wall_temperature: bool = True
    solve_canyon_temperature: bool = True
    solve_canyon_humidity: bool = True
    water_limited_evaporation: bool = True
    insulated_deep_boundary: bool = False

    def __post_init__(self) -> None:
        positive = {
            "dt": self.dt,
            "temperature_tolerance": self.temperature_tolerance,
            "specific_humidity_tolerance": self.specific_humidity_tolerance,
            "residual_tolerance": self.residual_tolerance,
            "max_temperature_step": self.max_temperature_step,
            "minimum_temperature": self.minimum_temperature,
            "maximum_temperature": self.maximum_temperature,
            "canyon_height": self.canyon_height,
            "ground_deep_temperature": self.ground_deep_temperature,
            "spinup_temperature_tolerance": self.spinup_temperature_tolerance,
            "spinup_moisture_tolerance": self.spinup_moisture_tolerance,
            "spinup_specific_humidity_tolerance": self.spinup_specific_humidity_tolerance,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_temperature >= self.maximum_temperature:
            raise ValueError("minimum_temperature must be below maximum_temperature")
        if not math.isfinite(self.relaxation) or not 0.0 < self.relaxation <= 1.0:
            raise ValueError("relaxation must be in (0, 1]")
        for name in ("ventilation_coefficient", "minimum_exchange_velocity"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.interior_temperature) or self.interior_temperature <= 0.0:
            raise ValueError("interior_temperature must be finite and positive")
        for name in ("max_coupling_iterations", "spinup_min_cycles", "spinup_max_cycles"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.spinup_min_cycles > self.spinup_max_cycles:
            raise ValueError("spinup_min_cycles cannot exceed spinup_max_cycles")
        for name in (
            "strict_convergence",
            "solve_wall_temperature",
            "solve_canyon_temperature",
            "solve_canyon_humidity",
            "water_limited_evaporation",
            "insulated_deep_boundary",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")


@dataclass(frozen=True)
class UrbanFacetGeometry:
    """Facet areas and reciprocal view areas on a common plan-area grid.

    ``area`` and ``sky_view_area`` have shape ``(7, rows, cols)``.  The
    exchange tensor has shape ``(7, 7, rows, cols)`` and must be symmetric:
    ``exchange_area[i,j] == exchange_area[j,i]``.  For every active facet,
    sky-view area plus all exchange areas must equal its area.  This area form
    enforces view-factor reciprocity by construction.
    """

    area: torch.Tensor
    sky_view_area: torch.Tensor
    exchange_area: torch.Tensor

    @classmethod
    def from_view_factors(
        cls,
        area: torch.Tensor,
        sky_view_factor: torch.Tensor,
        view_factor: torch.Tensor,
    ) -> "UrbanFacetGeometry":
        """Convert conventional view factors to reciprocal exchange areas.

        The conversion deliberately validates, rather than repairs, reciprocity
        and closure.  Geometry/ray-tracing errors must be corrected upstream.
        """
        if area.ndim != 3 or area.shape[0] != N_FACETS:
            raise ValueError("area must have shape (7, rows, cols)")
        if sky_view_factor.shape != area.shape:
            raise ValueError("sky_view_factor shape must match area")
        if view_factor.shape != (
            N_FACETS, N_FACETS, area.shape[1], area.shape[2]
        ):
            raise ValueError("view_factor must have shape (7, 7, rows, cols)")
        geometry = cls(
            area=area,
            sky_view_area=area * sky_view_factor,
            exchange_area=area.unsqueeze(1) * view_factor,
        )
        geometry.validate()
        return geometry

    def validate(self, tolerance: float = 2.0e-4) -> tuple[int, int]:
        if self.area.ndim != 3 or self.area.shape[0] != N_FACETS:
            raise ValueError("area must have shape (7, rows, cols)")
        if not self.area.is_floating_point():
            raise ValueError("geometry tensors must use a floating-point dtype")
        rows, cols = self.area.shape[1:]
        if self.sky_view_area.shape != self.area.shape:
            raise ValueError("sky_view_area shape must match area")
        if self.exchange_area.shape != (N_FACETS, N_FACETS, rows, cols):
            raise ValueError("exchange_area must have shape (7, 7, rows, cols)")
        tensors = {
            "area": self.area,
            "sky_view_area": self.sky_view_area,
            "exchange_area": self.exchange_area,
        }
        devices = {value.device for value in tensors.values()}
        if len(devices) != 1:
            raise ValueError("geometry tensors must share one device")
        dtypes = {value.dtype for value in tensors.values()}
        if len(dtypes) != 1 or any(not value.is_floating_point() for value in tensors.values()):
            raise ValueError("geometry tensors must share one floating-point dtype")
        for name, value in tensors.items():
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains non-finite values")
            if bool((value < 0.0).any()):
                raise ValueError(f"{name} cannot contain negative values")
        asymmetry = (self.exchange_area - self.exchange_area.transpose(0, 1)).abs().max()
        if float(asymmetry.item()) > tolerance:
            raise ValueError(
                f"exchange_area violates reciprocity; max asymmetry={float(asymmetry.item()):.3g}"
            )
        closure = self.sky_view_area + self.exchange_area.sum(dim=1)
        error = (closure - self.area).abs()
        scale = torch.clamp(self.area.abs(), min=1.0)
        if bool((error > tolerance * scale).any()):
            worst = float((error / scale).max().item())
            raise ValueError(f"facet view areas do not close; max relative error={worst:.3g}")
        return rows, cols


@dataclass(frozen=True)
class UrbanForcing:
    """Meteorology and external irradiance for one model timestep."""

    air_temperature: torch.Tensor | float
    vapor_pressure_kpa: torch.Tensor | float
    pressure_kpa: float
    wind_speed: torch.Tensor | float
    sky_longwave: torch.Tensor | float
    shortwave_irradiance: torch.Tensor
    precipitation_rate: torch.Tensor | float = 0.0
    anthropogenic_heat: torch.Tensor | float = 0.0
    rain_capture_fraction: Optional[torch.Tensor] = None


@dataclass
class UrbanEBState:
    """Prognostic temperatures and water stores."""

    surface_temperature: torch.Tensor
    layer_temperature: torch.Tensor
    canyon_air_temperature: torch.Tensor
    canyon_specific_humidity: torch.Tensor
    water_storage: torch.Tensor

    def clone(self) -> "UrbanEBState":
        return UrbanEBState(
            surface_temperature=self.surface_temperature.clone(),
            layer_temperature=self.layer_temperature.clone(),
            canyon_air_temperature=self.canyon_air_temperature.clone(),
            canyon_specific_humidity=self.canyon_specific_humidity.clone(),
            water_storage=self.water_storage.clone(),
        )


@dataclass(frozen=True)
class UrbanEBResult:
    state: UrbanEBState
    net_radiation: torch.Tensor
    sensible_heat: torch.Tensor
    latent_heat: torch.Tensor
    storage_heat: torch.Tensor
    residual: torch.Tensor
    shortwave_absorbed: torch.Tensor
    longwave_net: torch.Tensor
    iterations: int
    converged: bool
    max_temperature_change: float
    max_specific_humidity_change: float
    max_energy_residual: float
    water_drainage: torch.Tensor
    shortwave_escape: torch.Tensor
    longwave_to_sky: torch.Tensor


@dataclass(frozen=True)
class SpinupResult:
    state: UrbanEBState
    cycles: int
    converged: bool
    maximum_temperature_drift: float
    maximum_moisture_drift: float
    maximum_specific_humidity_drift: float


class CoupledUrbanEnergyBalance:
    """Solve all urban facets and canyon air in one nonlinear iteration."""

    def __init__(
        self,
        geometry: UrbanFacetGeometry,
        solid_materials: Sequence[MaterialProperties],
        canopy_properties: CanopyProperties,
        water_capacity: torch.Tensor,
        config: CoupledUrbanEBConfig = CoupledUrbanEBConfig(),
        *,
        material_ids: Optional[torch.Tensor] = None,
        material_table: Optional[Sequence[MaterialProperties]] = None,
    ):
        self.rows, self.cols = geometry.validate()
        if len(solid_materials) != N_SOLID_FACETS:
            raise ValueError("solid_materials must contain ground, roof and four wall materials")
        n_layers = solid_materials[0].n_layers
        if any(material.n_layers != n_layers for material in solid_materials):
            raise ValueError("all solid facets must use the same number of thermal layers")
        if (material_ids is None) != (material_table is None):
            raise ValueError("material_ids and material_table must be supplied together")
        if material_table is not None:
            if len(material_table) < 1:
                raise ValueError("material_table cannot be empty")
            if any(material.n_layers != n_layers for material in material_table):
                raise ValueError("all material-table entries must use the same number of thermal layers")
        canopy_properties.validate()
        if water_capacity.shape != geometry.area.shape:
            raise ValueError("water_capacity shape must match geometry area")
        if water_capacity.device != geometry.area.device:
            raise ValueError("water_capacity and geometry must share one device")
        if not bool(torch.isfinite(water_capacity).all()) or bool((water_capacity < 0.0).any()):
            raise ValueError("water_capacity must be finite and nonnegative")

        self.geometry = geometry
        self.materials = tuple(solid_materials)
        self.canopy = canopy_properties
        self.device = geometry.area.device
        self.dtype = geometry.area.dtype
        self.water_capacity = water_capacity.to(dtype=self.dtype)
        self.config = config
        self.n_layers = n_layers
        self._build_property_tensors(material_ids, material_table)

    def _build_property_tensors(
        self,
        material_ids: Optional[torch.Tensor] = None,
        material_table: Optional[Sequence[MaterialProperties]] = None,
    ) -> None:
        dev = self.device
        dtype = self.dtype
        shape = (N_SOLID_FACETS, self.rows, self.cols, self.n_layers)
        attributes = (
            ("albedo", "albedo"),
            ("emissivity", "emissivity"),
            ("max_conductance", "max_conductance"),
            ("g1", "g1_radiation"),
            ("g3", "g3_vpd"),
            ("g4", "g4_temp"),
            ("t_opt", "temp_optimal"),
        )
        if material_ids is None or material_table is None:
            thickness = torch.empty(shape, dtype=dtype, device=dev)
            conductivity = torch.empty_like(thickness)
            capacity = torch.empty_like(thickness)
            for index, material in enumerate(self.materials):
                thickness[index] = torch.tensor(material.thickness, dtype=dtype, device=dev)
                conductivity[index] = torch.tensor(material.conductivity, dtype=dtype, device=dev)
                capacity[index] = torch.tensor(material.heat_capacity, dtype=dtype, device=dev)
            solid_properties = {
                name: torch.tensor(
                    [getattr(material, attribute) for material in self.materials],
                    dtype=dtype,
                    device=dev,
                ).view(N_SOLID_FACETS, 1, 1).expand(N_SOLID_FACETS, self.rows, self.cols)
                for name, attribute in attributes
            }
            self.material_ids = None
        else:
            if tuple(material_ids.shape) != (N_SOLID_FACETS, self.rows, self.cols):
                raise ValueError(
                    "material_ids shape must be (6, rows, cols); "
                    f"got {tuple(material_ids.shape)}"
                )
            if material_ids.dtype not in {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }:
                raise ValueError("material_ids must use an integer dtype")
            ids = material_ids.to(device=dev, dtype=torch.long)
            if bool(((ids < 0) | (ids >= len(material_table))).any()):
                raise ValueError(f"material_ids must be between 0 and {len(material_table) - 1}")
            thickness_lut = torch.tensor(
                [material.thickness for material in material_table], dtype=dtype, device=dev
            )
            conductivity_lut = torch.tensor(
                [material.conductivity for material in material_table], dtype=dtype, device=dev
            )
            capacity_lut = torch.tensor(
                [material.heat_capacity for material in material_table], dtype=dtype, device=dev
            )
            thickness = thickness_lut[ids]
            conductivity = conductivity_lut[ids]
            capacity = capacity_lut[ids]
            solid_properties = {}
            for name, attribute in attributes:
                lookup = torch.tensor(
                    [getattr(material, attribute) for material in material_table],
                    dtype=dtype,
                    device=dev,
                )
                solid_properties[name] = lookup[ids]
            self.material_ids = ids

        self.thickness = thickness
        self.conductivity = conductivity
        self.heat_capacity = capacity
        canopy_properties = {
            "albedo": self.canopy.albedo_leaf,
            "emissivity": self.canopy.emissivity_leaf,
            "max_conductance": self.canopy.max_stomatal_conductance,
            "g1": self.canopy.g1,
            "g3": self.canopy.g3,
            "g4": self.canopy.g4,
            "t_opt": self.canopy.temp_optimal,
        }
        for name, canopy_value in canopy_properties.items():
            canopy_grid = torch.full(
                (1, self.rows, self.cols), canopy_value, dtype=dtype, device=dev
            )
            setattr(self, name, torch.cat((solid_properties[name], canopy_grid), dim=0))

    def initialize_state(
        self,
        air_temperature: torch.Tensor | float,
        *,
        vapor_pressure_kpa: torch.Tensor | float = 1.5,
        pressure_kpa: float = 101.3,
        initial_water_fraction: float = 0.0,
    ) -> UrbanEBState:
        if not 0.0 <= initial_water_fraction <= 1.0:
            raise ValueError("initial_water_fraction must be between 0 and 1")
        air = self._grid(air_temperature, "air_temperature", positive=True)
        vapor = self._grid(vapor_pressure_kpa, "vapor_pressure_kpa", nonnegative=True)
        if not math.isfinite(pressure_kpa) or pressure_kpa <= 0.0:
            raise ValueError("pressure_kpa must be finite and positive")
        if bool((vapor >= pressure_kpa).any()):
            raise ValueError("vapor_pressure_kpa must be below pressure_kpa")
        surface = air.unsqueeze(0).expand(N_FACETS, -1, -1).clone()
        layers = air.view(1, self.rows, self.cols, 1).expand(
            N_SOLID_FACETS, -1, -1, self.n_layers
        ).clone()
        water = self.water_capacity * initial_water_fraction
        humidity = self._specific_humidity_from_vapor(vapor, pressure_kpa)
        return UrbanEBState(surface, layers, air.clone(), humidity, water)

    @staticmethod
    def _specific_humidity_from_vapor(vapor_pressure: torch.Tensor, pressure_kpa: float):
        return EPSILON_RATIO * vapor_pressure / torch.clamp(
            pressure_kpa - (1.0 - EPSILON_RATIO) * vapor_pressure,
            min=1.0e-9,
        )

    @staticmethod
    def _vapor_from_specific_humidity(specific_humidity: torch.Tensor, pressure_kpa: float):
        return pressure_kpa * specific_humidity / torch.clamp(
            EPSILON_RATIO + (1.0 - EPSILON_RATIO) * specific_humidity,
            min=1.0e-9,
        )

    def _grid(self, value, name: str, *, nonnegative: bool = False, positive: bool = False):
        tensor = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        if tensor.ndim == 0:
            tensor = tensor.expand(self.rows, self.cols)
        if tensor.shape != (self.rows, self.cols):
            raise ValueError(f"{name} must be scalar or have shape {(self.rows, self.cols)}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} contains non-finite values")
        if positive and bool((tensor <= 0.0).any()):
            raise ValueError(f"{name} must be positive")
        if nonnegative and bool((tensor < 0.0).any()):
            raise ValueError(f"{name} must be nonnegative")
        return tensor

    def _validate_forcing(self, forcing: UrbanForcing):
        air = self._grid(forcing.air_temperature, "air_temperature", positive=True)
        vapor = self._grid(forcing.vapor_pressure_kpa, "vapor_pressure_kpa", nonnegative=True)
        wind = self._grid(forcing.wind_speed, "wind_speed", nonnegative=True)
        sky_lw = self._grid(forcing.sky_longwave, "sky_longwave", nonnegative=True)
        rain = self._grid(forcing.precipitation_rate, "precipitation_rate", nonnegative=True)
        anthropogenic = self._grid(forcing.anthropogenic_heat, "anthropogenic_heat")
        if forcing.shortwave_irradiance.shape != self.geometry.area.shape:
            raise ValueError("shortwave_irradiance shape must match geometry area")
        shortwave = forcing.shortwave_irradiance.to(device=self.device, dtype=self.dtype)
        if not bool(torch.isfinite(shortwave).all()) or bool((shortwave < 0.0).any()):
            raise ValueError("shortwave_irradiance must be finite and nonnegative")
        if not math.isfinite(forcing.pressure_kpa) or forcing.pressure_kpa <= 0.0:
            raise ValueError("pressure_kpa must be finite and positive")
        if bool((vapor >= forcing.pressure_kpa).any()):
            raise ValueError("vapor_pressure_kpa must be below pressure_kpa")
        capture = forcing.rain_capture_fraction
        if capture is None:
            capture = torch.zeros_like(self.geometry.area)
            capture[GROUND] = 1.0
            capture[ROOF] = 1.0
        else:
            if capture.shape != self.geometry.area.shape:
                raise ValueError("rain_capture_fraction shape must match geometry area")
            capture = capture.to(device=self.device, dtype=self.dtype)
            if not bool(torch.isfinite(capture).all()) or bool(((capture < 0.0) | (capture > 1.0)).any()):
                raise ValueError("rain_capture_fraction must be between 0 and 1")
        plan_capture = (self.geometry.area * capture).sum(dim=0)
        if bool((plan_capture > 1.0 + 1.0e-5).any()):
            raise ValueError(
                "area-weighted rain_capture_fraction exceeds one; precipitation would be duplicated"
            )
        return air, vapor, wind, sky_lw, rain, anthropogenic, shortwave, capture

    def _view_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        area = self.geometry.area
        active = area > 0.0
        safe_area = torch.where(active, area, torch.ones_like(area))
        factors = self.geometry.exchange_area / safe_area.unsqueeze(1)
        sky = self.geometry.sky_view_area / safe_area
        return torch.where(active.unsqueeze(1), factors, torch.zeros_like(factors)), torch.where(
            active, sky, torch.zeros_like(sky)
        )

    def _radiosity(
        self,
        surface_temperature: torch.Tensor,
        shortwave_external: torch.Tensor,
        sky_longwave: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        factors, sky_factor = self._view_factors()
        f_batch = factors.permute(2, 3, 0, 1)
        identity = torch.eye(N_FACETS, dtype=self.dtype, device=self.device).expand(
            self.rows, self.cols, N_FACETS, N_FACETS
        )
        alpha = self.albedo.expand_as(surface_temperature)
        emissivity = self.emissivity.expand_as(surface_temperature)

        sw_matrix = identity - alpha.permute(1, 2, 0).unsqueeze(-1) * f_batch
        sw_rhs = (alpha * shortwave_external).permute(1, 2, 0).unsqueeze(-1)
        sw_j = torch.linalg.solve(sw_matrix, sw_rhs).squeeze(-1).permute(2, 0, 1)
        sw_in = shortwave_external + torch.einsum("ijrc,jrc->irc", factors, sw_j)
        sw_absorbed = (1.0 - alpha) * sw_in
        sw_escape = (self.geometry.sky_view_area * sw_j).sum(dim=0)

        reflected = 1.0 - emissivity
        lw_matrix = identity - reflected.permute(1, 2, 0).unsqueeze(-1) * f_batch
        blackbody = STEFAN_BOLTZMANN * surface_temperature.pow(4)
        lw_external = sky_factor * sky_longwave.unsqueeze(0)
        lw_rhs = (emissivity * blackbody + reflected * lw_external).permute(1, 2, 0).unsqueeze(-1)
        lw_j = torch.linalg.solve(lw_matrix, lw_rhs).squeeze(-1).permute(2, 0, 1)
        lw_in = lw_external + torch.einsum("ijrc,jrc->irc", factors, lw_j)
        lw_net = emissivity * (lw_in - blackbody)
        lw_to_sky = (self.geometry.sky_view_area * (lw_j - sky_longwave.unsqueeze(0))).sum(dim=0)
        active = self.geometry.area > 0.0
        return (
            torch.where(active, sw_absorbed, torch.zeros_like(sw_absorbed)),
            torch.where(active, lw_net, torch.zeros_like(lw_net)),
            sw_escape,
            lw_to_sky,
        )

    def _conduction_affine(self, state: UrbanEBState, air: torch.Tensor):
        equivalent = torch.empty(
            (N_SOLID_FACETS, self.rows, self.cols), dtype=self.dtype, device=self.device
        )
        conductance = torch.empty_like(equivalent)
        n_pixels = self.rows * self.cols
        for facet in range(N_SOLID_FACETS):
            old = state.layer_temperature[facet].reshape(n_pixels, self.n_layers)
            old_surface = state.surface_temperature[facet].reshape(n_pixels)
            if facet in (ROOF, WALL_NORTH, WALL_EAST, WALL_SOUTH, WALL_WEST):
                deep = torch.full_like(old_surface, self.config.interior_temperature)
            else:
                deep = torch.full_like(old_surface, self.config.ground_deep_temperature)
            thickness = self.thickness[facet].reshape(n_pixels, self.n_layers)
            conductivity = self.conductivity[facet].reshape(n_pixels, self.n_layers)
            heat_capacity = self.heat_capacity[facet].reshape(n_pixels, self.n_layers)
            reference = solve_conduction_substepped(
                old,
                old_surface,
                deep,
                thickness,
                conductivity,
                heat_capacity,
                self.config.dt,
                self.config.insulated_deep_boundary,
            ).reshape(self.rows, self.cols, self.n_layers)
            plus = solve_conduction_substepped(
                old,
                old_surface + 1.0,
                deep,
                thickness,
                conductivity,
                heat_capacity,
                self.config.dt,
                self.config.insulated_deep_boundary,
            ).reshape(self.rows, self.cols, self.n_layers)
            response = plus[:, :, 0] - reference[:, :, 0]
            one_minus = torch.clamp(1.0 - response, min=1.0e-6)
            intercept = reference[:, :, 0] - response * state.surface_temperature[facet]
            equivalent[facet] = intercept / one_minus
            interface = 2.0 * self.conductivity[facet, :, :, 0] / self.thickness[facet, :, :, 0]
            conductance[facet] = interface * one_minus
        return equivalent, conductance

    def _heat_transfer(
        self,
        temperature: torch.Tensor,
        air: torch.Tensor,
        canyon_air: torch.Tensor,
        wind: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros_like(temperature)
        target = canyon_air.unsqueeze(0).expand_as(temperature).clone()
        target[ROOF] = air
        h[GROUND] = calculate_httc_ground(
            wind, canyon_air, temperature[GROUND], 2.0, 0.03
        )
        h[ROOF] = calculate_httc_ground(wind, air, temperature[ROOF], 2.0, 0.1)
        wall_h = torch.clamp(
            AIR_DENSITY * AIR_SPECIFIC_HEAT * (11.8 + 4.2 * torch.clamp(wind, min=0.1)) / 1000.0 - 4.0,
            min=5.0,
            max=100.0,
        )
        h[WALL_NORTH:WALL_WEST + 1] = wall_h
        leaf_wind = torch.clamp(wind, min=0.1)
        boundary_resistance = 200.0 * torch.sqrt(
            torch.as_tensor(self.canopy.d_leaf, dtype=self.dtype, device=self.device) / leaf_wind
        )
        h[CANOPY] = AIR_DENSITY * AIR_SPECIFIC_HEAT / boundary_resistance
        return h, target

    def _latent_heat(
        self,
        temperature: torch.Tensor,
        air_by_facet: torch.Tensor,
        vapor_above: torch.Tensor,
        vapor_canyon: torch.Tensor,
        pressure_kpa: float,
        h: torch.Tensor,
        water: torch.Tensor,
        shortwave: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return wet-surface evaporation, not root-supplied transpiration.

        The Jarvis-style response limits surface conductance, while ``wetness``
        and the available-water cap restrict flux to the explicit interception
        store.  A root-zone water state is deliberately absent from this model.
        """
        if self.config.water_limited_evaporation:
            wetness = torch.where(
                self.water_capacity > 0.0,
                torch.clamp(
                    water / torch.clamp(self.water_capacity, min=1.0e-12),
                    0.0,
                    1.0,
                ),
                torch.zeros_like(water),
            )
        else:
            wetness = torch.where(
                self.max_conductance > 0.0,
                torch.ones_like(water),
                torch.zeros_like(water),
            )
        vapor = vapor_canyon.unsqueeze(0).expand_as(temperature).clone()
        vapor[ROOF] = vapor_above
        vpd = torch.clamp(e_sat_kpa(air_by_facet) - vapor, min=0.0)
        shortwave_positive = torch.clamp(shortwave, min=0.0)
        radiation_response = shortwave_positive / torch.clamp(
            shortwave_positive + self.g1, min=0.01
        )
        vapor_response = 1.0 / (1.0 + self.g3 * vpd)
        temperature_response = torch.clamp(
            1.0 - self.g4 * (air_by_facet - 273.15 - self.t_opt).pow(2),
            min=0.01,
        )
        conductance = (
            self.max_conductance * 0.001
            * radiation_response
            * vapor_response
            * temperature_response
            * wetness
        )
        conductance = torch.where(
            self.max_conductance <= 0.0,
            torch.zeros_like(conductance),
            torch.where(
                self.max_conductance > 900.0,
                self.max_conductance * 0.001 * wetness,
                conductance,
            ),
        )
        positive_conductance = conductance > 0.0
        surface_resistance = torch.where(
            positive_conductance,
            1.0 / torch.where(
                positive_conductance, conductance, torch.ones_like(conductance)
            ),
            torch.full_like(conductance, 1.0e30),
        )
        aerodynamic = AIR_DENSITY * AIR_SPECIFIC_HEAT / torch.clamp(h, min=1.0e-6)
        gamma = psychrometric_constant(pressure_kpa)
        coefficient = torch.where(
            positive_conductance,
            AIR_DENSITY * AIR_SPECIFIC_HEAT / (gamma * (aerodynamic + surface_resistance)),
            torch.zeros_like(surface_resistance),
        )
        vapor_gradient = e_sat_kpa(temperature) - vapor
        latent = torch.clamp(coefficient * vapor_gradient, min=0.0)
        slope = torch.where(
            latent > 0.0,
            coefficient * delta_e_sat(temperature),
            torch.zeros_like(latent),
        )
        if self.config.water_limited_evaporation:
            available_flux = water * LATENT_HEAT_VAP / self.config.dt
            latent = torch.minimum(latent, available_flux)
            slope = torch.where(latent < available_flux, slope, torch.zeros_like(slope))
        return latent, slope

    def _canyon_specific_humidity(
        self,
        previous: torch.Tensor,
        vapor_above: torch.Tensor,
        pressure_kpa: float,
        latent_heat: torch.Tensor,
        canyon_air_temperature: torch.Tensor,
        wind: torch.Tensor,
    ) -> torch.Tensor:
        area = self.geometry.area
        canyon_facets = torch.ones(N_FACETS, dtype=torch.bool, device=self.device)
        canyon_facets[ROOF] = False
        evaporation = (
            area[canyon_facets] * latent_heat[canyon_facets] / LATENT_HEAT_VAP
        ).sum(dim=0)
        q_above = self._specific_humidity_from_vapor(vapor_above, pressure_kpa)
        density = pressure_kpa * 1000.0 / torch.clamp(
            287.05 * canyon_air_temperature, min=1.0
        )
        storage_coefficient = density * self.config.canyon_height / self.config.dt
        exchange_velocity = torch.clamp(
            self.config.ventilation_coefficient * wind,
            min=self.config.minimum_exchange_velocity,
        )
        ventilation_mass = density * exchange_velocity
        humidity = (
            storage_coefficient * previous
            + evaporation
            + ventilation_mass * q_above
        ) / torch.clamp(storage_coefficient + ventilation_mass, min=1.0e-12)
        return torch.clamp(humidity, min=0.0, max=0.1)

    def _canyon_temperature(
        self,
        previous: torch.Tensor,
        air: torch.Tensor,
        temperature: torch.Tensor,
        h: torch.Tensor,
        wind: torch.Tensor,
        anthropogenic: torch.Tensor,
    ) -> torch.Tensor:
        area = self.geometry.area
        canyon_facets = torch.ones(N_FACETS, dtype=torch.bool, device=self.device)
        canyon_facets[ROOF] = False
        ah = area[canyon_facets] * h[canyon_facets]
        surface_conductance = ah.sum(dim=0)
        surface_source = (ah * temperature[canyon_facets]).sum(dim=0)
        air_capacity_dt = AIR_DENSITY * AIR_SPECIFIC_HEAT * self.config.canyon_height / self.config.dt
        exchange_velocity = torch.clamp(
            self.config.ventilation_coefficient * wind,
            min=self.config.minimum_exchange_velocity,
        )
        ventilation = AIR_DENSITY * AIR_SPECIFIC_HEAT * exchange_velocity
        denominator = air_capacity_dt + surface_conductance + ventilation
        return (
            air_capacity_dt * previous
            + surface_source
            + ventilation * air
            + anthropogenic
        ) / torch.clamp(denominator, min=1.0e-9)

    def _check_state(self, state: UrbanEBState) -> None:
        expected = {
            "surface_temperature": (N_FACETS, self.rows, self.cols),
            "layer_temperature": (N_SOLID_FACETS, self.rows, self.cols, self.n_layers),
            "canyon_air_temperature": (self.rows, self.cols),
            "canyon_specific_humidity": (self.rows, self.cols),
            "water_storage": (N_FACETS, self.rows, self.cols),
        }
        for name, shape in expected.items():
            value = getattr(state, name)
            if value.shape != shape:
                raise ValueError(f"{name} shape {tuple(value.shape)} does not match {shape}")
            if value.device != self.device:
                raise ValueError(f"{name} is on the wrong device")
            if value.dtype != self.dtype:
                raise ValueError(f"{name} has the wrong dtype")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains non-finite values")
        if bool((state.water_storage < 0.0).any()) or bool(
            (state.water_storage > self.water_capacity + 1.0e-6).any()
        ):
            raise ValueError("water_storage is outside its capacity")

    @torch.no_grad()
    def step(self, state: UrbanEBState, forcing: UrbanForcing) -> UrbanEBResult:
        """Advance one fully coupled timestep."""
        self._check_state(state)
        air, vapor, wind, sky_lw, rain, anthropogenic, shortwave, capture = self._validate_forcing(forcing)
        active = self.geometry.area > 0.0
        solved_temperature = active.clone()
        if not self.config.solve_wall_temperature:
            solved_temperature[WALL_NORTH:WALL_WEST + 1] = False
        equivalent, conduction_coefficient = self._conduction_affine(state, air)
        incoming_water = rain.unsqueeze(0) * capture * self.config.dt
        water_before_evaporation = state.water_storage + incoming_water
        rain_overflow = torch.clamp(water_before_evaporation - self.water_capacity, min=0.0)
        water_for_flux = torch.minimum(water_before_evaporation, self.water_capacity)
        temperature = state.surface_temperature.clone()
        canyon_air = state.canyon_air_temperature.clone()
        canyon_humidity = state.canyon_specific_humidity.clone()
        converged = False
        max_change = float("inf")
        max_residual = float("inf")
        max_humidity_change = float("inf")
        iterations = 0

        for iterations in range(1, self.config.max_coupling_iterations + 1):
            previous_temperature = temperature.clone()
            previous_canyon = canyon_air.clone()
            previous_humidity = canyon_humidity.clone()
            sw_absorbed, lw_net, sw_escape, lw_to_sky = self._radiosity(
                temperature, shortwave, sky_lw
            )
            net_radiation = sw_absorbed + lw_net
            h, target_air = self._heat_transfer(temperature, air, canyon_air, wind)
            sensible = h * (temperature - target_air)
            canyon_vapor = self._vapor_from_specific_humidity(
                canyon_humidity, forcing.pressure_kpa
            )
            latent, latent_slope = self._latent_heat(
                temperature, target_air, vapor, canyon_vapor, forcing.pressure_kpa, h,
                water_for_flux, shortwave,
            )
            storage = torch.zeros_like(temperature)
            storage[:N_SOLID_FACETS] = conduction_coefficient * (
                temperature[:N_SOLID_FACETS] - equivalent
            )
            residual = net_radiation - sensible - latent - storage
            diagonal = h + latent_slope + 4.0 * self.emissivity * STEFAN_BOLTZMANN * temperature.pow(3)
            diagonal[:N_SOLID_FACETS] += conduction_coefficient
            update = torch.clamp(
                residual / torch.clamp(diagonal, min=1.0e-6),
                -self.config.max_temperature_step,
                self.config.max_temperature_step,
            )
            updated_temperature = torch.clamp(
                temperature + self.config.relaxation * update,
                self.config.minimum_temperature,
                self.config.maximum_temperature,
            )
            temperature = torch.where(
                solved_temperature,
                updated_temperature,
                torch.where(active, temperature, air.unsqueeze(0)),
            )
            if self.config.solve_canyon_temperature:
                canyon_candidate = self._canyon_temperature(
                    state.canyon_air_temperature,
                    air,
                    temperature,
                    h,
                    wind,
                    anthropogenic,
                )
            else:
                canyon_candidate = air
            canyon_air = (
                (1.0 - self.config.relaxation) * canyon_air
                + self.config.relaxation * canyon_candidate
            )
            if self.config.solve_canyon_humidity:
                humidity_candidate = self._canyon_specific_humidity(
                    state.canyon_specific_humidity,
                    vapor,
                    forcing.pressure_kpa,
                    latent,
                    canyon_air,
                    wind,
                )
            else:
                humidity_candidate = self._specific_humidity_from_vapor(
                    vapor, forcing.pressure_kpa
                )
            canyon_humidity = (
                (1.0 - self.config.relaxation) * canyon_humidity
                + self.config.relaxation * humidity_candidate
            )
            temperature_change = torch.where(
                solved_temperature,
                (temperature - previous_temperature).abs(),
                torch.zeros_like(temperature),
            ).max()
            canyon_change = (canyon_air - previous_canyon).abs().max()
            max_change = float(torch.maximum(temperature_change, canyon_change).item())
            max_humidity_change = float((canyon_humidity - previous_humidity).abs().max().item())
            max_residual = float(
                torch.where(
                    solved_temperature, residual.abs(), torch.zeros_like(residual)
                ).max().item()
            )
            if (
                math.isfinite(max_change)
                and math.isfinite(max_residual)
                and math.isfinite(max_humidity_change)
                and max_change <= self.config.temperature_tolerance
                and max_humidity_change <= self.config.specific_humidity_tolerance
                and max_residual <= self.config.residual_tolerance
            ):
                converged = True
                break

        sw_absorbed, lw_net, sw_escape, lw_to_sky = self._radiosity(temperature, shortwave, sky_lw)
        net_radiation = sw_absorbed + lw_net
        h, target_air = self._heat_transfer(temperature, air, canyon_air, wind)
        sensible = h * (temperature - target_air)
        canyon_vapor = self._vapor_from_specific_humidity(
            canyon_humidity, forcing.pressure_kpa
        )
        latent, _ = self._latent_heat(
            temperature, target_air, vapor, canyon_vapor, forcing.pressure_kpa, h,
            water_for_flux, shortwave,
        )
        storage = torch.zeros_like(temperature)
        storage[:N_SOLID_FACETS] = conduction_coefficient * (
            temperature[:N_SOLID_FACETS] - equivalent
        )
        residual = net_radiation - sensible - latent - storage
        max_residual = float(
            torch.where(
                solved_temperature, residual.abs(), torch.zeros_like(residual)
            ).max().item()
        )
        converged = converged and max_residual <= self.config.residual_tolerance
        if not bool(torch.isfinite(temperature).all()) or not bool(torch.isfinite(residual).all()):
            converged = False
        if self.config.strict_convergence and not converged:
            raise RuntimeError(
                "coupled urban energy balance did not converge: "
                f"iterations={iterations}, max_dT={max_change:.4g} K, "
                f"max_dq={max_humidity_change:.4g} kg kg-1, "
                f"max_residual={max_residual:.4g} W m-2"
            )

        new_layers = torch.empty_like(state.layer_temperature)
        n_pixels = self.rows * self.cols
        for facet in range(N_SOLID_FACETS):
            old = state.layer_temperature[facet].reshape(n_pixels, self.n_layers)
            surface = temperature[facet].reshape(n_pixels)
            deep = (
                torch.full_like(surface, self.config.interior_temperature)
                if facet != GROUND
                else torch.full_like(surface, self.config.ground_deep_temperature)
            )
            new_layers[facet] = solve_conduction_substepped(
                old,
                surface,
                deep,
                self.thickness[facet].reshape(n_pixels, self.n_layers),
                self.conductivity[facet].reshape(n_pixels, self.n_layers),
                self.heat_capacity[facet].reshape(n_pixels, self.n_layers),
                self.config.dt,
                self.config.insulated_deep_boundary,
            ).reshape(self.rows, self.cols, self.n_layers)

        evaporated_water = latent * self.config.dt / LATENT_HEAT_VAP
        if self.config.water_limited_evaporation:
            water = torch.clamp(water_for_flux - evaporated_water, min=0.0)
        else:
            water = water_for_flux
        water = torch.minimum(water, self.water_capacity)
        new_state = UrbanEBState(temperature, new_layers, canyon_air, canyon_humidity, water)
        return UrbanEBResult(
            state=new_state,
            net_radiation=net_radiation,
            sensible_heat=sensible,
            latent_heat=latent,
            storage_heat=storage,
            residual=residual,
            shortwave_absorbed=sw_absorbed,
            longwave_net=lw_net,
            iterations=iterations,
            converged=converged,
            max_temperature_change=max_change,
            max_specific_humidity_change=max_humidity_change,
            max_energy_residual=max_residual,
            water_drainage=rain_overflow,
            shortwave_escape=sw_escape,
            longwave_to_sky=lw_to_sky,
        )

    @torch.no_grad()
    def spin_up(
        self,
        forcings: Sequence[UrbanForcing],
        initial_state: Optional[UrbanEBState] = None,
        *,
        initial_water_fraction: float = 0.0,
    ) -> SpinupResult:
        """Repeat a forcing cycle until thermal and moisture states are periodic."""
        if len(forcings) < 1:
            raise ValueError("spin-up requires at least one forcing timestep")
        state = initial_state.clone() if initial_state is not None else self.initialize_state(
            forcings[0].air_temperature,
            vapor_pressure_kpa=forcings[0].vapor_pressure_kpa,
            pressure_kpa=forcings[0].pressure_kpa,
            initial_water_fraction=initial_water_fraction,
        )
        self._check_state(state)
        temperature_drift = float("inf")
        moisture_drift = float("inf")
        humidity_drift = float("inf")
        converged = False
        cycles = 0
        active = self.geometry.area > 0.0
        for cycles in range(1, self.config.spinup_max_cycles + 1):
            start = state.clone()
            for forcing in forcings:
                state = self.step(state, forcing).state
            surface_drift = torch.where(
                active,
                (state.surface_temperature - start.surface_temperature).abs(),
                torch.zeros_like(state.surface_temperature),
            ).max()
            layer_drift = (state.layer_temperature - start.layer_temperature).abs().max()
            canyon_drift = (state.canyon_air_temperature - start.canyon_air_temperature).abs().max()
            temperature_drift = float(torch.maximum(surface_drift, torch.maximum(layer_drift, canyon_drift)).item())
            moisture_drift = float((state.water_storage - start.water_storage).abs().max().item())
            humidity_drift = float(
                (state.canyon_specific_humidity - start.canyon_specific_humidity).abs().max().item()
            )
            if (
                cycles >= self.config.spinup_min_cycles
                and temperature_drift <= self.config.spinup_temperature_tolerance
                and moisture_drift <= self.config.spinup_moisture_tolerance
                and humidity_drift <= self.config.spinup_specific_humidity_tolerance
            ):
                converged = True
                break
        if self.config.strict_convergence and not converged:
            raise RuntimeError(
                "urban energy-balance spin-up did not reach a periodic state: "
                f"cycles={cycles}, temperature_drift={temperature_drift:.4g} K, "
                f"moisture_drift={moisture_drift:.4g} kg m-2, "
                f"humidity_drift={humidity_drift:.4g} kg kg-1"
            )
        return SpinupResult(
            state, cycles, converged, temperature_drift, moisture_drift, humidity_drift
        )

    @torch.no_grad()
    def run(
        self,
        forcings: Sequence[UrbanForcing],
        *,
        state: Optional[UrbanEBState] = None,
        spin_up: bool = True,
        initial_water_fraction: float = 0.0,
    ) -> list[UrbanEBResult]:
        """Optionally spin up, then return one result for every forcing step."""
        if len(forcings) < 1:
            return []
        if spin_up:
            state = self.spin_up(
                forcings,
                state,
                initial_water_fraction=initial_water_fraction,
            ).state
        elif state is None:
            state = self.initialize_state(
                forcings[0].air_temperature,
                vapor_pressure_kpa=forcings[0].vapor_pressure_kpa,
                pressure_kpa=forcings[0].pressure_kpa,
                initial_water_fraction=initial_water_fraction,
            )
        results = []
        for forcing in forcings:
            result = self.step(state, forcing)
            results.append(result)
            state = result.state
        return results


__all__ = [
    "CANOPY",
    "FACET_NAMES",
    "GROUND",
    "N_FACETS",
    "ROOF",
    "WALL_EAST",
    "WALL_NORTH",
    "WALL_SOUTH",
    "WALL_WEST",
    "CoupledUrbanEBConfig",
    "CoupledUrbanEnergyBalance",
    "SpinupResult",
    "UrbanEBResult",
    "UrbanEBState",
    "UrbanFacetGeometry",
    "UrbanForcing",
]
