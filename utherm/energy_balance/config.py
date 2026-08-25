# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Configuration dataclasses for energy balance solver.

Port of Rust EnergyBalanceConfig, CanyonAirTempConfig, CanopyProperties.
Pure Python dataclasses — no GPU dependency.
"""

from dataclasses import dataclass
import math


@dataclass
class EnergyBalanceConfig:
    """Configuration for the energy balance solver."""
    convergence_threshold: float = 0.001
    max_iterations: int = 40
    surface_residual_tolerance: float = 0.1  # W/m2
    t_deep: float = 288.15       # Deep ground temperature (K)
    z_ref: float = 2.0           # Reference height (m)
    z0: float = 0.01             # Roughness length (m)
    dt: float = 3600.0           # Timestep (s)
    roof_t_deep: float = None    # None → air temp; set to e.g. 295.15 for conditioned
    roof_wetness: float = None   # 0–1, None → 1.0 (well-watered)
    ground_wetness: float = None # 0–1, None → 1.0
    z_wind: float = None         # Wind measurement height (m); None → no correction

    def __post_init__(self) -> None:
        positive = {
            "convergence_threshold": self.convergence_threshold,
            "surface_residual_tolerance": self.surface_residual_tolerance,
            "t_deep": self.t_deep,
            "z_ref": self.z_ref,
            "z0": self.z0,
            "dt": self.dt,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not isinstance(self.max_iterations, int)
            or isinstance(self.max_iterations, bool)
            or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")
        if self.z0 >= self.z_ref:
            raise ValueError("z0 must be smaller than z_ref")
        if self.roof_t_deep is not None and (
            not math.isfinite(self.roof_t_deep) or self.roof_t_deep <= 0
        ):
            raise ValueError("roof_t_deep must be finite and positive")
        for name, value in (("roof_wetness", self.roof_wetness), ("ground_wetness", self.ground_wetness)):
            if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be between 0 and 1")
        if self.z_wind is not None and (not math.isfinite(self.z_wind) or self.z_wind <= 0):
            raise ValueError("z_wind must be finite and positive")


@dataclass
class CanyonAirTempConfig:
    """Configuration for the VTUF-3D-inspired canyon box diagnostic."""
    z_h: float = 10.0             # Mean building height (m)
    canyon_air_height: float = 6.0 # Effective canyon air column height (m)
    z_ref: float = 20.0           # Reference height for above-canyon exchange (m)
    z0_roof: float = 0.1          # Roughness length at canyon top (m)
    dt: float = 3600.0            # Outer timestep (s)
    max_substeps: int = 100
    max_dtcan: float = 0.1        # Max delta-Tcan per substep (K)

    def __post_init__(self) -> None:
        for name in ("z_h", "canyon_air_height", "z_ref", "z0_roof", "dt", "max_dtcan"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not isinstance(self.max_substeps, int)
            or isinstance(self.max_substeps, bool)
            or self.max_substeps < 1
        ):
            raise ValueError("max_substeps must be a positive integer")
        if self.z0_roof >= self.z_ref:
            raise ValueError("z0_roof must be smaller than z_ref")


@dataclass
class CanopyProperties:
    """Canopy (leaf) properties for big-leaf energy balance model."""
    albedo_leaf: float = 0.20
    emissivity_leaf: float = 0.98
    d_leaf: float = 0.05          # Characteristic leaf width (m)
    max_stomatal_conductance: float = 10.0  # mm/s
    g1: float = 100.0
    g3: float = 0.5
    g4: float = 0.0016
    temp_optimal: float = 25.0    # °C
    clumping_factor: float = 0.7
    wind_extinction: float = 0.5
    allow_dew: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in ("albedo_leaf", "emissivity_leaf"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if not math.isfinite(self.d_leaf) or self.d_leaf <= 0:
            raise ValueError("d_leaf must be finite and positive")
        for name in ("max_stomatal_conductance", "g1", "g3", "g4", "wind_extinction"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.temp_optimal):
            raise ValueError("temp_optimal must be finite")
        if not math.isfinite(self.clumping_factor) or not 0.0 < self.clumping_factor <= 1.0:
            raise ValueError("clumping_factor must be greater than 0 and no more than 1")
        if not isinstance(self.allow_dew, bool):
            raise ValueError("allow_dew must be boolean")

    @staticmethod
    def deciduous():
        return CanopyProperties(
            albedo_leaf=0.20, emissivity_leaf=0.98, d_leaf=0.05,
            max_stomatal_conductance=10.0,
            g1=100.0, g3=0.6, g4=0.0016, temp_optimal=25.0,
            clumping_factor=0.7, wind_extinction=0.5, allow_dew=False,
        )

    @staticmethod
    def evergreen():
        return CanopyProperties(
            albedo_leaf=0.12, emissivity_leaf=0.97, d_leaf=0.01,
            max_stomatal_conductance=6.0,
            g1=80.0, g3=0.4, g4=0.0012, temp_optimal=20.0,
            clumping_factor=0.6, wind_extinction=0.8, allow_dew=False,
        )
