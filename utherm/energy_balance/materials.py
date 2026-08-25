# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Material thermal properties for multi-layer surface energy balance.

Port of Rust MaterialProperties struct with all 9 presets.
Pure Python dataclass — no GPU dependency.
"""

from dataclasses import dataclass, field
import math
from typing import List


@dataclass
class MaterialProperties:
    """Material thermal properties for a multi-layer surface.

    Each material has 4 conduction layers (top to bottom) and Jarvis-Stewart
    stomatal conductance parameters for latent heat.
    """
    n_layers: int = 4
    thickness: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.10, 0.20])
    conductivity: List[float] = field(default_factory=lambda: [0.75, 0.75, 0.75, 0.75])
    heat_capacity: List[float] = field(default_factory=lambda: [1.9e6, 1.9e6, 1.9e6, 1.9e6])
    albedo: float = 0.2
    emissivity: float = 0.95
    max_conductance: float = 0.0  # mm/s, 0 = impervious
    g1_radiation: float = 100.0   # W/m²
    g3_vpd: float = 0.5           # kPa⁻¹
    g4_temp: float = 0.0016       # K⁻²
    temp_optimal: float = 25.0    # °C

    def __post_init__(self) -> None:
        if not isinstance(self.n_layers, int) or isinstance(self.n_layers, bool) or self.n_layers < 1:
            raise ValueError("n_layers must be a positive integer")
        layers = {
            "thickness": self.thickness,
            "conductivity": self.conductivity,
            "heat_capacity": self.heat_capacity,
        }
        for name, values in layers.items():
            if len(values) != self.n_layers:
                raise ValueError(f"{name} must contain exactly n_layers values")
            if any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError(f"{name} values must be finite and positive")
        for name, value in (("albedo", self.albedo), ("emissivity", self.emissivity)):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name, value in (
            ("max_conductance", self.max_conductance),
            ("g1_radiation", self.g1_radiation),
            ("g3_vpd", self.g3_vpd),
            ("g4_temp", self.g4_temp),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.temp_optimal):
            raise ValueError("temp_optimal must be finite")

    @staticmethod
    def asphalt():
        return MaterialProperties(
            thickness=[0.02, 0.05, 0.10, 0.20],
            conductivity=[0.75, 0.75, 0.75, 0.75],
            heat_capacity=[1.9e6, 1.9e6, 1.9e6, 1.9e6],
            albedo=0.12, emissivity=0.95,
            max_conductance=0.0, g1_radiation=0.0,
            g3_vpd=0.0, g4_temp=0.0, temp_optimal=25.0,
        )

    @staticmethod
    def concrete():
        return MaterialProperties(
            thickness=[0.02, 0.05, 0.10, 0.20],
            conductivity=[1.4, 1.4, 1.4, 1.4],
            heat_capacity=[2.0e6, 2.0e6, 2.0e6, 2.0e6],
            albedo=0.25, emissivity=0.92,
            max_conductance=0.0, g1_radiation=0.0,
            g3_vpd=0.0, g4_temp=0.0, temp_optimal=25.0,
        )

    @staticmethod
    def brick():
        return MaterialProperties(
            thickness=[0.02, 0.05, 0.10, 0.20],
            conductivity=[0.84, 0.84, 0.84, 0.84],
            heat_capacity=[1.6e6, 1.6e6, 1.6e6, 1.6e6],
            albedo=0.30, emissivity=0.90,
            max_conductance=0.0, g1_radiation=0.0,
            g3_vpd=0.0, g4_temp=0.0, temp_optimal=25.0,
        )

    @staticmethod
    def soil():
        return MaterialProperties(
            thickness=[0.02, 0.05, 0.10, 0.20],
            conductivity=[1.0, 1.0, 1.0, 1.0],
            heat_capacity=[1.5e6, 1.5e6, 1.5e6, 1.5e6],
            albedo=0.25, emissivity=0.94,
            max_conductance=5.0, g1_radiation=80.0,
            g3_vpd=0.4, g4_temp=0.001, temp_optimal=25.0,
        )

    @staticmethod
    def grass():
        return MaterialProperties(
            thickness=[0.02, 0.05, 0.10, 0.20],
            conductivity=[0.25, 0.25, 0.25, 0.25],
            heat_capacity=[1.0e6, 1.0e6, 1.0e6, 1.0e6],
            albedo=0.25, emissivity=0.95,
            max_conductance=12.0, g1_radiation=100.0,
            g3_vpd=0.5, g4_temp=0.0016, temp_optimal=25.0,
        )

    @staticmethod
    def dry_soil():
        return MaterialProperties(
            thickness=[0.02, 0.05, 0.10, 0.20],
            conductivity=[0.35, 0.35, 0.35, 0.35],
            heat_capacity=[1.3e6, 1.3e6, 1.3e6, 1.3e6],
            albedo=0.25, emissivity=0.94,
            max_conductance=1.5, g1_radiation=50.0,
            g3_vpd=0.3, g4_temp=0.001, temp_optimal=25.0,
        )

    @staticmethod
    def water():
        return MaterialProperties(
            thickness=[0.02, 0.05, 0.10, 0.20],
            conductivity=[0.6, 0.6, 0.6, 0.6],
            heat_capacity=[4.18e6, 4.18e6, 4.18e6, 4.18e6],
            albedo=0.08, emissivity=0.98,
            max_conductance=999.0, g1_radiation=0.0,
            g3_vpd=0.0, g4_temp=0.0, temp_optimal=25.0,
        )

    @staticmethod
    def roof():
        """Flat roof: waterproofing membrane, concrete slab, XPS insulation, gypsum ceiling."""
        return MaterialProperties(
            thickness=[0.01, 0.05, 0.10, 0.016],
            conductivity=[0.17, 1.4, 0.04, 0.16],
            heat_capacity=[1.5e6, 2.0e6, 0.03e6, 0.87e6],
            albedo=0.25, emissivity=0.92,
            max_conductance=0.0, g1_radiation=0.0,
            g3_vpd=0.0, g4_temp=0.0, temp_optimal=25.0,
        )

    @staticmethod
    def green_roof():
        """Green roof: soil substrate, drainage layer, insulation, concrete slab."""
        return MaterialProperties(
            thickness=[0.05, 0.05, 0.05, 0.15],
            conductivity=[0.5, 0.15, 0.04, 1.4],
            heat_capacity=[1.5e6, 0.40e6, 0.03e6, 2.0e6],
            albedo=0.20, emissivity=0.95,
            max_conductance=15.0, g1_radiation=100.0,
            g3_vpd=0.5, g4_temp=0.0016, temp_optimal=25.0,
        )
