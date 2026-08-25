# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU-accelerated energy balance solver for urban surface temperature.

This package provides a PyTorch-based energy balance solver. Numerical fields
remain as device tensors; convergence checks synchronize with the host.

Modules:
    materials  — MaterialProperties dataclass with 9 surface presets
    config     — EnergyBalanceConfig, CanyonAirTempConfig, CanopyProperties
    physics    — Vectorized physics kernels (e_sat, httc, NR, Thomas, etc.)
    net_radiation — GPU net radiation calculation
    solver     — Main EBSolver class (ground + roof)
    canopy     — Big-leaf and two-big-leaf canopy energy balance
    canyon     — VTUF-3D-inspired scalar canyon air-temperature diagnostic

Example:
    from utherm.energy_balance import (
        MaterialProperties, EnergyBalanceConfig, CanopyProperties,
        EBSolver, solve_canopy_eb, solve_canyon_air_temperature,
    )
"""

from .materials import MaterialProperties
from .config import EnergyBalanceConfig, CanyonAirTempConfig, CanopyProperties
from .physics import (
    SurfaceSolveResult,
    e_sat_kpa,
    delta_e_sat,
    psychrometric_constant,
    jarvis_stewart_conductance,
    calculate_httc_ground,
    calculate_httc_wall,
    g_coupling_beta,
    solve_tsfc_newton_raphson,
    solve_conduction_thomas_batched,
    solve_conduction_substepped,
    STEFAN_BOLTZMANN,
    AIR_DENSITY,
    AIR_SPECIFIC_HEAT,
    VON_KARMAN,
    GRAVITY,
)
from .net_radiation import calculate_net_radiation
from .solver import EBSolver
from .canopy import solve_canopy_eb, solve_canopy_eb_two_leaf
from .canyon import solve_canyon_air_temperature, solve_canyon_air_temperature_coupled
from .coupled import (
    CANOPY,
    FACET_NAMES,
    GROUND,
    N_FACETS,
    ROOF,
    WALL_EAST,
    WALL_NORTH,
    WALL_SOUTH,
    WALL_WEST,
    CoupledUrbanEBConfig,
    CoupledUrbanEnergyBalance,
    SpinupResult,
    UrbanEBResult,
    UrbanEBState,
    UrbanFacetGeometry,
    UrbanForcing,
)

__all__ = [
    # Data classes
    'MaterialProperties',
    'EnergyBalanceConfig',
    'CanyonAirTempConfig',
    'CanopyProperties',
    'SurfaceSolveResult',
    # Solvers
    'EBSolver',
    'solve_canopy_eb',
    'solve_canopy_eb_two_leaf',
    'solve_canyon_air_temperature',
    'solve_canyon_air_temperature_coupled',
    'CoupledUrbanEBConfig',
    'CoupledUrbanEnergyBalance',
    'SpinupResult',
    'UrbanEBResult',
    'UrbanEBState',
    'UrbanFacetGeometry',
    'UrbanForcing',
    'GROUND',
    'ROOF',
    'WALL_NORTH',
    'WALL_EAST',
    'WALL_SOUTH',
    'WALL_WEST',
    'CANOPY',
    'N_FACETS',
    'FACET_NAMES',
    # Net radiation
    'calculate_net_radiation',
    # Physics kernels
    'e_sat_kpa',
    'delta_e_sat',
    'psychrometric_constant',
    'jarvis_stewart_conductance',
    'calculate_httc_ground',
    'calculate_httc_wall',
    'g_coupling_beta',
    'solve_tsfc_newton_raphson',
    'solve_conduction_thomas_batched',
    'solve_conduction_substepped',
    # Constants
    'STEFAN_BOLTZMANN',
    'AIR_DENSITY',
    'AIR_SPECIFIC_HEAT',
    'VON_KARMAN',
    'GRAVITY',
]
