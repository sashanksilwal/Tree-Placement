# Copyright (C) 2025-2026 Sashank Silwal
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared fixtures for energy balance tests.

Adds --device CLI option: pytest --device=cuda to run on GPU.
Default is CPU so tests work everywhere including CI.
"""

import pytest
import torch

from utherm.energy_balance import (
    MaterialProperties,
    EnergyBalanceConfig,
    CanyonAirTempConfig,
    CanopyProperties,
)


# ── CLI option ──────────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        help="Torch device to run tests on: cpu (default) or cuda",
    )


# ── Device fixture ──────────────────────────────────────────────

@pytest.fixture
def device(request):
    """Torch device from --device CLI flag, with CUDA availability check."""
    name = request.config.getoption("--device")
    if name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(name)


# ── Small 4x4 grid helpers ─────────────────────────────────────

ROWS, COLS = 4, 4


@pytest.fixture
def grid_shape():
    return (ROWS, COLS)


@pytest.fixture
def ones_grid(device):
    """4x4 grid of ones (float32)."""
    return torch.ones(ROWS, COLS, dtype=torch.float32, device=device)


@pytest.fixture
def zeros_grid(device):
    """4x4 grid of zeros (float32)."""
    return torch.zeros(ROWS, COLS, dtype=torch.float32, device=device)


@pytest.fixture
def buildings_all_ground(device):
    """UMEP building mask: all ground (value=1)."""
    return torch.ones(ROWS, COLS, dtype=torch.float32, device=device)


@pytest.fixture
def buildings_mixed(device):
    """UMEP building mask: top-left 2x2 are buildings (0), rest ground (1)."""
    b = torch.ones(ROWS, COLS, dtype=torch.float32, device=device)
    b[:2, :2] = 0.0
    return b


# ── Standard meteorological conditions ─────────────────────────

@pytest.fixture
def summer_meteo():
    """Typical clear-sky summer midday conditions."""
    return {
        "ta_c": 25.0,         # Air temperature (C)
        "ws": 2.0,            # Wind speed (m/s)
        "rh": 50.0,           # Relative humidity (%)
        "ea_hpa": 15.8,       # ~50% RH at 25C
        "pressure_kpa": 101.325,
        "kdown": 500.0,       # Shortwave (W/m2)
        "rad_i": 400.0,       # Direct beam
        "rad_d": 100.0,       # Diffuse
    }


# ── Material presets ────────────────────────────────────────────

@pytest.fixture
def asphalt():
    return MaterialProperties.asphalt()


@pytest.fixture
def grass():
    return MaterialProperties.grass()


@pytest.fixture
def green_roof_mat():
    return MaterialProperties.green_roof()


@pytest.fixture
def roof_mat():
    return MaterialProperties.roof()


# ── Config presets ──────────────────────────────────────────────

@pytest.fixture
def eb_config():
    return EnergyBalanceConfig()


@pytest.fixture
def canyon_config():
    return CanyonAirTempConfig()


@pytest.fixture
def canopy_props():
    return CanopyProperties.deciduous()


# ── SVF grids (uniform open sky) ───────────────────────────────

@pytest.fixture
def svf_open(device):
    """SVF = 1 everywhere (open sky)."""
    return torch.ones(ROWS, COLS, dtype=torch.float32, device=device)
